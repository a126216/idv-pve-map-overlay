# -*- coding: utf-8 -*-
"""sift_matcher.py —— 用 SIFT 特征匹配在整张地图上定位「屏幕抓图区」。

架构参考自 Game-Map-Tracker（https://github.com/761696148/Game-Map-Tracker）：
    CLAHE 纹理增强 -> SIFT 特征 -> FLANN kNN -> Lowe's ratio 过滤
    -> RANSAC 单应矩阵 -> 反算中心点坐标

本文件是针对本项目重写的实现：那边是"已知是哪张图、只做单图跟点"，
这边要做的是"在 N 张候选地图里先认出是哪一张"，所以多了逐图打分与 Top-2 判定。

相比原先「手工裁门模板 + 128x128 MAE」的做法，这里的优势：
  * 尺度不变：屏幕抓图区与地图图片的分辨率/缩放不必一致；
  * 旋转不变：小地图带了旋转角度也能匹配；
  * 不需要任何手工裁剪 —— 直接把完整的 N 张地图建成索引即可。

性能上的三个关键点（都是实测出来的）：
  1. FLANN 索引必须"建一次、用多次"。用 knnMatch(query, train) 的双参形式时，
     OpenCV 每次调用都会为 train 重建 KD 树；改成 add()+train() 预训练后，
     单次匹配从 39ms 降到个位数毫秒。
  2. 特征点坐标存成 numpy 数组而不是 cv2.KeyPoint 对象列表，
     省掉 11 万个 Python 包装对象的开销。
  3. 建索引用线程池并行：OpenCV 的 SIFT 会释放 GIL，多核能真正并行。

依赖：opencv-python, numpy（不需要 torch，LoFTR 那套是可选的高配路线）。
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np


class SiftResult(NamedTuple):
    """一次 SIFT 识别的结果。name 为 None 表示没认出来。"""

    name: Optional[str]
    inliers: int                       # RANSAC 内点数，越大越可信
    second_name: Optional[str]
    second_inliers: int
    location: Optional[Tuple[int, int]]     # 抓图区中心在整图上的坐标 (x, y)
    source_size: Optional[Tuple[int, int]]  # 抓图区的 (宽, 高)
    target_size: Optional[Tuple[int, int]]  # 命中地图的 (宽, 高)
    reason: str = ""


def _new_flann(trees, checks):
    """FLANN KD 树匹配器；每张地图各持一个，构建时预训练一次。"""
    return cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=int(trees)),      # 1 = FLANN_INDEX_KDTREE
        dict(checks=int(checks)),
    )


class _MapIndex(object):
    """一张地图的 SIFT 索引。points 是 (N,2) float32，比 KeyPoint 列表省得多。"""

    __slots__ = ("name", "path", "points", "descriptors", "matcher", "width", "height")

    def __init__(self, name, path, points, descriptors, matcher, width, height):
        self.name = name
        self.path = path
        self.points = points              # (N, 2) float32
        self.descriptors = descriptors    # (N, 128) float32
        self.matcher = matcher            # 已 train() 的 FlannBasedMatcher
        self.width = width
        self.height = height


class SiftMatcher(object):
    """把屏幕抓图区与 N 张完整地图做 SIFT 匹配，返回最像的那一张及其位置。"""

    def __init__(
        self,
        folder_path,
        clahe_limit=3.0,
        nfeatures=0,
        ratio=0.75,
        min_good=8,
        min_inliers=12,
        ransac_threshold=8.0,
        min_gap_ratio=1.0,
        flann_trees=5,
        flann_checks=50,
        build_workers=None,
    ):
        self._folder = folder_path
        self._clahe_limit = float(clahe_limit)
        self._nfeatures = int(nfeatures)
        self._ratio = float(ratio)
        self._min_good = int(min_good)
        self._min_inliers = int(min_inliers)
        self._ransac_threshold = float(ransac_threshold)
        # 1.0 = 关闭；设成 1.15 表示"最优内点数必须比次优高 15%"，用于压制近似地图误判
        self._min_gap_ratio = float(min_gap_ratio)
        self._flann_trees = int(flann_trees)
        self._flann_checks = int(flann_checks)
        self._build_workers = int(build_workers or min(8, os.cpu_count() or 4))

        self._lock = threading.RLock()
        self._index = []          # List[_MapIndex]
        self.last_build_seconds = 0.0

    # ------------------------------------------------------------------ 工具
    def _new_clahe(self):
        """每次新建，避免跨线程共享有状态的 CLAHE 对象。"""
        return cv2.createCLAHE(clipLimit=self._clahe_limit, tileGridSize=(8, 8))

    def _new_sift(self):
        return cv2.SIFT_create(nfeatures=self._nfeatures)

    def _list_images(self, folder):
        if not os.path.isdir(folder):
            return []
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        return sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))

    @staticmethod
    def _read_image(path):
        # 先读字节再 imdecode，兼容中文路径
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # ------------------------------------------------------------------ 构建
    @property
    def map_count(self):
        with self._lock:
            return len(self._index)

    @property
    def keypoint_total(self):
        with self._lock:
            return sum(len(m.points) for m in self._index)

    def _build_one(self, filename):
        """给一张地图建索引；返回 (_MapIndex, None) 或 (None, 跳过原因)。"""
        path = os.path.join(self._folder, filename)
        try:
            image = self._read_image(path)
        except Exception as exc:
            return None, "%s(%s)" % (filename, exc)
        if image is None:
            return None, "%s(解码失败)" % filename

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = self._new_clahe().apply(gray)

        try:
            keypoints, descriptors = self._new_sift().detectAndCompute(gray, None)
        except cv2.error as exc:
            return None, "%s(SIFT 失败: %s)" % (filename, exc)
        if descriptors is None or len(keypoints) == 0:
            return None, "%s(没提取到特征点)" % filename

        # 坐标转成 numpy 数组，之后不再需要 KeyPoint 对象
        points = cv2.KeyPoint_convert(keypoints)

        matcher = _new_flann(self._flann_trees, self._flann_checks)
        try:
            matcher.add([descriptors])
            matcher.train()          # 关键：KD 树只建这一次
        except cv2.error as exc:
            return None, "%s(FLANN 训练失败: %s)" % (filename, exc)

        return _MapIndex(
            name=os.path.splitext(filename)[0],
            path=path,
            points=points,
            descriptors=descriptors,
            matcher=matcher,
            width=width,
            height=height,
        ), None

    def build(self):
        """为目录下的每张地图提取 CLAHE + SIFT 特征，并并行预训练 FLANN 索引。"""
        started = time.time()
        filenames = self._list_images(self._folder)
        built, skipped = [], []

        if filenames:
            workers = max(1, min(self._build_workers, len(filenames)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for entry, reason in pool.map(self._build_one, filenames):
                    if entry is not None:
                        built.append(entry)
                    elif reason:
                        skipped.append(reason)

        built.sort(key=lambda item: item.name)   # 与文件名顺序对齐，方便看日志

        with self._lock:
            self._index = built
        self.last_build_seconds = time.time() - started
        return len(built), skipped

    # ------------------------------------------------------------------ 匹配
    @staticmethod
    def _ratio_filter(knn, ratio):
        """Lowe's ratio test，向量化实现。

        返回 (query_idx, train_idx) 两组 int32 下标，可直接用于索引坐标数组。
        """
        pairs = [p for p in knn if len(p) == 2]
        empty = np.empty(0, dtype=np.int32)
        if not pairs:
            return empty, empty

        count = len(pairs)
        best = np.fromiter((p[0].distance for p in pairs), np.float32, count)
        second = np.fromiter((p[1].distance for p in pairs), np.float32, count)
        keep = best < ratio * second
        if not keep.any():
            return empty, empty

        # 注意：查表坐标一律取 p[0]（最近邻，也就是真正的匹配）。
        # p[1] 只是用来算 Lowe 比值的次近邻，拿它的 trainIdx 会得到完全错误的对应关系。
        query_idx = np.fromiter((p[0].queryIdx for p in pairs), np.int32, count)
        train_idx = np.fromiter((p[0].trainIdx for p in pairs), np.int32, count)
        return query_idx[keep], train_idx[keep]

    def _score_one(self, query_points, query_des, entry):
        """把抓图区特征和一张地图比一次，返回 (inliers, homography)。"""
        try:
            knn = entry.matcher.knnMatch(query_des, k=2)   # 用预训练索引，不传 train
        except cv2.error:
            return 0, None

        query_idx, train_idx = self._ratio_filter(knn, self._ratio)
        if query_idx.size < self._min_good:
            return 0, None

        src = query_points[query_idx].reshape(-1, 1, 2)
        dst = entry.points[train_idx].reshape(-1, 1, 2)

        try:
            homography, mask = cv2.findHomography(src, dst, cv2.RANSAC,
                                                  self._ransac_threshold)
        except cv2.error:
            return 0, None
        if homography is None or mask is None:
            return 0, None

        inliers = int(mask.sum())
        if inliers <= 0:
            return 0, None
        return inliers, homography

    def identify(self, region_bgr):
        """对一块屏幕抓图做识别，返回 SiftResult。"""
        with self._lock:
            index = list(self._index)

        if not index:
            return SiftResult(None, 0, None, 0, None, None, None, "没有可用的地图索引")

        if region_bgr is None or region_bgr.size == 0:
            return SiftResult(None, 0, None, 0, None, None, None, "抓图区为空")

        region_h, region_w = region_bgr.shape[:2]
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._new_clahe().apply(gray)

        try:
            keypoints, query_des = self._new_sift().detectAndCompute(gray, None)
        except cv2.error as exc:
            return SiftResult(None, 0, None, 0, None, (region_w, region_h), None,
                              "特征提取失败：%s" % exc)

        if query_des is None or len(keypoints) < 2:
            count = 0 if keypoints is None else len(keypoints)
            return SiftResult(None, 0, None, 0, None, (region_w, region_h), None,
                              "抓图区特征点太少（%d 个）" % count)

        query_points = cv2.KeyPoint_convert(keypoints)

        scored = []
        homographies = {}
        for entry in index:
            inliers, homography = self._score_one(query_points, query_des, entry)
            scored.append((inliers, entry))
            if homography is not None:
                homographies[entry.name] = homography

        scored.sort(key=lambda item: item[0], reverse=True)
        best_inliers, best_entry = scored[0]
        if len(scored) > 1:
            second_inliers, second_name = scored[1][0], scored[1][1].name
        else:
            second_inliers, second_name = 0, None

        if best_inliers < self._min_inliers:
            return SiftResult(None, best_inliers, second_name, second_inliers,
                              None, (region_w, region_h), None,
                              "最高内点数 %d 低于阈值 %d" % (best_inliers, self._min_inliers))

        # 可选的 Top-2 间距判定：压制"两张图长得很像"导致的误认
        if (self._min_gap_ratio > 1.0 and second_name is not None
                and best_inliers < second_inliers * self._min_gap_ratio):
            return SiftResult(None, best_inliers, second_name, second_inliers,
                              None, (region_w, region_h), None,
                              "%s 与 %s 太接近（内点 %d vs %d），拒绝判定"
                              % (best_entry.name, second_name, best_inliers, second_inliers))

        # 反算抓图区中心在整图上的坐标
        location = None
        homography = homographies.get(best_entry.name)
        if homography is not None:
            center = np.float32([[[region_w / 2.0, region_h / 2.0]]])
            try:
                mapped = cv2.perspectiveTransform(center, homography)
                cx, cy = int(mapped[0][0][0]), int(mapped[0][0][1])
                if 0 <= cx < best_entry.width and 0 <= cy < best_entry.height:
                    location = (cx, cy)
            except cv2.error:
                location = None

        if location is None:
            return SiftResult(None, best_inliers, second_name, second_inliers,
                              None, (region_w, region_h),
                              (best_entry.width, best_entry.height),
                              "单应矩阵投影出的坐标落在地图外，判定不可信")

        return SiftResult(best_entry.name, best_inliers, second_name, second_inliers,
                          location, (region_w, region_h),
                          (best_entry.width, best_entry.height))
