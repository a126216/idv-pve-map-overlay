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

依赖：opencv-python, numpy（不需要 torch，LoFTR 那套是可选的高配路线）。
"""

import os
import threading
import time
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np


class SiftResult(NamedTuple):
    """一次 SIFT 识别的结果。name 为 None 表示没认出来。"""

    name: Optional[str]
    inliers: int                       # RANSAC 内点数，越大越可信
    second_name: Optional[str]
    second_inliers: int
    location: Optional[Tuple[int, int]]   # 抓图区中心在整图上的坐标 (x, y)
    source_size: Optional[Tuple[int, int]]  # 抓图区的 (宽, 高)
    target_size: Optional[Tuple[int, int]]  # 命中地图的 (宽, 高)
    reason: str = ""


class _MapIndex(object):
    """一张地图的 SIFT 索引。"""

    __slots__ = ("name", "path", "keypoints", "descriptors", "width", "height")

    def __init__(self, name, path, keypoints, descriptors, width, height):
        self.name = name
        self.path = path
        self.keypoints = keypoints
        self.descriptors = descriptors
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
        flann_trees=5,
        flann_checks=50,
    ):
        self._folder = folder_path
        self._ratio = float(ratio)
        self._min_good = int(min_good)
        self._min_inliers = int(min_inliers)
        self._ransac_threshold = float(ransac_threshold)

        self._clahe = cv2.createCLAHE(clipLimit=float(clahe_limit), tileGridSize=(8, 8))
        self._sift = cv2.SIFT_create(nfeatures=int(nfeatures))
        self._flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=int(flann_trees)),      # 1 = FLANN_INDEX_KDTREE
            dict(checks=int(flann_checks)),
        )

        self._lock = threading.RLock()
        self._index = []          # List[_MapIndex]
        self.last_build_seconds = 0.0

    # ------------------------------------------------------------------ 构建
    @property
    def map_count(self):
        with self._lock:
            return len(self._index)

    @property
    def keypoint_total(self):
        with self._lock:
            return sum(len(m.keypoints) for m in self._index)

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

    def build(self):
        """为目录下的每张地图提取 CLAHE + SIFT 特征。返回建立索引的地图数。"""
        started = time.time()
        built = []
        skipped = []

        for filename in self._list_images(self._folder):
            path = os.path.join(self._folder, filename)
            try:
                image = self._read_image(path)
            except Exception as exc:
                skipped.append("%s(%s)" % (filename, exc))
                continue
            if image is None:
                skipped.append("%s(解码失败)" % filename)
                continue

            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = self._clahe.apply(gray)

            try:
                keypoints, descriptors = self._sift.detectAndCompute(gray, None)
            except cv2.error as exc:
                skipped.append("%s(SIFT 失败: %s)" % (filename, exc))
                continue

            if descriptors is None or len(keypoints) == 0:
                skipped.append("%s(没提取到特征点)" % filename)
                continue

            built.append(_MapIndex(
                name=os.path.splitext(filename)[0],
                path=path,
                keypoints=keypoints,
                descriptors=descriptors,
                width=width,
                height=height,
            ))

        with self._lock:
            self._index = built
        self.last_build_seconds = time.time() - started
        return len(built), skipped

    # ------------------------------------------------------------------ 匹配
    @staticmethod
    def _lowe_ratio(knn_pairs, ratio):
        """Lowe's ratio test：最近邻距离必须明显小于次近邻。"""
        good = []
        for pair in knn_pairs:
            if len(pair) == 2:
                best, second = pair
                if best.distance < ratio * second.distance:
                    good.append(best)
        return good

    def _score_one(self, query_kp, query_des, entry):
        """把抓图区特征和一张地图比一次，返回 (inliers, location) 。"""
        try:
            knn = self._flann.knnMatch(query_des, entry.descriptors, k=2)
        except cv2.error:
            return 0, None

        good = self._lowe_ratio(knn, self._ratio)
        if len(good) < self._min_good:
            return 0, None

        src = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([entry.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        try:
            homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, self._ransac_threshold)
        except cv2.error:
            return 0, None
        if homography is None or mask is None:
            return 0, None

        inliers = int(mask.sum())
        if inliers <= 0:
            return 0, None

        # 中心点坐标由调用方用这个单应矩阵反算
        return inliers, homography

    def identify(self, region_bgr):
        """对一块屏幕抓图做识别，返回 SiftResult。"""
        with self._lock:
            index = list(self._index)

        if not index:
            return SiftResult(None, 0, None, 0, None, None, None,
                              "没有可用的地图索引")

        if region_bgr is None or region_bgr.size == 0:
            return SiftResult(None, 0, None, 0, None, None, None, "抓图区为空")

        region_h, region_w = region_bgr.shape[:2]
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)

        try:
            query_kp, query_des = self._sift.detectAndCompute(gray, None)
        except cv2.error as exc:
            return SiftResult(None, 0, None, 0, None, (region_w, region_h), None,
                              "特征提取失败：%s" % exc)

        if query_des is None or len(query_kp) < 2:
            return SiftResult(None, 0, None, 0, None, (region_w, region_h), None,
                              "抓图区特征点太少（%d 个）" % len(query_kp))

        scores = []
        homographies = {}
        for entry in index:
            inliers, homography = self._score_one(query_kp, query_des, entry)
            scores.append((inliers, entry.name))
            if homography is not None:
                homographies[entry.name] = homography

        scores.sort(key=lambda item: item[0], reverse=True)
        best_inliers, best_name = scores[0]
        second_inliers, second_name = scores[1] if len(scores) > 1 else (0, None)

        if best_inliers < self._min_inliers:
            return SiftResult(None, best_inliers, second_name, second_inliers,
                              None, (region_w, region_h), None,
                              "最高内点数 %d 低于阈值 %d" % (best_inliers, self._min_inliers))

        # 反算抓图区中心在整图上的坐标
        entry = next(m for m in index if m.name == best_name)
        homography = homographies.get(best_name)
        location = None
        if homography is not None:
            center = np.float32([[[region_w / 2.0, region_h / 2.0]]])
            try:
                mapped = cv2.perspectiveTransform(center, homography)
                cx, cy = int(mapped[0][0][0]), int(mapped[0][0][1])
                if 0 <= cx < entry.width and 0 <= cy < entry.height:
                    location = (cx, cy)
            except cv2.error:
                location = None

        if location is None:
            return SiftResult(None, best_inliers, second_name, second_inliers,
                              None, (region_w, region_h), (entry.width, entry.height),
                              "单应矩阵投影出的坐标落在地图外，判定不可信")

        return SiftResult(best_name, best_inliers, second_name, second_inliers,
                          location, (region_w, region_h), (entry.width, entry.height))
