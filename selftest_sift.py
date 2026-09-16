# -*- coding: utf-8 -*-
"""selftest_sift.py —— 验证 SIFT 引擎能否在 18 张地图里认出「这是哪张、在哪一块」。

不需要开游戏：从每张地图上裁一块区域当作"屏幕抓图区"，先测纯净的，
再人为加旋转 / 缩放 / 亮度 / 噪声模拟真实小地图，交给 SiftMatcher 去认。

判读：
  * 全部认对（TOP1 正确、坐标偏差小）  -> 引擎可用，问题只剩"屏幕上到底是不是地图内容"；
  * 大量认错                          -> 地图纹理区分度不够，或裁剪太靠边（纯色区没特征点）。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import main
from sift_matcher import SiftMatcher

REGION_W, REGION_H = 583, 300
SEED = 20260916


def degrade(patch, angle, scale, rng):
    """模拟真实小地图：旋转 + 缩放 + 亮度对比度漂移 + 噪声。"""
    h, w = patch.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    out = cv2.warpAffine(patch, matrix, (w, h),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.80, 1.25), beta=rng.uniform(-30, 30))
    noise = rng.normal(0, 6, out.shape).astype(np.int16)
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def run_round(matcher, samples, label, tolerance):
    print()
    print("== %s ==" % label)
    ok_name = ok_loc = 0
    elapsed = 0.0
    for stem, patch, truth_xy in samples:
        t0 = time.time()
        result = matcher.identify(patch)
        elapsed += time.time() - t0

        name_ok = (result.name == stem)
        loc_err = None
        if result.location and truth_xy:
            loc_err = ((result.location[0] - truth_xy[0]) ** 2
                       + (result.location[1] - truth_xy[1]) ** 2) ** 0.5
        loc_ok = (loc_err is not None and loc_err <= tolerance)

        ok_name += 1 if name_ok else 0
        ok_loc += 1 if loc_ok else 0
        print("   %s %-10s -> %-10s inliers=%-4d 2nd=%-10s(%-3d) loc=%-14s err=%s"
              % ("OK " if name_ok else "BAD", stem, result.name, result.inliers,
                 result.second_name, result.second_inliers,
                 str(result.location), "%.0f" % loc_err if loc_err is not None else "n/a"))

    n = len(samples)
    print("   ---- %s: 名称正确 %d/%d, 坐标正确(<=%dpx) %d/%d, 平均 %.2fs/次"
          % (label, ok_name, n, tolerance, ok_loc, n, elapsed / max(1, n)))
    return ok_name, ok_loc, n


def main_test():
    print("正在建立 SIFT 索引……")
    matcher = SiftMatcher(
        main.MAP_FOLDER,
        clahe_limit=main.cfg_float("sift_clahe_limit") if "sift_clahe_limit" in main.CONFIG else 3.0,
        ratio=0.75,
        min_good=8,
        min_inliers=12,
        ransac_threshold=8.0,
    )
    count, skipped = matcher.build()
    print("索引地图 %d 张，特征点合计 %d，耗时 %.2fs"
          % (count, matcher.keypoint_total, matcher.last_build_seconds))
    if skipped:
        print("跳过：%s" % ("；".join(skipped),))

    rng = np.random.RandomState(SEED)
    samples_clean = []
    samples_hard = []

    for filename in matcher._list_images(main.MAP_FOLDER):
        stem = os.path.splitext(filename)[0]
        image = matcher._read_image(os.path.join(main.MAP_FOLDER, filename))
        if image is None:
            continue
        h, w = image.shape[:2]
        cw, ch = min(REGION_W, w), min(REGION_H, h)
        x = rng.randint(0, w - cw)
        y = rng.randint(0, h - ch)
        patch = image[y:y + ch, x:x + cw].copy()
        truth = (x + cw // 2, y + ch // 2)
        samples_clean.append((stem, patch, truth))
        samples_hard.append((stem, degrade(patch, rng.uniform(-25, 25), rng.uniform(0.6, 1.3), rng), truth))

    a = run_round(matcher, samples_clean, "A. 纯净裁剪（无任何退化）", 12)
    b = run_round(matcher, samples_hard, "B. 加了旋转/缩放/亮度/噪声", 60)

    print()
    print("DONE")
    return 0 if (a[0] == a[2] and b[0] == b[2]) else 1


if __name__ == "__main__":
    sys.exit(main_test())
