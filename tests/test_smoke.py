# -*- coding: utf-8 -*-
"""tests/test_smoke.py —— 无素材冒烟测试（合成图）。

不依赖任何游戏地图素材，也不依赖显示器/PyQt5：
  * test_sift_identify  —— 用合成图验证 SIFT 引擎能「在 N 张里认出是哪张 + 坐标」。
  * test_template_match —— 若环境装了 PyQt5（开发机/完整依赖），顺带验证 template(MAE)
                          模式；CI 里只装 opencv-python-headless，这部分会自动跳过。

运行：
    python tests/test_smoke.py        # 直接跑，失败以非零码退出
    pytest tests/test_smoke.py        # 或走 pytest
"""
import os
import sys
import tempfile

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sift_matcher import SiftMatcher

N_MAPS = 8
MAP_W, MAP_H = 320, 180
SEED_BASE = 1007


def _make_map(path, seed):
    """画大量高对比几何形状，保证任意中央裁剪都有充足 SIFT 特征且彼此可区分。"""
    rng = np.random.RandomState(seed)
    img = np.full((MAP_H, MAP_W, 3), rng.randint(30, 200, size=3).tolist(), dtype=np.uint8)
    for _ in range(24):
        color = rng.randint(0, 255, size=3).tolist()
        pt1 = (rng.randint(0, MAP_W), rng.randint(0, MAP_H))
        pt2 = (rng.randint(0, MAP_W), rng.randint(0, MAP_H))
        cv2.line(img, pt1, pt2, color, rng.randint(2, 5))
    for _ in range(14):
        color = rng.randint(0, 255, size=3).tolist()
        c = (rng.randint(0, MAP_W), rng.randint(0, MAP_H))
        cv2.circle(img, c, rng.randint(8, 40), color, rng.randint(2, 4))
    cv2.imwrite(path, img)
    return img


def _seed(i):
    return SEED_BASE + i * 1000


def _write_maps(folder):
    paths = {}
    for i in range(1, N_MAPS + 1):
        p = os.path.join(folder, "map%d.png" % i)
        _make_map(p, _seed(i))
        paths[i] = p
    return paths


def test_sift_identify():
    """SIFT 引擎应在合成图里正确认出裁剪来源，并反算出合理坐标。"""
    tmp = tempfile.mkdtemp(prefix="mm_sift_")
    paths = _write_maps(tmp)

    matcher = SiftMatcher(tmp, ratio=0.75, min_good=6, min_inliers=8, ransac_threshold=8.0)
    count, skipped = matcher.build()
    assert count == N_MAPS, "索引数量应为 %d，实际 %d（跳过 %s）" % (N_MAPS, count, skipped)

    cw, ch = 200, 120
    bad = 0
    for i in range(1, N_MAPS + 1):
        img = cv2.imread(paths[i])
        x = (img.shape[1] - cw) // 2
        y = (img.shape[0] - ch) // 2
        patch = img[y:y + ch, x:x + cw].copy()
        res = matcher.identify(patch)
        if res.name != "map%d" % i or res.location is None:
            bad += 1
            print("  BAD map%d -> %s loc=%s" % (i, res.name, res.location))
    assert bad == 0, "%d/%d 张合成图识别错误" % (bad, N_MAPS)
    print("test_sift_identify: %d/%d OK" % (N_MAPS, N_MAPS))


def test_template_match():
    """template(MAE) 模式：裁出与模板完全相同的区域应 MAE≈0 且命中自身。

    需要 PyQt5（main 模块顶层 import PyQt5）。CI 仅装 opencv-python-headless 时跳过。
    """
    try:
        import main
    except Exception as exc:  # 多为 CI 环境缺 PyQt5 / 显示器
        print("test_template_match: 跳过（无法导入 main：%s）" % exc)
        return

    main.CONFIG["matcher_mode"] = "template"
    main.CONFIG["ref_width"] = MAP_W
    main.CONFIG["ref_height"] = MAP_H
    main.CONFIG["door_coords"] = {"left": 60, "top": 30, "width": 200, "height": 120}
    main.CONFIG["use_hist_equalization"] = True
    main.CONFIG["error_threshold"] = 12.0
    main.CONFIG["top2_gap_threshold"] = 2.5

    tmp = tempfile.mkdtemp(prefix="mm_tmpl_")
    paths = _write_maps(tmp)

    matcher = main.MapMatcher(tmp)
    assert matcher.template_count == N_MAPS, "template 模式模板数应为 %d" % N_MAPS

    img3 = cv2.imread(paths[3])
    feat, err = main.extract_door_feature(img3)
    assert err is None, err
    dc = main.CONFIG["door_coords"]
    crop = img3[dc["top"]:dc["top"] + dc["height"], dc["left"]:dc["left"] + dc["width"]]
    res = matcher.match(crop)
    assert res.name == "map3", "template 模式应命中自身，实际 %r" % res.name
    assert res.error < 1e-3, "自身 MAE 应≈0，实际 %r" % res.error
    print("test_template_match: map3 命中自身，MAE=%.4f" % res.error)


if __name__ == "__main__":
    test_sift_identify()
    test_template_match()
    print("ALL SMOKE TESTS PASSED")
