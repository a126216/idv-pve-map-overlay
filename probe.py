# -*- coding: utf-8 -*-
"""probe.py —— 标定/诊断工具：屏幕抓图区到底和 maps/ 里的地图图片有没有对应关系？

用法（把游戏停在待识别的画面上，然后运行）：

    .venv\\Scripts\\python.exe probe.py [抓图保存路径]

它会做三件事：

  1. 抓取 config.json 里 door_coords 指定的屏幕区域并存成 PNG ——
     方便你用眼睛把它和 maps/door/*.png 摆在一起比对。
  2. 把这张抓图和 18 张门模板逐一算 MAE，列出排名（越接近 0 越像）。
  3. 把抓图在多个缩放倍率下与 18 张完整地图做模板匹配，
     报告每张地图的最佳相关系数与倍率。
     若某张地图的相关系数明显偏高（> 0.5），说明屏幕那块确实就是该地图图片的一部分，
     那么"多尺度模板匹配"这条路线可行，压根不需要手工裁门模板。

诊断要点：
  * 面板 A 里所有 MAE 都很大且彼此接近  -> 抓图区和门模板对不上（取景/缩放不一致）；
  * 面板 B 里所有相关系数都很低（< 0.3） -> 屏幕上根本不是地图图片的内容。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import main

# 抓图保存位置：默认放系统临时目录（可用命令行参数覆盖），不写死本机路径
DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "MapMatcher_screen_grab.png")

# 模板匹配的候选缩放倍率（屏幕像素 -> 地图图片像素）
SCALES = [0.45, 0.6, 0.8, 1.0, 1.25, 1.6]
# 匹配前统一降采样的比例，纯粹为了跑得快
WORK = 0.25


def grab_screen_region():
    crop = main.get_screen_crop()
    with main.new_mss() as sct:
        raw = sct.grab(crop)
    return cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR), crop


def save_png(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    buf.tofile(path)          # tofile 而不是 imwrite，兼容中文路径


def panel_a(grab, matcher):
    print("== A. 抓图 vs 18 张门模板（MAE，越小越像）==")
    feat = main.bgr_to_feature(grab)
    stack = matcher._model.stack
    if stack is None:
        print("   （没有加载到任何模板）")
        return
    errors = np.abs(stack - feat).mean(axis=(1, 2))
    order = np.argsort(errors)
    for rank, i in enumerate(order):
        mark = "  <== 阈值 12 以内" if errors[i] <= main.cfg_float("error_threshold") else ""
        print("   %2d. %-10s MAE = %8.3f%s" % (rank + 1, matcher._model.names[i], errors[i], mark))
    print("   最优 %.3f / 次优 %.3f / 间距 %.3f（判定要求间距 >= %.2f）"
          % (errors[order[0]], errors[order[1]] if order.size > 1 else float("nan"),
             errors[order[1]] - errors[order[0]] if order.size > 1 else float("nan"),
             main.cfg_float("top2_gap_threshold")))


def panel_b(grab, matcher):
    print()
    print("== B. 抓图 vs 18 张完整地图（多尺度模板匹配，相关系数越大越像）==")
    gh, gw = grab.shape[:2]
    grab_gray = cv2.cvtColor(grab, cv2.COLOR_BGR2GRAY)
    results = []

    for i, stem in enumerate(matcher._model.names):
        path = matcher._model.display_paths[i]
        img = main.read_image(path)
        if img is None:
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(img_gray, None, fx=WORK, fy=WORK, interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]

        best = (-2.0, None, None)
        for z in SCALES:
            tw = int(round(gw * z * WORK))
            th = int(round(gh * z * WORK))
            if tw < 8 or th < 8 or tw > sw or th > sh:
                continue
            tmpl = cv2.resize(grab_gray, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(small, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best[0]:
                best = (max_val, z, max_loc)

        results.append((best[0], stem, best[1], best[2]))

    results.sort(reverse=True)
    for rank, (score, stem, z, loc) in enumerate(results):
        print("   %2d. %-10s corr = %6.3f   scale = %-5s  yx = %s"
              % (rank + 1, stem, score, z, loc))
    top = results[0][0] if results else -2.0
    print()
    if top > 0.5:
        print("   >>> 最高相关系数 %.3f：屏幕那块很可能就是地图图片的一部分，多尺度匹配路线可行。" % top)
    elif top > 0.3:
        print("   >>> 最高相关系数 %.3f：有一定相关性但不明确，可能需要更多缩放倍率或先做预处理。" % top)
    else:
        print("   >>> 最高相关系数仅 %.3f：屏幕那块和地图图片基本没有对应关系。" % top)


def main_probe():
    out_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    grab, crop = grab_screen_region()
    print("抓图区域 = %r  实际尺寸 = %dx%d" % (crop, grab.shape[1], grab.shape[0]))
    print("expected aspect = %.3f" % main.door_aspect())
    try:
        save_png(grab, out_path)
        print("抓图已保存 -> %s" % out_path)
    except Exception as exc:
        print("抓图保存失败：%s" % exc)
    print()

    matcher = main.MapMatcher(main.MAP_FOLDER)
    print("模板数 = %d" % matcher.template_count)
    print()
    panel_a(grab, matcher)
    panel_b(grab, matcher)
    print()
    print("DONE")


if __name__ == "__main__":
    main_probe()
