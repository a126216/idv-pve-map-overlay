# -*- coding: utf-8 -*-
"""selftest.py —— 无界面自检：模板能不能加载、每张门模板能不能匹配到它自己。

改动 maps/ 之后跑一下这个，比开 GUI 按热键快得多：

    .venv\\Scripts\\python.exe selftest.py

判读标准：
  * 每张门模板都应"匹配到它自己"，且 MAE ≈ 0；
  * "次优误差"应当明显大于 0（越大说明模板之间区分度越好）；
  * 如果你的门模板宽高比和 door_coords 差太多，开头会有一串 WARNING。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main


def main_selftest():
    door_dir = os.path.join(main.MAP_FOLDER, main.DOOR_SUBDIR)
    door_files = dict((os.path.splitext(f)[0], os.path.join(door_dir, f))
                      for f in main.list_images(door_dir))

    print("door_coords = %r" % (main.CONFIG["door_coords"],))
    print("屏幕抓图区  = %r" % (main.get_screen_crop(),))
    print("期望宽高比  = %.3f" % main.door_aspect())
    print()

    matcher = main.MapMatcher(main.MAP_FOLDER)
    model = matcher._model
    print("加载模板数 = %d" % matcher.template_count)
    print("特征矩阵   = %s" % (None if model.stack is None else model.stack.shape,))
    print()

    if not model.names:
        print("!! 没有加载到任何模板，检查 maps/ 与 maps/door/ 目录")
        return 1

    bad = 0
    missing_display = 0
    for i, stem in enumerate(model.names):
        disp = model.display_paths[i]
        if not os.path.exists(disp):
            missing_display += 1
        if stem not in door_files:
            print("   %-10s 缺少门模板文件，跳过" % stem)
            continue
        img = main.read_image(door_files[stem])
        res = matcher.match(img)
        ok = (res.name == stem)
        if not ok:
            bad += 1
        print("   %s %-10s -> %-10s MAE=%7.3f  次优=%-10s 次优MAE=%8.3f  %s"
              % ("OK " if ok else "BAD", stem, res.name, res.error,
                 res.second_name, res.second_error, res.reason))

    print()
    print("自匹配失败 = %d / %d" % (bad, len(model.names)))
    print("显示图缺失 = %d" % missing_display)
    return 1 if (bad or missing_display) else 0


if __name__ == "__main__":
    sys.exit(main_selftest())
