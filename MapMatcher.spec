# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（onedir / 无控制台窗口）。

代码已为冻结环境设计（main.py 里用了 sys.frozen / sys._MEIPASS），本 spec 让
`pyinstaller MapMatcher.spec` 一键产出可分发目录。

注意：
  * maps/ 是 gitignore 的，本地有素材才会被打进 dist（没有也不影响启动，
    只是首次运行提示“没有可用的地图模板”）。
  * icon.ico 可选，存在则用作 exe 图标；不存在自动跳过。
"""
import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
)

ROOT = os.path.dirname(os.path.abspath(SPECPATH))
block_cipher = None

# ---- 资源：本地 maps/ 与图标（按需）----
datas = []
maps_dir = os.path.join(ROOT, "maps")
if os.path.isdir(maps_dir):
    datas.append((maps_dir, "maps"))
icon_path = os.path.join(ROOT, "icon.ico")
if os.path.exists(icon_path):
    datas.append((icon_path, "."))

hiddenimports = [
    "sift_matcher",
    "cv2", "numpy", "mss", "keyboard",
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets",
]

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=collect_dynamic_libs("cv2") + collect_dynamic_libs("PyQt5"),
    datas=datas + collect_data_files("PyQt5"),
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MapMatcher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                 # GUI 程序，不弹控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path if os.path.exists(icon_path) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MapMatcher",
)
