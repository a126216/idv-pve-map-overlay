# -*- coding: utf-8 -*-
"""
MapMatcher —— 游戏地图悬浮识别工具（单文件版）

工作原理
--------
  1. 启动时把 maps/ 下每张地图模板的"门"区域裁出来，灰度化并归一化成
     FEATURE_SIZE x FEATURE_SIZE 的特征矩阵，一次性缓存。
  2. 按下热键（默认 F2）时，后台线程用 mss 抓取屏幕上同比例的"门"区域，
     做完全相同的预处理，再与全部模板特征求平均绝对误差（MAE）。
  3. 双阈值判定：最优误差必须 <= error_threshold；且最优与次优的误差之差
     必须 >= top2_gap_threshold，否则认为两张地图过于相似，拒绝判定以防误认。
  4. 命中后在地图悬浮窗里显示对应的那一整张地图。

模板目录布局（maps/）
---------------------
  maps/door/*         门区域裁剪图：只用于识别。整张图直接归一化，绝不再裁一次。
  maps/full/*         整图：只用于显示（识别出是哪张，就显示哪张）。
  maps/*              旧结构兼容位：
                        * 若存在同名的门裁剪图，则此文件被当作"显示用整图"；
                        * 否则按旧逻辑当作模板，按其自身尺寸等比裁出门区域再识别。

  识别模板与显示图片通过【同名】配对（扩展名可不同），例如：
      maps/door/map1.jpg  ↔  maps/full/map1.jpg  ↔  maps/map1.jpg
  找不到同名整图时，退化为直接显示那张门裁剪图本身。

  门裁剪图的像素尺寸可以各不相同（都会被统一缩放到 FEATURE_SIZE 见方），
  但宽高比应尽量接近 door_coords 的宽高比，否则会被压扁导致精度下降——
  程序会对偏差超过 15% 的门模板发出告警。

线程模型（最容易踩坑的地方）
----------------------------
  * GUI 线程：OverlayWindow、托盘、QApplication。
  * MatcherWorker(QThread)：唯一接触 mss 与 OpenCV 的线程；mss 实例在线程内部
    创建、复用、释放，绝不跨线程共享。
  * keyboard 钩子线程：回调只允许 emit Qt 信号。跨线程直接操作控件或调用
    QApplication.quit() 会随机崩溃，必须投递到 GUI 线程执行。

config.json 位于 exe 同目录；缺失项、非法项、废弃项都会自动回退/清理并回写，
保证配置文件不会成为崩溃来源。
"""

import copy
import ctypes
import json
import logging
import os
import sys
import threading
import time
from typing import NamedTuple, Optional

import cv2
import numpy as np
import mss
import keyboard
from PyQt5.QtCore import Qt, QPoint, QRect, QSharedMemory, QThread, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QLabel,
    QMenu,
    QSizePolicy,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

# SIFT 引擎（架构参考 Game-Map-Tracker，见 sift_matcher.py）；导入失败时自动降级
try:
    from sift_matcher import SiftMatcher
except Exception as _sift_exc:      # 任何导入问题都不该让整个程序起不来
    SiftMatcher = None
    SIFT_IMPORT_ERROR = _sift_exc
else:
    SIFT_IMPORT_ERROR = None

# ==============================================================================
# 1. 运行环境与路径
# ==============================================================================
def _enable_dpi_awareness():
    """开启 Per-Monitor DPI 感知，避免 Windows 125%/150% 缩放导致抓图坐标偏移。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


_enable_dpi_awareness()

IS_FROZEN = getattr(sys, "frozen", False)
# 打包后 sys.executable 指向 exe 自身，__file__ 指向解包目录，两者必须区分
EXE_DIR = os.path.dirname(sys.executable) if IS_FROZEN else os.path.dirname(os.path.abspath(__file__))
BASE_DIR = getattr(sys, "_MEIPASS", EXE_DIR) if IS_FROZEN else EXE_DIR
MAP_FOLDER = os.path.join(BASE_DIR, "maps")
CONFIG_PATH = os.path.join(EXE_DIR, "config.json")
LOG_PATH = os.path.join(EXE_DIR, "MapMatcher_Debug.log")


def _setup_logging():
    """双层日志（文件 + 控制台）；目录不可写时降级为仅控制台，不让启动失败。"""
    handlers = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(LOG_PATH, encoding="utf-8"))
    except OSError as exc:
        print("[WARN] 无法写入日志文件 %s: %s" % (LOG_PATH, exc), file=sys.stderr)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


_setup_logging()
logger = logging.getLogger("MapMatcher")

# ==============================================================================
# 2. 配置：默认值 -> 合并磁盘配置 -> 校验 -> 自愈回写
# ==============================================================================
DEFAULT_CONFIG = {
    "hotkey": "f2",
    "exit_hotkey": "f12",
    "region_hotkey": "f3",    # 进入截图区域框选模式（可视化选择 door_coords）
    "error_threshold": 12.0,
    "top2_gap_threshold": 2.5,
    "use_hist_equalization": True,
    "ref_width": 2560,
    "ref_height": 1440,
    "door_coords": {"left": 1134, "top": 550, "width": 583, "height": 300},
    "window_pos": [50, 600],
    "window_size": [400, 320],
    # matcher_mode: sift = SIFT 特征匹配（尺度/旋转不变，无需手工裁门模板）
    #               template = 旧的 128x128 门区域 MAE 比对
    "matcher_mode": "sift",
    "sift_clahe_limit": 3.0,
    "sift_ratio": 0.75,
    "sift_min_good": 8,
    "sift_min_inliers": 12,
    "sift_ransac_threshold": 8.0,
    "sift_nfeatures": 0,
}
VALID_HOTKEYS = ["f%d" % i for i in range(2, 13)]
VALID_MODES = ("sift", "template")
DOOR_KEYS = ("left", "top", "width", "height")

CONFIG = {}


def _to_float(value):
    """尽量转成 float；失败返回 None（bool 不算合法数值）。"""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value, low, high):
    if high < low:
        return low
    return max(low, min(value, high))


def cfg_float(key):
    """安全读取数值型配置：缺失或非法时一律回退默认值，杜绝 KeyError。"""
    value = _to_float(CONFIG.get(key, DEFAULT_CONFIG.get(key)))
    if value is None:
        return float(DEFAULT_CONFIG[key])
    return value


def current_mode():
    """解析并校验 matcher_mode；非法值或 SIFT 不可用时自动降级。"""
    mode = str(CONFIG.get("matcher_mode", "sift")).lower()
    if mode not in VALID_MODES:
        mode = "sift"
    if mode == "sift" and SiftMatcher is None:
        mode = "template"
    return mode


def _read_config_file():
    """读取 config.json；文件缺失或损坏时返回空 dict，由默认值兜底。"""
    if not os.path.exists(CONFIG_PATH):
        logger.info("未找到 config.json，将生成默认配置。")
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
            raw = json.load(fp)
    except (OSError, ValueError) as exc:
        logger.error("config.json 读取失败（%s），本次使用默认配置。", exc)
        return {}
    if not isinstance(raw, dict):
        logger.error("config.json 顶层必须是 JSON 对象，本次使用默认配置。")
        return {}
    return raw


def save_config():
    """原子写入配置：先写 .tmp 再 os.replace，避免写一半崩溃把配置写坏。"""
    tmp_path = CONFIG_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(CONFIG, fp, indent=4, ensure_ascii=False)
        os.replace(tmp_path, CONFIG_PATH)
    except OSError as exc:
        logger.error("配置保存失败：%s", exc)


def _merge_config(raw):
    """以默认值为骨架合并磁盘配置：补齐缺失键、剔除已废弃键。"""
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for key in DEFAULT_CONFIG:
        if key in raw:
            merged[key] = raw[key]

    dropped = sorted(set(raw) - set(DEFAULT_CONFIG))
    if dropped:
        logger.warning("忽略 config.json 中已废弃的配置项：%s", "、".join(dropped))

    missing = sorted(set(DEFAULT_CONFIG) - set(raw))
    if missing and raw:
        logger.warning("config.json 缺少配置项，已使用默认值补齐：%s", "、".join(missing))
    return merged


def _sanitize_config():
    """校验关键配置，非法值一律回退默认值，防止用户手改配置把程序搞崩。"""
    for key in ("hotkey", "exit_hotkey", "region_hotkey"):
        if CONFIG.get(key) not in VALID_HOTKEYS:
            logger.warning("%s = %r 非法，已重置为 %s。", key, CONFIG.get(key), DEFAULT_CONFIG[key])
            CONFIG[key] = DEFAULT_CONFIG[key]
    # 三个热键不能两两相同
    if CONFIG["region_hotkey"] in (CONFIG["hotkey"], CONFIG["exit_hotkey"]):
        logger.warning("region_hotkey 与识别/退出热键冲突，已重置为 %s。", DEFAULT_CONFIG["region_hotkey"])
        CONFIG["region_hotkey"] = DEFAULT_CONFIG["region_hotkey"]

    for key in ("ref_width", "ref_height", "error_threshold", "top2_gap_threshold"):
        value = _to_float(CONFIG.get(key))
        invalid = value is None or value < 0
        if key in ("ref_width", "ref_height", "error_threshold"):
            invalid = invalid or value == 0
        if invalid:
            logger.warning("%s = %r 非法，已重置为 %s。", key, CONFIG.get(key), DEFAULT_CONFIG[key])
            CONFIG[key] = DEFAULT_CONFIG[key]
        else:
            CONFIG[key] = int(value) if key in ("ref_width", "ref_height") else float(value)

    door = CONFIG.get("door_coords")
    if (not isinstance(door, dict)
            or any(_to_float(door.get(k)) is None for k in DOOR_KEYS)
            or _to_float(door["left"]) < 0 or _to_float(door["top"]) < 0
            or _to_float(door["width"]) <= 0 or _to_float(door["height"]) <= 0):
        logger.warning("door_coords = %r 非法，已重置为默认值。", door)
        CONFIG["door_coords"] = copy.deepcopy(DEFAULT_CONFIG["door_coords"])
    else:
        CONFIG["door_coords"] = dict((k, int(door[k])) for k in DOOR_KEYS)

    pos = CONFIG.get("window_pos")
    if (not isinstance(pos, (list, tuple)) or len(pos) != 2
            or _to_float(pos[0]) is None or _to_float(pos[1]) is None):
        CONFIG["window_pos"] = list(DEFAULT_CONFIG["window_pos"])
    else:
        CONFIG["window_pos"] = [int(pos[0]), int(pos[1])]

    size = CONFIG.get("window_size")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or _to_float(size[0]) is None or _to_float(size[1]) is None
            or _to_float(size[0]) < 64 or _to_float(size[1]) < 64):
        logger.warning("window_size = %r 非法，已重置为默认值。", size)
        CONFIG["window_size"] = list(DEFAULT_CONFIG["window_size"])
    else:
        CONFIG["window_size"] = [int(size[0]), int(size[1])]

    CONFIG["use_hist_equalization"] = bool(CONFIG.get("use_hist_equalization", True))


def _load_config():
    raw = _read_config_file()
    CONFIG.update(_merge_config(raw))
    _sanitize_config()
    if CONFIG != raw:
        # 自愈：把补齐/修正后的配置写回磁盘，下次启动不再重复告警
        save_config()


_load_config()

# ==============================================================================
# 3. 图像特征：模板与实时截图共用同一套预处理，避免两侧口径不一致
# ==============================================================================
FEATURE_SIZE = 128
DOOR_SUBDIR = "door"    # 门区域裁剪图：仅用于识别
FULL_SUBDIR = "full"    # 整图：仅用于显示
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def list_images(folder):
    """列出目录下的图片文件名并排序，保证每次加载顺序一致。"""
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))


def read_image(path):
    """读图：先读字节再 imdecode，兼容中文/特殊字符路径。"""
    buffer = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def new_mss():
    """创建 mss 截图实例。

    mss 10 起把 mss.mss() 标记为废弃并改名 mss.MSS()，这里做一层兼容：
    老版本走 mss.mss()，新版本走 mss.MSS()，两边都不报警告。
    """
    factory = getattr(mss, "MSS", None) or getattr(mss, "mss")
    return factory()


def door_aspect():
    """door_coords 的宽高比，用于校验门裁剪图的形状是否合理。"""
    door = CONFIG["door_coords"]
    return float(door["width"]) / float(door["height"])


def bgr_to_feature(crop_bgr):
    """把"门"区域裁图统一转成 FEATURE_SIZE 见方的灰度特征矩阵。"""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if CONFIG.get("use_hist_equalization", True):
        # 直方图均衡化，消除光影/亮度差异带来的干扰
        gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)


def extract_door_feature(image_bgr):
    """按参考分辨率等比裁出模板的"门"区域并转成特征。

    返回 (feature, error)：成功时 error 为 None，失败时 feature 为 None。
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, "图像为空"
    height, width = image_bgr.shape[:2]
    scale_x = width / cfg_float("ref_width")
    scale_y = height / cfg_float("ref_height")
    door = CONFIG["door_coords"]

    left = _clamp(int(round(door["left"] * scale_x)), 0, width - 1)
    top = _clamp(int(round(door["top"] * scale_y)), 0, height - 1)
    right = _clamp(left + int(round(door["width"] * scale_x)), left + 1, width)
    bottom = _clamp(top + int(round(door["height"] * scale_y)), top + 1, height)

    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        return None, "门区域 (%d,%d)-(%d,%d) 无效" % (left, top, right, bottom)
    return bgr_to_feature(crop), None


_crop_cache = None
_crop_lock = threading.Lock()


def get_screen_crop(force_refresh=False):
    """按主显示器物理分辨率换算出"门"区域的抓图坐标。

    结果带缓存（分辨率通常不变），返回副本以免调用方改坏缓存。
    force_refresh=True 用于分辨率切换 / 托盘重载。
    """
    global _crop_cache
    with _crop_lock:
        if _crop_cache is not None and not force_refresh:
            return dict(_crop_cache)

        ref_w = cfg_float("ref_width")
        ref_h = cfg_float("ref_height")
        try:
            with new_mss() as sct:
                monitor = sct.monitors[1]
                screen_w, screen_h = int(monitor["width"]), int(monitor["height"])
        except Exception as exc:
            logger.warning("获取屏幕分辨率失败（%s），回退到参考分辨率。", exc)
            screen_w, screen_h = int(ref_w), int(ref_h)
        if screen_w <= 0 or screen_h <= 0:
            screen_w, screen_h = int(ref_w), int(ref_h)

        door = CONFIG["door_coords"]
        scale_x = screen_w / ref_w
        scale_y = screen_h / ref_h
        raw_width = int(round(door["width"] * scale_x))
        raw_height = int(round(door["height"] * scale_y))

        left = _clamp(int(round(door["left"] * scale_x)), 0, screen_w - 1)
        top = _clamp(int(round(door["top"] * scale_y)), 0, screen_h - 1)
        width = _clamp(raw_width, 1, screen_w - left)
        height = _clamp(raw_height, 1, screen_h - top)
        if (width, height) != (raw_width, raw_height):
            logger.warning("door_coords 换算后超出屏幕 %dx%d，已裁剪为 %dx%d。",
                           screen_w, screen_h, width, height)

        _crop_cache = {"left": left, "top": top, "width": width, "height": height}
        logger.info("抓图区域 %dx%d @ (%d, %d)，屏幕 %dx%d。", width, height, left, top, screen_w, screen_h)
        return dict(_crop_cache)


# ==============================================================================
# 4. 匹配器：特征向量化 + Top-2 双阈值防误判
# ==============================================================================
class MatchResult(NamedTuple):
    """一次识别的结果；name 为 None 表示未命中，reason 说明原因。"""

    name: Optional[str]
    error: float
    second_name: Optional[str]
    second_error: float
    reason: str = ""
    display_path: str = ""       # 命中后要显示的整图路径
    score_text: str = ""         # 状态栏显示的分数描述（两种模式口径不同）
    location: Optional[tuple] = None   # 抓图区中心在整图上的坐标 (x, y)
    map_size: Optional[tuple] = None   # 整图尺寸，画标记点用


class _TemplateSet(NamedTuple):
    """模板快照：三个字段整体替换，读取方一次取值就拿到自洽的数据。"""

    stack: Optional[np.ndarray]      # (N, S, S) float32，一次性向量化比对
    names: list                      # 与 stack 第 0 维一一对应的显示名
    display_paths: list              # 与 stack 第 0 维一一对应的显示图路径


class MapMatcher:
    """地图模板特征缓存 + 向量化匹配。"""

    def __init__(self, folder_path, build_features=True):
        self._folder = folder_path
        self._build_features = bool(build_features)
        self._lock = threading.RLock()   # 仅用于串行化 load_templates 的写入
        # 整体替换而不是分字段赋值：读取方无需加锁，也不会看到"名字与特征错位"的中间态
        self._model = _TemplateSet(stack=None, names=[], display_paths=[])
        self.load_templates()   # 构造即加载，调用方不必记得再调一次

    @property
    def template_count(self):
        return len(self._model.names)

    def display_path_for(self, name):
        """按名字取"显示用整图"路径；SIFT 模式复用同一套 门图/整图 配对规则。

        没有配对记录时（比如只放了整图、没放门裁剪图），直接去 full/ 与根目录找同名文件。
        """
        model = self._model
        try:
            return model.display_paths[model.names.index(name)]
        except ValueError:
            pass
        for folder in (os.path.join(self._folder, FULL_SUBDIR), self._folder):
            for ext in IMAGE_EXTS:
                candidate = os.path.join(folder, name + ext)
                if os.path.exists(candidate):
                    return candidate
        return ""

    def load_templates(self):
        """（重新）加载全部模板；单个文件损坏只跳过并告警，不影响其余模板。

        模板来源分两类：
          * maps/door/* —— 门区域裁剪图，整张直接归一化成特征，绝不再裁一次；
          * maps/*      —— 旧结构整图。若存在同名门裁剪图，它只当"显示用整图"；
                           否则按旧逻辑从整图里等比裁出门区域再识别。

        build_features=False 时（SIFT 模式）只建"名字 -> 显示整图"的映射，
        不读图也不提特征 —— SIFT 的识别完全走 sift_matcher，用不到这些模板特征。
        """
        if not self._build_features:
            self._load_display_map_only()
            return len(self._model.names)

        door_dir = os.path.join(self._folder, DOOR_SUBDIR)
        full_dir = os.path.join(self._folder, FULL_SUBDIR)
        door_files = list_images(door_dir)
        root_files = list_images(self._folder)
        full_files = list_images(full_dir)

        if not os.path.isdir(self._folder):
            logger.error("地图目录不存在：%s", self._folder)

        # 名字 -> 显示用整图路径；maps/full/ 优先于 maps/ 根目录同名文件
        display_map = {}
        for filename in root_files:
            display_map[os.path.splitext(filename)[0]] = os.path.join(self._folder, filename)
        for filename in full_files:
            display_map[os.path.splitext(filename)[0]] = os.path.join(full_dir, filename)

        names, features, display_paths, skipped = [], [], [], []
        door_stems = set(os.path.splitext(f)[0] for f in door_files)
        expected_aspect = door_aspect()
        door_loaded = 0
        aspect_offenders = []

        # ---- A. 门区域裁剪图：整张直接当特征，不再二次裁剪 ----
        for filename in door_files:
            path = os.path.join(door_dir, filename)
            image = self._safe_read(path, filename, skipped)
            if image is None:
                continue
            height, width = image.shape[:2]
            aspect = float(width) / float(height) if height else 0.0
            if expected_aspect > 0 and abs(aspect - expected_aspect) / expected_aspect > 0.15:
                aspect_offenders.append("%s(%.2f)" % (filename, aspect))
            stem = os.path.splitext(filename)[0]
            names.append(stem)
            features.append(bgr_to_feature(image))
            # 找不到同名整图时，退化为显示这张门裁剪图本身
            display_paths.append(display_map.get(stem, path))
            door_loaded += 1

        # ---- B. 旧结构整图：没有同名门裁剪图的，按旧逻辑当模板 ----
        for filename in root_files:
            stem = os.path.splitext(filename)[0]
            if stem in door_stems:
                continue          # 已被当作上面某个门模板的显示整图
            path = os.path.join(self._folder, filename)
            image = self._safe_read(path, filename, skipped)
            if image is None:
                continue
            feature, error = extract_door_feature(image)
            if feature is None:
                skipped.append("%s(%s)" % (filename, error))
                continue
            names.append(stem)
            features.append(feature)
            display_paths.append(path)

        stack = np.stack(features) if features else None
        with self._lock:
            self._model = _TemplateSet(stack=stack, names=names, display_paths=display_paths)

        if aspect_offenders:
            # SIFT 模式靠整图特征匹配，这些门模板压根不参与识别，没必要报警
            report = logger.info if CONFIG.get("matcher_mode") == "sift" else logger.warning
            report("有 %d 张门模板的宽高比偏离 door_coords 的 %.2f 超过 15%%（%s%s）；"
                   "当前模式是 %s，只有 template 模式才会受此影响。",
                   len(aspect_offenders), expected_aspect,
                   "、".join(aspect_offenders[:5]),
                   "…" if len(aspect_offenders) > 5 else "",
                   CONFIG.get("matcher_mode"))

        logger.info("成功缓存 %d 张地图特征（门裁剪图 %d 张，整图 %d 张）。",
                    len(names), door_loaded, len(names) - door_loaded)
        if skipped:
            logger.warning("跳过 %d 个不可用模板：%s", len(skipped), "；".join(skipped))
        return len(names)

    def _load_display_map_only(self):
        """只建立 名字 -> 显示整图 的映射；不读图、不提特征（SIFT 模式足够用）。"""
        full_dir = os.path.join(self._folder, FULL_SUBDIR)
        mapping = {}
        for filename in list_images(self._folder):
            mapping[os.path.splitext(filename)[0]] = os.path.join(self._folder, filename)
        for filename in list_images(full_dir):     # full/ 优先覆盖根目录同名文件
            mapping[os.path.splitext(filename)[0]] = os.path.join(full_dir, filename)
        names = sorted(mapping)
        with self._lock:
            self._model = _TemplateSet(stack=None, names=names,
                                       display_paths=[mapping[n] for n in names])
        logger.info("已建立 %d 条『地图名 -> 显示整图』映射（SIFT 模式不需要模板特征）。",
                    len(names))

    @staticmethod
    def _safe_read(path, filename, skipped):
        """读图失败不抛异常，只记入 skipped 列表。"""
        try:
            image = read_image(path)
        except Exception as exc:
            skipped.append("%s(%s)" % (filename, exc))
            return None
        if image is None:
            skipped.append("%s(解码失败)" % filename)
            return None
        return image

    def match(self, screen_crop_bgr):
        """对实时抓取的"门"区域做匹配，返回 MatchResult。"""
        try:
            feature = bgr_to_feature(screen_crop_bgr)
        except cv2.error as exc:
            return MatchResult(name=None, error=float("inf"), second_name=None,
                               second_error=float("inf"), reason="特征提取失败：%s" % exc)

        # 一次属性读取拿到自洽快照，无需加锁
        model = self._model
        stack, names, display_paths = model.stack, model.names, model.display_paths

        if stack is None or not names:
            return MatchResult(name=None, error=float("inf"), second_name=None,
                               second_error=float("inf"), reason="maps 目录下没有可用的地图模板")

        # 一次算出与所有模板的平均绝对误差，避免 Python 层逐张循环
        errors = np.abs(stack - feature).mean(axis=(1, 2))
        order = np.argsort(errors, kind="stable")
        best_index = int(order[0])
        second_index = int(order[1]) if order.size > 1 else -1

        best_error = float(errors[best_index])
        second_error = float(errors[second_index]) if second_index >= 0 else float("inf")
        second_name = names[second_index] if second_index >= 0 else None

        threshold = cfg_float("error_threshold")
        if best_error > threshold:
            return MatchResult(name=None, error=best_error, second_name=second_name,
                               second_error=second_error,
                               reason="最优误差 %.2f 超过阈值 %.2f" % (best_error, threshold))

        gap = cfg_float("top2_gap_threshold")
        if second_error - best_error < gap:
            return MatchResult(
                name=None, error=best_error, second_name=second_name, second_error=second_error,
                reason="%s 与 %s 过于相似（误差间距 %.2f < %.2f），拒绝判定"
                       % (names[best_index], second_name, second_error - best_error, gap))

        return MatchResult(name=names[best_index], error=best_error,
                           second_name=second_name, second_error=second_error,
                           display_path=display_paths[best_index])


# ==============================================================================
# 5. 识别线程：mss 线程内单例复用 + 连发保护 + 预检查 + 可控退出
# ==============================================================================
class MatcherWorker(QThread):
    """后台识别线程，唯一接触 mss / OpenCV 的地方。"""

    result_ready = pyqtSignal(object)   # 载荷为 MatchResult

    # 轮询间隔：_trigger 的 clear 语义与退出信号共用同一个 Event，
    # 用带超时的 wait 可以保证 stop() 永远不会被漏掉（纯阻塞 wait 会死等）。
    POLL_INTERVAL = 0.2
    RETRY_INTERVAL = 0.5

    def __init__(self, matcher, parent=None):
        super().__init__(parent)
        self._matcher = matcher          # 模板匹配引擎（旧方案，保留作备用/降级）
        self._sift = None                # SIFT 引擎，在线程里构建
        self._mode = current_mode()
        if str(CONFIG.get("matcher_mode", "sift")).lower() not in VALID_MODES:
            logger.warning("matcher_mode = %r 非法，按 sift 处理。", CONFIG.get("matcher_mode"))
        if self._mode == "template" and SiftMatcher is None:
            logger.error("SIFT 模块不可用（%s），已降级为 template 模式。", SIFT_IMPORT_ERROR)
        self._trigger = threading.Event()
        self._running = True

    # ---------------- 供 GUI 线程调用 ----------------
    def trigger(self):
        """热键触发一次识别；上一次尚未完成时忽略本次请求（连发保护）。"""
        if not self._running or self._trigger.is_set():
            return
        self._trigger.set()

    def stop(self):
        """请求退出并等待线程回收；幂等，可在退出流程里重复调用。"""
        self._running = False
        self._trigger.set()   # 立即唤醒可能正阻塞在 wait() 上的线程
        if not self.wait(3000):
            logger.warning("识别线程未在 3 秒内退出，继续退出流程。")

    # ---------------- 线程体 ----------------
    def run(self):
        self._build_engines()
        logger.info("识别线程已启动（模式 %s），等待 %s 触发……",
                    self._mode, CONFIG["hotkey"].upper())
        sct = None
        try:
            while self._running:
                self._trigger.wait(timeout=self.POLL_INTERVAL)
                if not self._trigger.is_set():
                    continue
                if not self._running:
                    # 退出时不要再白做一次抓图 + 匹配
                    break
                try:
                    if sct is None:
                        sct = self._create_sct()
                        if sct is None:
                            time.sleep(self.RETRY_INTERVAL)
                            continue
                    self._handle_once(sct)
                except Exception:
                    logger.exception("本轮识别出现未预期异常，已跳过。")
                    sct = self._discard_sct(sct)   # 下次重建，避免坏实例反复报错
                finally:
                    # 处理结束后才复位，保证处理期间的热键连发被丢弃
                    self._trigger.clear()
        except Exception:
            logger.exception("识别线程异常退出。")
        finally:
            self._discard_sct(sct)
            logger.info("识别线程已退出。")

    @staticmethod
    def _create_sct():
        """在当前线程内创建 mss 实例——mss 不是线程安全的，绝不能跨线程共享。"""
        try:
            sct = new_mss()
            logger.info("mss 截图实例初始化成功。")
            return sct
        except Exception as exc:
            logger.error("mss 初始化失败：%s", exc)
            return None

    @staticmethod
    def _discard_sct(sct):
        if sct is not None:
            try:
                sct.close()
                logger.info("mss 实例已释放。")
            except Exception as exc:
                logger.warning("释放 mss 实例失败：%s", exc)
        return None

    @staticmethod
    def _precheck(image):
        """抓图有效性预检：全黑/全白/无纹理通常意味着窗口模式或坐标不对，直接短路。"""
        if image is None or image.size == 0:
            return False, "截图为空"
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = float(gray.mean())
        std_val = float(gray.std())
        if mean_val < 10 or mean_val > 245:
            return False, "画面全黑或全白（均值 %.1f），请确认游戏为无边框窗口" % mean_val
        if std_val < 5:
            return False, "画面无纹理（标准差 %.1f），请检查 door_coords 是否抓偏" % std_val
        return True, ""

    def _handle_once(self, sct):
        crop = get_screen_crop()
        try:
            raw = sct.grab(crop)
        except Exception as exc:
            logger.error("截图失败：%s", exc)
            self.result_ready.emit(MatchResult(name=None, error=float("inf"), second_name=None,
                                               second_error=float("inf"),
                                               reason="截图失败：%s" % exc))
            return

        image = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)

        ok, reason = self._precheck(image)
        if not ok:
            logger.error("【预检查】%s", reason)
            self.result_ready.emit(MatchResult(name=None, error=float("inf"), second_name=None,
                                               second_error=float("inf"), reason=reason))
            return

        result = self._identify(image)
        if result.name:
            logger.info("【命中】%s（%s，次优 %s）", result.name,
                        result.score_text or "误差 %.2f" % result.error,
                        result.second_name or "无")
        else:
            logger.warning("【未匹配】%s", result.reason)
        self.result_ready.emit(result)

    def _build_engines(self):
        """把 SIFT 索引的构建放在识别线程里做，避免拖住 GUI 启动。"""
        if self._mode != "sift":
            return
        try:
            engine = SiftMatcher(
                MAP_FOLDER,
                clahe_limit=cfg_float("sift_clahe_limit"),
                nfeatures=int(cfg_float("sift_nfeatures")),
                ratio=cfg_float("sift_ratio"),
                min_good=int(cfg_float("sift_min_good")),
                min_inliers=int(cfg_float("sift_min_inliers")),
                ransac_threshold=cfg_float("sift_ransac_threshold"),
            )
            count, skipped = engine.build()
            if skipped:
                logger.warning("SIFT 跳过 %d 张地图：%s", len(skipped), "；".join(skipped))
            if count == 0:
                logger.error("SIFT 一张地图都没索引成功，降级为 template 模式。")
                self._mode = "template"
                return
            self._sift = engine
            logger.info("SIFT 索引就绪：%d 张地图、%d 个特征点，耗时 %.1fs。",
                        count, engine.keypoint_total, engine.last_build_seconds)
        except Exception:
            logger.exception("SIFT 引擎初始化失败，降级为 template 模式。")
            self._mode = "template"

    def _identify(self, image):
        """按当前模式识别，统一返回 MatchResult。"""
        if self._mode == "sift" and self._sift is not None:
            hit = self._sift.identify(image)
            if hit.name:
                return MatchResult(
                    name=hit.name,
                    error=float(hit.inliers),
                    second_name=hit.second_name,
                    second_error=float(hit.second_inliers),
                    display_path=self._matcher.display_path_for(hit.name),
                    score_text="内点 %d" % hit.inliers,
                    location=hit.location,
                    map_size=hit.target_size,
                )
            return MatchResult(name=None, error=float(hit.inliers),
                               second_name=hit.second_name,
                               second_error=float(hit.second_inliers),
                               reason=hit.reason)
        return self._matcher.match(image)


# ==============================================================================
# 6. 悬浮窗与托盘：位置记忆 + 鼠标穿透 + 结果可视化
# ==============================================================================

class RegionSelector(QWidget):
    """全屏半透明遮罩，鼠标拖拽框选截图区域；Enter/双击确认，Esc 取消。"""

    def __init__(self, overlay, parent=None):
        super().__init__(parent)
        self._overlay = overlay
        self._start = QPoint()
        self._end = QPoint()
        self._dragging = False

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 用 virtualGeometry 覆盖所有显示器，而不是只覆盖主屏
        self.setGeometry(QApplication.primaryScreen().virtualGeometry())
        self.setCursor(Qt.CrossCursor)
        # 无边框 Tool 窗口默认拿不到键盘焦点；不设这个 Enter/Esc 会失效，只剩双击能确认
        self.setFocusPolicy(Qt.StrongFocus)
        self.setWindowTitle("MapMatcher - 选择截图区域")

    def mousePressEvent(self, event):
        self._start = event.globalPos()
        self._end = self._start
        self._dragging = True
        self.update()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._end = event.globalPos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._end = event.globalPos()
            self._dragging = False
            self.update()

    def mouseDoubleClickEvent(self, event):
        self._confirm()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm()
        elif event.key() == Qt.Key_Escape:
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
        # _norm_rect() 存的是全局坐标（要写进 door_coords），绘制前必须换算成本窗口的局部坐标；
        # 主屏在原点时两者恰好相等，多显示器下不换算就会画错位置。
        rect = self._norm_rect().translated(-self.pos())
        if rect.width() > 4 and rect.height() > 4:
            # 挖空选框，露出底下游戏画面，方便对齐
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(rect, Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(255, 90, 90), 2))
            painter.drawRect(rect)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect.x() + 4, rect.y() + rect.height() - 8,
                             "%d×%d @ (%d, %d)" % (rect.width(), rect.height(), rect.x(), rect.y()))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(16, 28, "拖拽选择截图区域 ｜ Enter / 双击 确认 ｜ Esc 取消")

    def _norm_rect(self):
        x1, y1 = self._start.x(), self._start.y()
        x2, y2 = self._end.x(), self._end.y()
        return QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def _confirm(self):
        rect = self._norm_rect()
        if rect.width() < 8 or rect.height() < 8:
            self.close()
            return
        # 选框是 Qt 的逻辑像素，而 door_coords 要交给 mss 用，必须是物理像素。
        # 本机 100% 缩放下 ratio == 1.0，这里是个安全的空操作。
        ratio = self.devicePixelRatioF() or 1.0
        self._overlay.apply_region(int(rect.x() * ratio), int(rect.y() * ratio),
                                   int(rect.width() * ratio), int(rect.height() * ratio))
        self.close()


class OverlayWindow(QWidget):
    """鼠标穿透的悬浮窗，附带托盘菜单。"""

    # keyboard 的回调运行在它自己的钩子线程里，必须经信号投递到 GUI 线程执行
    sig_trigger = pyqtSignal()
    sig_quit = pyqtSignal()
    sig_select_region = pyqtSignal()

    def __init__(self, matcher, parent=None):
        super().__init__(parent)
        self.matcher = matcher
        self._pixmap = None
        self._location = None      # 抓图区中心在整图上的坐标
        self._map_size = None      # 整图尺寸
        self._closing = False

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)   # 鼠标穿透，不挡游戏操作
        self.setWindowTitle("MapMatcher")

        self._build_ui()

        pos = CONFIG.get("window_pos", DEFAULT_CONFIG["window_pos"])
        self.move(int(pos[0]), int(pos[1]))

        self.worker = MatcherWorker(matcher, self)
        self.worker.result_ready.connect(self.update_display)
        self.worker.start()

        self._build_tray()

        self.sig_trigger.connect(self.worker.trigger)
        self.sig_quit.connect(self.close_program)
        self.sig_select_region.connect(self.open_region_selector)

    # ---------------- 界面 ----------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.image_label = QLabel("准备就绪", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 150); color: #DDDDDD; font-size: 13px;")

        # 悬浮窗是无边框 Tool 窗口，标题栏不可见——分数/原因必须显示在这里
        self.status_label = QLabel("按 %s 识别" % CONFIG["hotkey"].upper(), self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 200); color: #FFFFFF; font-size: 12px; padding: 3px;")

        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.status_label, 0)

        size = CONFIG.get("window_size", DEFAULT_CONFIG["window_size"])
        self.resize(int(size[0]), int(size[1]))

    def _build_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        icon_path = os.path.join(EXE_DIR, "icon.ico")
        if os.path.exists(icon_path):
            self.tray_icon.setIcon(QIcon(icon_path))
        else:
            self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.tray_icon.setToolTip("MapMatcher —— 地图识别")

        # QSystemTrayIcon 不接管菜单所有权，必须用 self 属性持有；
        # 存成局部变量会被 GC 回收，菜单随即失效。
        self.tray_menu = QMenu(self)
        action_show = QAction("显示 / 隐藏悬浮窗", self)
        action_reload = QAction("重新加载地图模板", self)
        action_region = QAction("选择截图区域", self)
        action_quit = QAction("退出程序", self)
        action_show.triggered.connect(self.toggle_window)
        action_reload.triggered.connect(self.reload_templates)
        action_region.triggered.connect(self.open_region_selector)
        action_quit.triggered.connect(self.close_program)
        self.tray_menu.addAction(action_show)
        self.tray_menu.addAction(action_reload)
        self.tray_menu.addAction(action_region)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(action_quit)

        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.show()

    def toggle_window(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()

    def reload_templates(self):
        """热重载地图模板，新增地图后无需重启；顺带刷新抓图区域缓存。"""
        get_screen_crop(force_refresh=True)
        count = self.matcher.load_templates()
        message = "已加载 %d 张地图模板" % count if count else "maps 目录下没有可用的地图模板"
        self.status_label.setText(message)
        logger.info("托盘触发重新加载：%s", message)
        self.tray_icon.showMessage("MapMatcher", message, QSystemTrayIcon.Information, 3000)

    # ---------------- 结果显示 ----------------
    def update_display(self, result):
        if result.name:
            # 识别用的是门区域/SIFT 特征，显示用的是配对好的那张完整地图
            pixmap = QPixmap(result.display_path) if result.display_path else QPixmap()
            if pixmap.isNull():
                logger.error("命中 %s，但显示图无法读取：%s", result.name, result.display_path)
                self._show_text("地图图片读取失败")
                self.status_label.setText("命中 %s，但显示图无法读取" % result.name)
                return
            self._location = result.location
            self._map_size = result.map_size
            self._show_pixmap(pixmap)
            score = result.score_text or ("误差 %.2f" % result.error)
            self.status_label.setText("命中 %s ｜ %s" % (result.name, score))
            self.setWindowTitle("MapMatcher - %s (%s)" % (result.name, score))
        else:
            self._location = None
            self._map_size = None
            self._show_text("未找到匹配的地图")
            self.status_label.setText(result.reason or "未找到匹配的地图")
            self.setWindowTitle("MapMatcher - 未匹配")

    def _show_pixmap(self, pixmap):
        self._pixmap = pixmap
        self.image_label.setText("")
        self._refresh_pixmap()

    def _show_text(self, text):
        self._pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(text)

    def _refresh_pixmap(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.image_label.size(),
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if self._location and self._map_size and self._map_size[0] and self._map_size[1]:
            # 把整图坐标换算成缩放后的像素坐标，画一个红点标出抓图区在整图上的位置
            sx = scaled.width() / float(self._map_size[0])
            sy = scaled.height() / float(self._map_size[1])
            point = QPoint(int(self._location[0] * sx), int(self._location[1] * sy))
            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QBrush(QColor(255, 40, 40)))
            painter.drawEllipse(point, 7, 7)
            painter.end()
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()   # 窗口尺寸变化后重新缩放，避免拉伸失真

    # ---------------- 退出 ----------------
    def close_program(self):
        """退出流程：保存窗口位置 -> 停线程 -> 关托盘 -> 退出（幂等）。"""
        if self._closing:
            return
        self._closing = True
        logger.info("正在退出程序……")
        try:
            CONFIG["window_pos"] = [self.x(), self.y()]
            save_config()
            self.worker.stop()
        except Exception:
            logger.exception("退出清理过程中出现异常。")
        finally:
            self.tray_icon.hide()
            QApplication.quit()

    # ---------------- 截图区域框选 ----------------
    def open_region_selector(self):
        """进入全屏框选模式，让用户拖拽选择截图区域（替代写死的 door_coords）。"""
        if getattr(self, "_selector", None) and self._selector.isVisible():
            return
        self._selector = RegionSelector(self, self)
        self._selector.show()
        # 必须显式抢焦点，否则 keyPressEvent 收不到 Enter/Esc
        self._selector.raise_()
        self._selector.activateWindow()
        self._selector.setFocus()

    def apply_region(self, left, top, width, height):
        """把用户框选的屏幕真实像素区域写入配置并刷新抓图缓存。"""
        try:
            with new_mss() as sct:
                monitor = sct.monitors[1]
                sw, sh = int(monitor["width"]), int(monitor["height"])
        except Exception as exc:
            logger.warning("获取屏幕分辨率失败（%s），以选框推断参考分辨率。", exc)
            sw, sh = left + width, top + height
        CONFIG["door_coords"] = {"left": int(left), "top": int(top),
                                 "width": int(width), "height": int(height)}
        CONFIG["ref_width"], CONFIG["ref_height"] = sw, sh
        save_config()
        get_screen_crop(force_refresh=True)
        msg = "截图区域已更新：%dx%d @ (%d, %d)" % (width, height, left, top)
        self.status_label.setText(msg)
        self.tray_icon.showMessage("MapMatcher", msg, QSystemTrayIcon.Information, 3000)
        logger.info(msg)
        # 立刻用新区域试识别一次，用户不必再按一次热键就能看到效果
        self.worker.trigger()

# ==============================================================================
# 7. 程序入口
# ==============================================================================
def main():
    # 单实例锁必须在创建 QApplication 之前持有；对象要活到进程结束
    shared_memory = QSharedMemory("MapMatcher_SingleInstance_v1")
    if not shared_memory.create(1):
        logger.error("程序已在运行，请勿重复打开！")
        return 0

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)   # 悬浮窗只是隐藏，不应连带退出程序

    icon_path = os.path.join(EXE_DIR, "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    mode = current_mode()
    # SIFT 模式的识别完全由 sift_matcher 负责，这里只需要"名字 -> 显示整图"的映射
    matcher = MapMatcher(MAP_FOLDER, build_features=(mode != "sift"))
    if mode == "template" and matcher.template_count == 0:
        logger.error("maps 目录下没有可用的地图模板。可把门区域裁剪图放进 maps/%s/，"
                     "整图放进 maps/%s/（按同名配对）；旧结构下 maps/ 里的整图也能直接用。",
                     DOOR_SUBDIR, FULL_SUBDIR)

    overlay = OverlayWindow(matcher)
    overlay.show()

    try:
        # 注意：传的是信号 emit 而不是槽函数本身——回调发生在钩子线程，
        # 直接执行会跨线程碰 GUI 对象（尤其 QApplication.quit()）导致随机崩溃。
        keyboard.add_hotkey(CONFIG["hotkey"], overlay.sig_trigger.emit)
        keyboard.add_hotkey(CONFIG["exit_hotkey"], overlay.sig_quit.emit)
        keyboard.add_hotkey(CONFIG["region_hotkey"], overlay.sig_select_region.emit)
    except Exception:
        logger.exception("热键注册失败。")
        return 1

    app.aboutToQuit.connect(keyboard.unhook_all)
    app.aboutToQuit.connect(overlay.worker.stop)   # 兜底：保证线程一定被回收

    logger.info("程序启动成功！")
    logger.info("热键：按 %s 识别，按 %s 退出，按 %s 选择截图区域。",
                CONFIG["hotkey"].upper(), CONFIG["exit_hotkey"].upper(), CONFIG["region_hotkey"].upper())
    logger.info("配置路径：%s", CONFIG_PATH)
    logger.info("地图目录：%s", MAP_FOLDER)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
