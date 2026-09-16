# MapMatcher

按热键 → 截取游戏屏幕上一块固定区域 → 认出这是哪张地图 → 悬浮窗显示该地图整图，并在对应位置画一个红点。

## 两种识别模式

| 模式 | 配置 | 原理 | 说明 |
|---|---|---|---|
| **sift**（默认） | `"matcher_mode": "sift"` | CLAHE 增强 → SIFT 特征 → FLANN kNN → Lowe ratio → RANSAC 单应 | **尺度/旋转不变，不需要手工裁任何模板**。直接把完整地图建成索引 |
| template | `"matcher_mode": "template"` | 128×128 灰度 MAE + Top-2 双阈值 | 兜底方案，需要 `maps/door/` 下的门区域裁剪图 |

SIFT 引擎的架构参考 [Game-Map-Tracker](https://github.com/761696148/Game-Map-Tracker)
（CLAHE / SIFT / FLANN / Lowe ratio / RANSAC 单应这套配方），代码为本项目独立重写，
见 `sift_matcher.py`；那边是「已知哪张图、只做单图跟点」，这边扩展成了「在 N 张候选图里先认出是哪一张」。

## 快速开始

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

- `F2` 识别，`F12` 退出；托盘图标右键可「显示/隐藏悬浮窗」「重新加载地图模板」「退出程序」。
- 启动后界面立刻出现，SIFT 索引在后台线程建立（18 张图约 4~5 秒），期间日志会打印
  `SIFT 索引就绪：18 张地图、114973 个特征点`。
- 识别出的地图会显示在悬浮窗里，红点标出抓图区在整图上的位置。
- 悬浮窗尺寸由 `config.json` 的 `window_size` 控制（默认 `[400, 320]`）。

## 从仓库运行

仓库不含游戏地图素材与用户配置（均已 gitignore）。克隆后：

```
pip install -r requirements.txt
cp config.example.json config.json   # 可选，首次启动会自动生成默认配置
python main.py                       # Linux/macOS
.venv\Scripts\python.exe main.py     # Windows
```

首次启动若 `maps/` 下没有素材，悬浮窗会提示"没有可用的地图模板"，属正常现象——
把你的地图图片放进 `maps/` 根目录（SIFT 模式）即可，详见下文「地图目录」。

## 地图素材与版权

`maps/` 下的游戏地图素材版权归各自游戏厂商所有，**不随本仓库分发**，仅供个人本地使用。
本项目的核心价值在于识别引擎与工具链代码；地图图片请自行准备，或用 `probe.py` 辅助标定。

## 目录结构

```
main.py           主程序（GUI / 识别线程 / 配置管理）
sift_matcher.py   SIFT 识别引擎
selftest.py       无界面自检：模板加载 + 门模板自匹配
selftest_sift.py  无界面自检：SIFT 引擎认图能力（含旋转/缩放/噪声）
probe.py          标定工具：判断屏幕抓图区到底是不是地图内容
maps/             地图图片
config.json       配置（缺失/非法项自动回退并回写）
```

## 地图目录

```
maps/map1.jpg ... 18 张完整地图   ← SIFT 模式直接索引这些
maps/door/*.png   门区域裁剪图     ← 仅 template 模式使用
maps/full/*.jpg   整图覆盖位       ← 可选，优先于根目录同名文件
```

**SIFT 模式下 `maps/` 根目录的完整地图就是识别依据**，`maps/door/` 不参与识别
（所以那里的裁剪宽高比是否标准已经无所谓了）。

## 主要配置项

| 键 | 默认 | 说明 |
|---|---|---|
| `hotkey` / `exit_hotkey` | `f2` / `f12` | 识别 / 退出热键（f2~f12） |
| `matcher_mode` | `sift` | `sift` 或 `template` |
| `door_coords` | 2560×1440 屏上的 `(1134,550) 583×300` | **抓图区**。SIFT 模式下它就是「小地图/待识别区域」 |
| `ref_width` / `ref_height` | 2560 / 1440 | 参考分辨率，`door_coords` 按屏幕实际分辨率等比换算 |
| `sift_min_inliers` | 12 | RANSAC 内点数下限，认不出就调低 |
| `sift_ratio` | 0.75 | Lowe's ratio，调大=更宽松 |
| `sift_clahe_limit` | 3.0 | CLAHE 对比度增强强度（弱纹理地形有用） |
| `window_size` | `[400, 320]` | 悬浮窗尺寸 |

## 自检 / 标定

```
.venv\Scripts\python.exe selftest_sift.py   # SIFT 引擎认图能力（正常应 18/18 全对）
.venv\Scripts\python.exe selftest.py        # 门模板加载与自匹配（template 模式用）
.venv\Scripts\python.exe probe.py           # 游戏开着时跑，判断抓图区里到底是不是地图内容
```

`probe.py` 会把抓图存到系统临时目录（可用命令行参数指定路径），并输出两块诊断面板：
A. 抓图 vs 门模板的 MAE 排名；B. 抓图与 `maps/` 下各张完整地图的多尺度相关系数。
若 B 面板某张图相关系数明显偏高，说明抓图区确实是地图内容。

## 测试与 CI

无素材冒烟测试：用合成图验证 SIFT 引擎（以及装有 PyQt5 时的 template 模式）能正确工作，
不依赖游戏素材、也不需要显示器：

```
python tests/test_smoke.py
```

GitHub Actions 在每次推送 / PR 时自动运行（见 `.github/workflows/ci.yml`），仅安装
`opencv-python-headless` + `numpy` 即可完成（template 测试在缺 PyQt5 时自动跳过）。

## 打包为可执行文件

已提供 PyInstaller 配置 `MapMatcher.spec`（窗口化、无控制台窗口）：

```
pip install pyinstaller
pyinstaller MapMatcher.spec
```

产出 `dist/MapMatcher/` 目录；本地的 `maps/` 与 `icon.ico` 存在时会一并打入。

## 发布 Release

1. 按上节用 PyInstaller 构建出 `dist/MapMatcher/`；将其压缩为 `MapMatcher-win.zip`。
2. 在 GitHub 仓库页面 → **Releases** → **Draft a new release**：
   - 填 Tag（如 `v1.0.0`）与标题；
   - 把 zip 作为附件上传；
   - 正文注明「地图素材需用户自备（版权归游戏厂商，不随发行包分发）」。
3. 点 **Publish release** 即可。

> 本仓库当前为 Private，需在 **Settings → Change visibility** 改为 Public，简历链接才会对外可访问。

## 版权提示

- `maps/` 下的游戏地图素材版权归游戏厂商所有，仅供个人本地使用。
- SIFT 引擎的**思路**参考 Game-Map-Tracker（仓库内 LICENSE 为 Apache-2.0，属其集成的 LoFTR 组件）；
  该项目自身代码未明确声明许可，因此本项目的 `sift_matcher.py` 是独立实现，未复制其源码。

---

# MapMatcher (English)

Press a hotkey → capture a fixed region of the game screen → recognize which map it is →
show the full map in an overlay window, with a red dot marking where the captured region lies on that map.

## Two recognition modes

| Mode | Config | Principle | Notes |
|---|---|---|---|
| **sift** (default) | `"matcher_mode": "sift"` | CLAHE → SIFT features → FLANN kNN → Lowe ratio → RANSAC homography | **Scale/rotation invariant, no manual templates needed.** Index the full maps directly |
| template | `"matcher_mode": "template"` | 128×128 grayscale MAE + Top-2 dual threshold | Fallback; requires door-region crops under `maps/door/` |

The SIFT engine's architecture is inspired by
[Game-Map-Tracker](https://github.com/761696148/Game-Map-Tracker)
(CLAHE / SIFT / FLANN / Lowe ratio / RANSAC homography); the code here is an independent
rewrite — see `sift_matcher.py`. That project tracks a point on a *known* image; this one
extends it to *first decide which of N candidate maps* the capture belongs to.

## Quick start

```
pip install -r requirements.txt
python main.py
```

- `F2` to recognize, `F12` to quit. Right-click the tray icon for "Show/Hide overlay",
  "Reload maps", and "Quit".
- The UI appears immediately; the SIFT index is built in a background thread
  (≈4–5 s for 18 maps). Logs print `SIFT 索引就绪：18 张地图、114973 个特征点`.
- The recognized map is shown in the overlay; the red dot marks the capture region on the full map.
- Overlay size is controlled by `config.json`'s `window_size` (default `[400, 320]`).

## Run from a clone

The repo ships without game-map assets or user config (both gitignored). After cloning:

```
pip install -r requirements.txt
cp config.example.json config.json   # optional; a default config is auto-generated on first run
python main.py
```

On first run, if `maps/` has no assets the overlay shows "no usable map templates" — that's
expected. Drop your own map images into `maps/` (root dir for SIFT mode); see "Map directory" below.

## Map assets & copyright

Game-map assets under `maps/` are owned by their respective publishers and are **not distributed**
with this repo — for personal local use only. The value of this project is the recognition engine
and tooling; supply your own images, or use `probe.py` to calibrate.

## Project layout

```
main.py           entry point (GUI / worker thread / config management)
sift_matcher.py   SIFT recognition engine
selftest.py       headless self-test: template loading + door-template self-match
selftest_sift.py  headless self-test: SIFT recognition (with rotation/scale/noise)
probe.py          calibration tool: is the captured region actually map content?
maps/             map images (gitignored)
config.json       config (missing/invalid keys auto-fallback and rewritten)
```

## Map directory

```
maps/map1.jpg ... full maps        <- indexed directly by SIFT mode
maps/door/*.png  door-region crops  <- template mode only
maps/full/*.jpg  full-image override <- optional, takes precedence over same-named root file
```

**In SIFT mode the full maps under `maps/` are the recognition basis**; `maps/door/` is not used
(so crop aspect ratios there no longer matter).

## Key config items

| Key | Default | Notes |
|---|---|---|
| `hotkey` / `exit_hotkey` | `f2` / `f12` | recognize / quit hotkeys (f2–f12) |
| `matcher_mode` | `sift` | `sift` or `template` |
| `door_coords` | `(1134,550) 583×300` on a 2560×1440 screen | **capture region**; in SIFT mode it's the "minimap / region to identify" |
| `ref_width` / `ref_height` | 2560 / 1440 | reference resolution; `door_coords` is scaled to the actual screen |
| `sift_min_inliers` | 12 | RANSAC inlier floor; lower it if recognition fails |
| `sift_ratio` | 0.75 | Lowe's ratio; higher = looser |
| `sift_clahe_limit` | 3.0 | CLAHE contrast strength (helps low-texture terrain) |
| `window_size` | `[400, 320]` | overlay window size |

## Self-tests / calibration

```
python selftest_sift.py   # SIFT recognition (should be 100% correct)
python selftest.py        # door-template loading & self-match (template mode)
python probe.py           # run while the game is open to check the capture region
```

## Tests & CI

A synthetic-image smoke test runs in CI without any game assets or a display:

```
python tests/test_smoke.py
```

It builds a SIFT index from generated images and asserts each cropped region is recognized
correctly (and, on machines with PyQt5, that the template/MAE path matches itself).

## Building a standalone executable

A PyInstaller spec is provided (`MapMatcher.spec`):

```
pip install pyinstaller
pyinstaller MapMatcher.spec
```

Produces a `dist/MapMatcher/` directory (windowed, no console). Local `maps/` and `icon.ico`
are bundled when present.

## Releasing

1. Build with PyInstaller as above to get `dist/MapMatcher/`; zip it as `MapMatcher-win.zip`.
2. On the GitHub repo page → **Releases** → **Draft a new release**:
   - Set a Tag (e.g. `v1.0.0`) and a title;
   - Upload the zip as an asset;
   - Note in the body that map assets are **not** bundled (copyright belongs to the publishers).
3. Click **Publish release**.

> This repo is currently Private. Change it to Public under **Settings → Change visibility** so the link is reachable from your resume.

## License

This project is released under the [MIT License](LICENSE). Game-map assets are not included
and remain the property of their respective publishers.
