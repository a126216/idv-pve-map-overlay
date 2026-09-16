# 贡献指南 / Contributing

感谢你考虑为 **MapMatcher** 贡献力量。本文件说明如何本地搭建、运行测试、提交改动。

Thanks for considering contributing to **MapMatcher**. This guide covers local setup, tests, and how to submit changes.

---

## 一、开发环境 / Development setup

需要 Python 3.11+（本仓库 CI 覆盖 3.11 / 3.12）。

Requires Python 3.11+ (CI covers 3.11 / 3.12).

```bash
# 1) 克隆
git clone https://github.com/a126216/idv-pve-map-overlay.git
cd idv-pve-map-overlay

# 2) 创建虚拟环境（推荐）
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate    # Linux / macOS

# 3) 安装依赖
pip install -r requirements.txt
```

依赖说明：

- `opencv-python` 或 `opencv-python-headless`：SIFT 引擎与冒烟测试所需。开发机建议装完整版（含 GUI 后端）。
- `PyQt5`：悬浮窗与托盘 UI 所需。
- `mss` / `keyboard`：屏幕抓取与全局热键。
- `numpy`：矩阵运算。

---

## 二、地图素材 / Map assets（重要）

**仓库不包含任何游戏地图素材**，也不会接收它们。原因：

- 游戏地图版权归各自厂商所有，公开分发存在 DMCA 风险；
- 避免把个人素材混入开源仓库。

请自行把素材放到本地 `maps/` 目录（结构见 `maps/README.md`），它已被 `.gitignore` 忽略，不会进入版本控制。

The repository **ships no game-map assets** and will not accept them (copyright / DMCA). Place your own assets in the local `maps/` directory (see `maps/README.md`); it is gitignored.

---

## 三、运行 / Run

```bash
python main.py        # 启动悬浮识别工具（需先放好 maps/ 素材）
python probe.py       # 打开游戏时运行，检查当前截图区是否匹配
```

自定义截图区域：按 <kbd>F3</kbd>（或托盘右键「选择截图区域」）拖拽框选屏幕区域，
确认后自动写入 `config.json`，换分辨率/换电脑无需改坐标。

---

## 四、测试 / Tests

冒烟测试使用**合成图**，不依赖任何素材或显示器：

```bash
python tests/test_smoke.py     # 直接运行，失败以非零码退出
pytest tests/test_smoke.py     # 或走 pytest
```

- `test_sift_identify`：验证 SIFT 引擎在合成图里正确认出来源并反算坐标。
  仅需 `numpy` + `opencv-python-headless`，可在无 GUI 的 CI 环境运行。
- `test_template_match`：验证 template(MAE) 模式；仅当环境装有 PyQt5 时运行，
  CI 中自动跳过。

GitHub Actions 会在每次 push / PR 时于 Ubuntu + Python 3.11/3.12 上自动运行该测试。

---

## 五、代码风格 / Code style

- 默认使用 UTF-8 编码，保留中文注释（项目刻意如此）。
- 工具为**单文件** `main.py` + 可独立复用的 `sift_matcher.py`，请勿无必要地拆分。
- 提交前请本地跑通冒烟测试。

---

## 六、提交改动 / Submitting changes

1. Fork 本仓库并新建分支：`git checkout -b fix/your-topic`
2. 做出改动并确保 `python tests/test_smoke.py` 通过。
3. 提交信息用简洁的中文或英文描述「做了什么 + 为什么」。
4. 发起 Pull Request，简述改动动机。

我们会在 CI 通过后审阅。欢迎 Issue 讨论想法。

1. Fork and branch: `git checkout -b fix/your-topic`
2. Make changes; ensure `python tests/test_smoke.py` passes locally.
3. Write a concise commit message describing *what* and *why*.
4. Open a Pull Request describing the motivation.

We review once CI is green. Issues for discussion are welcome.
