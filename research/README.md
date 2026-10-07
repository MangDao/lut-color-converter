# research/ —— 验证脚本

这些脚本不是应用的一部分，是用来**证明应用说的数字是真的**。
每个都能独立跑，跑完自己打印结论。

---

## 怎么跑

这些脚本 import 的是 `app/` 里的共享模块（`camera_spaces`、`lutlib`、
`formats`、`serving`、`demos`），但那几个文件**不在这个目录里**——
仓库里每个模块只有一份，放在 `app/`。

所以跑之前把 `app/` 加进 import 路径：

```bash
# bash / Git Bash
PYTHONPATH=app python research/format_tests.py

# PowerShell
$env:PYTHONPATH="app"; python research\format_tests.py

# cmd
set PYTHONPATH=app && python research\format_tests.py
```

依赖和 `app/` 一样：

```bash
pip install numpy colour-science PyOpenColorIO
```

---

## 脚本清单

### `format_tests.py` —— 文件格式校验

`.3dl` 和 `.vlt` 两个 colour-science 不支持的格式，30 项检查。
包括用 **OpenColorIO 自己的读取器当独立裁判**——因为网格顺序写反了
不会崩溃，只会静默产生一个转置的 LUT，必须靠外部参照才能发现。

```bash
PYTHONPATH=app python research/format_tests.py
```

期望：**30/30 checks passed**

### `p0_hub_roundtrip.py` / `p0_vendor_hub.py` —— 枢纽本身

验证「以 ACES2065-1 为枢纽」这个设计：650 条厂商间路径能不能压成 52 条
而不产生误差。

```bash
PYTHONPATH=app python research/p0_hub_roundtrip.py
```

期望：误差 1.3e-10（浮点极限）

### `usecase_*.py` —— 转换到底会丢多少

三个脚本回答同一个问题——「我有相机 A 的 LUT，我拍相机 B，会丢东西吗」，
用三种不同的采样口径，答案递进：

| 脚本 | 采样方式 | 结果 | 说明 |
|---|---|---|---|
| `usecase_realistic.py` | MacBeth 色卡 + 灰阶 + 色相扫描 | **0 / 4709 损失** | 最接近用户实际遇到的情况 |
| `usecase_extreme.py` | 极端高亮 / 色域边界 / 窄带 | 0～60% | 主动施压，找边界在哪 |
| `usecase_check.py` | 均匀码值立方体 | 23～72% | **上界**。角点是纯红纯绿满曝光，不是真实颜色 |

三个都跑：

```bash
PYTHONPATH=app python research/usecase_realistic.py
PYTHONPATH=app python research/usecase_extreme.py
PYTHONPATH=app python research/usecase_check.py
```

⚠️ 后两个需要本机装有 DaVinci Resolve（去读它随附的厂商 LUT 做对照）。
没装的话会打印一行提示然后正常退出，不会崩。路径可用 `LUT_DIR` 覆盖。

### `make_clf.py` / `make_clf_fujifilm.py` —— CLF / LMT 包装

把 LUT 包成 ACES 的 CLF / LMT，遵守「进 AP0 出 AP0」的契约。
底层用 `ociomakeclf`，其中 Fujifilm 的路线官方缺，是手工补的。

```bash
PYTHONPATH=app python research/make_clf.py
```

### `derive_space.py` —— 反解厂商原色

从官方 OCIO 的 IDT 反推出厂商色彩空间的原色坐标，
用来交叉验证注册表里的数值。

---

## 和厂商 LUT 有关的说明

`usecase_*.py` 读的是**本机 DaVinci Resolve 随附的厂商 LUT**：

```
C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\LUT
```

**只读、不复制、不打包、不分发。** 这个仓库里所有 `.cube` / `.3dl` / `.vlt`
都是脚本自己合成的，**没有任何厂商文件**。

换路径：

```bash
LUT_DIR=/path/to/your/luts PYTHONPATH=app python research/usecase_realistic.py
```

---

## 为什么这些脚本值钱

不是因为它们跑了多少用例，而是因为**它们的验收标准是「对照」而不是「看起来对」**：

- `format_tests.py` 用 OpenColorIO 当裁判，而不是自己判自己
- `usecase_*.py` 用三种采样口径，因为单一口径会给出误导性的结论
- `app/demos.py` 的自检有一列叫 **residual**——把总误差拆成
  「固有损失 + 网格下限 + 转换自身的贡献」，只有最后一项才是代码的责任

一个数字如果和肉眼可见的行为不符，**先怀疑测量代码**。
这个项目里测量口径错过四次，每次都表现为「一个看起来很好的数字」或
「一个看起来灾难性的数字」。
