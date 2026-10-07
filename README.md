# LUT 跨色域转换

把一枚 LUT 的**输入或输出色彩空间**换掉，产出新的一枚。

最常见的用法：**你有一枚为别的相机做的 LUT，想用在自己的相机上。**

```
原来：  Sony S-Log3 ──[LUT]──> Rec.709
改成：  Canon Log 3 ──[新 LUT]──> Rec.709
```

覆盖 **13 家厂商 / 29 个色彩空间**：Sony、Canon、Nikon、Fujifilm、Panasonic、
ARRI、RED、DJI、Blackmagic、DaVinci、Apple、FilmLight、GoPro。

> 不需要装 Python，不需要联网，不需要管理员权限。
> 解压 → 双击安装 → 桌面出图标。

---

## 下载

到 [Releases](../../releases) 页面下载 **`LUT-Converter-v1.0.zip`**（37 MB）。

解压后：

1. 把整个文件夹拷到一个固定的位置
2. 双击 **`点我安装.bat`**
3. 回桌面，双击新出现的图标

详细图文说明在包里的 `docs/使用说明.txt`（6 页），更快的版本在 `docs/先看这里.txt`。

> **⚠️ 只有 Windows 能用。**
> 那个包里内置的是 Windows 版的 Python 运行时。
> macOS / Linux 请从源码运行，见下面的「从源码运行」一节。

> **资产文件名用 ASCII**（`LUT-Converter-v1.0.zip`）是刻意的：
> 中文文件名在上传过程中被浏览器/服务器的编码处理弄坏过，
> 变成过 `LUT.-.zip` 这种没法认的名字。

---

## 它做什么

| 能力 | 说明 |
|---|---|
| 换输入色彩空间 | 让一枚为相机 A 做的 LUT 接受相机 B 的素材 |
| 换输出色彩空间 | 比如从 Rec.709 换成别的显示空间 |
| 读写格式 | `.cube` · `.csp` · `.spi1d` · `.spi3d` · `.spimtx` · `.3dl` · `.vlt` |
| 导出网格 | 17³ / 33³ / 65³ |
| 精度报告 | 超界比例、输出范围、同空间还原精度 |

界面上会显示一份检查报告，**只需要看两个地方**：

- **同空间还原** —— 显示「逐位精确」就说明换算链路正确
- **均匀采样越界** —— 这个百分比是上界，**基本可以忽略**（见下）

---

## 转换的代价（实测）

这是最该关心的一件事，所以直接给数字。

用三种口径测同一件事——「我有相机 A 的 LUT，我拍相机 B，会丢东西吗」：

| 采样方式 | 结果 |
|---|---|
| **真实拍摄内容**（MacBeth 色卡 + 灰阶 + 色相扫描，4709 色） | **0 损失** |
| 13 家厂商互相转换，全部 17 种组合 | **全部 0 损失** |
| 极端：过曝 4～64 倍的纯色物体 | 会截断，最多 60% |
| 均匀码值立方体（界面上显示的那个百分比） | 23～72% |

**为什么最后一行比第一行难看得多？** 因为均匀立方体的角点是纯红、纯绿、
满曝光的组合——**现实里不存在这种颜色**。界面上显示的就是这个数，
它是个**上界**，不是你会遇到的比例。

**结论：正常画面不会丢东西。** 只有画面里有又亮又艳的东西（过曝的霓虹灯、
阳光直射的纯红物体）时，那一小块会被压住。

原因是各家的 log 曲线能表达的光线范围差很多：

| 色彩空间 | 可表示的线性范围 |
|---|---|
| ARRI LogC4 | −0.018 … **469.8** |
| ARRI LogC3 | −0.017 … 55.1 |
| Panasonic V-Log | −0.022 … 46.1 |
| Sony S-Log3 | −0.014 … 38.4 |
| Canon Log 3 | −0.074 … 14.7 |
| GoPro Protune | 0 … **1.0** |

往窄的空间转就装不下。**加密网格救不回来**（33³ 和 65³ 测下来一样）。
另一层原因：一个相机原色下的饱和色，在另一个相机原色下可能是**负值**，
那也需要截断。

复现这三组测量：

```bash
PYTHONPATH=app python research/usecase_realistic.py    # 真实内容
PYTHONPATH=app python research/usecase_extreme.py      # 极端色彩施压
PYTHONPATH=app python research/usecase_check.py        # 均匀网格（上界）
```

---

## 它不做什么：不自动识别输入色彩空间

**输入空间由你指定，程序不猜。**

技术上做得到，这个项目也试过，然后砍掉了。原因很直接：

> 用 19 枚**真实厂商 LUT**（Sony / Panasonic / DJI / Blackmagic / ARRI）实测：
> **正确 0 枚、自信答错 3 枚。**

根本原因是识别需要一个参考库，而参考库用的显示变换与厂商自用的不一致；
试过两种修法，都没救回来。剩下的方向还有一个硬上限——真实厂商 LUT 里
「相机 log → 显示」这个方向只覆盖 6/13 厂商，另外 7 家的 LUT 拿不到。

所以设计上就不猜。而且猜错的代价比不猜大：**程序答错的时候是很确信的样子。**

手动选一次只要两秒。`docs/使用说明.txt` 里写了怎么从文件名判断。

---

## 从源码运行

```bash
pip install numpy colour-science PyOpenColorIO

python app/lut_hub.py             # 探端口、开浏览器
python app/server.py              # 或者只开服务 -> http://127.0.0.1:8770
```

自检（15 个重定向用例的误差账本）：

```bash
python app/demos.py
```

期望输出：

```
  1. retarget to the SAME spaces reproduces the LUT bit-for-bit        15/15
  2. .cube / .3dl / .vlt round trip within one 12-bit step            15/15
  3. residual attributable to the retarget <= 0.005                   15/15
     worst residual: +0.00460
```

第三行是**误差归属账本**：把总误差拆成三份——用完美解析函数当中间级时的固有
损失、网格分辨率的下限、以及转换本身真正贡献的部分。最后一项最深 +0.0046
（不到半个百分点），说明换算逻辑本身没有引入误差。

格式校验（30 项，含用 OpenColorIO 做独立裁判）：

```bash
PYTHONPATH=app python research/format_tests.py       # 期望 30/30
```

---

## 重新打包发行版

```bash
python build/build_package.py --python "C:\path\to\a\cpython"
```

它从指定的本地 CPython 复制一个运行时出来，装好三个依赖，生成启动器和图标，
然后**验证装出来的东西真的能跑**（起服务、调 `/api/selftest`、跑一次真实转换），
最后打成 zip。

两个刻意的选择：

- **不下载 Python**，用本地的。python.org 的 embeddable zip（11 MB）
  在这个网络环境下试过三次，三次都被截断在 5.8 MB。
- **不做 MSI**。这个工具解压双击就够用；而 WiX 装不下中文快捷方式名
  （`LGHT0311`，code page 1252），安装界面会被迫写成英文。

---

## 目录

```
app/            发行的应用源码
  lut_hub.py      入口（探端口、开浏览器）
  serving.py      服务层：读 LUT、重定向、误差账本
  server.py       标准库 HTTP，无框架依赖
  gui.html        单文件界面
  demos.py        5 个合成样例 + 自检
  camera_spaces.py  29 个色彩空间的注册表
  lutlib.py       共享管线（枢纽、插值）
  formats.py      .3dl / .vlt 读写
  lut_io.py       LUT 读取 + 文件名线索

build/          发行包怎么造出来的
  setup.py          在用户机器上跑的安装器
  build_package.py  从源码重建发行 zip

docs/           面向使用者的说明 + 5 张界面截图

research/       验证脚本
  format_tests.py     30 项格式校验
  usecase_*.py        三种口径的用户场景实测
  p0_*.py             枢纽往返验证
  make_clf*.py        CLF / LMT 包装
  derive_space.py     从官方 OCIO IDT 反解厂商原色
```

---

## 许可证

MIT，见 [LICENSE](LICENSE)。

**本仓库不含任何厂商 LUT 文件，也不分发厂商数据。** 它做的是色彩数学：
以 ACES2065-1 为枢纽，在各家公开的色彩空间定义之间换算。
你在使用时需要的厂商 LUT 由你自己提供。

`research/usecase_*.py` 会去读本机 DaVinci Resolve 随附的厂商 LUT 做对照，
**只读、不复制、不打包**；那个目录不存在时脚本打印一行提示就正常退出。
路径可以用环境变量覆盖：

```bash
LUT_DIR=/path/to/your/luts PYTHONPATH=app python research/usecase_realistic.py
```

第三方组件：[colour-science](https://www.colour-science.org/)（BSD-3）、
[OpenColorIO](https://opencolorio.org/)（BSD-3）、[NumPy](https://numpy.org/)（BSD-3）、
[pylnk3](https://github.com/strayge/pylnk3)（MIT）。发行包里还内置了一个
[CPython](https://www.python.org/)（PSF License）。

---

## 已知的没做

- HDR 输出（PQ / HLG）
- 批处理（目前一次一枚）
- `.dat` / `.ilut` / `.olut` 读取
- 自动识别 LUT 的色彩空间（**已决定不做**，见上）
- 代码签名 —— 所以 Windows 首次运行会弹 SmartScreen，
  点「更多信息」→「仍要运行」
