"""
build_package.py - rebuild the distributable from source
========================================================

    python build/build_package.py --python "C:\\path\\to\\a\\cpython"

Produces `LUT跨色域转换.zip` next to the repo: a self-contained folder with a
vendored CPython, the three dependencies, launchers, icon and docs. That is what
gets attached to a GitHub Release.

WHY IT TAKES --python INSTEAD OF DOWNLOADING ONE
------------------------------------------------
The obvious approach is python.org's embeddable zip (11 MB). It was attempted
three times from this network and truncated all three times at ~5.8 MB. A local
CPython install is also strictly better for this purpose: it ships `Lib/`,
`DLLs/` and `vcruntime140.dll`, so there is no `._pth` to patch and no missing
runtime to chase.

WHY IT IS NOT AN MSI
--------------------
The tool is a folder you unzip and double-click. An MSI adds a harvest step, a
compile step and a four-way uninstall verification for no user benefit here. And
WiX cannot carry Chinese display strings in the shortcut name at all - `LGHT0311`,
code page 1252 - so the installer's own UI would have to be in English while the
app is in Chinese.

WHAT IT VERIFIES BEFORE DECLARING SUCCESS
-----------------------------------------
A packaging script that only copies files happily produces a folder that does not
run. This one, after assembling:

  1. imports all three dependencies from the vendored runtime
  2. starts the service and calls /api/app and /api/selftest
  3. checks the launchers are pure ASCII with CRLF - cmd parses .bat in the OEM
     code page and mangles any non-ASCII byte, and LF line endings are unreliable
     around parenthesised blocks
  4. checks the text files have a BOM, so an older Notepad does not render the
     Chinese as mojibake
  5. checks the icon carries all seven rasters - Pillow's ICO writer takes ONE
     image and resizes it into every slot, which silently collapses a
     size-specific icon set down to a single entry
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILD = os.path.join(ROOT, "build_temp")
TOP = "LUT跨色域转换"
ZIP = os.path.join(ROOT, f"{TOP}.zip")

DEPS = ["numpy", "colour", "PyOpenColorIO"]
# Shared libraries that wheels install BESIDE the package rather than inside it.
# Copying only `numpy` and not `numpy.libs` gives:
#   ImportError: DLL load failed while importing _multiarray_umath
SIBLING_LIBS = ["numpy.libs"]
# stdlib that looks unused and is not. colour-science does
#   from unittest.mock import MagicMock
# at module level in colour/algebra/interpolation.py. Removing unittest for
# "slimming" breaks the import chain outright.
STDLIB_KEEP = {"unittest"}

INSTALL_BAT = r"""@echo off
rem ===========================================================================
rem  LUT Converter - setup launcher
rem
rem  Content is ASCII-only on purpose: cmd.exe parses .bat in the OEM code page,
rem  so a non-ASCII byte here gets mangled. Chinese the user reads is printed by
rem  the Python side, which handles Unicode properly.
rem ===========================================================================
setlocal
title LUT Converter - Setup

set "HERE=%~dp0"
set "PY=%HERE%runtime\python\python.exe"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONWARNINGS=ignore"

if not exist "%PY%" (
    echo.
    echo   Cannot find the bundled Python runtime:
    echo     %PY%
    echo.
    echo   The folder looks incomplete. Please copy the WHOLE folder again,
    echo   not just part of it.
    echo.
    pause
    exit /b 1
)

"%PY%" "%HERE%app\setup.py"
exit /b %ERRORLEVEL%
"""

RUN_BAT = r"""@echo off
rem ===========================================================================
rem  LUT Converter - run in place, without installing
rem
rem  Content is ASCII-only on purpose, same reason as the setup launcher.
rem ===========================================================================
setlocal
title LUT Converter

set "HERE=%~dp0"
set "PY=%HERE%runtime\python\python.exe"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONWARNINGS=ignore"

if not exist "%PY%" (
    echo.
    echo   Cannot find the bundled Python runtime:
    echo     %PY%
    echo.
    echo   The folder looks incomplete. Please copy the WHOLE folder again,
    echo   not just part of it.
    echo.
    pause
    exit /b 1
)

"%PY%" "%HERE%app\lut_hub.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo   Program exited with code %RC%.
    echo.
    pause
)
exit /b %RC%
"""


def say(msg=""):
    print(msg, flush=True)


def find_site_packages(py_home):
    """Where pip put things. A venv and a full install differ."""
    for cand in (os.path.join(py_home, "Lib", "site-packages"),
                 os.path.join(py_home, "lib", "site-packages")):
        if os.path.isdir(cand):
            return cand
    return None


def vendor_runtime(py_home, dst):
    """Copy a CPython tree, then the three dependencies into its site-packages."""
    if not os.path.exists(os.path.join(py_home, "python.exe")):
        raise SystemExit(f"不是 CPython 安装目录（找不到 python.exe）：{py_home}")

    shutil.copytree(py_home, dst,
                    ignore=shutil.ignore_patterns("__pycache__", "test", "tests",
                                                  "*.pyc", "*.lib", "*.pdb"))
    sp_src = find_site_packages(py_home)
    if not sp_src:
        raise SystemExit(f"找不到 site-packages：{py_home}")
    sp = os.path.join(dst, "Lib", "site-packages")
    os.makedirs(sp, exist_ok=True)

    for pkg in DEPS:
        s = os.path.join(sp_src, pkg)
        if not os.path.isdir(s):
            raise SystemExit(f"依赖没装：{pkg}（在 {sp_src} 里找不到）")
        shutil.copytree(s, os.path.join(sp, pkg),
                        ignore=shutil.ignore_patterns("__pycache__", "test",
                                                      "tests", "*.pyc", "*.lib"))
        say(f"    依赖 {pkg}")
    for lib in SIBLING_LIBS:
        s = os.path.join(sp_src, lib)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(sp, lib))
            say(f"    同级库 {lib}   ← 漏了这个 numpy 会 ImportError")
    for name in os.listdir(sp_src):
        if name.endswith(".dist-info") and any(
                name.lower().startswith(p.lower()) for p in DEPS):
            shutil.copytree(os.path.join(sp_src, name), os.path.join(sp, name),
                            ignore=shutil.ignore_patterns("__pycache__"))

    # sanity: the stdlib we were tempted to delete
    for keep in STDLIB_KEEP:
        if not os.path.isdir(os.path.join(dst, "Lib", keep)):
            raise SystemExit(f"stdlib 缺少 {keep}，colour-science 的 import 会断")


def write_bats(dst):
    findings = []
    for name, body in (("点我安装.bat", INSTALL_BAT), ("启动软件.bat", RUN_BAT)):
        data = body.replace("\n", "\r\n").encode("ascii")
        with open(os.path.join(dst, name), "wb") as fh:
            fh.write(data)
        findings.append((name, sum(1 for b in data if b > 127),
                         data.count(b"\r\n"),
                         data.count(b"\n") - data.count(b"\r\n")))
    return findings


def check_text_encoding(dst):
    """CRLF + BOM on the Chinese text files, for the benefit of old Notepad."""
    fixed = []
    for f in ("先看这里.txt", "使用说明.txt"):
        p = os.path.join(dst, f)
        if not os.path.exists(p):
            continue
        d = open(p, "rb").read()
        body = d[3:] if d[:3] == b"\xef\xbb\xbf" else d
        body = body.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        out = b"\xef\xbb\xbf" + body
        if out != d:
            open(p, "wb").write(out)
            fixed.append(f)
    return fixed


def check_icon(dst):
    """All seven rasters must be present. See the note in the docstring."""
    import struct

    p = os.path.join(dst, "app", "icon.ico")
    if not os.path.exists(p):
        raise SystemExit("缺少 app/icon.ico")
    d = open(p, "rb").read()
    _, _, n = struct.unpack("<HHH", d[:6])
    sizes, off = [], 6
    for _ in range(n):
        w, h = struct.unpack("<BB", d[off:off + 2])
        sizes.append(w or 256)
        off += 16
    want = [16, 24, 32, 48, 64, 128, 256]
    if sorted(sizes) != want:
        raise SystemExit(f"图标尺寸不全：{sizes}，应为 {want}")
    say(f"    icon.ico  {n} 个尺寸 {sizes}  {len(d):,} bytes")


def verify(dst):
    """Prove the assembled folder actually runs. Copying files is not evidence."""
    py = os.path.join(dst, "runtime", "python", "python.exe")
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(k, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONWARNINGS"] = "ignore"

    r = subprocess.run(
        [py, "-c", "import numpy, colour, PyOpenColorIO as o; "
                   "print(numpy.__version__, colour.__version__, o.__version__)"],
        capture_output=True, text=True, env=env, timeout=180)
    if r.returncode:
        raise SystemExit(f"依赖导入失败：\n{r.stdout}\n{r.stderr}")
    say(f"    依赖导入 OK：{r.stdout.strip()}")

    proc = subprocess.Popen([py, os.path.join(dst, "app", "lut_hub.py")],
                            cwd=dst, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        import json
        import urllib.request

        port = None
        for _ in range(40):
            time.sleep(1)
            for p in range(8770, 8790):
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{p}/api/app", timeout=2) as resp:
                        app = json.load(resp)
                        port = p
                        break
                except Exception:
                    continue
            if port:
                break
        if not port:
            raise SystemExit("服务没起来")

        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/selftest", timeout=180) as resp:
            st = json.load(resp)
        exact = sum(1 for row in st["rows"] if row["verify_same_space"]["exact"])
        if exact != len(st["rows"]):
            raise SystemExit(f"自检没全过：{exact}/{len(st['rows'])}")
        say(f"    自检 {exact}/{len(st['rows'])}   {len(app['spaces'])} 个色彩空间")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True,
                    help="一个本地 CPython 安装目录（含 python.exe）")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    say("=" * 70)
    say("  重建发行包")
    say("=" * 70)

    if os.path.isdir(BUILD):
        shutil.rmtree(BUILD)
    os.makedirs(BUILD)

    say(f"  1. 内置运行时      ← {args.python}")
    vendor_runtime(args.python, os.path.join(BUILD, "runtime", "python"))

    say("  2. 应用源码")
    shutil.copytree(os.path.join(ROOT, "app"), os.path.join(BUILD, "app"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # make_icon.py needs Pillow, which is NOT one of the runtime dependencies, so
    # the icon is committed as a built artifact and only regenerated on demand.
    # Copying the generator next to the build script keeps that reproducible.
    shutil.copy2(os.path.join(HERE, "make_icon.py"),
                 os.path.join(BUILD, "make_icon.py"))
    shutil.copy2(os.path.join(HERE, "setup.py"),
                 os.path.join(BUILD, "app", "setup.py"))
    shutil.copy2(os.path.join(HERE, "pylnk3.py"),
                 os.path.join(BUILD, "pylnk3.py"))

    say("  3. 文档与截图")
    os.makedirs(os.path.join(BUILD, "图片说明"), exist_ok=True)
    for f in ("先看这里.txt", "使用说明.txt"):
        src = os.path.join(ROOT, "docs", f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(BUILD, f))
    img = os.path.join(ROOT, "docs", "images")
    if os.path.isdir(img):
        for f in os.listdir(img):
            shutil.copy2(os.path.join(img, f), os.path.join(BUILD, "图片说明", f))

    say("  4. 启动器与编码")
    for name, na, crlf, lf in write_bats(BUILD):
        say(f"    {name}  非ASCII={na}  CRLF={crlf}  裸LF={lf}")
        if na or lf:
            raise SystemExit("启动器编码不合格")
    fixed = check_text_encoding(BUILD)
    say(f"    文本文件编码修正：{fixed or '本来就是对的'}")
    check_icon(BUILD)

    say("  5. 运行验证")
    if args.skip_verify:
        say("    跳过（--skip-verify）")
    else:
        verify(BUILD)

    say("  6. 打包")
    for dirpath, dirnames, files in os.walk(BUILD):
        for f in files:
            if f.endswith(".pyc"):
                os.unlink(os.path.join(dirpath, f))
    if os.path.exists(ZIP):
        os.remove(ZIP)
    n = 0
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for dirpath, dirnames, filenames in os.walk(BUILD):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for f in filenames:
                full = os.path.join(dirpath, f)
                z.write(full, os.path.join(TOP, os.path.relpath(full, BUILD)))
                n += 1
    size = os.path.getsize(ZIP)
    shutil.rmtree(BUILD, ignore_errors=True)

    say()
    say("=" * 70)
    say(f"  完成：{os.path.basename(ZIP)}")
    say(f"  {n} 个文件，{size/1e6:.1f} MB")
    say("=" * 70)
    say()
    say("  把它作为 GitHub Release 的附件发出去：")
    say("    gh release create v1.0 LUT跨色域转换.zip -t 'v1.0' -n '首个版本'")
    say("  或者在网页上 Releases → Draft a new release → 拖进去")


if __name__ == "__main__":
    main()
