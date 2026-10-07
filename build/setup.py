"""
setup.py - one-click setup for the shared folder
================================================

Lives in app/ and is launched by 点我安装.bat in the folder above. What it does,
and why each step is here:

  1. find the REAL desktop. On a machine with OneDrive folder backup the desktop
     is NOT %USERPROFILE%\\Desktop - putting a shortcut there means the user never
     sees it. Asked of Windows via SHGetFolderPathW, with a registry fallback.
  2. create the shortcut with pylnk3, shipped in the folder above. It has to be
     pylnk3 and not a hand-written .lnk: Explorer locates the target through the
     LinkTargetIDList, and a shortcut can look structurally valid in every flag
     and still do nothing when double-clicked.
  3. copy the program somewhere permanent. Running from a USB stick and then
     pulling the stick out would break the shortcut, so the default is
     %LOCALAPPDATA%\\LUTConverter - per-user, no admin rights, no UAC prompt.

Safe to run twice: an existing shortcut is replaced, an existing copy of the
program is refreshed rather than duplicated.

Nothing here needs elevation. If a step fails the script says which one, and the
folder stays runnable by hand - 启动软件.bat works in place.
"""

import os
import shutil
import struct
import sys
import traceback

APP_DIR = os.path.dirname(os.path.abspath(__file__))     # .../app
HERE = os.path.dirname(APP_DIR)                          # the shared folder

APP_NAME = "LUT 跨色域转换"
# Installed per-user, no elevation. Two separate places on purpose:
#   Programs\LUTConverter  the program - safe to delete, nothing is written to it
#   LUTConverter           the data dir - uploads and scratch files the user makes
# Keeping them apart means deleting the program does not throw away the user's
# files. Collapsing them into one folder was the first attempt and it did exactly
# that: uploads/ ended up inside the program folder.
INSTALL_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA") or HERE, "Programs", "LUTConverter")
DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or HERE, "LUTConverter")
SHORTCUT_NAME = APP_NAME + ".lnk"


def say(msg=""):
    print(msg, flush=True)


def banner(step, total, text):
    say()
    say("=" * 62)
    say(f"  第 {step} 步 / 共 {total} 步：{text}")
    say("=" * 62)


# ------------------------------------------------------------------ the desktop
def real_desktop():
    """The desktop the user actually looks at.

    %USERPROFILE%\\Desktop is wrong on any machine using OneDrive folder backup,
    which is most consumer Windows installs now. Ask Windows first.
    """
    try:
        import ctypes
        from ctypes import wintypes

        CSIDL_DESKTOPDIRECTORY = 0x0010
        SHGFP_TYPE_CURRENT = 0
        buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
        ok = ctypes.windll.shell32.SHGetFolderPathW(
            None, CSIDL_DESKTOPDIRECTORY, None, SHGFP_TYPE_CURRENT, buf)
        if ok == 0 and buf.value and os.path.isdir(buf.value):
            return buf.value, "SHGetFolderPathW"
    except Exception:
        pass

    try:                                    # registry can hold the redirected path
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders")
        raw, _ = winreg.QueryValueEx(key, "Desktop")
        path = os.path.expandvars(raw)
        if os.path.isdir(path):
            return path, "registry"
    except Exception:
        pass

    fallback = os.path.join(os.environ.get("USERPROFILE", HERE), "Desktop")
    return fallback, "guess (%USERPROFILE%\\Desktop)"


# ----------------------------------------------------------------- the shortcut
def make_shortcut(desktop, target):
    """Returns (ok, path, note)."""
    try:
        import importlib.util
    except Exception:
        pass

    # pylnk3 is a single module shipped next to this script
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        import pylnk3
    except Exception as exc:
        return False, None, f"pylnk3 无法加载：{exc}"

    lnk = os.path.join(desktop, SHORTCUT_NAME)
    icon = os.path.join(target, "app", "icon.ico")
    try:
        pylnk3.for_file(target, lnk,
                        description="LUT cross-gamut converter",
                        work_dir=target,
                        icon_file=icon if os.path.exists(icon) else None,
                        icon_index=0)
    except Exception as exc:
        return False, None, f"创建失败：{exc}"

    # Verify the two things that actually matter: the IDList Explorer needs, and
    # that the target it points at exists.
    try:
        data = open(lnk, "rb").read()
        flags = struct.unpack("<I", data[20:24])[0]
        if not (flags & 0x01):
            return False, lnk, ("快捷方式缺少目标清单（HasLinkTargetIDList），"
                                "双击可能没反应")
    except Exception as exc:
        return False, lnk, f"校验失败：{exc}"

    return True, lnk, f"{len(data)} 字节，目标清单完整"


# ------------------------------------------------------------------- the files
def copy_program(dest):
    """Copy app/ + runtime/ + the launchers into `dest`. Refresh if present.

    setup.py itself is skipped - it is a one-shot installer and has no business
    sitting in an installed copy.
    """
    os.makedirs(dest, exist_ok=True)
    for name in ("app", "runtime"):
        src = os.path.join(HERE, name)
        if not os.path.isdir(src):
            raise RuntimeError(f"缺少文件夹：{name}")
        dst = os.path.join(dest, name)
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                      "setup.py"))
    for f in ("启动软件.bat", "使用说明.txt"):
        src = os.path.join(HERE, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, f))
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(dest) for f in fs)
    return total


def main():
    say()
    say("=" * 62)
    say(f"  {APP_NAME}  —  安装")
    say("=" * 62)
    say()
    say("  这个程序不需要联网，不需要管理员权限，")
    say("  也不会改动系统里的任何东西。装出来的东西可以随时删掉。")

    total_steps = 3

    # ---------------------------------------------------------------- step 1
    banner(1, total_steps, "查找桌面")
    desktop, how = real_desktop()
    say(f"  桌面位置：{desktop}")
    say(f"  （查询方式：{how}）")
    if not os.path.isdir(desktop):
        say()
        say("  ⚠ 这个位置不存在，将跳过快捷方式，只完成程序安装。")

    # ---------------------------------------------------------------- step 2
    banner(2, total_steps, "复制程序")
    if os.path.normcase(INSTALL_DIR) == os.path.normcase(HERE):
        say("  已经在目标位置运行，跳过复制。")
        target = HERE
    else:
        say(f"  复制到：{INSTALL_DIR}")
        say("  （放在这里是为了：U 盘拔掉后程序还能用；不需要管理员权限）")
        try:
            size = copy_program(INSTALL_DIR)
            target = INSTALL_DIR
            say(f"  ✓ 完成，共 {size / 1e6:.0f} MB")
        except Exception as exc:
            say(f"  ✗ 复制失败：{exc}")
            say()
            say("  没关系 —— 你可以直接在当前这个文件夹里用：")
            say(f"    双击  {os.path.join(HERE, '启动软件.bat')}")
            target = HERE

    # ---------------------------------------------------------------- step 3
    banner(3, total_steps, "在桌面创建快捷方式")
    launcher = os.path.join(target, "启动软件.bat")
    if not os.path.exists(launcher):
        ok, lnk, note = False, None, f"找不到启动文件：{launcher}"
    elif not os.path.isdir(desktop):
        ok, lnk, note = False, None, "桌面目录不存在"
    else:
        ok, lnk, note = make_shortcut(desktop, target)
    if ok:
        say(f"  ✓ 已创建：{lnk}")
        say(f"     {note}")
    else:
        say(f"  ⚠ 没能创建快捷方式：{note}")
        say()
        say("  不影响使用，你随时可以自己拖一个：")
        say(f"    右键 {launcher}")
        say("    → 发送到 → 桌面快捷方式")

    # ------------------------------------------------------------------ done
    say()
    say("=" * 62)
    if ok:
        say("  安装完成 ✔")
    else:
        say("  安装完成（快捷方式那一步没成功，程序本身没问题）")
    say("=" * 62)
    say()
    say("  ▶ 怎么用：回到桌面，双击「" + APP_NAME + "」")
    say()
    say("  ▶ 会弹出一个黑色窗口，等一下浏览器会自己打开。")
    say("    那个黑窗口是程序本体，用的时候不要关。")
    say("    用完了直接关掉黑窗口就是退出。")
    say()
    say(f"  ▶ 详细的图文说明在：{os.path.join(target, '使用说明.txt')}")
    say(f"    也可以看文件夹里的「图片说明」文件夹。")
    say()
    say("  ▶ 想卸载：删掉这三处就干净了")
    say(f"    ① 桌面上的快捷方式")
    say(f"    ② 程序文件夹  {target}")
    say(f"    ③ 数据文件夹  {DATA_DIR}")
    say(f"       （③ 里只有你自己导出的临时文件，删不删都行）")
    say()
    say("    如果想连安装痕迹一起清掉，再按 Win+R 粘贴下面这行回车：")
    say(f"      %LOCALAPPDATA%")
    say("    然后把 LUTConverter 和 Programs\\LUTConverter 两个文件夹删掉。")
    say()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print()
        print("  安装过程中出了点问题：")
        traceback.print_exc()
        print()
        print("  程序本身可能仍然可用，试试直接双击：启动 LUT 转换器.bat")
    print()
    input("  按回车键关闭这个窗口…")
