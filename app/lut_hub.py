"""
lut_hub.py - entry point for the packaged application
=====================================================

Starts the local server and opens a browser at it. Double-clicking the launcher
runs this.

Two things it deliberately does NOT do:

  * write anything into the program directory. Uploads and scratch files go to
    %LOCALAPPDATA%\\LUTConverter, so uninstalling leaves nothing behind and the
    program directory can stay read-only.
  * require the working directory to be the app folder. Everything is resolved
    from __file__, so the launcher works from a shortcut with any start-in.

The port is probed rather than fixed: if 8770 is taken (a previous copy still
running, or something else using it) the next free one is used, and the console
prints which.
"""

import os
import socket
import sys
import threading
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

APP = "LUT 跨色域转换"
PORTS = list(range(8770, 8790))


def free_port():
    for p in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"no free port in {PORTS[0]}-{PORTS[-1]}")


def main():
    print("=" * 70)
    print(f"  {APP}")
    print("=" * 70)

    try:
        import serving as SV
        import server as SRV
    except Exception as exc:
        print(f"\n  程序文件不完整，或依赖缺失：{exc}")
        print("  请重新解压 / 重装。")
        input("\n  按回车退出…")
        return 1

    port = free_port()
    SRV.PORT = port
    SRV.get_converter()

    print(f"\n  正在启动，浏览器会自动打开。")
    print(f"  如果没打开，手动访问： http://127.0.0.1:{port}")
    print(f"\n  数据目录（可安全删除）：{SV.OUT}")
    print(f"  关闭本窗口即退出程序。")
    print("-" * 70)

    url = f"http://127.0.0.1:{port}"

    def open_later():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_later, daemon=True).start()

    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", port), SRV.Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  已退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
