"""
server.py - local web interface for the LUT conversion tool
==========================================================

Standard library only.

    python server.py            -> http://127.0.0.1:8770

Endpoints
---------
  GET  /                 the interface
  GET  /api/app          spaces, formats, timing
  POST /api/inspect      body = LUT bytes, or ?demo=<id>   -> what the file says
  GET  /api/convert      ?key=&sin=&sout=&sin_new=&sout_new=&fmt=  -> download
  GET  /api/convert&dry=1                                  -> the numbers only

`sin`/`sout` are the spaces the LUT is currently on; `sin_new`/`sout_new` are the
ones wanted. All four are declared by the caller - nothing is inferred from the
file, which is the whole point of the redesign.
"""

import json
import os
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import demos as DM
import formats as F
import serving as SV

PORT = int(os.environ.get("LUT_HUB_PORT", "8770"))
# The packaged launcher starts the server itself and stays quiet; running this file
# directly is the developer path, where the startup lines are useful.
VERBOSE = __name__ == "__main__"
CONVERTER = None
CONVERTER_ERR = None
CACHE = {}                       # token -> path on disk
CACHE_LOCK = threading.Lock()


def get_converter():
    global CONVERTER, CONVERTER_ERR
    if CONVERTER is None and CONVERTER_ERR is None:
        try:
            CONVERTER = SV.Converter(verbose=VERBOSE)
        except Exception as exc:
            import traceback
            CONVERTER_ERR = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
    return CONVERTER


def doc(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "lut-hub"

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            print(f"  {self.command} {self.path.split('?')[0]} "
                  f"-> {args[1] if len(args) > 1 else ''}", flush=True)

    def _send(self, body, ctype="application/json; charset=utf-8", code=200, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, ensure_ascii=False, default=float), code=code)

    def _fail(self, msg, code=400):
        self._json({"error": str(msg)}, code=code)

    def _params(self):
        return {k: unquote(v[0]) for k, v in parse_qs(urlparse(self.path).query).items()}

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    # -------------------------------------------------------------------- GET
    def do_GET(self):
        p = urlparse(self.path).path
        try:
            if p in ("/", "/index.html"):
                return self._send(doc("gui.html"), "text/html; charset=utf-8")
            if p in ("/icon.png", "/favicon.ico"):
                # served so the browser tab shows the app icon rather than a
                # blank page glyph
                fname = "icon.png" if p == "/icon.png" else "icon.ico"
                fp = os.path.join(HERE, fname)
                if not os.path.exists(fp):
                    return self._fail("no icon", 404)
                mime = ("image/png" if fname.endswith(".png")
                        else "image/x-icon")
                with open(fp, "rb") as fh:
                    return self._send(fh.read(), mime, extra={
                        "Cache-Control": "public, max-age=86400"})
            if p == "/api/app":
                c = get_converter()
                if c is None:
                    return self._json({"error": f"converter failed to start: {CONVERTER_ERR}"},
                                      code=503)
                info = c.app_info()
                info["demos"] = [{k: d[k] for k in ("id", "label", "note", "spaces")}
                                 for d in DM.CASES]
                return self._json(info)
            if p == "/api/selftest":
                c = get_converter()
                if c is None:
                    return self._fail(f"converter failed to start: {CONVERTER_ERR}", 503)
                rows = DM.selftest(c)
                exact = sum(1 for r in rows if r["verify_same_space"]["exact"])
                res_ok = sum(1 for r in rows if r["tool_residual"] <= 0.005)
                worst = max(r["tool_residual"] for r in rows)
                mean_lo = min(r["table_mean"] for r in rows)
                mean_hi = max(r["table_mean"] for r in rows)
                return self._json({
                    "rows": rows,
                    "summary": (
                        f"{exact}/{len(rows)} 例同空间还原逐位精确。"
                        f"残余（扣掉解析组合损失与网格下限后，转换本身的贡献）"
                        f"最深 {worst:+.5f}，{res_ok}/{len(rows)} 例在 0.005 以内。"
                        f"平均误差 {mean_lo:.5f}–{mean_hi:.5f}。"
                        f"「最大误差」一列由显示变换的裁切折角决定，"
                        f"加密网格收敛很慢，看平均误差更有意义。"),
                })
            if p == "/api/convert":
                return self._convert(self._params())
            return self._fail("not found", 404)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return self._fail(exc, 500)

    # ------------------------------------------------------------------- POST
    def do_POST(self):
        p = urlparse(self.path).path
        try:
            q = self._params()
            c = get_converter()
            if c is None:
                return self._fail(f"converter failed to start: {CONVERTER_ERR}", 503)

            if p == "/api/inspect":
                demo = q.get("demo")
                if demo:
                    path, case = DM.materialise(demo)
                else:
                    body = self._body()
                    if not body:
                        return self._fail("empty body: send the LUT bytes, or ?demo=<id>")
                    path = SV.save_upload(q.get("name", "upload.cube"), body)
                    case = None
                r = c.inspect(path)
                token = secrets.token_urlsafe(9)
                with CACHE_LOCK:
                    CACHE[token] = path
                    if len(CACHE) > 200:
                        CACHE.pop(next(iter(CACHE)))
                r["key"] = token
                if case:
                    r["demo"] = {"id": case["id"], "label": case["label"],
                                 "note": case["note"], "spaces": case["spaces"]}
                return self._json(r)

            return self._fail("not found", 404)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return self._fail(exc, 500)

    # --------------------------------------------------------------- convert
    def _convert(self, q):
        c = get_converter()
        with CACHE_LOCK:
            path = CACHE.get(q.get("key") or "")
        if not path or not os.path.exists(path):
            return self._fail("unknown key: inspect the LUT first")

        allowed = set(c.spaces) | {SV.DSP}
        need = ["sin", "sout", "sin_new", "sout_new"]
        miss = [k for k in need if not q.get(k)]
        if miss:
            return self._fail(f"missing parameter(s): {', '.join(miss)}")
        bad = [q[k] for k in need if q[k] not in allowed]
        if bad:
            return self._fail(f"unknown space(s): {', '.join(bad)}")

        fn, info, _ = F.load(path)
        fmt = (q.get("fmt") or "cube").lstrip(".").lower()
        note = None
        grid = int(q.get("grid") or c.grid_for(fmt))
        if fmt == "vlt" and grid != 17:
            note = f"grid {grid} -> 17; the .vlt specification defines 17³ only"
            grid = 17

        table, stats = c.retarget(fn, q["sin"], q["sout"],
                                  q["sin_new"], q["sout_new"], n=grid)
        stats["verify_same_space"] = c.verify(fn, q["sin"], q["sout"])
        stats["format"] = fmt
        if note:
            stats["note"] = note

        if q.get("dry"):
            return self._json(stats)

        data, fname, mime = c.build_output(table, fmt, os.path.basename(path))
        return self._send(data, mime, extra={
            "Content-Disposition": f'attachment; filename="{fname}"'})


def main():
    print("=" * 90)
    print("LUT hub - conversion tool")
    print("=" * 90)
    get_converter()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"  http://127.0.0.1:{PORT}")
    print("  Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
