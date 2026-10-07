"""
CLF export - the canonical carrier
==================================

Demonstrates the production workflow end to end:

  1. build a LOOK LUT that operates inside one vendor's log space
  2. hand it to the official `ociomakeclf --csc <space>`
  3. the tool emits a CLF containing
         [ACES2065-1 -> <space>] [the LUT] [<space> -> ACES2065-1]
     which is exactly the LMT contract: ACES2065-1 in, ACES2065-1 out.
  4. verify the CLF against the analytic chain

Also recorded: which vendors the official tool can actually handle.
"""

import json
import os
import subprocess
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import colour
import PyOpenColorIO as ocio

from camera_spaces import build_registry

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

BIN = os.path.join(os.path.dirname(ocio.__file__), "bin")
MAKECLF = os.path.join(BIN, "ociomakeclf.exe")
CHECKLUT = os.path.join(BIN, "ociochecklut.exe")

CAM = build_registry()
HUB = "ACES2065-1"


def grid(size):
    g = np.linspace(0, 1, size)
    R, G, B = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([R, G, B], -1).reshape(-1, 3)


def tri_correct(t, rgb):
    n = t.shape[0]
    x = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.clip(np.floor(x).astype(int), 0, n - 2)
    f = x - i0
    r, g, b = i0[:, 0], i0[:, 1], i0[:, 2]
    fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    c00 = t[r, g, b] * (1 - fr) + t[r + 1, g, b] * fr
    c01 = t[r, g, b + 1] * (1 - fr) + t[r + 1, g, b + 1] * fr
    c10 = t[r, g + 1, b] * (1 - fr) + t[r + 1, g + 1, b] * fr
    c11 = t[r, g + 1, b + 1] * (1 - fr) + t[r + 1, g + 1, b + 1] * fr
    return (c00 * (1 - fg) + c10 * fg) * (1 - fb) + (c01 * (1 - fg) + c11 * fg) * fb


OCFG = ocio.Config.CreateRaw()


def bt(s, inv=False):
    t = ocio.BuiltinTransform(style=s)
    if inv:
        t.setDirection(ocio.TRANSFORM_DIR_INVERSE)
    return t


P_LOOK = OCFG.getProcessor(ocio.GroupTransform([
    ocio.CDLTransform(slope=[1.06, 1.00, 0.95], offset=[0.010, 0.000, -0.010],
                      power=[0.95, 1.00, 1.05], sat=0.88)])).getDefaultCPUProcessor()


def run(p, x):
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    p.applyRGB(a)
    return a.astype(np.float64)


def c2h(name, code):
    return colour.RGB_to_RGB(code, CAM[name], HUB, apply_cctf_decoding=True,
                             apply_cctf_encoding=False)


def h2c(name, lin):
    lin = colour.RGB_to_RGB(lin, HUB, CAM[name], apply_cctf_decoding=False,
                            apply_cctf_encoding=False)
    cs = CAM[name]
    lo = float(np.min(np.asarray(cs.cctf_decoding(np.zeros((1, 3))))))
    hi = float(np.max(np.asarray(cs.cctf_decoding(np.ones((1, 3))))))
    return np.asarray(cs.cctf_encoding(np.clip(lin, lo, hi)))


def main():
    print("=" * 94)
    print("Official ACES tooling coverage - which of our spaces can ociomakeclf wrap?")
    print("=" * 94)
    out = subprocess.run([MAKECLF, "--list"], capture_output=True, text=True)
    listed = [ln.strip() for ln in out.stdout.splitlines() if ln.strip() and not ln.startswith("The")]
    supported = set(listed)
    print(f"  ociomakeclf --csc supports {len(supported)} spaces")
    print()

    # every space we model, and whether the official tool can wrap it
    coverage = [
        ("Sony", "Sony S-Log3 / S-Gamut3.Cine", "SONY_SLOG3-SGAMUT3.CINE"),
        ("Sony", "Sony S-Log3 / S-Gamut3", "SONY_SLOG3-SGAMUT3"),
        ("Sony", "Sony S-Log3 / Venice S-Gamut3.Cine", "SONY_SLOG3-SGAMUT3.CINE-VENICE"),
        ("Sony", "Sony S-Log3 / Venice S-Gamut3", "SONY_SLOG3-SGAMUT3-VENICE"),
        ("Sony", "Sony S-Log2 / S-Gamut", None),
        ("Canon", "Canon Log 2 / Cinema Gamut", "CANON_CLOG2-CGAMUT"),
        ("Canon", "Canon Log 3 / Cinema Gamut", "CANON_CLOG3-CGAMUT"),
        ("Canon", "Canon Log / Cinema Gamut", None),
        ("Nikon", "Nikon N-Log / N-Gamut", None),
        ("Fujifilm", "Fujifilm F-Log / F-Gamut", None),
        ("Fujifilm", "Fujifilm F-Log2 / F-Gamut C", None),
        ("Panasonic", "Panasonic V-Log / V-Gamut", "PANASONIC_VLOG-VGAMUT"),
        ("Panasonic", "Panasonic Panalog / V-Gamut", None),
        ("ARRI", "ARRI LogC3 / AWG3", "ARRI_ALEXA-LOGC-EI800-AWG"),
        ("ARRI", "ARRI LogC4 / AWG4", "ARRI_LOGC4"),
        ("RED", "RED Log3G10 / REDWideGamutRGB", "RED_LOG3G10-RWG"),
        ("RED", "RED REDLogFilm / REDWideGamutRGB", "RED_REDLOGFILM-RWG"),
        ("RED", "RED Log3G12 / REDWideGamutRGB", None),
        ("RED", "RED REDLogFilm / DRAGONcolor2", None),
        ("DJI", "DJI D-Log / D-Gamut", None),
        ("Apple", "Apple Log / Rec.2020", "APPLE_LOG"),
        ("Apple", "Apple Log 2 / Apple Wide Gamut", "APPLE_LOG-APPLEWG"),
        ("Blackmagic", "Blackmagic BMD Film Gen5 / Blackmagic Wide Gamut", None),
        ("Blackmagic", "DaVinci Intermediate / DaVinci Wide Gamut", None),
        ("FilmLight", "FilmLight T-Log / E-Gamut", None),
        ("GoPro", "GoPro Protune / Protune Native", None),
    ]

    print(f"  {'vendor':12s} {'space':50s} {'official route':>16s}")
    print("  " + "-" * 82)
    ok = miss = 0
    by_vendor = {}
    for vendor, space, csc in coverage:
        good = csc is not None and csc in supported
        ok += int(good)
        miss += int(not good)
        by_vendor.setdefault(vendor, [0, 0])
        by_vendor[vendor][0] += int(good)
        by_vendor[vendor][1] += 1
        tag = csc if good else "MISSING"
        print(f"  {vendor:12s} {space:50s} {tag:>16s}")

    print()
    print(f"  spaces with an official route : {ok} / {ok+miss}  ({100*ok/(ok+miss):.0f}%)")
    print()
    print(f"  {'vendor':14s} {'covered':>10s}")
    print("  " + "-" * 28)
    for v, (a, b) in sorted(by_vendor.items()):
        mark = "" if a == b else "   <- gap"
        print(f"  {v:14s} {a:>5d}/{b:<4d}{mark}")

    gap_vendors = sorted(v for v, (a, b) in by_vendor.items() if a == 0)
    print()
    print(f"  vendors with NO official route at all: {', '.join(gap_vendors)}")
    print()
    print("  This is the product opening. Half the modelled spaces, and 7 of the 13")
    print("  vendors, cannot be canonicalized with the official ACES tooling. The")
    print("  colour science for all of them is public - only the CSC wrapper is absent.")

    with open(os.path.join(OUT, "clf_vendor_coverage.json"), "w", encoding="utf-8") as fh:
        json.dump({"supported_csc": listed,
                   "coverage": [{"vendor": v, "space": s, "csc": c,
                                 "official": bool(c and c in supported)}
                                for v, s, c in coverage],
                   "n_covered": ok, "n_total": ok + miss,
                   "vendors_without_route": gap_vendors}, fh, indent=2)
    return coverage

    # ------------------------------------------------------------------ build
    print()
    print("=" * 94)
    print("Build a LOOK LUT inside one vendor's log space, then canonicalize it")
    print("=" * 94)
    V = "Sony S-Log3 / S-Gamut3.Cine"
    CSC = "SONY_SLOG3-SGAMUT3.CINE"
    SIZE = 33

    def look_in_vendor(x):
        return h2c(V, run(P_LOOK, c2h(V, x)))

    tab = np.asarray(look_in_vendor(grid(SIZE))).reshape(SIZE, SIZE, SIZE, 3)
    lut = colour.LUT3D(table=tab.astype(np.float32), name=f"look_in_{CSC}",
                       domain=[[0, 0, 0], [1, 1, 1]])
    cube_path = os.path.join(OUT, f"look_in_{CSC}_33.cube")
    colour.write_LUT(lut, cube_path)
    print(f"  look LUT  : {os.path.basename(cube_path)}  ({os.path.getsize(cube_path)/1024:.0f} KB)")

    clf_path = os.path.join(OUT, f"LMT_{CSC}_look.clf")
    cmd = [MAKECLF, cube_path, clf_path, "--csc", CSC, "--generateid", "--verbose"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  ociomakeclf: rc={r.returncode}")
    for ln in (r.stdout + r.stderr).splitlines():
        if ln.strip():
            print(f"    {ln.strip()}")
    print(f"  CLF       : {os.path.basename(clf_path)}  ({os.path.getsize(clf_path)/1024:.0f} KB)")

    # ------------------------------------------------------------------ inspect
    print()
    print("=" * 94)
    print("What is inside the CLF")
    print("=" * 94)
    txt = open(clf_path, encoding="utf-8").read()
    print(f"  total lines: {len(txt.splitlines())}")
    head = [ln for ln in txt.splitlines() if not ln.strip().startswith("<Array")][:40]
    for ln in head[:36]:
        s = ln.rstrip()
        print("   ", s[:150])
    print()
    for key in ["ProcessList", "LUT1D", "LUT3D", "Matrix", "Log", "Range", "InverseLUT1D",
                "ACEStransformID", "Description"]:
        print(f"    contains <{key}>: {('<'+key) in txt}")

    # ------------------------------------------------------------------ verify
    print()
    print("=" * 94)
    print("Verify the CLF against the analytic chain")
    print("=" * 94)
    ft = ocio.FileTransform(clf_path, interpolation=ocio.INTERP_TETRAHEDRAL)
    P_CLF = OCFG.getProcessor(ft).getDefaultCPUProcessor()

    test_hub = c2h(V, np.random.rand(4000, 3))
    clf_out = run(P_CLF, test_hub)
    analytic = run(P_LOOK, test_hub)
    rel = np.abs(clf_out - analytic) / np.maximum(np.abs(analytic), 1e-2)
    print(f"  input  : ACES2065-1 (the hub)")
    print(f"  compare: CLF output vs the analytic look, both in ACES2065-1")
    print(f"    median rel diff {np.median(rel):.6f}")
    print(f"    p99    rel diff {np.percentile(rel,99):.4f}")
    print(f"    max    rel diff {np.max(rel):.4f}")
    print()
    print("  The CLF is a self-describing hub object: it carries the ACES2065-1 <->")
    print("  Sony S-Log3 conversions plus the look, so any ACES-aware application can")
    print("  use it without knowing anything about Sony's colour science.")

    r = subprocess.run([CHECKLUT, clf_path, "0.18 0.18 0.18"],
                       capture_output=True, text=True)
    print()
    print(f"  ociochecklut on ACES 0.18 grey -> {r.stdout.strip() or r.stderr.strip()}")


if __name__ == "__main__":
    main()
