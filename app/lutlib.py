"""
lutlib - shared colour plumbing for the LUT hub prototype
=========================================================

One place for: the 8-vendor space registry, the OCIO hub pipeline,
grid/interpolation helpers, and perceptual error measurement.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import colour
import PyOpenColorIO as ocio

from camera_spaces import build_registry, repr_range, encode_clamped

HUB = "ACES2065-1"
DSP = "ITU-R BT.709"

CAM = build_registry()


# ------------------------------------------------------------------ OCIO chain
OCFG = ocio.Config.CreateRaw()


def bt(style, inverse=False):
    t = ocio.BuiltinTransform(style=style)
    if inverse:
        t.setDirection(ocio.TRANSFORM_DIR_INVERSE)
    return t


ODT = ocio.GroupTransform([
    bt("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0"),
    bt("DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.709"),
])
ODT_INV = ocio.GroupTransform([
    bt("DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.709", True),
    bt("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0", True),
])
GAMUT_COMPRESS = ocio.GroupTransform([
    bt("ACES-LMT - ACES 1.3 Reference Gamut Compression")])
CCT_TO_AP0 = bt("ACEScct_to_ACES2065-1")
AP0_TO_CCT = bt("ACEScct_to_ACES2065-1", True)


def proc(group):
    return OCFG.getProcessor(group).getDefaultCPUProcessor()


P_ODT = proc(ODT)
P_ODT_INV = proc(ODT_INV)
P_GC = proc(GAMUT_COMPRESS)
P_CCT_TO_AP0 = proc(CCT_TO_AP0)
P_AP0_TO_CCT = proc(AP0_TO_CCT)


def run(p, x):
    """OCIO's applyRGB edits in place; always hand it a fresh float32 copy."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    p.applyRGB(a)
    return a.astype(np.float64)


def look_processor(cdl):
    return proc(ocio.GroupTransform([AP0_TO_CCT, cdl, CCT_TO_AP0]))


# ------------------------------------------------------------------- geometry
def grid_points(n):
    """n^3 lattice in [0,1]^3, ordered [R,G,B] major (row-vector convention)."""
    g = np.linspace(0.0, 1.0, n)
    R, G, B = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([R, G, B], axis=-1).reshape(-1, 3)


def bake(fn, n):
    return np.asarray(fn(grid_points(n)), dtype=np.float64).reshape(n, n, n, 3)


def tri(t, rgb, lo=0.0, hi=1.0):
    n = t.shape[0]
    x = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0) * (n - 1)
    i0 = np.clip(np.floor(x).astype(np.int64), 0, n - 2)
    f = x - i0
    r, g, b = i0[:, 0], i0[:, 1], i0[:, 2]
    fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    c00 = t[r, g, b] * (1 - fr) + t[r + 1, g, b] * fr
    c01 = t[r, g, b + 1] * (1 - fr) + t[r + 1, g, b + 1] * fr
    c10 = t[r, g + 1, b] * (1 - fr) + t[r + 1, g + 1, b] * fr
    c11 = t[r, g + 1, b + 1] * (1 - fr) + t[r + 1, g + 1, b + 1] * fr
    return (c00 * (1 - fg) + c10 * fg) * (1 - fb) + (c01 * (1 - fg) + c11 * fg) * fb


def tetra(t, rgb, lo=0.0, hi=1.0):
    n = t.shape[0]
    x = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0) * (n - 1)
    i0 = np.clip(np.floor(x).astype(np.int64), 0, n - 2)
    f = x - i0
    r, g, b = i0[:, 0], i0[:, 1], i0[:, 2]
    fr, fg, fb = f[:, 0], f[:, 1], f[:, 2]
    c000 = t[r, g, b]
    c111 = t[r + 1, g + 1, b + 1]
    out = np.empty_like(c000)
    cases = [
        ((fr >= fg) & (fg >= fb), 1, 0, 0, 1, 1, 0, 1, 1, 1, fr, fg, fb),
        ((fr >= fb) & (fb > fg), 1, 0, 0, 1, 0, 1, 1, 1, 1, fr, fb, fg),
        ((fb > fr) & (fr >= fg), 0, 0, 1, 1, 0, 1, 1, 1, 1, fb, fr, fg),
        ((fb > fg) & (fg > fr), 0, 0, 1, 0, 1, 1, 1, 1, 1, fb, fg, fr),
        ((fg > fr) & (fr >= fb), 0, 1, 0, 1, 1, 0, 1, 1, 1, fg, fr, fb),
        ((fg >= fb) & (fb > fr), 0, 1, 0, 0, 1, 1, 1, 1, 1, fg, fb, fr),
    ]
    for m, a1, b1, c1, a2, b2, c2, a3, b3, c3, w1, w2, w3 in cases:
        if not m.any():
            continue
        i = np.where(m)[0]
        v1 = t[r[i] + a1, g[i] + b1, b[i] + c1]
        v2 = t[r[i] + a2, g[i] + b2, b[i] + c2]
        v3 = t[r[i] + a3, g[i] + b3, b[i] + c3]
        out[i] = (c000[i] + (v1 - c000[i]) * w1[i][:, None]
                  + (v2 - v1) * w2[i][:, None] + (v3 - v2) * w3[i][:, None])
    return out


# ---------------------------------------------------------------- space maths
def space_obj(name):
    return CAM[name] if name in CAM else colour.RGB_COLOURSPACES[name]


def repr_range_space(cs):
    """Linear-light range representable by code values 0.0 .. 1.0."""
    return repr_range(cs)


def cam_to_hub(cs, code):
    return colour.RGB_to_RGB(code, cs, HUB, apply_cctf_decoding=True,
                             apply_cctf_encoding=False)


def hub_to_cam(cs, lin_hub):
    lin = colour.RGB_to_RGB(np.asarray(lin_hub), HUB, cs, apply_cctf_decoding=False,
                            apply_cctf_encoding=False)
    return encode_clamped(cs, lin)


def lin_to_lin(src, dst, lin):
    return colour.RGB_to_RGB(np.asarray(lin), space_obj(src), space_obj(dst),
                             apply_cctf_decoding=False, apply_cctf_encoding=False)


# --------------------------------------------------------------- measurement
def to_lab(codes):
    lin = np.clip(np.clip(codes, 0.0, 1.0), 0.0, 1.0) ** 2.4
    return colour.XYZ_to_Lab(colour.RGB_to_XYZ(lin, DSP))


def dE(a, b):
    return colour.delta_E(to_lab(a), to_lab(b), method="CIE 2000")


def stat(de):
    de = np.asarray(de)
    return {
        "mean": round(float(np.mean(de)), 4),
        "median": round(float(np.median(de)), 4),
        "p95": round(float(np.percentile(de, 95)), 4),
        "max": round(float(np.max(de)), 4),
        "pct_gt_1": round(float(100 * np.mean(de > 1.0)), 2),
    }
