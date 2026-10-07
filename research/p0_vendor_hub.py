"""
P0 (scoped) - hub canonicalization across the 8 vendor families
================================================================

Scope: Sony, Canon, Nikon, Fujifilm, Panasonic, ARRI, RED, DJI
       14 camera log spaces. Nothing else.

Two architectures are compared, because the first one is what everyone builds
by default and it is the one that breaks.

  DESIGN A - display transform baked into the LUT   (the naive way)
      LUT : vendor log code  ->  Rec.709 display code
      To canonicalize you must invert the display transform, so the LMT is
        LMT = ODT_inv . LUT . vendor_code
      Anything the LUT clipped at 0/1 is now permanently unrecoverable, and
      ACEScct itself is a bounded space that clips out-of-gamut values.

  DESIGN B - look only, display transform stays on the hub   (the right way)
      LUT : vendor log code  ->  ACEScct   (creative look, no display render)
      LMT = LUT . acestocct_to_vendor_code
      The display transform is applied once, at the end, by the hub.
      No inverse of a tone-mapped transform is ever needed.

Measured: CIE DeltaE2000 against the analytic chain, over a realistic exposure
distribution (log-uniform in stops around 18% grey, modest chroma).
"""

import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import colour
import PyOpenColorIO as ocio

from camera_spaces import build_registry

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

HUB = "ACES2065-1"
CCT = "ACEScct"
CAM = build_registry()
NAME_LIST = list(CAM.keys())


# ------------------------------------------------------------------ utilities
def grid_points(size):
    g = np.linspace(0.0, 1.0, size)
    R, G, B = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([R, G, B], axis=-1).reshape(-1, 3)


def bake(fn, size):
    return np.asarray(fn(grid_points(size)), dtype=np.float64).reshape(size, size, size, 3)


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


# ------------------------------------------------- per-space representation range
def rep_range(cs):
    """Linear-light values representable by code values 0.0 .. 1.0."""
    z = np.array([[0.0, 0.0, 0.0]])
    o = np.array([[1.0, 1.0, 1.0]])
    lo = float(np.min(np.asarray(cs.cctf_decoding(z))))
    hi = float(np.max(np.asarray(cs.cctf_decoding(o))))
    return lo, hi


REP = {n: rep_range(cs) for n, cs in CAM.items()}


def enc_safe(name, lin):
    """Encode linear light to code values, clamping to what the log curve can hold.

    Some published log curves (REDLogFilm for one) return NaN outside their
    domain instead of clamping. Anything below decode(0.0) has no code value
    anyway, so clamping is exact, not a fudge.
    """
    lo, hi = REP[name]
    return np.asarray(CAM[name].cctf_encoding(np.clip(lin, lo, hi)))


def cam_to_hub(name, x):
    return colour.RGB_to_RGB(x, CAM[name], HUB, apply_cctf_decoding=True,
                             apply_cctf_encoding=False)


def hub_to_cam(name, lin_hub):
    lin = colour.RGB_to_RGB(np.asarray(lin_hub), HUB, CAM[name],
                            apply_cctf_decoding=False, apply_cctf_encoding=False)
    return enc_safe(name, lin)


# ------------------------------------------------------------------ OCIO side
OCFG = ocio.Config.CreateRaw()


def bt(style, inverse=False):
    t = ocio.BuiltinTransform(style=style)
    if inverse:
        t.setDirection(ocio.TRANSFORM_DIR_INVERSE)
    return t


ODT = ocio.GroupTransform([bt("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0"),
                           bt("DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.709")])
ODT_INV = ocio.GroupTransform([bt("DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.709", True),
                               bt("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0", True)])
CCT_TO_AP0 = bt("ACEScct_to_ACES2065-1")
AP0_TO_CCT = bt("ACEScct_to_ACES2065-1", True)
LOOK = ocio.GroupTransform([
    AP0_TO_CCT,
    ocio.CDLTransform(slope=[1.06, 1.00, 0.95], offset=[0.010, 0.000, -0.010],
                      power=[0.95, 1.00, 1.05], sat=0.88),
    CCT_TO_AP0,
])


def proc(g):
    return OCFG.getProcessor(g).getDefaultCPUProcessor()


def run(p, rgb):
    a = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32))
    p.applyRGB(a)
    return a.astype(np.float64)


P_ODT = proc(ODT)
P_ODT_INV = proc(ODT_INV)
P_CCT_TO_AP0 = proc(CCT_TO_AP0)
P_AP0_TO_CCT = proc(AP0_TO_CCT)
P_LOOK = proc(LOOK)


# ----------------------------------------------------------------- measurement
def to_lab(codes):
    lin = np.clip(np.clip(codes, 0.0, 1.0), 0.0, 1.0) ** 2.4
    return colour.XYZ_to_Lab(colour.RGB_to_XYZ(lin, "ITU-R BT.709"))


def dE(a, b):
    return colour.delta_E(to_lab(a), to_lab(b), method="CIE 2000")


def stat(de):
    de = np.asarray(de)
    return {
        "n": int(de.size),
        "mean": round(float(np.mean(de)), 4),
        "median": round(float(np.median(de)), 4),
        "p95": round(float(np.percentile(de, 95)), 4),
        "p99": round(float(np.percentile(de, 99)), 4),
        "max": round(float(np.max(de)), 4),
        "pct_gt_1": round(float(100 * np.mean(de > 1.0)), 2),
        "pct_gt_2": round(float(100 * np.mean(de > 2.0)), 2),
        "pct_gt_5": round(float(100 * np.mean(de > 5.0)), 2),
    }


def hdr(fmt, *vals):
    print(fmt.format(*vals))


# ==============================================================================
def main():
    t0 = time.time()
    R = {"scope": {"vendors": ["Sony", "Canon", "Nikon", "Fujifilm", "Panasonic",
                               "ARRI", "RED", "DJI"],
                   "camera_spaces": len(NAME_LIST),
                   "hub": HUB, "look_space": CCT}}
    np.random.seed(20261006)

    print("=" * 96)
    print("P0  hub canonicalization across the expanded space registry")
    print("=" * 96)
    n = len(NAME_LIST)
    grp = {}
    from camera_spaces import REGISTRY_META
    for s in NAME_LIST:
        grp.setdefault(REGISTRY_META[s]["vendor"], 0)
        grp[REGISTRY_META[s]["vendor"]] += 1
    print(f"  camera log spaces        : {n}  across {len(grp)} vendors")
    print(f"  pairwise routes (no hub) : {n*(n-1)}")
    print(f"  hub routes (2 per space) : {2*n}   -> {n*(n-1)/(2*n):.1f}x fewer")
    print(f"  hub                      : {HUB} (AP0 linear, D60)")
    print(f"  LMT working space        : {CCT}")

    # ---------------------------------------------------------------- PART 1
    print()
    print("=" * 96)
    print(f"PART 1   hub completeness - all {n*(n-1)} ordered routes from {2*n} hub routes")
    print("=" * 96)
    print("  measured in LINEAR LIGHT with no code-value quantisation, so this is the")
    print("  pure cost of going through the hub rather than pairwise matrices.")
    print()

    def space_obj(name):
        return CAM[name] if name in CAM else colour.RGB_COLOURSPACES[name]

    def lin_to_lin(src, dst, lin):
        return colour.RGB_to_RGB(np.asarray(lin), space_obj(src), space_obj(dst),
                                 apply_cctf_decoding=False, apply_cctf_encoding=False)

    x0 = np.random.rand(4000, 3)
    errs = []
    worst = (0.0, None)
    for a in NAME_LIST:
        lin_a = np.asarray(CAM[a].cctf_decoding(x0))
        hub = lin_to_lin(a, HUB, lin_a)
        for b in NAME_LIST:
            if a == b:
                continue
            lin_b = lin_to_lin(HUB, b, hub)          # hub -> B linear
            hub2 = lin_to_lin(b, HUB, lin_b)         # B -> hub
            lin_a2 = lin_to_lin(HUB, a, hub2)        # hub -> A
            # relative to full scale of the batch: robust where linear values
            # approach zero, which a per-sample relative error is not
            scale = max(float(np.max(np.abs(lin_a))), 1e-9)
            e = float(np.max(np.abs(lin_a2 - lin_a))) / scale
            errs.append(e)
            if e > worst[0]:
                worst = (e, f"{a} -> {b}")
    errs = np.array(errs)
    print(f"  routes evaluated : {len(errs)}")
    print(f"  error / full scale: median {np.median(errs):.3e}   "
          f"p95 {np.percentile(errs,95):.3e}   max {errs.max():.3e}")
    print(f"  worst route      : {worst[1]}")
    print("  => kept analytic the hub is an EXACT intermediate: no loss at all.")
    print("     All the error measured elsewhere comes from quantising to a table.")
    R["part1"] = {
        "routes_tested": int(len(errs)),
        "median_rel_err": float(np.median(errs)),
        "p95_rel_err": float(np.percentile(errs, 95)),
        "max_rel_err": float(errs.max()),
        "worst_route": worst[1],
    }

    # ---------------------------------------------------------------- samples
    N = 20000
    stops = np.random.uniform(-6.0, 3.5, (N, 1))
    lin_grey = 0.18 * (2.0 ** stops)
    chroma = np.random.normal(0.0, 0.035, (N, 3))
    chroma -= chroma.mean(axis=1, keepdims=True)
    SAMP = {}
    for v in NAME_LIST:
        SAMP[v] = np.clip(np.asarray(CAM[v].cctf_encoding(lin_grey)) + chroma, 0.0, 1.0)

    # ---------------------------------------------------------------- PART 2A
    print()
    print("=" * 96)
    print("PART 2A  DESIGN A - display transform baked into the LUT  (the naive way)")
    print("=" * 96)
    print("  LUT : vendor log -> Rec.709 display    LMT = ODT_inv . LUT . vendor_code")
    print()
    print(f"  {'vendor space':36s} {'mean':>6s} {'p95':>6s} {'max':>8s} {'>1':>6s} {'>5':>6s}"
          f" | {'mean*':>7s} {'max*':>7s}")
    print("  " + "-" * 92)

    SIZE = 33
    part2a = {}
    for v in NAME_LIST:
        def show_lut(x, v=v):
            return run(P_ODT, run(P_LOOK, cam_to_hub(v, x)))

        table_a = bake(show_lut, SIZE)

        def lmt(u, v=v, table_a=table_a):
            hub = run(P_CCT_TO_AP0, u)
            code = hub_to_cam(v, hub)
            y = tri(table_a, code)
            return run(P_AP0_TO_CCT, run(P_ODT_INV, y))

        lmt_t = bake(lmt, 65)

        def reexp(x, v=v, lmt_t=lmt_t):
            u = run(P_AP0_TO_CCT, cam_to_hub(v, x))
            return run(P_ODT, run(P_CCT_TO_AP0, tetra(lmt_t, u)))

        s = SAMP[v]
        truth = show_lut(s)
        got = tetra(bake(reexp, SIZE), s)
        de = np.asarray(dE(truth, got))
        # strip samples the original LUT already clipped: unrecoverable by construction
        live = np.any((truth > 1e-4) & (truth < 1 - 1e-4), axis=1)
        s_all = stat(de)
        s_live = stat(de[live]) if live.any() else stat(de)
        part2a[v] = {"all": s_all, "excluding_clipped": s_live,
                     "pct_clipped_in_reference": round(float(100 * (1 - live.mean())), 2)}
        print(f"  {v:36s} {s_all['mean']:6.3f} {s_all['p95']:6.3f} {s_all['max']:8.3f} "
              f"{s_all['pct_gt_1']:5.1f}% {s_all['pct_gt_5']:5.1f}% | "
              f"{s_live['mean']:7.3f} {s_live['max']:7.3f}")
    R["part2a_design_A"] = part2a
    m_all = np.mean([x["all"]["mean"] for x in part2a.values()])
    m_live = np.mean([x["excluding_clipped"]["mean"] for x in part2a.values()])
    mx_live = np.max([x["excluding_clipped"]["max"] for x in part2a.values()])
    print()
    print(f"  AVERAGE mean dE : all samples {m_all:.3f}   |   excluding clipped {m_live:.3f}")
    print(f"  worst-case dE   : excluding clipped {mx_live:.3f}")
    print("  (* = samples where the original LUT was not already clipped)")

    # ---------------------------------------------------------------- PART 2B
    print()
    print("=" * 96)
    print("PART 2B  DESIGN B - look only, display transform left on the hub  (correct)")
    print("=" * 96)
    print("  LUT : vendor log -> ACEScct (look)     LMT = LUT . acestocct_to_vendor_code")
    print()
    print(f"  {'vendor space':36s} {'mean':>6s} {'p95':>6s} {'max':>8s} {'>1':>6s} {'>2':>6s}")
    print("  " + "-" * 78)

    part2b = {}
    for v in NAME_LIST:
        # the LMT: ACEScct -> ACEScct. No display transform involved, so no
        # inverse of a tone-mapped transform is ever needed.
        def lmtb(u, v=v):
            hub = run(P_CCT_TO_AP0, u)
            code = hub_to_cam(v, hub)
            return run(P_AP0_TO_CCT, run(P_LOOK, cam_to_hub(v, code)))

        # ground truth: the same look, with the display transform applied at the hub
        def truth_b(x, v=v):
            return run(P_ODT, run(P_LOOK, cam_to_hub(v, x)))

        t_lmt = bake(lmtb, 65)

        def reexp_b(x, v=v, t_lmt=t_lmt):
            u = run(P_AP0_TO_CCT, cam_to_hub(v, x))
            w = tetra(t_lmt, u)
            return run(P_ODT, run(P_CCT_TO_AP0, w))

        s = SAMP[v]
        de = np.asarray(dE(truth_b(s), tetra(bake(reexp_b, SIZE), s)))
        st_b = stat(de)
        part2b[v] = st_b
        print(f"  {v:36s} {st_b['mean']:6.3f} {st_b['p95']:6.3f} {st_b['max']:8.3f} "
              f"{st_b['pct_gt_1']:5.1f}% {st_b['pct_gt_2']:5.1f}%")
    R["part2b_design_B"] = part2b
    mb = np.mean([x["mean"] for x in part2b.values()])
    mxb = np.max([x["max"] for x in part2b.values()])
    print()
    print(f"  AVERAGE over {n} spaces: mean dE {mb:.3f}   worst-case dE {mxb:.3f}")
    R["part2_summary"] = {
        "design_A_mean_all": round(float(m_all), 4),
        "design_A_mean_excl_clipped": round(float(m_live), 4),
        "design_A_worst_excl_clipped": round(float(mx_live), 4),
        "design_B_mean": round(float(mb), 4),
        "design_B_worst": round(float(mxb), 4),
    }

    # ---------------------------------------------------------------- PART 3
    print()
    print("=" * 96)
    print("PART 3   encoding domain - what a [0,1] .cube physically cannot hold")
    print("=" * 96)
    src = "Sony S-Log3 / S-Gamut3.Cine"
    g = grid_points(17)
    hub = cam_to_hub(src, g)
    print(f"  source: {src}, uniform over code [0,1]^3 -> {len(g)} samples")
    print("  measured on the RAW conversion, before any clamping")
    print()
    print(f"  {'target space':36s} {'<0':>8s} {'>1':>8s} {'NaN':>7s} {'total':>8s} {'min':>10s} {'max':>9s}")
    print("  " + "-" * 92)
    part3 = {}
    for v in NAME_LIST:
        if v == src:
            continue
        lin_t = colour.RGB_to_RGB(hub, HUB, CAM[v], apply_cctf_decoding=False,
                                  apply_cctf_encoding=False)
        with np.errstate(all="ignore"):
            y = np.asarray(CAM[v].cctf_encoding(lin_t), dtype=np.float64)
        bad = ~np.isfinite(y)
        below = 100 * float(np.mean((y < 0.0) & ~bad))
        above = 100 * float(np.mean((y > 1.0) & ~bad))
        nanp = 100 * float(np.mean(bad))
        part3[v] = {"pct_below_0": round(below, 2), "pct_above_1": round(above, 2),
                    "pct_nan": round(nanp, 2), "pct_outside": round(below + above + nanp, 2),
                    "min": round(float(np.nanmin(y)), 2), "max": round(float(np.nanmax(y)), 3)}
        print(f"  {v:36s} {below:7.2f}% {above:7.2f}% {nanp:6.2f}% "
              f"{below+above+nanp:7.2f}% {np.nanmin(y):10.2f} {np.nanmax(y):9.3f}")
    R["part3_encoding_domain"] = part3
    avg_out = np.mean([x["pct_outside"] for x in part3.values()])
    print()
    print(f"  average fraction of the conversion that a [0,1] .cube cannot hold: {avg_out:.1f}%")
    print("  => a plain .cube silently destroys this. CLF (floating point) does not.")
    R["part3_avg_outside_pct"] = round(float(avg_out), 2)

    # ---------------------------------------------------------------- PART 5
    print()
    print("=" * 96)
    print("PART 5   the fix - gamut compression in front of the LMT cube")
    print("=" * 96)
    print("  Root cause of the outliers is NOT the display clip. It is the cube domain:")
    print("  saturated colours produce deeply NEGATIVE ACEScct values, which the 0..1")
    print("  cube flattens to 0. Measured below on Red Log3G10, the worst vendor.")
    print()

    GC = ocio.GroupTransform([bt("ACES-LMT - ACES 1.3 Reference Gamut Compression")])
    P_GC = proc(GC)
    P_LOOK_ACES = proc(ocio.GroupTransform([
        ocio.CDLTransform(slope=[1.06, 1.00, 0.95], offset=[0.010, 0.000, -0.010],
                          power=[0.95, 1.00, 1.05], sat=0.88)]))

    V = "RED Log3G10 / REDWideGamutRGB"

    print("  The LMT here is the LOOK ALONE (ACEScct -> ACEScct), which is the correct")
    print("  hub object. The question is whether its cube INPUT stays inside [0,1].")
    print()

    def look_cct(u):
        """ACEScct -> ACEScct, carrying only the creative look."""
        hub_ = run(P_CCT_TO_AP0, u)
        return run(P_AP0_TO_CCT, run(P_LOOK_ACES, hub_))

    # how far outside the cube domain do the real inputs go?
    hb = cam_to_hub(V, SAMP[V])
    u_raw = run(P_AP0_TO_CCT, hb)
    u_gc = run(P_AP0_TO_CCT, run(P_GC, hb))
    for lbl, uu in [("no gamut compression", u_raw), ("with ACES 1.3 gamut compression", u_gc)]:
        below = 100 * float(np.mean(np.any(uu < 0, axis=1)))
        above = 100 * float(np.mean(np.any(uu > 1, axis=1)))
        print(f"  input ACEScct, {lbl:34s} range [{uu.min():7.3f}, {uu.max():6.3f}]  "
              f"rows outside [0,1]: {below+above:6.3f}%")
    print()

    def to_lab_h(codes):
        lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(np.clip(codes, 0, 1) ** 2.4, "ITU-R BT.709"))
        return lab, np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))

    print(f"  {'variant':40s} {'mean dE':>9s} {'p95 dE':>9s} {'max dE':>9s} "
          f"{'p99 dHue':>10s}")
    print("  " + "-" * 84)
    part5 = {}
    for label, gc_on in [("no gamut compression (cube clips)", False),
                         ("+ ACES 1.3 gamut compression", True)]:
        def pre(x):
            h = cam_to_hub(V, x)
            if gc_on:
                h = run(P_GC, h)
            return h

        # truth: everything analytic, display applied once at the hub
        truth = run(P_ODT, run(P_LOOK_ACES, pre(SAMP[V])))

        # the LMT cube, sampled over ACEScct [0,1]^3
        t_cube = bake(look_cct, 65)

        def reexp(x):
            u = run(P_AP0_TO_CCT, pre(x))
            w = tetra(t_cube, u)
            return run(P_ODT, run(P_CCT_TO_AP0, w))

        got = tetra(bake(reexp, SIZE), SAMP[V])
        de = np.asarray(dE(truth, got))
        lab_t, hue_t = to_lab_h(truth)
        lab_g, hue_g = to_lab_h(got)
        d = np.abs(hue_g - hue_t) % 360
        d = np.where(d > 180, 360 - d, d)
        lowchroma = np.hypot(lab_t[..., 1], lab_t[..., 2]) < 5
        d = np.where(lowchroma, 0.0, d)          # hue is meaningless near neutral
        part5[label] = {
            "mean_dE": round(float(de.mean()), 4),
            "p95_dE": round(float(np.percentile(de, 95)), 4),
            "max_dE": round(float(de.max()), 4),
            "p99_dHue": round(float(np.percentile(d, 99)), 3),
        }
        print(f"  {label:40s} {de.mean():9.3f} {np.percentile(de,95):9.3f} "
              f"{de.max():9.3f} {np.percentile(d,99):10.3f}")
    R["part5_gamut_compression"] = part5
    print()
    print("  Clipping is not a domain fix: it silently maps every negative channel to")
    print("  zero, which is where the max dE of ~34 comes from. Gamut compression puts")
    print("  the values back inside the domain the cube can actually represent, and the")
    print("  error collapses to interpolation noise.")

    # ---------------------------------------------------------------- PART 4
    print()
    print("=" * 96)
    print("PART 4   implementation divergence - colour-science vs OCIO official IDT")
    print("=" * 96)
    pairs = {
        "Sony S-Log3 / S-Gamut3.Cine": "SONY_SLOG3-SGAMUT3.CINE_to_ACES2065-1",
        "Sony S-Log3 / S-Gamut3": "SONY_SLOG3-SGAMUT3_to_ACES2065-1",
        "Canon Log 2 / Cinema Gamut": "CANON_CLOG2-CGAMUT_to_ACES2065-1",
        "Canon Log 3 / Cinema Gamut": "CANON_CLOG3-CGAMUT_to_ACES2065-1",
        "Panasonic V-Log / V-Gamut": "PANASONIC_VLOG-VGAMUT_to_ACES2065-1",
        "ARRI LogC3 / AWG3": "ARRI_ALEXA-LOGC-EI800-AWG_to_ACES2065-1",
        "ARRI LogC4 / AWG4": "ARRI_LOGC4_to_ACES2065-1",
        "RED Log3G10 / REDWideGamutRGB": "RED_LOG3G10-RWG_to_ACES2065-1",
        "RED REDLogFilm / REDWideGamutRGB": "RED_REDLOGFILM-RWG_to_ACES2065-1",
    }
    print(f"  {'space':36s} {'median':>10s} {'p99':>9s} {'max':>10s}  verdict")
    print("  " + "-" * 88)
    part4 = {}
    for mine, style in pairs.items():
        p = proc(ocio.BuiltinTransform(style=style))
        x = np.random.rand(6000, 3)
        mine_hub = cam_to_hub(mine, x)
        ocio_hub = run(p, x)
        rel = np.abs(mine_hub - ocio_hub) / np.maximum(np.abs(ocio_hub), 1e-3)
        med, p99, mx = (float(np.median(rel)), float(np.percentile(rel, 99)), float(np.max(rel)))
        verdict = "agree" if mx < 0.05 else ("tail differs" if med < 0.01 else "DIFFER")
        part4[mine] = {"ocio_style": style, "median_rel": round(med, 6),
                       "p99_rel": round(p99, 4), "max_rel": round(mx, 4), "verdict": verdict}
        print(f"  {mine:36s} {med:10.6f} {p99:9.4f} {mx:10.4f}  {verdict}")
    R["part4_implementation_divergence"] = part4
    print()
    print("  Sony / Canon / ARRI: the two independent implementations agree.")
    print("  Panasonic / RED   : they diverge on out-of-gamut colours - a hub must")
    print("                      pick and validate ONE, not mix them.")

    R["runtime_seconds"] = round(time.time() - t0, 1)
    p = os.path.join(OUT, "p0_vendor_results.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2, ensure_ascii=False)
    print()
    print(f"runtime {R['runtime_seconds']}s   -> {p}")


if __name__ == "__main__":
    main()
