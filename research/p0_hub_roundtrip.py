"""
P0 - Hub canonicalization round-trip experiment
================================================

Question
--------
If every LUT is canonicalized onto ONE hub (ACES2065-1) instead of being
converted pairwise, does the round trip lose unacceptable quality?

Everything below is built from SPEC-PUBLISHED transforms shipped with
OpenColorIO 2.6 - no vendor LUT files are used or redistributed.

Transforms used (OCIO builtins)
-------------------------------
  IDT   SONY_SLOG3-SGAMUT3.CINE_to_ACES2065-1
  ODT   ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0   (ACES 2.0)
  DISP  DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.709
  CCT   ACEScct_to_ACES2065-1

Ground truth chain for the "original LUT" (a show LUT for a Sony camera):

  S-Log3 code  --IDT-->  ACES2065-1        (hub)
               --LOOK--> ACES2065-1        (creative look, applied in ACEScct)
               --ODT-->  CIE-XYZ-D65
               --DISP--> Rec.709 code      (display)

That chain is baked to a 33^3 .cube  ->  LUT_A.

Canonicalization (the "LMT sandwich"):
  LMT_cct(u) = to_cct( inv_output( LUT_A( inv_idt( to_hub(u) ) ) ) )
  where u is an ACEScct value, and the result is again ACEScct.
  The LMT is what lives on the hub. Sampled 33^3 / 65^3.

Re-export:
  LUT_B(x) = DISP( ODT( from_cct_to_hub( LMT_cct( to_cct( IDT(x) ) ) ) ) )

Metric: CIE DeltaE 2000 on Rec.709 display code values, evaluated on
lattice / random / neutral-axis / anchor samples.

Ablations
  1. no shaper  - sample the sandwich on raw scene-linear ACES2065-1 in a 0..1 cube
  2. grid size  - internal 33^3 vs 65^3
  3. interpolation - trilinear vs tetrahedral
"""

import json
import os
import time

import numpy as np
import PyOpenColorIO as ocio
import colour

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- LUT utilities
def grid_points(size):
    g = np.linspace(0.0, 1.0, size)
    R, G, B = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([R, G, B], axis=-1).reshape(-1, 3)


def bake(fn, size):
    pts = grid_points(size)
    return np.asarray(fn(pts), dtype=np.float64).reshape(size, size, size, 3)


def apply_trilinear(t, rgb, lo=0.0, hi=1.0):
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


def apply_tetrahedral(t, rgb, lo=0.0, hi=1.0):
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
        out[i] = (
            c000[i]
            + (v1 - c000[i]) * w1[i][:, None]
            + (v2 - v1) * w2[i][:, None]
            + (v3 - v2) * w3[i][:, None]
        )
    return out


# ------------------------------------------------------------------ OCIO setup
CFG = ocio.Config.CreateRaw()


def bt(style, inverse=False):
    t = ocio.BuiltinTransform(style=style)
    if inverse:
        t.setDirection(ocio.TRANSFORM_DIR_INVERSE)
    return t


IDT = bt("SONY_SLOG3-SGAMUT3.CINE_to_ACES2065-1")
IDT_INV = bt("SONY_SLOG3-SGAMUT3.CINE_to_ACES2065-1", inverse=True)
ODT = bt("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0")
ODT_INV = bt("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0", inverse=True)
DISP = bt("DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.709")
DISP_INV = bt("DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.709", inverse=True)
CCT_TO_AP0 = bt("ACEScct_to_ACES2065-1")
AP0_TO_CCT = bt("ACEScct_to_ACES2065-1", inverse=True)


def cpu(group):
    return CFG.getProcessor(group).getDefaultCPUProcessor()


def run(processor, rgb):
    """OCIO applyRGB edits the array in place; make a float32 working copy."""
    arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32))
    processor.applyRGB(arr)
    return arr.astype(np.float64)


# the creative look: a warm, slightly desaturated filmic grade, applied in ACEScct
LOOK_INNER = ocio.CDLTransform(
    slope=[1.06, 1.00, 0.95],
    offset=[0.010, 0.000, -0.010],
    power=[0.95, 1.00, 1.05],
    sat=0.88,
)

# LOOK: ACES2065-1 -> ACES2065-1   (this is what the hub object should recover)
LOOK = ocio.GroupTransform([AP0_TO_CCT, LOOK_INNER, CCT_TO_AP0])

# the "original show LUT" chain: S-Log3 code -> Rec.709 display code
CHAIN_A = ocio.GroupTransform([IDT, AP0_TO_CCT, LOOK_INNER, CCT_TO_AP0, ODT, DISP])

# canonicalization plumbing
LMT_PRE = ocio.GroupTransform([CCT_TO_AP0, IDT_INV])            # ACEScct -> S-Log3
LMT_POST = ocio.GroupTransform([DISP_INV, ODT_INV, AP0_TO_CCT])  # Rec.709 -> ACEScct

# re-export plumbing
PRE_B = ocio.GroupTransform([IDT, AP0_TO_CCT])                   # S-Log3 -> ACEScct
POST_B = ocio.GroupTransform([CCT_TO_AP0, ODT, DISP])            # ACES2065-1 -> Rec.709

# no-shaper variant: hub values used directly, clipped into the cube domain
PRE_NS = ocio.GroupTransform([IDT])                              # S-Log3 -> ACES2065-1
POST_NS = ocio.GroupTransform([ODT, DISP])                       # ACES2065-1 -> Rec.709


# ---------------------------------------------------------------- measurement
def to_lab(codes):
    lin = np.clip(np.clip(codes, 0.0, 1.0), 0.0, 1.0) ** 2.4     # BT.1886 Lb=0 Lw=1
    XYZ = colour.RGB_to_XYZ(lin, "ITU-R BT.709")
    return colour.XYZ_to_Lab(XYZ)


def dE(a, b):
    return colour.delta_E(to_lab(a), to_lab(b), method="CIE 2000")


def st(de):
    de = np.asarray(de)
    return {
        "mean": round(float(np.mean(de)), 4),
        "median": round(float(np.median(de)), 4),
        "p95": round(float(np.percentile(de, 95)), 4),
        "p99": round(float(np.percentile(de, 99)), 4),
        "max": round(float(np.max(de)), 4),
        "pct_gt_1": round(float(100 * np.mean(de > 1.0)), 2),
        "pct_gt_2": round(float(100 * np.mean(de > 2.0)), 2),
        "pct_gt_5": round(float(100 * np.mean(de > 5.0)), 2),
    }


def line(name, s):
    return (
        f"{name:26s} mean {s['mean']:6.3f}  med {s['median']:6.3f}  "
        f"p95 {s['p95']:7.3f}  max {s['max']:8.3f}  "
        f">1:{s['pct_gt_1']:5.1f}%  >2:{s['pct_gt_2']:5.1f}%"
    )


# ==============================================================================
def main():
    t_start = time.time()
    report = {"environment": {}}
    np.random.seed(20261006)

    report["environment"] = {
        "ocio": ocio.__version__,
        "colour_science": colour.__version__,
        "numpy": np.__version__,
        "idt": "SONY_SLOG3-SGAMUT3.CINE_to_ACES2065-1",
        "odt": "ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0",
        "display": "DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.709",
        "hub": "ACES2065-1",
        "lmt_internal_space": "ACEScct",
    }

    print("=" * 88)
    print("STEP 0  verify OCIO round trips are exact (invertibility of the plumbing)")
    print("=" * 88)
    probe = np.array([[0.1, 0.1, 0.1], [0.41, 0.41, 0.41], [0.6, 0.6, 0.6],
                      [0.9, 0.9, 0.9], [0.55, 0.47, 0.42]])
    f = run(cpu(IDT), probe)
    r = run(cpu(IDT_INV), f)
    print(f"  S-Log3 -> ACES2065-1 -> S-Log3   max err {np.max(np.abs(r - probe)):.3e}")
    report["environment"]["idt_invert_err"] = float(np.max(np.abs(r - probe)))

    xyz = run(cpu(ODT), f)
    back = run(cpu(ODT_INV), xyz)
    print(f"  ACES2065-1 -> XYZ -> ACES2065-1  max err {np.max(np.abs(back - f)):.3e}")
    report["environment"]["odt_invert_err"] = float(np.max(np.abs(back - f)))

    d = run(cpu(DISP), xyz)
    backd = run(cpu(DISP_INV), d)
    print(f"  XYZ -> Rec.709 -> XYZ            max err {np.max(np.abs(backd - xyz)):.3e}")
    print(f"  Rec.709 display codes for the probe: {np.round(d,4).tolist()}")

    # ---------------------------------------------------------------- LUT_A
    print()
    print("=" * 88)
    print("STEP 1  bake the 'original LUT'  S-Log3/S-Gamut3.Cine -> Rec.709  (33^3)")
    print("=" * 88)
    SIZE = 33
    proc_a = cpu(CHAIN_A)

    def lut_a(x):
        return run(proc_a, x)

    t0 = time.time()
    table_a = bake(lut_a, SIZE)
    print(f"  baked {SIZE}^3 in {time.time()-t0:.2f}s")
    print(f"  output range [{table_a.min():.4f}, {table_a.max():.4f}]")
    clipped_frac = float(np.mean((table_a <= 1e-6) | (table_a >= 1 - 1e-6)))
    print(f"  fraction of channels pinned at 0.0/1.0 : {100*clipped_frac:.2f}%")
    report["lut_a"] = {
        "size": SIZE,
        "range": [round(float(table_a.min()), 4), round(float(table_a.max()), 4)],
        "pct_channels_pinned": round(100 * clipped_frac, 2),
    }

    lutA = colour.LUT3D(table=table_a.astype(np.float32),
                        name="P0_showLUT_SLog3_to_Rec709",
                        domain=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    p = os.path.join(OUT, "A_show_lut_slog3_to_rec709_33.cube")
    colour.write_LUT(lutA, p)
    print(f"  wrote {os.path.basename(p)}  ({os.path.getsize(p)/1024:.1f} KB)")

    # ---------------------------------------------------------------- samples
    lat17 = grid_points(17)
    rnd = np.random.rand(20000, 3)
    neut = np.stack([np.linspace(0, 1, 513)] * 3, axis=-1)
    anchors = np.array([
        [0.41, 0.41, 0.41],   # ~18% grey in S-Log3
        [0.60, 0.60, 0.60],
        [0.09, 0.09, 0.09],   # deep shadow
        [0.90, 0.90, 0.90],   # highlight
        [0.55, 0.47, 0.42],   # warm skin
        [0.45, 0.42, 0.48],   # cool skin
        [0.35, 0.50, 0.30],   # foliage
    ])

    # a realistic exposure distribution: log-uniform in stops around 18% grey,
    # with modest chroma added in code space. This is what real footage looks
    # like, and it is very different from a uniform sweep of the code range.
    N = 20000
    stops = np.random.uniform(-7.0, 4.0, (N, 1))
    lin = 0.18 * (2.0 ** stops)
    grey_code = colour.cctf_encoding(lin, "S-Log3")
    chroma = np.random.normal(0.0, 0.035, (N, 3))
    chroma -= chroma.mean(axis=1, keepdims=True)
    real = np.clip(grey_code + chroma, 0.0, 1.0)

    samples = {
        "lattice_17": lat17,
        "random_20k": rnd,
        "realistic_20k": real,
        "neutral_axis": neut,
        "anchors": anchors,
    }
    truth = {k: lut_a(v) for k, v in samples.items()}

    # a reference sample is "degenerate" if the original LUT output is pinned
    # at 0.0 or 1.0 - in that region any sane implementation matches trivially
    alive = {k: np.any((v > 1e-6) & (v < 1 - 1e-6), axis=1) for k, v in truth.items()}
    print()
    print("  sample sets and how much of each is actually informative:")
    for k in samples:
        print(f"    {k:16s} n={len(samples[k]):6d}  "
              f"non-degenerate {100*alive[k].mean():5.1f}%")
    report["sample_sets"] = {
        k: {"n": int(len(samples[k])), "pct_non_degenerate": round(float(100 * alive[k].mean()), 2)}
        for k in samples
    }

    # ---------------------------------------------------------------- STEP 2
    print()
    print("=" * 88)
    print("STEP 2  reference floor: the original 33^3 LUT vs the analytic chain")
    print("=" * 88)
    report["floor"] = {}
    for iname, interp in [("trilinear", apply_trilinear), ("tetrahedral", apply_tetrahedral)]:
        s = st(dE(truth["lattice_17"], interp(table_a, lat17)))
        report["floor"][iname] = s
        print("  " + line(f"33^3 self, {iname}", s))
    print("  (this is the LUT's own quantization error - the floor nothing can beat)")

    # ---------------------------------------------------------------- STEP 3
    print()
    print("=" * 88)
    print("STEP 3  canonicalize onto the hub  (build the LMT in ACEScct)")
    print("=" * 88)
    proc_pre = cpu(LMT_PRE)
    proc_post = cpu(LMT_POST)

    def lmt_cct(u, interp):
        slog = run(proc_pre, u)               # ACEScct -> S-Log3 code
        y = interp(table_a, slog)             # the original LUT
        return run(proc_post, y)              # Rec.709 display -> ACEScct

    hub_stats = {}
    for size in (33, 65):
        t0 = time.time()
        tbl = bake(lambda p_: lmt_cct(p_, apply_trilinear), size)
        np.save(os.path.join(OUT, f"LMT_acescct_{size}.npy"), tbl)
        hub_stats[str(size)] = {
            "range": [round(float(tbl.min()), 4), round(float(tbl.max()), 4)],
            "seconds": round(time.time() - t0, 2),
        }
        print(f"  LMT {size}^3 baked in {time.time()-t0:.2f}s   "
              f"range [{tbl.min():.4f}, {tbl.max():.4f}]")

    # identity recovery check: LMT should equal the analytic LOOK
    idn = np.random.rand(5000, 3)
    lmt_sampled = apply_trilinear(np.load(os.path.join(OUT, "LMT_acescct_65.npy")), idn)
    look_true = run(cpu(LOOK), run(cpu(CCT_TO_AP0), idn))
    look_true = run(cpu(AP0_TO_CCT), look_true)
    de_id = dE(np.clip(lmt_sampled, 0, 1), np.clip(look_true, 0, 1))
    print(f"  LMT vs analytic LOOK (ACEScct): mean dE {de_id.mean():.3f}  "
          f"p95 {np.percentile(de_id,95):.3f}  max {de_id.max():.3f}")
    print("  NOTE: that dE is NOT meaningful as a colour error (ACEScct is not a")
    print("        display space). It only shows the LMT is close to the LOOK.")
    report["lmt_vs_look"] = {
        "mean": round(float(de_id.mean()), 4),
        "p95": round(float(np.percentile(de_id, 95)), 4),
        "max": round(float(de_id.max()), 4),
        "caveat": "measured in ACEScct, not a perceptual display space",
    }
    report["lmt"] = hub_stats

    # ---------------------------------------------------------------- STEP 4
    print()
    print("=" * 88)
    print("STEP 4  re-export from the hub and compare against the original LUT")
    print("=" * 88)
    proc_preb = cpu(PRE_B)
    proc_postb = cpu(POST_B)

    results = {}
    for size in (33, 65):
        tbl = np.load(os.path.join(OUT, f"LMT_acescct_{size}.npy"))
        for iname, interp in [("trilinear", apply_trilinear), ("tetrahedral", apply_tetrahedral)]:
            def lut_b(x):
                w = run(proc_preb, x)
                y = interp(tbl, w)
                return run(proc_postb, y)

            tbl_b = bake(lut_b, SIZE)
            colour.write_LUT(
                colour.LUT3D(table=tbl_b.astype(np.float32),
                             name=f"P0_reexport_from_hub_{size}_{iname}",
                             domain=[0.0, 1.0]),
                os.path.join(OUT, f"B_reexport_hub_{size}_{iname}.cube"),
            )

            key = f"hub_{size}_{iname}"
            results[key] = {}
            print()
            print(f"  --- internal LMT {size}^3, {iname} ---")
            for sname in samples:
                s = st(dE(truth[sname], interp(tbl_b, samples[sname])))
                results[key][sname] = s
                print("    " + line(sname, s))
    report["roundtrip"] = results

    # ---------------------------------------------------------------- STEP 5
    print()
    print("=" * 88)
    print("ABLATION 1  no shaper: raw scene-linear ACES2065-1 into a 0..1 cube")
    print("=" * 88)
    proc_prens = cpu(PRE_NS)
    proc_postns = cpu(POST_NS)

    lmt_ns = bake(
        lambda p_: run(cpu(CCT_TO_AP0),
                       run(proc_postns, apply_trilinear(
                           table_a, run(proc_pre, run(cpu(AP0_TO_CCT), p_))))),
        65,
    )
    # simpler construction matching the scripts intent:
    def lmt_hub_ns(x_hub):
        clipped = np.clip(x_hub, 0.0, 1.0)
        slog = run(cpu(IDT_INV), clipped)
        return run(cpu(ODT), run(cpu(DISP), apply_trilinear(table_a, slog)))

    lmt_ns = bake(lmt_hub_ns, 65)

    def lut_b_ns(x):
        hub = run(proc_prens, x)                # S-Log3 -> ACES2065-1 (linear, >1!)
        clipped = np.clip(hub, 0.0, 1.0)        # <-- the bug
        y_hub = apply_trilinear(lmt_ns, clipped)
        return run(proc_postns, y_hub)

    tbl_ns = bake(lut_b_ns, SIZE)
    report["ablation_no_shaper"] = {}
    for sname in samples:
        s = st(dE(truth[sname], apply_trilinear(tbl_ns, samples[sname])))
        report["ablation_no_shaper"][sname] = s
        print("  " + line(sname, s))

    # ---------------------------------------------------------------- STEP 6
    print()
    print("=" * 88)
    print("EVIDENCE  how far outside 0..1 do hub values actually go?")
    print("=" * 88)
    hub_vals = run(proc_prens, lat17)
    pct_c = 100 * float(np.mean(hub_vals > 1.0))
    pct_r = 100 * float(np.mean(np.any(hub_vals > 1.0, axis=1)))
    pct_neg = 100 * float(np.mean(hub_vals < 0.0))
    print(f"  lattice_17 (S-Log3 0..1) -> ACES2065-1:")
    print(f"    max value            : {hub_vals.max():.2f}")
    print(f"    channels > 1.0       : {pct_c:.2f}%")
    print(f"    samples with any >1.0: {pct_r:.2f}%")
    print(f"    channels < 0.0       : {pct_neg:.2f}%")
    print("  => a bounding 0..1 3D LUT destroys this range. A shaper is mandatory.")
    report["hub_domain"] = {
        "max": round(float(hub_vals.max()), 3),
        "pct_channels_gt1": round(pct_c, 2),
        "pct_samples_any_gt1": round(pct_r, 2),
        "pct_channels_negative": round(pct_neg, 2),
    }

    report["runtime_seconds"] = round(time.time() - t_start, 1)
    rp = os.path.join(OUT, "p0_results.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print()
    print(f"total runtime {report['runtime_seconds']}s")
    print(f"results -> {rp}")


if __name__ == "__main__":
    main()
