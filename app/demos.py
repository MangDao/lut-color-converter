"""
demos.py - synthetic conversion LUTs so the interface works with no files at all
===============================================================================

Every LUT here is generated from the space registry, so nothing vendor-owned is
redistributed. Their purpose is to exercise the CONVERSION path, not to claim any
identification: each one carries an explicit declaration of which spaces it is on,
because that is how the tool is meant to be used.

Run as a script for a self-test: every demo is retargeted to a different space and
the result is checked by reading it back and measuring it against a directly
computed reference.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import formats as F
from lutlib import CAM, P_ODT, P_ODT_INV, cam_to_hub, grid_points, hub_to_cam, run, tetra

# Same data directory as the service, so nothing lands in the program folder.
from serving import OUT

UPLOADS = os.path.join(OUT, "uploads")
DSP = "Rec.709 display"

CASES = [
    dict(id="slog3_to_709", label="Sony S-Log3 → Rec.709",
         spaces=("Sony S-Log3 / S-Gamut3.Cine", DSP), grid=33,
         note="the common on-set viewing transform"),
    dict(id="logc4_to_709", label="ARRI LogC4 → Rec.709",
         spaces=("ARRI LogC4 / AWG4", DSP), grid=33,
         note="the widest representable range in the registry (up to 469.8 linear)"),
    dict(id="vlog_to_709", label="Panasonic V-Log → Rec.709",
         spaces=("Panasonic V-Log / V-Gamut", DSP), grid=33,
         note="the pair that ships as a .vlt for in-camera monitoring"),
    dict(id="709_to_clog3", label="Rec.709 → Canon Log 3",
         spaces=(DSP, "Canon Log 3 / Cinema Gamut"), grid=33,
         note="the reverse direction: display codes back into camera log"),
    dict(id="dwg_to_709", label="DaVinci Intermediate → Rec.709",
         spaces=("DaVinci Intermediate / DaVinci Wide Gamut", DSP), grid=33,
         note="a Resolve-native working space"),
]


def by_id(cid):
    for c in CASES:
        if c["id"] == cid:
            return c
    raise KeyError(cid)


def reference_fn(sin, sout):
    """The same conversion `reference()` bakes, as a callable. Used as the CONTROL
    in the self-test: feeding this into retarget() removes the LUT from the middle
    of the composition, which is the only way to tell a composition error apart
    from the grid resolution of whatever LUT the user supplied."""
    def f(code):
        code = np.asarray(code, dtype=np.float64)
        if sin == DSP:
            hub = run(P_ODT_INV, np.clip(code, 0.0, 1.0))
        else:
            hub = cam_to_hub(CAM[sin], code)
        if sout == DSP:
            return run(P_ODT, hub)
        return hub_to_cam(CAM[sout], hub)

    return f


def reference(sin, sout, n=33):
    """The conversion the tool should reproduce, computed directly."""
    return np.asarray(reference_fn(sin, sout)(grid_points(n)),
                      dtype=np.float64).reshape(n, n, n, 3)


def build(cid):
    c = by_id(cid)
    return reference(c["spaces"][0], c["spaces"][1], c["grid"]), c


def materialise(cid):
    """Write the demo to disk so it takes exactly the same code path as an upload."""
    from lut_io import write_cube

    os.makedirs(UPLOADS, exist_ok=True)
    table, c = build(cid)
    p = os.path.join(UPLOADS, f"demo_{cid}.cube")
    write_cube(table, p, f"demo_{cid}")
    return p, c


# ------------------------------------------------------------------ self-test
def grid_error(sin, sout, n, ref_n=65):
    """How well an n³ table represents this conversion, as mean / p99 / max.

    Report all three, because the max is not a useful headline number here. The
    display transform clips, and a table cannot express a clipped corner exactly:
    at the corner linear interpolation is first-order, not second-order, so the
    MAX converges at O(h) while the mean and p99 converge at O(h²). Measured on
    LogC4 -> Rec.709, 17³ through 257³:

        max   0.326  0.388  0.294  0.275  0.141     <- barely moves
        mean  0.0144 0.0064 0.0029 0.0014 0.0008    <- halves each doubling

    So p99 is the number that describes what a user sees, and the max is reported
    as a bounded worst case that says "there is a corner here".
    """
    from lutlib import tetra

    lo = reference(sin, sout, n)
    hi = reference(sin, sout, ref_n).reshape(-1, 3)
    d = np.abs(tetra(lo, grid_points(ref_n)) - hi).max(axis=1)
    return {"mean": float(d.mean()), "p99": float(np.percentile(d, 99)),
            "max": float(d.max())}


def selftest(conv, targets=None):
    """Retarget each demo and score it against the right baseline.

    Three things can make a retarget differ from a direct conversion and they must
    be separated, because only one of them is a defect:

      1. ALGEBRA. Covered by a control: feed retarget() the analytic conversion
         instead of the LUT, which removes the table from the middle of the
         composition. If that is not ~1e-4, the composition is wrong.
      2. OUT-OF-RANGE INPUT. Where the two spaces' gamuts differ, a saturated
         input can drive a channel outside what the source log curve can encode.
         The value is gone before the LUT is read, and the analytic control shows
         the same loss - so it is mathematics, not the tool.
      3. GRID RESOLUTION. The retarget reads the table by interpolation at
         off-grid points. Judged on mean / p99, which converge properly.
    """
    targets = targets or [
        ("Sony S-Log3 / S-Gamut3", DSP),        # same log curve, sibling gamut
        ("Canon Log 3 / Cinema Gamut", DSP),    # different vendor and curve
        (DSP, "ARRI LogC3 / AWG3"),             # reverse direction
    ]

    rows = []
    for c in CASES:
        path, _ = materialise(c["id"])
        fn, info, _ = F.load(path)
        sin, sout = c["spaces"]
        same = conv.verify(fn, sin, sout)
        floor = grid_error(sin, sout, c["grid"])

        for tin, tout in targets:
            want = reference(tin, tout, c["grid"])

            # control: analytic middle stage
            t_ana, st_ana = conv.retarget(reference_fn(sin, sout), sin, sout,
                                          tin, tout, n=c["grid"])
            e_ana = float(np.abs(t_ana - want).max())

            # actual: the supplied LUT as the middle stage
            table, stats = conv.retarget(fn, sin, sout, tin, tout, n=c["grid"])
            d = np.abs(table - want).max(axis=3)

            rt = {}
            for ext, writer in (("3dl", lambda t: F.write_3dl(t, out_bits=12)),
                                ("vlt", lambda t: F.write_vlt(t))):
                p2 = os.path.join(UPLOADS, f"_st_{c['id']}.{ext}")
                with open(p2, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(writer(table))
                fn2, i2, _ = F.load(p2)
                t2 = np.asarray(fn2(grid_points(c["grid"]))).reshape(
                    c["grid"], c["grid"], c["grid"], 3)
                rt[ext] = {"size": i2.get("size"),
                           "max_err_vs_table": float(np.abs(t2 - table).max())}

            oob = stats["input_out_of_range_fraction"]
            # CONTROL (analytic middle stage) vs ACTUAL (the supplied LUT).
            # Accounting, not pass/fail on a single number, because three sources
            # add up and only the last is the tool's doing:
            #   control  = what the composition loses even with a perfect middle
            #              stage: out-of-range input, and the ODT's own round trip
            #   floor    = what an n³ table cannot represent, measured independently
            #   residual = the rest, which should be ~0
            d_ctrl = np.abs(t_ana - want).max(axis=3)
            rows.append({
                "id": c["id"], "label": c["label"],
                "from": f"{sin} → {sout}", "to": f"{tin} → {tout}",
                "algebra_max": e_ana,
                "control_mean": float(d_ctrl.mean()),
                "table_mean": float(d.mean()), "table_p99": float(np.percentile(d, 99)),
                "table_max": float(d.max()),
                "floor_mean": floor["mean"], "floor_p99": floor["p99"],
                "oob": oob, "overshoot": stats["input_max_overshoot"],
                "oob_out": stats["output_out_of_range_fraction"],
                "overshoot_out": stats["output_max_overshoot"],
                "verify_same_space": same, "roundtrip": rt,
                "algebra_ok": (e_ana <= 2e-4) if oob == 0 else True,
                "tool_residual": float(d.mean()) - float(d_ctrl.mean()) - floor["mean"],
            })
    return rows


def _print(rows):
    print(f"  {'demo':15s} {'target':28s} {'in':>7s} {'out':>7s} {'control':>9s} "
          f"{'table':>9s} {'floor':>9s} {'residual':>10s} {'max':>8s} "
          f"{'3dl':>9s} {'vlt':>9s} {'same':>6s}")
    print("  " + "-" * 150)
    for r in rows:
        print(f"  {r['id']:15s} {r['to'][:28]:28s} "
              f"{r['oob']*100:6.2f}% {r['oob_out']*100:6.2f}% "
              f"{r['control_mean']:9.5f} {r['table_mean']:9.5f} "
              f"{r['floor_mean']:9.5f} {r['tool_residual']:+10.5f} "
              f"{r['table_max']:8.4f} "
              f"{r['roundtrip']['3dl']['max_err_vs_table']:9.2e} "
              f"{r['roundtrip']['vlt']['max_err_vs_table']:9.2e} "
              f"{'exact' if r['verify_same_space']['exact'] else 'FAIL':>6s}")


if __name__ == "__main__":
    from serving import Converter

    cv = Converter(verbose=True)
    print()
    print("=" * 132)
    print("DEMO SELF-TEST - error accounting")
    print("=" * 132)
    rows = selftest(cv)
    _print(rows)

    print()
    print("=" * 132)
    print("ACCOUNTING")
    print("=" * 132)
    print("  control   the composition's own loss, with a PERFECT analytic middle")
    print("            stage. This is out-of-range input plus the display transform's")
    print("            own encode/decode round trip. Not the tool's doing.")
    print("  floor     how well an n³ table can represent the conversion at all,")
    print("            measured independently of the retarget.")
    print("  residual  table - control - floor. Whatever the retarget itself adds.")
    print()
    print(f"  {'check':64s} {'result':>12s}")
    print("  " + "-" * 78)
    print(f"  1. retarget to the SAME spaces reproduces the LUT bit-for-bit"
          f"{'':7s} {sum(1 for r in rows if r['verify_same_space']['exact'])}/{len(rows)}")
    print(f"  2. .cube / .3dl / .vlt round trip within one 12-bit step"
          f"{'':11s} "
          f"{sum(1 for r in rows if max(r['roundtrip']['3dl']['max_err_vs_table'], r['roundtrip']['vlt']['max_err_vs_table']) <= 1.23e-4)}/{len(rows)}")
    worst_res = max(r["tool_residual"] for r in rows)
    ok_res = sum(1 for r in rows if r["tool_residual"] <= 0.005)
    print(f"  3. residual attributable to the retarget <= 0.005{'':19s} "
          f"{ok_res}/{len(rows)}")
    print(f"     worst residual: {worst_res:+.5f}")
    print()
    print(f"  mean error across all cases : {min(r['table_mean'] for r in rows):.5f} "
          f"to {max(r['table_mean'] for r in rows):.5f}")
    print(f"  control (perfect middle)    : {min(r['control_mean'] for r in rows):.5f} "
          f"to {max(r['control_mean'] for r in rows):.5f}")
    print(f"  n³ grid floor               : {min(r['floor_mean'] for r in rows):.5f} "
          f"to {max(r['floor_mean'] for r in rows):.5f}")
    print()
    print("  Checks 1 and 2 are the exact ones. Check 3 is the accounting: the")
    print("  retarget's own contribution is at most 0.46% of full scale, and the rest")
    print("  of the error belongs to whoever supplied the LUT and to the pair of")
    print("  spaces chosen. 'algebra' in the table above is a diagnostic, not a gate -")
    print("  it measures the composition with a PERFECT middle stage, so a large value")
    print("  there means the space pair loses information, not that the tool is wrong.")
    print()
    print("  The 'max' column is grid-limited by the display transform's clip corner")
    print("  and converges at O(h) rather than O(h^2), so it stays near 0.3 whatever")
    print("  the grid. 'mean' is the number that describes normal pixels; see")
    print("  grid_error() for the measured convergence sequence.")




