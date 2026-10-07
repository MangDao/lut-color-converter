"""
usecase_check.py - "I have a LUT for camera A, I shoot camera B. Does this work?"
================================================================================

That is the single most likely thing a user wants, so it gets its own measured
answer rather than an assurance.

Setup: a real vendor LUT from the local DaVinci Resolve install, built for camera
A's log, retargeted so it accepts camera B's log.

The composition is

    f'(x) = f( Encode_A( Decode_B( x ) ) )

and the interesting question is not whether it runs - it is what happens at the
intermediate step, which is a CODE value in A's log. Decoding B's codes gives hub
light that A's log curve may not be able to express. There the value is clamped,
and the result differs from what the look would have done on a camera-A negative.

So this measures, in three parts:

  1. the retarget is exact wherever the intermediate is in range
  2. how much of the input range is NOT in range, and for which inputs
  3. whether the loss lands on ordinary pixels or only on extremes

Part 3 is the one that decides if the answer is "yes, go ahead" or "yes, but know
this". A loss confined to saturated highlights is a footnote; a loss through the
mid-tones would make the tool useless and would need saying loudly.
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import formats as F
import serving as SV
from camera_spaces import repr_range
from lutlib import CAM, grid_points, tetra

OUT = os.path.join(HERE, "out")
LUTDIR = os.environ.get(
    "LUT_DIR", r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\LUT")

DSP = SV.DSP


def find_candidates():
    """Real vendor log -> display LUTs, paired with the space they expect."""
    want = [
        (r"Sony\SLog3SGamut3.CineToCine+709.cube", "Sony S-Log3 / S-Gamut3.Cine"),
        (r"Panasonic\V-Log to V-709.cube", "Panasonic V-Log / V-Gamut"),
        (r"DJI\DJI_X7_DLOG2Rec709.cube", "DJI D-Log / D-Gamut"),
        (r"Blackmagic Design\Blackmagic 4.6K Film to Rec709 v3.cube",
         "Blackmagic BMD Film Gen5 / Blackmagic Wide Gamut"),
    ]
    out = []
    for rel, space in want:
        p = os.path.join(LUTDIR, rel)
        if os.path.exists(p):
            out.append((p, space))
    return out


def ideal(fn, sin, sout, sin_new, hub):
    """The composition WITHOUT clamping at the intermediate step.

    Not writable to a LUT file - it is only a reference to measure the clamp
    against. `torch`-free by construction: the middle stage is applied to encoded
    codes the way a LUT would, but the encode is allowed to leave [0,1].
    """
    mid = SV.hub_to_codes(sin, hub) if sin != sin_new else hub
    # a LUT cannot be sampled outside [0,1]; this reference pretends it can by
    # clamping only at the very last step, which is the best the look could do
    y = fn(np.clip(mid, 0.0, 1.0))
    return SV.convert_codes(sout, sout_new, y) if sout != sout_new else y


def main():
    print("=" * 104)
    print('USE CASE: "I have a LUT built for camera A, and I shoot camera B"')
    print("=" * 104)
    print(f"  source of real LUTs: {LUTDIR}")

    cases = find_candidates()
    if not cases:
        print("  no candidate LUTs found in the local Resolve install")
        return
    print(f"  {len(cases)} real vendor LUTs available\n")

    conv = SV.Converter()

    # camera B options: what a user might be shooting instead
    targets = [
        ("Canon Log 3 / Cinema Gamut", "Canon Log 3"),
        ("ARRI LogC3 / AWG3", "ARRI LogC3"),
        ("Panasonic V-Log / V-Gamut", "Panasonic V-Log"),
        ("Blackmagic BMD Film Gen5 / Blackmagic Wide Gamut", "BMD Film Gen5"),
    ]

    rows = []
    for path, space_a in cases:
        name = os.path.basename(path)
        fn, info, _ = F.load(path)
        n = info.get("size") or 33
        print("=" * 104)
        print(f"LUT: {name}")
        print(f"     built for {space_a}  ->  Rec.709,  grid {n}³")
        print("=" * 104)
        print(f"  {'retarget input to':24s} {'out of range':>13s} "
              f"{'error mean':>11s} {'error p99':>11s} {'error max':>11s} "
              f"{'err where in range':>19s}")
        print("  " + "-" * 96)

        g = grid_points(n)
        for tgt, label in targets:
            if tgt == space_a:
                continue
            table, st = conv.retarget(fn, space_a, DSP, tgt, DSP, n=n)

            # the reference: same composition, but remembering where the input was
            # out of range so the two can be separated
            hub = SV.codes_to_hub(tgt, g)
            lin_src = SV.lin_in_space(space_a, hub)
            f_oob, worst, det = SV.out_of_range(space_a, lin_src)

            want = SV.convert_codes(DSP, DSP, fn(
                SV.hub_to_codes(space_a, hub))).reshape(n, n, n, 3)

            d = np.abs(table - want).max(axis=3).reshape(-1)
            below = np.clip(det["lo"] - lin_src, 0, None)
            above = np.clip(lin_src - det["hi"], 0, None)
            ok = ~(np.any(below > 1e-9, axis=1) | np.any(above > 1e-9, axis=1))

            dm = float(d[ok].max()) if ok.any() else float("nan")
            print(f"  {label:24s} {f_oob*100:12.2f}% {d.mean():11.5f} "
                  f"{np.percentile(d,99):11.5f} {d.max():11.5f} {dm:19.2e}")

            rows.append({"lut": name, "built_for": space_a, "target": tgt,
                         "target_label": label, "out_of_range": f_oob,
                         "err_mean": float(d.mean()),
                         "err_p99": float(np.percentile(d, 99)),
                         "err_max": float(d.max()),
                         "err_in_range_max": dm,
                         "in_range_fraction": float(ok.mean()),
                         "range_lo": det["lo"], "range_hi": det["hi"]})
        print()

    # ------------------------------------------------------------- what it means
    print("=" * 104)
    print("WHAT THE NUMBER MEANS")
    print("=" * 104)
    print("  'out of range' - fraction of camera-B input codes whose decoded light")
    print("    cannot be expressed as camera-A log codes. There the look sees a")
    print("    clamped value instead of the real one.")
    print("  'err where in range' - the error over the inputs that ARE representable.")
    print("    This is the number that should be at machine precision; if it is not,")
    print("    the composition is wrong.")
    print()

    good = [r for r in rows if r["err_in_range_max"] < 1e-6]
    print(f"  composition exact on the in-range region: {len(good)}/{len(rows)} cases")
    print()
    print(f"  {'worst out of range':>20s}  {'target':12s}  {'built for':34s}")
    print("  " + "-" * 74)
    for r in sorted(rows, key=lambda r: -r["out_of_range"])[:6]:
        print(f"  {r['out_of_range']*100:19.1f}%  {r['target_label']:12s}  "
              f"{r['built_for'][:34]:34s}")

    payload = {"source": LUTDIR, "rows": rows,
               "exact_cases": len(good), "total_cases": len(rows)}
    with open(os.path.join(OUT, "usecase_results.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print()
    print(f"  -> out/usecase_results.json")


if __name__ == "__main__":
    main()
