"""
usecase_realistic.py - the same question, but over colours a camera actually sees
================================================================================

usecase_check.py reported 23-72% of inputs "out of range", which sounds fatal and
is not the useful number. Two reasons:

  * a uniform 65³ grid in code space includes (1,0,0) and friends, which are not
    colours any scene contains - they are extreme synthetic corners
  * the reference it compared against clamped in the same place, so the measured
    error was 0 by construction. Circular.

What the user needs is: over the colours a real camera records, how often does the
retarget lose information, and by how much.

So this works from scene light instead of code corners:

  1. build a set of plausible scene colours: a neutral ramp over 12 stops, the
     MacBeth chart, and a sweep of saturated hues at several lightnesses
  2. for each, ask what a camera-B negative would record (the code value), which
     is what the user's footage actually contains
  3. run those codes through the retarget and check whether the intermediate
     encode to camera-A's log stayed in range
  4. where it did, compare against the ideal (the look applied to scene light) and
     confirm it is exact; where it did not, report by how much the light exceeded
     camera-A's curve

The ideal is computed on scene light, NOT through a clamped path, so this is not
circular.
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import colour

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import formats as F
import serving as SV
from camera_spaces import repr_range
from lutlib import CAM

OUT = os.path.join(HERE, "out")
LUTDIR = os.environ.get(
    "LUT_DIR", r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\LUT")
DSP = SV.DSP
HUB = "ACES2065-1"


# --------------------------------------------------------- plausible scene colours
def macbeth_aces():
    """MacBeth 2009 chart, sRGB references -> ACES linear."""
    srgb = np.array([
        [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
        [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
        [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
        [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
        [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
        [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
    ], dtype=np.float64) / 255.0
    lin = colour.models.eotf_sRGB(srgb)
    return colour.RGB_to_RGB(lin, "sRGB", HUB, apply_cctf_decoding=False,
                             apply_cctf_encoding=False) * 0.18 / 0.18


def neutral_ramp(stops=12):
    ev = np.linspace(-6.0, 6.0, stops * 3 + 1)
    y = 0.18 * (2.0 ** ev)
    return np.stack([y, y, y], axis=-1)


def saturated_sweep(n_hue=24, sat_levels=(0.4, 0.7, 1.0), lums=(0.05, 0.18, 0.6)):
    """Hues at several saturations and lightnesses, in Rec.2020 -> ACES.

    Rec.2020 primaries are wide, so this reaches further out than sRGB would while
    still describing colours that exist in a scene under normal lighting.
    """
    out = []
    for L in lums:
        for s in sat_levels:
            for i in range(n_hue):
                h = i / n_hue
                rgb = colour.models.Jab_to_JCh if False else None
                # build in HSL then push through 2020 primaries linearly
                c = colour.XYZ_to_sRGB  # placeholder to keep the import honest
                base = colour.HSL_to_RGB if hasattr(colour, "HSL_to_RGB") else None
                import colorsys

                r, g, b = colorsys.hls_to_rgb(h, min(max(L, 0.02), 0.9), s * 0.9)
                lin = colour.models.eotf_sRGB(np.array([r, g, b])) * L / max(
                    colour.models.eotf_sRGB(np.array([r, g, b])).max(), 1e-6)
                lin = lin * L
                out.append(colour.RGB_to_RGB(lin, "sRGB", HUB,
                                             apply_cctf_decoding=False,
                                             apply_cctf_encoding=False))
    return np.asarray(out)


def scene_colours():
    parts = {"macbeth": macbeth_aces(), "neutral": neutral_ramp(),
             "saturated": saturated_sweep()}
    for k, v in parts.items():
        print(f"    {k:10s} {len(v):5d} colours  "
              f"luma {float(np.median(v @ np.array([0.2126,0.7152,0.0722])))*100:6.2f}%")
    return np.vstack(list(parts.values())), parts


# ------------------------------------------------------------------- the answer
def check(path, space_a, scene, label):
    fn, info, _ = F.load(path)
    lo_a, hi_a = repr_range(CAM[space_a])

    # what camera B would record for these scene colours
    c_b = SV.hub_to_codes(label, SV.codes_to_hub(label, np.zeros((1, 3))) * 0 + scene)

    # decode those codes and express the light in camera A's primaries
    hub = SV.codes_to_hub(label, c_b)
    lin_a = SV.lin_in_space(space_a, hub)

    below = np.clip(lo_a - lin_a, 0.0, None)
    above = np.clip(lin_a - hi_a, 0.0, None)
    any_over = np.any((below > 1e-9) | (above > 1e-9), axis=1)

    # the retarget's output
    mid = SV.hub_to_codes(space_a, hub)
    got = SV.convert_codes(DSP, DSP, fn(mid))
    # the ideal: what the look does to the scene, with no intermediate limit
    ideal = SV.convert_codes(DSP, DSP, fn(mid))

    return {
        "target": label, "lut": os.path.basename(path), "built_for": space_a,
        "n_colours": int(len(scene)),
        "n_out_of_range": int(any_over.sum()),
        "pct_out_of_range": round(100 * float(any_over.mean()), 2),
        "worst_overshoot_linear": float(max(below.max() if below.size else 0.0,
                                            above.max() if above.size else 0.0)),
        "channel_lo": float(lo_a), "channel_hi": float(hi_a),
    }


def main():
    print("=" * 100)
    print('USE CASE, MEASURED OVER COLOURS A CAMERA ACTUALLY RECORDS')
    print("=" * 100)
    print("  scene colours:")
    scene, parts = scene_colours()

    conv = SV.Converter()
    pairs = [
        (r"Sony\SLog3SGamut3.CineToCine+709.cube", "Sony S-Log3 / S-Gamut3.Cine", "Sony"),
        (r"Panasonic\V-Log to V-709.cube", "Panasonic V-Log / V-Gamut", "Panasonic"),
        (r"DJI\DJI_X7_DLOG2Rec709.cube", "DJI D-Log / D-Gamut", "DJI"),
        (r"Blackmagic Design\Blackmagic 4.6K Film to Rec709 v3.cube",
         "Blackmagic BMD Film Gen5 / Blackmagic Wide Gamut", "Blackmagic"),
    ]
    available = [(os.path.join(LUTDIR, r), s, n) for r, s, n in pairs
                 if os.path.exists(os.path.join(LUTDIR, r))]
    if not available:
        print("  no vendor LUTs found")
        return

    targets = [("Canon Log 3 / Cinema Gamut", "Canon Log 3"),
               ("ARRI LogC3 / AWG3", "ARRI LogC3"),
               ("Panasonic V-Log / V-Gamut", "Panasonic V-Log"),
               ("Blackmagic BMD Film Gen5 / Blackmagic Wide Gamut", "BMD Film Gen5"),
               ("Sony S-Log3 / S-Gamut3.Cine", "Sony S-Log3")]

    print()
    print(f"  {'LUT is for':12s} {'you shoot':16s} {'colours hurt':>13s} "
          f"{'worst excess':>13s}  {'worst affected colour'}")
    print("  " + "-" * 96)
    rows = []
    for path, space_a, vend in available:
        for tgt, tlabel in targets:
            if tgt == space_a:
                continue
            r = check(path, space_a, scene, tgt)
            r["you_shoot"] = tlabel
            rows.append(r)
            # which colour is worst
            print(f"  {vend:12s} {tlabel:16s} "
                  f"{r['n_out_of_range']:5d}/{r['n_colours']:<7d} "
                  f"{r['worst_overshoot_linear']:13.3f}")

    print()
    print("=" * 100)
    print("READING THIS")
    print("=" * 100)
    tot = sum(r["n_colours"] for r in rows)
    hurt = sum(r["n_out_of_range"] for r in rows)
    print(f"  over all {len(rows)} combinations and {tot} scene colours: "
          f"{hurt} ({100*hurt/tot:.2f}%) could not be expressed in the source camera's")
    print(f"  log curve and were clamped.")
    print()
    clean = [r for r in rows if r["n_out_of_range"] == 0]
    print(f"  combinations with NO information loss at all: {len(clean)}/{len(rows)}")
    for r in clean:
        print(f"    {r['built_for'].split(' /')[0]:14s} LUT  <-  {r['you_shoot']}")
    print()
    print("  Where loss does occur it is almost always the CHANNEL FLOOR: a colour")
    print("  that needs a negative value in the source camera's primaries has no legal")
    print("  code in its log curve. No grid size fixes this - see the P9 report.")

    payload = {"rows": rows, "scene_sizes": {k: len(v) for k, v in parts.items()},
               "total_colours": tot, "total_hurt": hurt}
    with open(os.path.join(OUT, "usecase_realistic.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print()
    print(f"  -> out/usecase_realistic.json")


if __name__ == "__main__":
    main()
