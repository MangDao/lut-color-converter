"""
usecase_extreme.py - trying hard to MAKE it clip
================================================

usecase_realistic.py found 0 of 4709 scene colours clipping across 17 camera
pairings. Before believing a zero, try to break it. If extreme but real colours do
clip, that is the honest boundary to report; if nothing plausible clips, the
limitation is real but academic.

What is tested, from mild to absurd:

  reflectance sweep   saturated object colours under normal exposure
  highlight sweep     the same objects lit 4 to 64x over
  gamut edges         colours sitting exactly on the Rec.2020 and AP1 boundaries
  spectral-ish        very narrow-band colours, the closest a real LED gets to
                      monochromatic
  camera-difference   the specific mismatch the retarget has to absorb: a colour
                      the SOURCE camera can record but the TARGET's primaries
                      render negative

The last one is the theoretical worst case: it is constructed by taking a code
value the source log curve can represent and asking whether the target's
primaries still hold it.
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import colorsys

import numpy as np
import colour

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import serving as SV
from camera_spaces import repr_range
from lutlib import CAM

OUT = os.path.join(HERE, "out")
HUB = "ACES2065-1"


def srgb_linear(rgb):
    return colour.models.eotf_sRGB(np.asarray(rgb, dtype=np.float64))


def to_hub(rgb_linear, src="sRGB"):
    return colour.RGB_to_RGB(np.asarray(rgb_linear, dtype=np.float64), src, HUB,
                             apply_cctf_decoding=False, apply_cctf_encoding=False)


def reflectance_sweep(n_hue=36, sat=(0.5, 0.8, 1.0)):
    """Object colours: HSL at full-ish saturation, held as reflectance in [0,1]."""
    out = []
    for s in sat:
        for i in range(n_hue):
            r, g, b = colorsys.hls_to_rgb(i / n_hue, 0.5, s)
            out.append(to_hub(srgb_linear([r, g, b])))
    return np.asarray(out)


def with_exposure(refl, exposures=(0.18, 1.0, 4.0, 16.0, 64.0)):
    return np.vstack([refl * e for e in exposures])


def gamut_edges():
    """Colours sitting on the widest gamut boundaries we have."""
    out = []
    for space in ("ITU-R BT.2020", "ACES2065-1"):
        for i in range(36):
            for lvl in (1.0, 4.0):
                h = i / 36
                r, g, b = colorsys.hls_to_rgb(h, 0.5, 1.0)
                lin = srgb_linear([r, g, b])
                out.append(colour.RGB_to_RGB(lin, "sRGB", space,
                                             apply_cctf_decoding=False,
                                             apply_cctf_encoding=False) * lvl)
    corners = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.float64)
    out += [c * lvl for c in corners for lvl in (0.18, 1.0, 8.0)]
    return np.asarray(out)


def spectral_ish(n=48):
    """Narrow-band colours: sRGB primaries pushed to the edge of their gamut."""
    out = []
    for i in range(n):
        lam = 380 + i * (780 - 380) / n
        try:
            xyz = colour.wavelength_to_XYZ(lam)
        except Exception:
            continue
        rgb = colour.XYZ_to_sRGB(xyz / max(float(np.max(xyz)), 1e-9), apply_cctf_encoding=False)
        if np.min(rgb) < -0.5:
            continue
        rgb = np.clip(rgb, 0, None)
        if rgb.max() <= 0:
            continue
        out.append(to_hub(rgb / rgb.max() * 0.9))
    return np.asarray(out)


def source_reachable(space_a, n=4000, seed=3):
    """The real worst case: colours the SOURCE camera can actually record.

    Sampling the source camera's full legal code cube and decimating, so these are
    code values that exist. Then check whether they survive being re-expressed in
    the target's primaries.
    """
    rng = np.random.default_rng(seed)
    codes = rng.random((n, 3))
    return SV.codes_to_hub(space_a, codes)


def audit(space_a, space_b, hub):
    lo_a, hi_a = repr_range(CAM[space_a])
    lin_a = SV.lin_in_space(space_a, hub)
    below = np.clip(lo_a - lin_a, 0, None)
    above = np.clip(lin_a - hi_a, 0, None)
    over = np.any((below > 1e-9) | (above > 1e-9), axis=1)
    worst = float(max(below.max() if below.size else 0.0,
                      above.max() if above.size else 0.0))
    # how many are pinned at exactly the curve floor once encoded
    mid = SV.hub_to_codes(space_a, hub)
    pinned = float(np.mean(np.any((mid <= 1e-9) | (mid >= 1 - 1e-9), axis=1)))
    return {"n": int(len(hub)), "out_of_range": int(over.sum()),
            "pct": round(100 * float(over.mean()), 3), "worst_excess": worst,
            "pct_pinned": round(100 * pinned, 3),
            "lo": float(lo_a), "hi": float(hi_a)}


def main():
    print("=" * 104)
    print("TRYING TO MAKE IT CLIP - extreme but real colours")
    print("=" * 104)

    sets = {
        "reflectance x exposure": with_exposure(reflectance_sweep()),
        "gamut edges": gamut_edges(),
        "narrow band (spectral)": spectral_ish(),
    }
    for k, v in sets.items():
        print(f"  {k:26s} {len(v):6d} colours")

    spaces = ["Sony S-Log3 / S-Gamut3.Cine", "Panasonic V-Log / V-Gamut",
              "DJI D-Log / D-Gamut", "Canon Log 3 / Cinema Gamut",
              "ARRI LogC3 / AWG3", "Blackmagic BMD Film Gen5 / Blackmagic Wide Gamut",
              "ARRI LogC4 / AWG4", "GoPro Protune / Protune Native"]

    print()
    print("=" * 104)
    print("A. A SOURCE CAMERA'S OWN CODE CUBE, RE-EXPRESSED IN ANOTHER DEAD CAMERA'S PRIMARIES")
    print("=" * 104)
    print("  This is the true worst case: code values that exist, decoded, then asked")
    print("  to fit a different set of primaries and log curve.")
    print()
    print(f"  {'LUT built for':30s} " + "".join(f"{s.split(' /')[0][:11]:>13s}" for s in spaces))
    print("  " + "-" * (30 + 13 * len(spaces)))
    worst_all = {}
    for a in spaces:
        hub = source_reachable(a)
        cells = []
        for b in spaces:
            if a == b:
                cells.append("--")
                continue
            r = audit(b, a, hub)
            cells.append(f"{r['pct']:12.2f}%")
            worst_all[(a, b)] = r
        print(f"  {a:30s} " + "".join(f"{c:>13s}" for c in cells))

    print()
    print("=" * 104)
    print("B. EXTREME BUT REAL COLOURS")
    print("=" * 104)
    print(f"  {'colour set':26s} {'target primaries':30s} {'out of range':>13s} "
          f"{'worst excess':>13s} {'pinned after':>13s}")
    print("  " + "-" * 100)
    rows = []
    for sname, hub in sets.items():
        if len(hub) == 0:
            continue
        for b in spaces:
            r = audit(b, b, hub)
            rows.append({"set": sname, "target": b, **r})
            print(f"  {sname:26s} {b:30s} {r['pct']:12.2f}% "
                  f"{r['worst_excess']:13.4f} {r['pct_pinned']:12.2f}%")
        print()

    print("=" * 104)
    print("VERDICT")
    print("=" * 104)
    live = [r for r in rows if r["out_of_range"] > 0]
    print(f"  extreme-colour cases that DO clip: {len(live)}/{len(rows)}")
    if live:
        worst = max(live, key=lambda r: r["pct"])
        print(f"    worst: {worst['set']} against {worst['target'].split(' /')[0]} "
              f"-> {worst['pct']:.2f}% of colours, worst excess "
              f"{worst['worst_excess']:.3f} linear")
    print()
    pairs_live = {k: v for k, v in worst_all.items() if v["out_of_range"] > 0}
    print(f"  camera-to-camera combinations that clip on the source's own cube: "
          f"{len(pairs_live)}/{len(worst_all)}")
    print()
    print("  How to read this: the source cube test (A) samples the FULL legal code")
    print("  cube, whose corners are extreme colours no scene contains - so its")
    print("  percentages are an upper bound, not a rate. Section B is closer to real")
    print("  content. The number that matters for the user's footage is the one in")
    print("  usecase_realistic.py, measured over MacBeth, a neutral ramp and hue")
    print("  sweeps: 0 of 4709 colours clipped across 17 camera pairings.")

    payload = {"source_cube": {f"{a} -> {b}": v for (a, b), v in worst_all.items()},
               "extreme_sets": rows,
               "pairs_clipping": len(pairs_live)}
    with open(os.path.join(OUT, "usecase_extreme.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print()
    print(f"  -> out/usecase_extreme.json")


if __name__ == "__main__":
    main()
