"""
serving.py - the conversion tool as a service
=============================================

Scope, after the identification feature was removed (2026-10-07): read a LUT,
let the user declare what it is, move its input and/or output space, write it
back out. Nothing here infers a colour space from the LUT's contents - see
`deprecated_identification/README.md` for why that was dropped.

The user declares the input space. That is not a limitation being hidden, it is
the honest interface: only the person who made or downloaded the LUT knows what
its input codes are, and asking costs one dropdown while guessing produced 3
confident errors in 19 real vendor LUTs.

What IS measured and reported, because it is knowable from the file:

  * the LUT's grid, type, and - for .3dl / .vlt - the grid order and where the
    output scale came from
  * its actual output range and how much of it is pinned at 0 or 1
  * whether retargeting to the spaces it is already in reproduces the input
    (must be exact, or nothing downstream is trustworthy)
  * how much of the input gets clipped when moving to a narrower input space
"""

import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import formats as F
from camera_spaces import repr_range
from lut_io import read_filename_hints, write_cube
from lutlib import (CAM, P_ODT, P_ODT_INV, cam_to_hub, grid_points, hub_to_cam,
                    run, tetra)

# Nothing is written into the program directory. Uploads and scratch files go to a
# per-user data folder, so uninstalling leaves nothing behind (and the program
# directory stays readable when installed under Program Files).
DATA = os.environ.get("LUT_HUB_DATA") or os.path.join(
    os.environ.get("LOCALAPPDATA") or tempfile.gettempdir(), "LUTConverter")
OUT = DATA
os.makedirs(OUT, exist_ok=True)

DSP = "Rec.709 display"

FORMAT_HELP = {
    ".cube": "Iridas / Resolve cube - read by colour-science",
    ".csp": "Cinespace - read by colour-science",
    ".spi1d": "Sony Pictures 1D - read by colour-science",
    ".spi3d": "Sony Pictures 3D - read by colour-science",
    ".spimtx": "Sony Pictures matrix - read by colour-science",
    ".3dl": "Autodesk Lustre - BLUE varies fastest, verified against OpenColorIO",
    ".vlt": "Panasonic VARICAM 3DLUT - RED varies fastest, non-uniform input axis",
}

WRITE_FORMATS = {".cube": "33³, the safe default",
                 ".3dl": "Lustre; 33³, B-fastest, declares its own bit depth",
                 ".vlt": "Panasonic VARICAM; forced to 17³, the only size the spec defines"}


# --------------------------------------------------------------- colour plumbing
def codes_to_hub(space, x):
    """codes in `space` -> linear ACES2065-1 (the hub)."""
    x = np.asarray(x, dtype=np.float64)
    if space == DSP:
        return run(P_ODT_INV, np.clip(x, 0.0, 1.0))
    return cam_to_hub(CAM[space], x)


def hub_to_codes(space, lin):
    """linear ACES2065-1 -> codes in `space`."""
    if space == DSP:
        return run(P_ODT, np.asarray(lin, dtype=np.float64))
    return hub_to_cam(CAM[space], np.asarray(lin, dtype=np.float64))


def convert_codes(src, dst, x):
    if src == dst:
        return np.array(x, dtype=np.float64, copy=True)
    return hub_to_codes(dst, codes_to_hub(src, x))


def lin_in_space(space, hub):
    """linear ACES2065-1 -> linear light in `space`'s own primaries. Unclamped, so
    out-of-gamut values stay visible as negative or over-range numbers."""
    import colour as _c

    hub = np.asarray(hub, dtype=np.float64)
    if space == DSP:
        return hub
    return _c.RGB_to_RGB(hub, "ACES2065-1", CAM[space],
                         apply_cctf_decoding=False, apply_cctf_encoding=False)


def out_of_range(space, lin):
    """Which samples fall outside what `space`'s log curve can encode.

    This has to be measured in the LINEAR domain, before encoding. The obvious
    check - look at the codes and test for 0 or 1 - reads zero every time, because
    `encode_clamped` has already pinned them. Measuring there is how the first
    version of this file reported 0.0000% clipping on a case that was clipping
    heavily: a conversion from S-Gamut3 into S-Gamut3.Cine drives the red channel
    negative for saturated colours, which lands on the curve's floor.

    Returns (fraction of samples with any channel out of range, max overshoot in
    linear units, and which channels).
    """
    lo, hi = repr_range(CAM[space])
    d = np.asarray(lin, dtype=np.float64)
    below = np.clip(lo - d, 0.0, None)
    above = np.clip(d - hi, 0.0, None)
    over = below + above
    any_over = np.any(over > 1e-9, axis=1)
    worst = float(np.max(over)) if over.size else 0.0
    return (float(np.mean(any_over)), worst,
            {"below_lo": float(np.mean(np.any(below > 1e-9, axis=1))),
             "above_hi": float(np.mean(np.any(above > 1e-9, axis=1))),
             "lo": float(lo), "hi": float(hi)})


def input_range_check(sin, hub):
    """Can the incoming hub values be expressed as codes in `sin`?

    The mirror of `output_range_check`, and the same asymmetry applies: a camera
    space is checked in linear light against its log curve's range, while the
    display space is not checked that way at all. A display code is an ODT output,
    which the ODT has already clamped into [0,1], so feeding it a hub value can
    never overshoot - what it can do is SATURATE, and that loss shows up in the
    analytic control rather than here.

    What the caller samples matters a lot, and the difference is easy to
    misread. On a uniform grid over the whole code cube - which is what retarget()
    uses, because the LUT must be filled in everywhere - the corners are things
    like fully-saturated pure red at full exposure. Those are not scene colours.
    Measured over MacBeth, a neutral ramp and hue sweeps, the same conversions clip
    nothing at all (0 of 4709 colours, 17 of 17 camera pairings). So the fraction
    returned here is an upper bound describing the grid, not a rate a user should
    expect on their footage, and the GUI says so.
    """
    if sin == DSP:
        return 0.0, 0.0, {"test": "none: the display's codes are ODT outputs in [0,1]"}
    return out_of_range(sin, lin_in_space(sin, hub))


def output_range_check(sout, sout_new, y):
    """Can the LUT's own output be re-expressed in `sout_new`?

    The display space needs a different test from a camera space, and getting this
    wrong is easy. A camera space is checked in LINEAR light against the range its
    log curve can encode. The display space cannot be checked that way: decoding a
    display code yields ACES linear, whose scale has nothing to do with the display
    re-encoding is what matters, so the check is on the CODES.

    Comparing display-space linear against [0,1] - the first thing this tried -
    reports ~76% out of range on cases that are perfectly fine.
    """
    y = np.asarray(y, dtype=np.float64)
    if sout == sout_new:
        return 0.0, 0.0, {"test": "none: nothing is re-encoded"}
    if sout_new == DSP:
        over = np.clip(-y, 0.0, None) + np.clip(y - 1.0, 0.0, None)
        over = over.max(axis=1) if over.ndim > 1 else over
        return (float(np.mean(over > 1e-9)), float(np.max(over)) if over.size else 0.0,
                {"test": "display codes against [0,1]"})
    lin = lin_in_space(sout_new, codes_to_hub(sout, y))
    f, w, d = out_of_range(sout_new, lin)
    return f, w, {**d, "test": "linear against the target log curve's range"}


def sample_table(fn, n=33):
    return np.asarray(fn(grid_points(n)), dtype=np.float64).reshape(n, n, n, 3)


# ------------------------------------------------------------------ the service
class Converter:
    def __init__(self, verbose=False):
        t0 = time.time()
        self.spaces = sorted(CAM)
        self.ready_ms = round(1000 * (time.time() - t0))
        if verbose:
            print(f"  converter ready in {self.ready_ms} ms ({len(self.spaces)} spaces)")

    # ------------------------------------------------------------------ info
    def app_info(self):
        return {
            "spaces": self.spaces,
            "display_space": DSP,
            "read_formats": sorted(F.OWN) + [".cube", ".csp", ".spi1d", ".spi3d", ".spimtx"],
            "write_formats": WRITE_FORMATS,
            "format_help": FORMAT_HELP,
            "ready_ms": self.ready_ms,
            "note": ("input colour space is declared by the user; this tool does not "
                     "infer it from the LUT contents"),
        }

    # --------------------------------------------------------------- inspect
    def inspect(self, path, name=None):
        """What the file itself says. No inference about colour spaces."""
        name = name or os.path.basename(path)
        t0 = time.time()
        fn, info, text = F.load(path)
        table = sample_table(fn, 33)

        pinned_lo = float(np.mean(table <= 1e-6))
        pinned_hi = float(np.mean(table >= 1.0 - 1e-6))
        hints = read_filename_hints(name, info.get("comments"))

        notes = []
        if info.get("type") == "LUT3x1D":
            notes.append("1D LUT: a transfer function has no channel crosstalk, so it "
                         "cannot carry a colour space. Converting it between spaces is "
                         "still meaningful only if you pick the right pair.")
        for k in ("warning", "input_axis_kind", "grid_order", "scale_source"):
            if info.get(k):
                notes.append(f"{k}: {info[k]}")

        return {
            "file": {
                "name": name,
                "format": info.get("format") or os.path.splitext(name)[1].lower(),
                "lut_type": info.get("type"),
                "size": info.get("size"),
                "bits": info.get("bits"),
                "grid_order": info.get("grid_order"),
                "scale_source": info.get("scale_source"),
                "input_axis_kind": info.get("input_axis_kind"),
                "domain_lo": info.get("domain_lo"),
                "domain_hi": info.get("domain_hi"),
                "comments": (info.get("comments") or [])[:6],
            },
            "measured": {
                "output_range": [round(float(table.min()), 5), round(float(table.max()), 5)],
                "pinned_at_0": round(pinned_lo, 5),
                "pinned_at_1": round(pinned_hi, 5),
                "identity_distance": round(float(np.mean(np.abs(
                    table.reshape(-1, 3) - grid_points(33)))), 5),
            },
            "filename_hints": hints,
            "notes": notes,
            "elapsed_ms": round(1000 * (time.time() - t0)),
        }

    # -------------------------------------------------------------- conversion
    def retarget(self, fn, sin, sout, sin_new, sout_new, n=33):
        """Move a LUT's input space and/or its output space.

        f maps codes in `sin` to codes in `sout`. We want f' on `sin_new` -> `sout_new`:

            f'(x) = Encode_sout_new( Decode_sout( f( Encode_sin( Decode_sin_new(x) ) ) ) )

        which reads as "convert sin_new -> sin, apply f, convert sout -> sout_new".

        The honest limit is in the middle step. `f` is a LUT in CODE space, so the
        intermediate value has to be encoded back to codes, and a code value cannot
        express something the source space's log curve cannot hold. Where the two
        gamuts differ, a saturated input can drive a channel negative in the source
        space; that channel lands on the curve floor and the value is gone before the
        LUT is ever consulted. The result is then NOT the same as converting
        directly, and no grid size fixes it - measured at both 33³ and 65³.

        So the amount is measured (in the linear domain, see `out_of_range`) and
        returned. Above a few percent, the retargeted LUT is a different transform,
        not a more convenient spelling of the same one.
        """
        g = grid_points(n)
        hub = codes_to_hub(sin_new, g)                 # unclamped
        frac, worst, detail = input_range_check(sin, hub)

        if sin_new == sin:
            mid_in = g                                 # identity: no encode round trip
            frac, worst, detail = 0.0, 0.0, {**detail, "test": "none: identity input"}
        else:
            mid_in = hub_to_codes(sin, hub)            # this is where clamping happens
        y = fn(mid_in)

        # The OUTPUT side needs its own check. The LUT's output is in `sout` codes
        # and they are about to be re-expressed in `sout_new`, which has a different
        # representable range. Missing this is how a case can report 0% input
        # clipping and still lose information: a Canon Log 3 code that decodes below
        # ARRI LogC3's floor has no ARRI code at all.
        ofrac, oworst, odetail = output_range_check(sout, sout_new, y)

        z = convert_codes(sout, sout_new, y)
        table = np.asarray(z, dtype=np.float64).reshape(n, n, n, 3)

        return table, {
            "sin": sin, "sout": sout, "sin_new": sin_new, "sout_new": sout_new,
            "grid": n,
            "input_out_of_range_fraction": round(frac, 6),
            "input_max_overshoot": worst,
            "input_range_detail": {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in detail.items()},
            "output_out_of_range_fraction": round(ofrac, 6),
            "output_max_overshoot": oworst,
            "output_range_detail": {k: (round(v, 6) if isinstance(v, float) else v)
                                    for k, v in odetail.items()},
            "exact_by_construction": frac == 0.0 and ofrac == 0.0,
            "output_range": [round(float(table.min()), 5), round(float(table.max()), 5)],
            "output_pinned_fraction": round(float(np.mean(
                (table <= 1e-6) | (table >= 1.0 - 1e-6))), 6),
            "unchanged": sin == sin_new and sout == sout_new,
        }

    def verify(self, fn, sin, sout, n=17):
        """Retargeting to the SAME spaces must reproduce the LUT. If it does not,
        the colour conversion is not round-tripping and nothing else is
        trustworthy.

        The tolerance is machine epsilon, not zero. A real vendor LUT goes through
        colour-science's readers and comes back with ~5e-17 of float noise, which a
        strict == 0 test reports as a failure. That is a property of floating point,
        not of the conversion.
        """
        g = grid_points(n)
        t, _ = self.retarget(fn, sin, sout, sin, sout, n=n)
        d = np.abs(tetra(t, g) - np.asarray(fn(g), dtype=np.float64))
        return {"max_abs_error": float(d.max()), "mean_abs_error": float(d.mean()),
                "exact": bool(d.max() <= 1e-12),
                "tolerance": 1e-12}

    def build_output(self, table, fmt, name):
        """(bytes, filename, mime) for the requested format."""
        stem = os.path.splitext(os.path.basename(name))[0] or "output"
        if fmt == "3dl":
            return (F.write_3dl(table, note="retargeted by lut-hub").encode("utf-8"),
                    f"{stem}_retargeted.3dl", "text/plain; charset=utf-8")
        if fmt == "vlt":
            return (F.write_vlt(table, axis=F.VLT_LAYOUTS["spec"]).encode("utf-8"),
                    f"{stem}_retargeted.vlt", "text/plain; charset=utf-8")
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".cube")
        os.close(fd)
        try:
            write_cube(table, tmp, f"{stem}_retargeted")
            with open(tmp, "rb") as fh:
                data = fh.read()
        finally:
            os.unlink(tmp)
        return data, f"{stem}_retargeted.cube", "application/octet-stream"

    def grid_for(self, fmt):
        return 17 if fmt == "vlt" else 33


def save_upload(name, data):
    """Uploads land in out/uploads. Kept on disk so the browser only sends the
    file once; the token cache in server.py maps back to the path."""
    d = os.path.join(OUT, "uploads")
    os.makedirs(d, exist_ok=True)
    safe = "".join(ch for ch in os.path.basename(name) if ch.isalnum() or ch in "._- ")
    safe = safe or "upload.cube"
    p = os.path.join(d, safe)
    with open(p, "wb") as fh:
        fh.write(data)
    return p
