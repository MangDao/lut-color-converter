"""
Filling the official gap: a hand-built CLF for Fujifilm F-Log2 / F-Gamut C
==========================================================================

`ociomakeclf --list` offers CSC spaces for Sony, Canon, ARRI, Panasonic and RED.
Nikon, Fujifilm and DJI have NO official route.

This script builds the equivalent CLF by hand, copying the exact node structure
that ociomakeclf emits:

    <Matrix>                        ACES2065-1  -> F-Gamut C  (linear)
    <Log style="cameraLinToLog">    linear      -> F-Log2 code
    <LUT3D interpolation="tetrahedral">   the look
    <Log style="cameraLogToLin">    F-Log2 code -> linear
    <Matrix>                        F-Gamut C   -> ACES2065-1

which is the LMT contract: ACES2065-1 in, ACES2065-1 out.

Everything is verified against colour-science at the end.
"""

import os
import uuid
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
V = "Fujifilm F-Log2 / F-Gamut C"
CAM = build_registry()
SPACE = CAM[V]

# F-Log2 constants, from the Fujifilm F-Log2 Data Sheet via colour-science
FLOG2 = dict(cut1=0.000889, a=5.555556, b=0.064829, c=0.245281,
             d=0.384316, e=8.799461, f=0.092864)
BASE = 10


def grid(size):
    g = np.linspace(0, 1, size)
    R, G, B = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([R, G, B], -1).reshape(-1, 3)


OCFG = ocio.Config.CreateRaw()
P_LOOK = OCFG.getProcessor(ocio.GroupTransform([
    ocio.CDLTransform(slope=[1.06, 1.00, 0.95], offset=[0.010, 0.000, -0.010],
                      power=[0.95, 1.00, 1.05], sat=0.88)])).getDefaultCPUProcessor()
P_TETRA = None


def run(p, x):
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    p.applyRGB(a)
    return a.astype(np.float64)


def hub_to_space_linear(lin_hub):
    """ACES2065-1 linear -> F-Gamut C linear."""
    return colour.RGB_to_RGB(lin_hub, HUB, SPACE, apply_cctf_decoding=False,
                             apply_cctf_encoding=False)


def space_to_hub_linear(lin_fgc):
    return colour.RGB_to_RGB(lin_fgc, SPACE, HUB, apply_cctf_decoding=False,
                             apply_cctf_encoding=False)


def repr_range(cs):
    lo = float(np.min(np.asarray(cs.cctf_decoding(np.zeros((1, 3))))))
    hi = float(np.max(np.asarray(cs.cctf_decoding(np.ones((1, 3))))))
    return lo, hi


def encode_flog2(lin):
    lo, hi = repr_range(SPACE)
    x = np.clip(lin, lo, hi)
    k = FLOG2
    return np.where(x >= k["cut1"],
                    k["c"] * np.log10(np.maximum(k["a"] * x + k["b"], 1e-12)) + k["d"],
                    k["e"] * x + k["f"])


# matrices ---------------------------------------------------------------
def matrices():
    """Return (AP0->F-Gamut C, F-Gamut C->AP0) as 3x3 in CLF column-vector form."""
    I = np.eye(3)
    b = np.asarray(colour.RGB_to_RGB(I, HUB, SPACE, apply_cctf_decoding=False,
                                     apply_cctf_encoding=False))
    # our data convention is row vectors; CLF acts on column vectors
    M1 = b.T
    b2 = np.asarray(colour.RGB_to_RGB(I, SPACE, HUB, apply_cctf_decoding=False,
                                      apply_cctf_encoding=False))
    M2 = b2.T
    return M1, M2


def fmt_matrix(m, indent=" " * 8):
    rows = []
    for r in m:
        rows.append(indent + " ".join(f"{v:.12g}" for v in r))
    return "\n".join(rows)


def fmt_cube(table, indent=" " * 8):
    """Serialise a colour.LUT3D table into a CLF LUT3D array.

    Ordering trap: a .cube stores entries with RED varying fastest, but a CLF
    LUT3D array stores them with RED varying SLOWEST (blue fastest). Getting
    this backwards silently swaps red and blue - verified against OCIO.
    colour.LUT3D.table is indexed [r, g, b], so a plain C-order flatten
    already yields r slowest / b fastest, which is exactly what CLF wants.
    """
    arr = table.reshape(-1, 3)
    return "\n".join(indent + " ".join(f"{v:.9g}" for v in row) for row in arr)


def build_clf(table, size, m1, m2, path):
    k = FLOG2
    logparams = (
        f'<LogParams base="{BASE}" linSideSlope="{k["a"]:.12g}" '
        f'linSideOffset="{k["b"]:.12g}" logSideSlope="{k["c"]:.12g}" '
        f'logSideOffset="{k["d"]:.12g}" linSideBreak="{k["cut1"]:.12g}" '
        f'linearSlope="{k["e"]:.12g}"/>'
    )
    uid = f"urn:uuid:{uuid.uuid4()}"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<ProcessList compCLFversion="3" xmlns="http://www.smpte-ra.org/ns/2136-1/2024" id="{uid}">
    <Id>{uid}</Id>
    <Description>ACES LMT for a look authored in Fujifilm F-Log2 / F-Gamut C.
Built by the hub prototype because ociomakeclf has no CSC route for Fujifilm.
Contract: ACES2065-1 in, ACES2065-1 out.</Description>
    <InputDescriptor>ACES2065-1</InputDescriptor>
    <OutputDescriptor>ACES2065-1</OutputDescriptor>
    <Matrix inBitDepth="32f" outBitDepth="32f">
        <Array dim="3 3">
{fmt_matrix(m1)}
        </Array>
    </Matrix>
    <Log inBitDepth="32f" outBitDepth="32f" style="cameraLinToLog">
        {logparams}
    </Log>
    <LUT3D inBitDepth="32f" outBitDepth="32f" interpolation="tetrahedral">
        <Array dim="{size} {size} {size} 3">
{fmt_cube(table)}
        </Array>
    </LUT3D>
    <Log inBitDepth="32f" outBitDepth="32f" style="cameraLogToLin">
        {logparams}
    </Log>
    <Matrix inBitDepth="32f" outBitDepth="32f">
        <Array dim="3 3">
{fmt_matrix(m2)}
        </Array>
    </Matrix>
</ProcessList>
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    return xml


def main():
    print("=" * 92)
    print("Hand-built CLF for Fujifilm F-Log2 / F-Gamut C  (official route: none)")
    print("=" * 92)

    m1, m2 = matrices()
    print("  AP0 -> F-Gamut C matrix:")
    for r in m1:
        print("   ", " ".join(f"{v:14.9f}" for v in r))
    print("  round trip M2 @ M1 max abs err:",
          f"{np.max(np.abs(m2 @ m1 - np.eye(3))):.3e}")

    SIZE = 33
    def look_in_space(x):
        lin = SPACE.cctf_decoding(np.asarray(x))
        hub = space_to_hub_linear(lin)
        hub = run(P_LOOK, hub)
        lin2 = hub_to_space_linear(hub)
        return encode_flog2(lin2)

    table = np.asarray(look_in_space(grid(SIZE))).reshape(SIZE, SIZE, SIZE, 3)
    print(f"  look cube baked at {SIZE}^3, range [{table.min():.4f}, {table.max():.4f}]")

    clf = os.path.join(OUT, "LMT_FUJIFILM_FLOG2-FGAMUTC_look.clf")
    build_clf(table, SIZE, m1, m2, clf)
    print(f"  wrote {os.path.basename(clf)}  ({os.path.getsize(clf)/1024:.0f} KB)")

    # also drop the plain .cube for comparison
    colour.write_LUT(
        colour.LUT3D(table=table.astype(np.float32),
                     name="look_in_FUJIFILM_FLOG2-FGAMUTC",
                     domain=[[0, 0, 0], [1, 1, 1]]),
        os.path.join(OUT, "look_in_FUJIFILM_FLOG2-FGAMUTC_33.cube"))

    # ---------------------------------------------------------------- verify
    print()
    print("=" * 92)
    print("VERIFY - does OCIO read it, and does it match colour-science?")
    print("=" * 92)
    ft = ocio.FileTransform(clf, interpolation=ocio.INTERP_TETRAHEDRAL)
    P_CLF = OCFG.getProcessor(ft).getDefaultCPUProcessor()

    test = np.random.rand(4000, 3) * 0.6           # ACES2065-1 linear values
    clf_out = run(P_CLF, test)
    truth = run(P_LOOK, test)

    # compare in a perceptual space: take both through the ACES 2.0 ODT + Rec.709
    P_ODT = OCFG.getProcessor(ocio.GroupTransform([
        ocio.BuiltinTransform(style="ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-REC709_2.0"),
        ocio.BuiltinTransform(style="DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.709")])
    ).getDefaultCPUProcessor()

    def lab(codes):
        return colour.XYZ_to_Lab(colour.RGB_to_XYZ(np.clip(codes, 0, 1) ** 2.4, "ITU-R BT.709"))

    d_clf = lab(run(P_ODT, clf_out))
    d_truth = lab(run(P_ODT, truth))
    de = colour.delta_E(d_clf, d_truth, method="CIE 2000")
    print(f"  OCIO read the hand-built CLF successfully")
    print(f"  deltaE2000 vs the analytic look (measured through ACES 2.0 -> Rec.709):")
    print(f"    mean   {np.mean(de):.4f}")
    print(f"    median {np.median(de):.4f}")
    print(f"    p95    {np.percentile(de,95):.4f}")
    print(f"    max    {np.max(de):.4f}")

    r = np.abs(clf_out - truth) / np.maximum(np.abs(truth), 1e-2)
    print(f"  relative difference in ACES2065-1: median {np.median(r):.6f}  "
          f"p99 {np.percentile(r,99):.4f}")
    print()
    print("  => the gap is fillable with open tooling. Only the CSC wrapper was missing.")
    print("     The colour science itself is fully published.")


if __name__ == "__main__":
    main()
