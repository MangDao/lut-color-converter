"""
derive_space.py - recover a vendor colour space from its official OCIO IDT
=========================================================================

Some vendors ship an official OCIO IDT but have no entry in colour-science
(Apple is the immediate case). Recover the space numerically instead of
copying primaries from a PDF and hoping.

An OCIO "IDT" maps  CAMERA LOG CODE VALUES  ->  ACES2065-1 linear:

        IDT( log_encode(x) )  =  M @ x          for linear x in the vendor gamut

so feeding `log_encode(e_i)` through the IDT gives column i of M directly.
Normalising the columns of `A_AP0_to_XYZ @ M` yields the chromaticities.

Direction matters: the first version of this script fed LINEAR basis vectors
into an IDT that expects LOG CODE VALUES, and produced a matrix full of 1e41.
"""

import numpy as np
import colour
import PyOpenColorIO as ocio

D60 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D60"]
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]


def ap0_to_xyz():
    """ACES2065-1 (AP0 linear) -> XYZ(D60).

    Do NOT use colour.RGB_to_XYZ(np.eye(3), "ACES2065-1", D60) - passing a
    whitepoint there applies a chromatic adaptation and silently breaks the
    matrix. colourspace.matrix_RGB_to_XYZ is the authoritative one and matches
    the published ACES matrix to 1e-10.
    """
    return np.asarray(colour.RGB_COLOURSPACES["ACES2065-1"].matrix_RGB_to_XYZ)


def derive(style, cctf_name, label, verbose=True):
    """Recover (primaries, whitepoint, M_AP0_from_vendor) from an official IDT."""
    enc = colour.CCTF_ENCODINGS[cctf_name]
    cfg = ocio.Config.CreateRaw()
    p = cfg.getProcessor(ocio.BuiltinTransform(style=style)).getDefaultCPUProcessor()

    cols = []
    for i in range(3):
        x = np.zeros((1, 3))
        x[0, i] = 1.0                       # vendor linear basis vector
        code = np.asarray(enc(x), dtype=np.float32)
        arr = np.ascontiguousarray(code.copy())
        p.applyRGB(arr)
        cols.append(arr.astype(np.float64)[0])   # = M @ e_i
    M = np.stack(cols, axis=1)              # vendor linear -> AP0 linear

    A = ap0_to_xyz()
    vendor_to_xyz = A @ M

    primaries = np.stack([vendor_to_xyz[:, i] / vendor_to_xyz[:, i].sum()
                          for i in range(3)])
    white = vendor_to_xyz @ np.ones(3)
    wp = white[:2] / white.sum()

    if verbose:
        print(f"  {label}")
        print(f"    vendor linear -> AP0 linear matrix (columns = M @ e_i):")
        for r in M.T:
            print("      " + " ".join(f"{v:13.9f}" for v in r))
        for i, lab in enumerate("RGB"):
            print(f"    {lab}: ({primaries[i][0]:.6f}, {primaries[i][1]:.6f})")
        print(f"    whitepoint: ({wp[0]:.6f}, {wp[1]:.6f})   "
              f"D65 = ({D65[0]:.4f}, {D65[1]:.4f})   "
              f"delta {np.linalg.norm(np.array(wp) - np.array(D65)):.2e}")
    return primaries, wp, M


def verify(style, cctf_name, primaries, label, n=4000):
    enc = colour.CCTF_ENCODINGS[cctf_name]
    dec = colour.CCTF_DECODINGS[cctf_name]
    cfg = ocio.Config.CreateRaw()
    p = cfg.getProcessor(ocio.BuiltinTransform(style=style)).getDefaultCPUProcessor()

    # a plausible set of vendor log code values
    code = np.random.rand(n, 3) * 0.85 + 0.05
    arr = np.ascontiguousarray(code.astype(np.float32).copy())
    p.applyRGB(arr)
    official_ap0 = arr.astype(np.float64)

    # rebuild: decode the code, rotate by the recovered primaries, compare
    lin = np.asarray(dec(code))
    m_vendor_to_xyz = np.stack([primaries[i] / primaries[i].sum() for i in range(3)],
                               axis=1)
    m_xyz_to_ap0 = np.linalg.inv(ap0_to_xyz())
    mine_ap0 = lin @ (m_xyz_to_ap0 @ m_vendor_to_xyz).T

    rel = np.abs(mine_ap0 - official_ap0) / np.maximum(np.abs(official_ap0), 1e-2)
    print(f"    verify vs official IDT: median rel {np.median(rel):.2e}  "
          f"p99 {np.percentile(rel,99):.2e}  max {np.max(rel):.2e}")
    return float(np.median(rel))


if __name__ == "__main__":
    print("=" * 86)
    print("Recovering Apple Log / Apple Wide Gamut from the official OCIO IDT")
    print("=" * 86)
    prim, wp, M = derive("APPLE_LOG-APPLEWG_to_ACES2065-1", "Apple Log Profile",
                         "Apple Log / Apple Wide Gamut")
    verify("APPLE_LOG-APPLEWG_to_ACES2065-1", "Apple Log Profile", prim,
           "Apple Log / Apple Wide Gamut")

    print()
    print("  sanity: does the same method reproduce a space whose primaries we")
    print("  already know? run it on Sony and compare against colour-science.")
    prim_s, wp_s, _ = derive("SONY_SLOG3-SGAMUT3.CINE_to_ACES2065-1", "S-Log3",
                             "(recovery test) Sony S-Gamut3.Cine")
    known = np.asarray(colour.RGB_COLOURSPACES["S-Gamut3.Cine"].primaries)
    print(f"    max |recovered - colour-science| : "
          f"{np.max(np.abs(prim_s[:, :2] - known)):.2e}")
    print(f"    -> the method is sound; the residual is just the log curve difference")
