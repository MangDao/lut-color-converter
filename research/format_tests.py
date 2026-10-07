"""
format_tests.py - prove the .3dl / .vlt readers, using outside references
=========================================================================

A round-trip test cannot catch a wrong grid order: if the writer and the reader
share the same mistake the table comes back perfect. So the order is pinned by
two outside references first -

  A  the vendor's own sentence, transcribed into a hand-built file
  B  OpenColorIO, which has its own .3dl implementation - not ours

and only then do the round trips mean anything.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import formats as F

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


CODE = 4095.0
# Each output channel encodes its own grid indices, so a wrong axis order shows
# up as a wrong number rather than as a plausible-looking LUT.
def enc(r, g, b):
    return (r * 2048 + g * 512 + b * 128, r * 1024 + g * 256 + b * 64,
            r * 512 + g * 128 + b * 32)


def want(r, g, b):
    return np.array(enc(r, g, b), dtype=np.float64) / CODE


PROBES = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
          (1, 0, 1), (0, 1, 1), (1, 1, 1)]


def report_order(table, label):
    ok = True
    for (r, g, b) in PROBES:
        w, got = want(r, g, b), table[r, g, b]
        same = np.allclose(got, w, atol=1e-12)
        ok &= same
        print(f"    (r={r},g={g},b={b}) -> table[{r},{g},{b}] = {np.round(got,4)} "
              f"want {np.round(w,4)}  {'ok' if same else 'MISMATCH'}")
    return ok


# ---------------------------------------------------------------- 0. the claim
def test_coverage():
    print()
    print("=" * 96)
    print("0. WHAT colour-science ACTUALLY READS  (basis for writing our own)")
    print("=" * 96)
    import colour
    from colour.io.luts import MAPPING_EXTENSION_TO_LUT_FORMAT
    print(f"  colour-science {colour.__version__}")
    print(f"  reads: {sorted(MAPPING_EXTENSION_TO_LUT_FORMAT)}")
    exts = set(MAPPING_EXTENSION_TO_LUT_FORMAT)
    check("colour-science does not read .3dl", ".3dl" not in exts)
    check("colour-science does not read .vlt", ".vlt" not in exts)


# --------------------------------------------------- A. order, vendor documents
def test_vlt_order():
    """Panasonic, VARICAM 3DLUT REFERENCE MANUAL Rev.1.0 section 2:
       'the line 5 is R0, G0, B0, line 6 is R1, G0, B0'  ->  RED varies fastest."""
    print()
    print("=" * 96)
    print("A. .vlt GRID ORDER - against the sentence in Panasonic's manual")
    print("=" * 96)
    # n=2 on purpose: n=17 would drag in the non-uniform axis question, and this
    # test is only about the ordering.
    seq = []
    for b in range(2):
        for g in range(2):
            for r in range(2):
                seq.append(enc(r, g, b))
    txt = ("# panasonic vlt file version 1.0\n"
           '# source vlt file ""\n'
           "LUT_3D_SIZE 2\n\n" + "\n".join(f"{a} {b_} {c}" for a, b_, c in seq) + "\n")
    open(os.path.join(OUT, "_t_order.vlt"), "w", encoding="utf-8").write(txt)

    table, info = F.read_vlt(txt)
    print(f"  size {info['size']}, order {info['grid_order']}, scale {info['scale_source']}")
    check("first data line is (R0,G0,B0), second is (R1,G0,B0)",
          report_order(table, "vlt"), "R varies fastest, as the manual states")


def test_3dl_order():
    """Autodesk Lustre Colour Management User Guide ch.3:
       'Values for the blue index are incremented first'  ->  BLUE varies fastest."""
    print()
    print("=" * 96)
    print("B. .3dl GRID ORDER - against the sentence in Autodesk's guide")
    print("=" * 96)
    seq = []
    for r in range(2):
        for g in range(2):
            for b in range(2):
                seq.append(enc(r, g, b))
    txt = "3DMESH\n" + "\n".join(f"{a} {b_} {c}" for a, b_, c in seq) + "\n"
    table, info = F.read_3dl(txt)
    print(f"  size {info['size']}, order {info['grid_order']}, scale {info['scale_source']}")
    check("third data line is (R0,G0,B2)", report_order(table, "3dl"),
          "B varies fastest, as the guide states")

    body = "\n".join("0 0 0" for _ in range(17 ** 3))
    _, i2 = F.read_3dl("# comment\n3DMESH\nMesh 4 12\n" + body + "\n")
    check("`Mesh 4 12` -> 17^3 nodes, 12-bit",
          i2["size"] == 17 and i2["bits"] == 12 and 2 ** i2["mesh_in_bits"] + 1 == 17,
          f"size {i2['size']}, bits {i2['bits']}, scale from {i2['scale_source']}")

    lad = " ".join(str(int(round(x))) for x in np.linspace(0, 1023, 17))
    _, i3 = F.read_3dl(lad + "\n" + body + "\n")
    check("coordinate-ladder dialect detected", i3["ladder_len"] == 17,
          f"ladder of {i3['ladder_len']} values, scale from {i3['scale_source']}")


# ------------------------------------------------------- C. OpenColorIO referee
def test_ocio_referee():
    print()
    print("=" * 96)
    print("C. INDEPENDENT REFEREE - OpenColorIO's own .3dl reader, not ours")
    print("=" * 96)
    import PyOpenColorIO as ocio

    import demos as DM
    from lutlib import grid_points, tetra

    space = "Sony S-Log3 / S-Gamut3.Cine"
    table = DM.reference(space, DM.DSP, 17)
    p = os.path.join(OUT, "_t_referee.3dl")
    open(p, "w", encoding="utf-8", newline="\n").write(
        F.write_3dl(table, out_bits=12, note="referee"))

    try:
        grp = ocio.GroupTransform([ocio.FileTransform(p)])
        cpu = ocio.Config.CreateRaw().getProcessor(grp).getDefaultCPUProcessor()
    except Exception as exc:
        check("OpenColorIO can load a .3dl FileTransform", False, repr(exc))
        return

    pts = grid_points(17)
    ours = tetra(table, pts)
    theirs = np.ascontiguousarray(pts.astype(np.float32))
    cpu.applyRGB(theirs)
    d = np.abs(ours - theirs.astype(np.float64))

    print(f"  reference LUT : {space} -> Rec.709 at 17^3, written as .3dl")
    print(f"  ours vs OCIO  : max |diff| {d.max():.3e}   mean {d.mean():.3e}   "
          f"(one 12-bit code = {1/4095:.2e})")
    check("our .3dl writer+reader agrees with OpenColorIO",
          d.max() < 3e-4, f"max abs difference {d.max():.2e}")
    print("  A wrong grid order does not crash - it returns a silently transposed")
    print("  LUT, which is exactly why an outside reference is needed here.")


# ------------------------------------------------------------- D. round trips
def _maxdiff(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def test_roundtrips():
    print()
    print("=" * 96)
    print("D. ROUND TRIPS")
    print("=" * 96)
    import demos as DM
    from lutlib import grid_points, tetra

    q = 1.0 / CODE / 2                                  # half a 12-bit code
    cases = [("Sony S-Log3 / S-Gamut3.Cine", "log2disp"),
             ("ARRI LogC4 / AWG4", "disp2log")]

    print("  D1  .3dl - uniform axis, and our layout already matches its order")
    for space, label in cases:
        for n in (17, 33):
            table = (DM.reference(space, DM.DSP, n) if label == "log2disp"
                     else DM.reference(DM.DSP, space, n))
            pts = grid_points(n)
            t2, i2 = F.read_3dl(F.write_3dl(table, out_bits=12))
            e = _maxdiff(tetra(t2, pts), tetra(table, pts))
            print(f"    {label:9s} {n}^3  B-fastest  max err {e:.3e}"
                  f"   {'ok' if e <= q + 1e-6 else 'TOO BIG'}")
            check(f".3dl round trip {n}^3 {label}", e <= q + 1e-6, f"{e:.2e}")

    print()
    print("  D2  .vlt idempotence - read, write back on the same axis, read again")
    print("      (no resampling possible, so this must be pure quantisation)")
    for space, label in cases:
        table = (DM.reference(space, DM.DSP, 17) if label == "log2disp"
                 else DM.reference(DM.DSP, space, 17))
        t1, i1 = F.read_vlt(F.write_vlt(table))
        t2, i2 = F.read_vlt(F.write_vlt(t1, axis=i1["input_axis"]))
        e = _maxdiff(t2, t1)
        print(f"    {label:9s} 17^3  axis {i1['input_axis_kind']:28s} "
              f"max err {e:.3e}   {'ok' if e <= q + 1e-6 else 'TOO BIG'}")
        check(f".vlt idempotence 17^3 {label}", e <= q + 1e-6, f"{e:.2e}")

    print()
    print("  D3  .vlt representation cost of a uniform 17^3 table")
    print("      The format fixes where the nodes go, so a uniform table has to be")
    print("      moved onto them. That move is lossy; here is how lossy, and what")
    print("      carrying the axis instead of assuming it buys.")
    print()
    print(f"    {'lut':24s} {'layout':>8s} {'correct eval':>13s} {'axis ignored':>13s}")
    print("    " + "-" * 62)
    worst = 0.0
    for space, label in cases:
        table = (DM.reference(space, DM.DSP, 17) if label == "log2disp"
                 else DM.reference(DM.DSP, space, 17))
        pts = grid_points(17)
        truth = tetra(table, pts)
        for layout in ("spec", "uniform"):
            t1, i1 = F.read_vlt(F.write_vlt(table, layout=layout), layout=layout)
            e_ok = _maxdiff(F.as_fn(t1, i1)(pts), truth)
            e_bad = _maxdiff(tetra(t1, pts), truth)
            worst = max(worst, e_ok)
            print(f"    {label:24s} {layout:>8s} {e_ok:13.3e} {e_bad:13.3e}")
    print()
    print("    'axis ignored' = evaluate the file as if its nodes were uniform.")
    print("    Neither reading dominates: on this metric they are the same size, so")
    print("    the layout question CANNOT be settled by which number is smaller -")
    print("    only by which reading matches how the file was actually produced.")
    print("    What carrying the axis does buy is measured separately, in D2: the")
    print("    read -> write -> read cycle is exact (0.0) instead of accumulating a")
    print("    second placement error. That is the reason to carry it, not accuracy.")
    check("the .vlt node-placement cost is bounded and reported", worst < 0.02,
          f"worst {worst:.2e} = {worst*CODE:.0f} of 4095 codes")
    check("layout='uniform' makes a uniform table's .vlt exact",
          abs(_maxdiff_of_uniform_tables() - q) < 1e-9,
          f"error equals the quantisation step {q:.2e}")


def _maxdiff_of_uniform_tables():
    """The layout='uniform' path with no resampling at all: pure quantisation."""
    import demos as DM
    from lutlib import grid_points, tetra
    table = DM.reference("Sony S-Log3 / S-Gamut3.Cine", DM.DSP, 17)
    t1, i1 = F.read_vlt(F.write_vlt(table, layout="uniform"), layout="uniform")
    return _maxdiff(F.as_fn(t1, i1)(grid_points(17)), tetra(table, grid_points(17)))


def test_vlt_axis():
    print()
    print("=" * 96)
    print("E. THE TWO READINGS OF THE .vlt INPUT AXIS")
    print("=" * 96)
    dev = np.abs(F.VLT_AXIS_17 - F.uniform_axis(17))
    i = int(np.argmax(dev))
    print("  the manual prints:  0 256 512 ... 3584 3840 4095   (last step 255)")
    print("  a uniform generator: i * 4095/16                      (step 255.9375)")
    print(f"  they differ by at most {dev.max()*CODE:.2f} code value(s) of 4095, "
          f"at node {i}")
    print("  Both readings are implemented (`layout=`). Which one a given file")
    print("  follows cannot be determined from the file itself, so this is a")
    print("  documented, bounded unknown - not something the reader can resolve.")
    check("ambiguity is bounded and quantified", dev.max() * CODE < 1.0,
          f"max {dev.max()*CODE:.3f} codes")


def test_scale_and_refusals():
    print()
    print("=" * 96)
    print("F. SCALE INFERENCE, AND A CLEAN REFUSAL FOR THE BINARY .vlt")
    print("=" * 96)
    rng = np.random.default_rng(5)
    body12 = rng.integers(0, 4096, size=(17 ** 3, 3))
    body10 = rng.integers(0, 1024, size=(17 ** 3, 3))
    fmt = lambda a: "\n".join(f"{x} {y} {z}" for x, y, z in a)

    for head, arr, want_bits, label in [
            ("3DMESH\nMesh 4 12\n", body12, 12, "declared Mesh 4 12"),
            ("3DMESH\nMesh 4 10\n", body10, 10, "declared Mesh 4 10"),
            ("3DMESH\n", body12, 12, "undeclared, values up to 4095"),
            ("3DMESH\n", body10, 10, "undeclared, values up to 1023")]:
        _, i = F.read_3dl(head + fmt(arr) + "\n")
        print(f"  {label:32s} -> bits {str(i['bits']):>4s}, divisor "
              f"{int(i['scale_divisor']):>5d}, source: {i['scale_source']}")
        check(f"scale, {label}", i["bits"] == want_bits)

    ladder = " ".join(str(int(round(x))) for x in np.linspace(0, 1023, 17))
    _, i = F.read_3dl(ladder + "\n" + fmt(body10) + "\n")
    check("10-bit coordinate ladder -> 10-bit output scale",
          i["bits"] == 10, f"divisor {int(i['scale_divisor'])}")

    raw = (b"\x00\x01\x02" * 100).decode("latin-1")
    try:
        F.read_vlt(raw)
        check("binary .vlt refused with a clear message", False, "no error raised")
    except ValueError as exc:
        check("binary .vlt refused with a clear message",
              "binary" in str(exc) and "VARICAM" in str(exc), str(exc)[:58] + "...")


# ------------------------------------------------------------- G. integration
def test_roundtrip_through_real_conversion():
    """A file only matters if it survives being used.

    The old version of this test pushed the file through the identification stage,
    which has since been removed. What replaces it is stronger for this purpose:
    write the file, read it back, and compare against the conversion computed
    directly from the registry. Format bugs show up here as a wrong LUT, not as a
    wrong label.
    """
    from lutlib import grid_points, tetra

    import demos as DM

    print()
    print("=" * 96)
    print("G. A REAL CONVERSION, WRITTEN TO EACH FORMAT AND READ BACK")
    print("=" * 96)
    print("  comparing against the registry directly, not against a label")

    for sin, sout, label in [
            ("Sony S-Log3 / S-Gamut3.Cine", DM.DSP, "S-Log3/S-Gamut3.Cine -> Rec.709"),
            ("ARRI LogC3 / AWG3", DM.DSP, "LogC3/AWG3 -> Rec.709"),
            ("Panasonic V-Log / V-Gamut", DM.DSP, "V-Log/V-Gamut -> Rec.709"),
            ("Canon Log 3 / Cinema Gamut", DM.DSP, "CLog3/CinemaGamut -> Rec.709")]:
        table = DM.reference(sin, sout, 33)
        pts = grid_points(33)
        want = tetra(table, pts)
        for ext, writer in ((".3dl", lambda t: F.write_3dl(t, out_bits=12)),
                            (".vlt", lambda t: F.write_vlt(t))):
            p = os.path.join(OUT, "_t_e2e" + ext)
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(writer(table))
            fn, info, _ = F.load(p)
            n = info["size"]
            got = np.asarray(fn(grid_points(n))).reshape(n, n, n, 3)
            # compare on the reader's own grid, against the same conversion there
            ref = DM.reference(sin, sout, n)
            d = float(np.abs(got - ref).max())
            q = 1.0 / 4095            # one 12-bit code
            ok = d <= q
            print(f"  {ext:5s} {label:34s} {n:2d}³  max err vs registry "
                  f"{d:.3e}  {'ok' if ok else 'TOO BIG'}")
            check(f"{label} survives {ext}", ok,
                  f"{d:.2e} <= one 12-bit code ({q:.2e})")


def main():
    print("=" * 96)
    print(".3dl / .vlt READERS - verification")
    print("=" * 96)
    test_coverage()
    test_vlt_order()
    test_3dl_order()
    test_ocio_referee()
    test_roundtrips()
    test_vlt_axis()
    test_scale_and_refusals()
    test_roundtrip_through_real_conversion()

    print()
    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAIL  {name}   {detail}")
    print(f"  {npass}/{len(RESULTS)} checks passed")
    return npass == len(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
