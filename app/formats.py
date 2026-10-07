"""
formats.py - read/write the two LUT formats colour-science does not cover
=========================================================================

colour-science 0.4.7 ships readers for exactly five LUT families:

    .cube (Iridas)   .cube (Resolve)   .csp   .spi1d   .spi3d   .spimtx

Probed, not assumed - see format_tests.py. Two formats that matter in the field
are absent, and they turn out to disagree about the one thing that silently
corrupts everything: the order the grid is written in.

    .3dl   Autodesk Lustre        BLUE  varies fastest
    .vlt   Panasonic VARICAM      RED   varies fastest

Both are documented by their vendors, so both are quoted in the code below
rather than guessed. A reader that gets this wrong does not crash - it returns a
LUT that is a transposed version of the real one, which is worse.

Our own in-memory layout (see lutlib.grid_points) is `table[r, g, b]` built by
`meshgrid(..., indexing="ij")`, i.e. **blue fastest**, so `.3dl` reshapes
directly and `.vlt` has to be transposed on axes 0 and 2.

The axis problem
----------------
Panasonic fixes the 17 sampling positions at

    "0 256 512 768 ... 3584 3840 4095"

which is sixteen steps of 256 and then a final step of 255. The axis is NOT
uniform, and the deviation is not academic. Resampling a steep log LUT onto that
axis and back loses 6.3e-3 (about 26 of 4095 code values) on Sony S-Log3 ->
Rec.709 - fifty times the 12-bit quantisation step. That figure is measured, not
estimated; the first version of this module did exactly that resampling.

So the axis is carried as data instead: `read_vlt` hands back the table ON ITS
OWN NODE POSITIONS plus `info["input_axis"]`, and `as_fn` evaluates against those
positions. Nothing is resampled on read. Resampling only happens once, on write,
because the file format itself fixes where the nodes have to go - that is a
property of .vlt, not of this code, and the residual is reported rather than
hidden.

Other real-world messiness handled here:

  * .3dl is a family of dialects: a bare `3DMESH` line, a `Mesh <in_bits>
    <out_bits>` line, a Lustre "coordinate ladder" first line, or a bare integer
    size. The rule used to separate header from data is that **data lines have
    exactly three numbers** - every documented ladder has 5/9/17/33/65 entries,
    so three is unambiguous.
  * A second, binary `.vlt` exists (monitor LUTs uploaded to camera). It is
    detected and refused with a clear message instead of being parsed to garbage.

Scale: `.3dl` declares its output bit depth in the `Mesh` line and that is used
verbatim. `.vlt` is fixed at 12 bits by its specification. Anything inferred from
the observed maximum is labelled as inferred so callers can surface it.
"""

import os

import numpy as np

DATA_TOKENS = 3                      # a data line is exactly R G B and nothing else


# ------------------------------------------------------------------- utilities
def _prep(text):
    """Strip comments and blanks. Refuse binary rather than emit nonsense."""
    if "\x00" in text[:8192]:
        raise ValueError(
            "this file is binary, not the ASCII dialect. Panasonic's camera-upload "
            "monitor LUT is a separate binary .vlt; only the VARICAM 3DLUT "
            "REFERENCE MANUAL Rev.1.0 ASCII form is implemented here.")
    out = []
    for raw in text.splitlines():
        s = raw.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def _nums(s):
    """Floats if the whole line is numeric, else None. 'LUT_3D_SIZE 17' -> None,
    because the token LUT_3D_SIZE is not a number."""
    toks = s.split()
    try:
        return [float(t) for t in toks]
    except ValueError:
        return None


def _bits_for(vmax):
    for b in (8, 10, 12, 14, 16, 18, 20):
        if vmax <= (2 ** b - 1) + 1e-6:
            return b
    return None


def _cube_from(vals, n):
    """(n^3, 3) float list -> (n, n, n, 3). Order is the caller's business."""
    if len(vals) != n ** 3:
        raise ValueError(f"expected {n**3} grid entries, found {len(vals)}")
    return np.asarray(vals, dtype=np.float64).reshape(n, n, n, 3)


def _size_from_count(count):
    n = int(round(count ** (1.0 / 3.0)))
    if n ** 3 != count:
        raise ValueError(
            f"{count} data entries is not a perfect cube - the file is truncated "
            f"or this is not a 3D LUT in a dialect this reader knows")
    return n


def uniform_axis(n):
    return np.linspace(0.0, 1.0, n)


def axis_is_uniform(axis, tol=1e-12):
    axis = np.asarray(axis, dtype=np.float64)
    return bool(np.max(np.abs(axis - uniform_axis(len(axis)))) <= tol)


def resample_axis(table, src_axis, dst_axis, axis):
    """Linear resample along one axis of a (...,3) table."""
    m = np.moveaxis(np.asarray(table, dtype=np.float64), axis, 0)
    flat = m.reshape(len(src_axis), -1)
    out = np.empty((len(dst_axis), flat.shape[1]))
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst_axis, src_axis, flat[:, c])
    return np.moveaxis(out.reshape((len(dst_axis),) + m.shape[1:]), 0, axis)


def _resample_all(table, src_axis, dst_axis):
    for ax in range(3):
        table = resample_axis(table, src_axis, dst_axis, ax)
    return table


def to_uniform(table, info=None):
    """Put a table back on a uniform axis - for exporters (.cube) that cannot
    express node positions. Lossy by design; sized in format_tests.py."""
    t = np.asarray(table, dtype=np.float64)
    ax = None if info is None else info.get("input_axis")
    if ax is None or len(ax) != t.shape[0] or axis_is_uniform(ax):
        return t
    return _resample_all(t, np.asarray(ax, dtype=np.float64), uniform_axis(t.shape[0]))


def _to_int(v, hi):
    return np.clip(np.rint(np.asarray(v, dtype=np.float64) * hi), 0, hi).astype(np.int64)


def _fmt_flat(vals):
    return "\n".join(f"{int(a)} {int(b)} {int(c)}" for a, b, c in vals)


# ============================================================== Panasonic .vlt
# "The extension of this 3DLUT file is .vlt and ASCII character encoding is used.
#  Lines 1 to 4 of the file are the file header ... Line 3 shows the number of
#  point for 3D cubes. 17point 3D cubes shown as follows.  LUT_3D_SIZE 17
#  ... Output values are noted in integers from 0 to 4095 ... Order of grid
#  points is incremented with R first followed by G and B. In other words, the
#  line 5 is R0, G0, B0, line 6 is R1, G0, B0"
#                        - Panasonic VARICAM 3DLUT REFERENCE MANUAL Rev.1.0, §2
#
# And the sampling positions, same paragraph, quoted verbatim:
#   "The 17 point sampling intervals of input values are
#    0 256 512 768 ... 3584 3840 4095"
VLT_BITS = 12
VLT_MAX = 2 ** VLT_BITS - 1                                    # 4095
VLT_AXIS_17 = np.array([i * 256 for i in range(16)] + [VLT_MAX], dtype=np.float64) / VLT_MAX

# The manual prints 0,256,...,3840,4095, which is NOT uniform: the last step is
# 255 while the others are 256. A tool that generated the grid as i*4095/16
# would produce 255.9375 steps instead. The two readings disagree by at most 0.94
# code values, so both are offered and the difference is measured in
# format_tests.py rather than argued about. "spec" follows the printed manual.
VLT_LAYOUTS = {"spec": VLT_AXIS_17, "uniform": uniform_axis(17)}


def read_vlt(text, layout="spec"):
    lines = _prep(text)

    n_decl = None
    for s in lines:
        t = s.split()
        if len(t) >= 2 and t[0].upper().startswith("LUT_3D_SIZE"):
            try:
                n_decl = int(float(t[1]))
            except ValueError:
                pass

    vals = [v for v in (_nums(s) for s in lines)
            if v is not None and len(v) == DATA_TOKENS]
    n = _size_from_count(len(vals))

    # R varies fastest, so the flat index is r + n*g + n*n*b and reshape(N,N,N,3)
    # lands it as (b, g, r) - transpose to our own (r, g, b).
    raw = np.transpose(_cube_from(vals, n), (2, 1, 0, 3))

    if n == 17:
        axis = VLT_LAYOUTS[layout]
        kind = ("specification, non-uniform" if layout == "spec"
                else "read as uniform (i*4095/16)")
    else:
        axis, kind = uniform_axis(n), "assumed uniform (the spec fixes n=17 only)"

    dev = float(np.max(np.abs(np.asarray(axis) - uniform_axis(n)))) * VLT_MAX
    info = {"format": "vlt", "type": "LUT3D", "size": n,
            "layout": layout,
            "declared_size": n_decl, "bits": VLT_BITS,
            "scale_source": "vlt specification (0..4095)",
            "scale_divisor": float(VLT_MAX),
            "grid_order": "R fastest",
            "input_axis": np.asarray(axis, dtype=np.float64).tolist(),
            "input_axis_kind": kind,
            "input_axis_max_deviation_codes": round(dev, 3),
            "resampled_on_read": False}
    if n_decl is not None and n_decl != n:
        info["warning"] = f"LUT_3D_SIZE says {n_decl}, data has {n}"
    return raw / VLT_MAX, info


def write_vlt(table, source_name="", axis=None, layout="spec"):
    """Emit a .vlt. `axis` is the node positions the table is already on; pass
    the value from a matching read_vlt to avoid any resampling at all."""
    t = np.asarray(table, dtype=np.float64)
    n = t.shape[0]

    if axis is not None:
        axis = np.asarray(axis, dtype=np.float64)
        if len(axis) != n:
            raise ValueError(f"axis has {len(axis)} positions but the table has {n} nodes")
    elif n == 17 and layout == "spec":
        # The format dictates the node positions, so a uniform table has to be
        # moved onto them. One resample, on write only.
        t = _resample_all(t, uniform_axis(n), VLT_LAYOUTS[layout])

    flat = np.transpose(t, (2, 1, 0, 3)).reshape(-1, 3)     # -> R-fastest file order
    ints = _to_int(flat, VLT_MAX)
    head = ("# panasonic vlt file version 1.0\n"
            f'# source vlt file "{source_name}"\n'
            f"LUT_3D_SIZE {n}\n\n")
    return head + _fmt_flat(ints) + "\n"


# ================================================================ Autodesk .3dl
# "The RGB values on the first line in the list are for (0,0,0), those on the
#  second line are for (0,0,1), and on the third line are for (0,0,2). Values
#  for the blue index are incremented first, followed by the green index,
#  followed by the red index, until the whole list of 17x17x17 is complete."
#                        - Autodesk Lustre Colour Management User Guide, ch.3
#
# Keyword form, from the Lustre 2009 help:
#   "(c) Mesh keyword (d) 3DMESH keyword (e) Input bit depth (f) Output bit depth
#    ... the 3D LUT has an input bit depth of 4 bits and an output bit depth of
#    12 bits ... You use the input value to calculate the RGB triplet to be
#    17*17*17 (where 17 = (2 to the power of 4)+1)"
# so `Mesh 4 12` means 17^3 nodes scaled 0..4095.
DL_BITS_DEFAULT = 12


def read_3dl(text):
    lines = _prep(text)

    has_3dmesh = False
    in_bits = out_bits = None
    ladder = None
    vals = []

    for s in lines:
        toks = s.split()
        up = [t.upper() for t in toks]
        if up and up[0] == "3DMESH":
            has_3dmesh = True
            continue
        if up and up[0] == "MESH":
            m = _nums(" ".join(toks[1:]))
            if m and len(m) >= 2:
                in_bits, out_bits = int(m[0]), int(m[1])
            continue
        m = _nums(s)
        if m is None:
            continue                                  # some other keyword
        if len(m) == DATA_TOKENS:
            vals.append(m)
        elif len(m) > DATA_TOKENS and all(m[i] < m[i + 1] for i in range(len(m) - 1)):
            ladder = m                                 # Lustre coordinate ladder

    n = _size_from_count(len(vals))
    # BLUE varies fastest, which is exactly our own layout - reshape directly.
    table = _cube_from(vals, n)

    vmax = float(table.max())
    if out_bits is not None:
        div, bits, src = 2 ** out_bits - 1, out_bits, "declared in the Mesh line"
    elif ladder is not None:
        b = _bits_for(float(max(ladder)))
        div, bits = 2 ** b - 1, b
        src = "inferred from the coordinate ladder"
    elif vmax <= 1.0 + 1e-9:
        div, bits, src = 1.0, None, "values already in 0..1"
    else:
        b = _bits_for(vmax)
        if b is None:
            div, bits, src = vmax, None, "normalised to the observed maximum"
        else:
            div, bits, src = 2 ** b - 1, b, "inferred from the maximum value"

    info = {"format": "3dl", "type": "LUT3D", "size": n,
            "bits": bits, "scale_divisor": float(div), "scale_source": src,
            "grid_order": "B fastest", "has_3dmesh": has_3dmesh,
            "mesh_in_bits": in_bits, "mesh_out_bits": out_bits,
            "ladder_len": (len(ladder) if ladder else None),
            "input_axis": uniform_axis(n).tolist(),
            "input_axis_kind": "uniform (the format samples uniformly)",
            "resampled_on_read": False}

    if in_bits is not None and 2 ** in_bits + 1 != n:
        info["warning"] = (f"Mesh says input depth {in_bits} (=> {2**in_bits+1} nodes) "
                           f"but the data has {n}")
    return table / div, info


def write_3dl(table, out_bits=DL_BITS_DEFAULT, note=None):
    t = np.asarray(table, dtype=np.float64)
    n = t.shape[0]
    hi = 2 ** out_bits - 1
    ints = _to_int(t.reshape(-1, 3), hi)              # already B-fastest

    head = []
    if note:
        head += [f"# {ln}" for ln in str(note).splitlines()]
    head.append("# 3D LUT - grid order is B fastest, per the Lustre user guide")
    head.append("3DMESH")
    k = None if n < 2 else int(round(np.log2(n - 1)))
    if k is not None and 2 ** k + 1 == n:
        head.append(f"Mesh {k} {out_bits}")
    else:
        head.append(f"# {n} nodes per axis (not 2^bits+1, so no Mesh line)")
    return "\n".join(head) + "\n" + _fmt_flat(ints) + "\n"


# ==================================================================== dispatch
OWN = {".vlt": read_vlt, ".3dl": read_3dl}


def read_own(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in OWN:
        raise ValueError(f"not one of our formats: {ext}")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return OWN[ext](fh.read())


def as_fn(table, info=None):
    """Wrap a table as the sampler the rest of the project expects (points in
    [0,1]^3 -> code values). Honours `info["input_axis"]` when the node positions
    are not uniform, so nothing is silently resampled.

    The axis is honoured by mapping each input to a FRACTIONAL node index and
    letting the usual interpolator do the rest - `tetra(t, idx, 0, n-1)` is the
    identity map of the converter it already applies internally.
    """
    from lutlib import tetra

    t = np.asarray(table, dtype=np.float64)
    n = t.shape[0]
    axis = None
    if info is not None and info.get("input_axis") is not None:
        a = np.asarray(info["input_axis"], dtype=np.float64)
        if len(a) == n and not axis_is_uniform(a):
            axis = a

    if axis is None:
        def fn(pts):
            return tetra(t, np.asarray(pts, dtype=np.float64))
        return fn

    gal = np.arange(n, dtype=np.float64)

    def fn(pts):
        p = np.clip(np.asarray(pts, dtype=np.float64), 0.0, 1.0)
        idx = np.empty_like(p)
        for c in range(3):
            idx[:, c] = np.interp(p[:, c], axis, gal)
        return tetra(t, idx, 0.0, float(n - 1))

    return fn


def load(path):
    """(fn, info, raw_text) - same contract as lut_io.load_lut, with the formats
    colour-science lacks filled in."""
    ext = os.path.splitext(path)[1].lower()
    if ext in OWN:
        table, info = read_own(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        return as_fn(table, info), info, text

    from lut_io import load_lut
    return load_lut(path)
