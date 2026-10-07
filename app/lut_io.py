"""
lut_io.py - LUT file reading and writing for the conversion tool
===============================================================

Extracted from the retired identification module (see
`deprecated_identification/README.md`). Only the pieces the conversion chain
needs live here:

  load_lut             colour-science's readers, wrapped into a sampler
  write_cube           .cube output
  read_filename_hints  reads the LITERAL text a file already carries

The last one deserves a note, because it is the only thing in this repo that
looks like the feature that was removed. It is not the same thing:

  * the removed feature INFERRED a colour space from the LUT's numerical content.
    Measured on 19 real vendor LUTs: 0 correct, 3 confidently wrong.
  * this function READS tokens a human already wrote in the filename or comment
    header. It does not score, rank, or guess. It fills a dropdown, and the user
    confirms.

Reading literal text is reliable but rarely available; inferring from numbers is
always available but was wrong. Only the first is shipped.
"""

import os
import re

import numpy as np
import colour

from lutlib import tetra

# ---------------------------------------------------------------- LUT loading
def load_lut(path):
    """(sampler, info, raw_text).

    The sampler maps points in [0,1]^3 to output code values. `info` describes
    what the file itself declares: LUT kind, grid size and domain.
    """
    text = ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except Exception:
        pass

    lut = colour.read_LUT(path)
    info = {"path": os.path.basename(path),
            "comments": list(getattr(lut, "comments", []) or []),
            "domain": np.asarray(lut.domain).tolist()}

    if isinstance(lut, colour.LUT3D):
        table = np.asarray(lut.table, dtype=np.float64)
        dom = np.asarray(lut.domain, dtype=np.float64)
        info.update(type="LUT3D", size=int(table.shape[0]),
                    domain_lo=[float(dom[0][0]), float(dom[0][1]), float(dom[0][2])],
                    domain_hi=[float(dom[1][0]), float(dom[1][1]), float(dom[1][2])])

        def fn(pts):
            return tetra(table, pts, float(dom[0][0]), float(dom[1][0]))

        return fn, info, text

    if isinstance(lut, colour.LUT3x1D):
        table = np.asarray(lut.table, dtype=np.float64)
        xs = np.linspace(0.0, 1.0, table.shape[0])
        info.update(type="LUT3x1D", size=table.shape[0])

        def fn(pts):
            out = np.empty_like(np.asarray(pts, dtype=np.float64))
            for c in range(3):
                out[:, c] = np.interp(pts[:, c], xs, table[:, c])
            return out

        return fn, info, text

    raise ValueError(f"unsupported LUT type: {type(lut)}")


def write_cube(table, path, name):
    colour.write_LUT(colour.LUT3D(table=np.asarray(table, dtype=np.float64),
                                  name=name,
                                  domain=[[0, 0, 0], [1, 1, 1]]), path)


def cube_text(table, name):
    """The .cube body without touching the filesystem."""
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".cube")
    os.close(fd)
    try:
        write_cube(table, tmp, name)
        with open(tmp, "r", encoding="utf-8") as fh:
            return fh.read()
    finally:
        os.unlink(tmp)


# ------------------------------------------------------------- filename hints
# Token -> the space name it refers to. LONGEST MATCH WINS, so 'flog2' beats
# 'flog' and 's-gamut3.cine' beats 's-gamut3'.
#
# These are matched against the filename and the comment header only. A hit means
# "the file says this", not "the numbers look like this".
HINTS = [
    ("Sony S-Log3 / Venice S-Gamut3.Cine", ["venice s-gamut3.cine", "venice sgamut3.cine"]),
    ("Sony S-Log3 / Venice S-Gamut3", ["venice s-gamut3", "venice sgamut3"]),
    ("Sony S-Log3 / S-Gamut3.Cine", ["s-gamut3.cine", "sgamut3.cine", "slog3cine",
                                     "s-gamut3cine", "venice-sgamut3cine"]),
    ("Sony S-Log3 / S-Gamut3", ["s-gamut3", "sgamut3", "slog3", "s-log3"]),
    ("Sony S-Log2 / S-Gamut", ["slog2", "s-log2", "s-gamut", "sgamut"]),
    ("Canon Log 2 / Cinema Gamut", ["canon log 2", "clog2", "c-log2"]),
    ("Canon Log 3 / Cinema Gamut", ["canon log 3", "clog3", "c-log3"]),
    ("Canon Log / Cinema Gamut", ["canon log", "clog", "c-log"]),
    ("Nikon N-Log / N-Gamut", ["n-log", "nlog", "n-gamut", "ngamut"]),
    ("Fujifilm F-Log2 / F-Gamut C", ["f-log2", "flog2", "f-gamut c", "fgamut c"]),
    ("Fujifilm F-Log / F-Gamut", ["f-log", "flog", "f-gamut", "fgamut"]),
    ("Panasonic V-Log / V-Gamut", ["v-log", "vlog", "v-gamut", "vgamut", "v709"]),
    ("Panasonic Panalog / V-Gamut", ["panalog"]),
    ("ARRI LogC4 / AWG4", ["logc4", "log c4", "awg4"]),
    ("ARRI LogC3 / AWG3", ["logc3", "log c3", "awg3", "logc to"]),
    ("RED Log3G12 / REDWideGamutRGB", ["log3g12"]),
    ("RED Log3G10 / REDWideGamutRGB", ["log3g10", "l3g10"]),
    ("RED REDLogFilm / DRAGONcolor2", ["dragoncolor", "dragon colour", "dragon color"]),
    ("RED REDLogFilm / REDWideGamutRGB", ["redlogfilm"]),
    ("DJI D-Log / D-Gamut", ["d-log", "dlog", "d-gamut", "dgamut", "dlog2"]),
    ("Blackmagic BMD Film Gen5 / Blackmagic Wide Gamut",
     ["bmd film gen 5", "bmd film gen5", "generation 5", "gen 5",
      "blackmagic wide gamut", "bmdfilm", "bmd film", "film to rec709", "film to video"]),
    ("DaVinci Intermediate / DaVinci Wide Gamut",
     ["davinci wide gamut", "davinci intermediate", "dwg"]),
    ("Apple Log 2 / Apple Wide Gamut", ["apple log 2", "applelog2", "apple wide gamut"]),
    ("Apple Log / Rec.2020", ["apple log", "applelog"]),
    ("FilmLight T-Log / E-Gamut", ["t-log", "tlog", "e-gamut", "egamut"]),
    ("GoPro Protune / Protune Native", ["protune", "gopro"]),
]

DISPLAY_TOKENS = ["rec709", "rec.709", "bt709", "bt.709", "709", "srgb",
                  "display", "video", "rgba"]


def _norm(s):
    """Normalise for containment tests only: 'f-log' and 'flog' should be seen as
    the same token, so hyphens and dots are dropped when comparing."""
    return re.sub(r"[-.\s_]", "", s)


def read_filename_hints(path, comments=None, max_hits=4):
    """Literal token matches in the filename and comment header.

    Longest match wins, by greedy containment: a matching token that is a substring
    of a longer matching token is shadowed. That is what makes 'flog2' beat 'flog'
    and 's-gamut3.cine' beat 's-gamut3' - the two cases the token table was written
    around.

    Returns {hints, display_mentioned}. No scores and no ranking, because this is
    exact text matching against text a human wrote; there is nothing to weigh.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    hay = name.lower()
    if comments:
        hay += " " + " ".join(str(c) for c in comments).lower()

    found = []                                    # (space, token) for every match
    for space, toks in HINTS:
        for t in toks:
            if t in hay:
                found.append((space, t))

    # greedy longest-first, dropping anything shadowed by a longer accepted token
    found.sort(key=lambda st: -len(st[1]))
    accepted, out = [], []
    for space, tok in found:
        nt = _norm(tok)
        if any(nt in _norm(a) for a in accepted):
            continue
        if space in [s for s, _ in out]:
            continue
        accepted.append(tok)
        out.append({"space": space, "token": tok})
        if len(out) >= max_hits:
            break

    return {"hints": out,
            "display_mentioned": any(t in hay for t in DISPLAY_TOKENS),
            "matched_text": [h["token"] for h in out]}
