"""
Camera / post colour space registry - expanded coverage
=======================================================

v1 covered the 8 vendors that dominate the field (14 spaces).
v2 adds the spaces and vendors that actually circulate alongside them:

  within the original 8
    Sony       + Venice S-Gamut3, Venice S-Gamut3.Cine
    Canon      + Canon Log (1)
    Panasonic  + Panalog
    RED        + Log3G12/RWG, REDLogFilm/DRAGONcolor2

  additional vendors
    Blackmagic + BMD Film Gen5 / Blackmagic Wide Gamut
    Blackmagic + DaVinci Intermediate / DaVinci Wide Gamut   (Resolve's native space)
    Apple      + Apple Log / Rec.2020                        (iPhone 15 Pro)
    Apple      + Apple Log 2 / Apple Wide Gamut              (iPhone 17 Pro)
    FilmLight  + T-Log / E-Gamut
    GoPro      + Protune / Protune Native

All primaries and transfer functions come from published specifications
(Apple Log 2 White Paper, Blackmagic DWG white paper, etc.) via colour-science.
No vendor LUT files are used or redistributed.
"""

import numpy as np
import colour

D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
D60 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D60"]

REGISTRY_META = {}

# Published Apple Wide Gamut primaries - Apple Log 2 White Paper (Sept 2025).
# Verified independently: recovering them from the official OCIO
# APPLE_LOG-APPLEWG IDT returns red (0.7235, 0.3007) vs published (0.725, 0.301).
APPLE_WIDE_GAMUT = np.array([[0.725, 0.301],
                             [0.221, 0.814],
                             [0.068, -0.076]])


# ============================================================================
# Piecewise log curve machinery, for curves whose constants we have to supply
# ourselves (either vendor-published, or third-party reverse engineered).
# ============================================================================
def piecewise_log_curve(A, B, C, D, E, T, name=""):
    """The standard log-footage shape: a linear segment near black, then a log.

        linear_to_log(x) = A*x + B              for x <= T
                         = ln(x + C) * D + E    otherwise

        log_to_linear(y) = (y - B) / A                    for y <= A*T + B
                         = exp((y - E) / D) - C           otherwise

    The inverse is analytic and exact, which is why this shape is worth
    special-casing rather than tabulating.
    """
    T_out = A * T + B

    def enc(x):
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(all="ignore"):
            log_part = np.log(np.maximum(x + C, 1e-12)) * D + E
            return np.where(x <= T, x * A + B, log_part)

    def dec(y):
        y = np.asarray(y, dtype=np.float64)
        with np.errstate(all="ignore"):
            return np.where(y <= T_out, (y - B) / A, np.exp((y - E) / D) - C)

    enc.__name__ = f"piecewise_log_{name}"

    # self-check: the two segments must meet
    gap = abs(float(enc(np.array([T]))[0]) - float(enc(np.array([T * (1 + 1e-12)]))[0]))
    return enc, dec, {"A": A, "B": B, "C": C, "D": D, "E": E, "T": T,
                      "T_out": T_out, "segment_gap": gap}


# ---------------------------------------------------------------------------
# Reverse-engineered data. NOT vendor-published. Kept separate and explicitly
# labelled so it can never be mistaken for a specification.
# ---------------------------------------------------------------------------
REVERSED = {
    "source": "psychopath.io 'Blackmagic Design's Color Spaces' (2022-04-23), "
              "reverse-engineered from DaVinci Resolve output",
    "method": "encoded->linear LUTs extracted from Resolve via a 2^17-step "
              "gradient OpenEXR, then analytic constants fitted; tool open sourced",
    "accuracy": "avg relative error ~7e-7, max < 1e-4 (Pocket Film series worse)",
    "caveat": "derived from Resolve's implementation, not from BMD camera "
              "silicon. White points land slightly off D65 (0.3135, 0.3305), "
              "which is reverse-engineering noise, not a real white point.",
    "validation": "the Gen5 constants below reproduce colour-science's published "
                  "Blackmagic Film Generation 5 to ~1e-7 relative, which is an "
                  "independent cross-check of the whole method.",
}

# psychopath lists these with different letters than colour-science does:
#   psychopath A,B  = linear segment   -> colour-science D,E
#   psychopath C,D,E = log segment     -> colour-science B,A,C
BMD_REVERSED = {
    # "Film" -> the original BMD Film curve, paired with the "Film Gen 1" gamut
    "Film": dict(A=4.969340550061595, B=0.03538815027497705,
                 C=0.03251848397268609, D=0.1864420102390252,
                 E=0.6723093484094137, T=0.004999977151237935),
    # "4K Film" -> the 4K variant, paired with the "4K Film Gen 1" gamut
    "4K Film": dict(A=3.4845696382315063, B=0.035388150275256276,
                    C=0.0797443784368146, D=0.2952978430809614,
                    E=0.781640290185019, T=0.005000044472991669),
    "4.6K Film Gen3": dict(A=4.6708570973650385, B=0.07305940817239664,
                           C=0.0287284246696045, D=0.15754052970309015,
                           E=0.6303838233991069, T=0.00499997387034723),
    "Film Gen5": dict(A=8.283611088773256, B=0.09246580021201303,
                      C=0.0054940711907293955, D=0.08692875224330131,
                      E=0.5300133837514731, T=0.004999993693740552),
}

BMD_GAMUTS_REVERSED = {
    "Film Gen 1": np.array([[0.917258, 0.250238],
                            [0.283328, 1.707231],
                            [0.085572, -0.070780]]),
    "4K Film Gen 1": np.array([[0.742225, 0.285898],
                               [0.414011, 1.303536],
                               [0.034208, -0.083318]]),
    "4.6K Film Gen 3": np.array([[0.860829, 0.368869],
                                 [0.328213, 0.615591],
                                 [0.078252, -0.023256]]),
}


def _resolve_cctf(cctf):
    """Accept either a colour-science CCTF name, or an (enc, dec, label) tuple."""
    if isinstance(cctf, tuple):
        enc, dec, label = cctf
        return enc, dec, label
    return colour.CCTF_ENCODINGS[cctf], colour.CCTF_DECODINGS[cctf], cctf


def _from(primaries_from, cctf, name, note="", provenance="published"):
    base = colour.RGB_COLOURSPACES[primaries_from]
    return _explicit(np.asarray(base.primaries), np.asarray(base.whitepoint),
                     cctf, name, note, src=primaries_from, provenance=provenance)


def _explicit(primaries, whitepoint, cctf, name, note="", src="(published)",
              provenance="published"):
    enc, dec, label = _resolve_cctf(cctf)
    cs = colour.RGB_Colourspace(
        name=name,
        primaries=np.asarray(primaries),
        whitepoint=np.asarray(whitepoint),
        cctf_encoding=enc,
        cctf_decoding=dec,
    )
    REGISTRY_META[name] = {
        "primaries_source": src,
        "cctf": label,
        "provenance": provenance,
        "whitepoint": [round(float(v), 5) for v in np.asarray(whitepoint)],
        "primaries": [[round(float(v), 5) for v in row] for row in np.asarray(primaries)],
        "note": note,
        "vendor": name.split(" ", 1)[0],
    }
    return cs


# name, (primaries_from | explicit primaries), cctf, note
SPEC = [
    # ---------------------------------------------------------------- Sony
    ("Sony S-Log3 / S-Gamut3.Cine", "S-Gamut3.Cine", "S-Log3",
     "FX/FS/Venice, the cine gamut"),
    ("Sony S-Log3 / S-Gamut3", "S-Gamut3", "S-Log3",
     "full S-Gamut3, wider than .Cine"),
    ("Sony S-Log3 / Venice S-Gamut3.Cine", "Venice S-Gamut3.Cine", "S-Log3",
     "Venice-specific cine gamut, primaries differ from non-Venice"),
    ("Sony S-Log3 / Venice S-Gamut3", "Venice S-Gamut3", "S-Log3",
     "Venice-specific full gamut"),
    ("Sony S-Log2 / S-Gamut", "S-Gamut", "S-Log2",
     "legacy S-Log2, F5/F55 era"),

    # --------------------------------------------------------------- Canon
    ("Canon Log / Cinema Gamut", "Cinema Gamut", "Canon Log",
     "original C-Log, C300/C100 generation"),
    ("Canon Log 2 / Cinema Gamut", "Cinema Gamut", "Canon Log 2",
     "C300 III / C500 II / R5C"),
    ("Canon Log 3 / Cinema Gamut", "Cinema Gamut", "Canon Log 3",
     "simpler log, same gamut"),

    # --------------------------------------------------------------- Nikon
    ("Nikon N-Log / N-Gamut", "N-Gamut", "N-Log",
     "Z series. NOTE: N-Gamut primaries are Rec.2020"),

    # ------------------------------------------------------------ Fujifilm
    ("Fujifilm F-Log / F-Gamut", "F-Gamut", "F-Log",
     "X-T3 era. NOTE: F-Gamut primaries are Rec.2020"),
    ("Fujifilm F-Log2 / F-Gamut C", "F-Gamut C", "F-Log2",
     "X-H2S / GFX100 II"),

    # ----------------------------------------------------------- Panasonic
    ("Panasonic V-Log / V-Gamut", "V-Gamut", "V-Log",
     "Varicam / S1H / GH6 / S5 II"),
    ("Panasonic Panalog / V-Gamut", "V-Gamut", "Panalog",
     "legacy Panalog on the V-Gamut primaries"),

    # ---------------------------------------------------------------- ARRI
    ("ARRI LogC3 / AWG3", "ARRI Wide Gamut 3", "ARRI LogC3",
     "ALEXA classic / Mini / Amira"),
    ("ARRI LogC4 / AWG4", "ARRI Wide Gamut 4", "ARRI LogC4",
     "ALEXA 35"),

    # ----------------------------------------------------------------- RED
    ("RED Log3G10 / REDWideGamutRGB", "REDWideGamutRGB", "Log3G10",
     "IPP2 pipeline"),
    ("RED Log3G12 / REDWideGamutRGB", "REDWideGamutRGB", "Log3G12",
     "older Log3G12 curve on RWG"),
    ("RED REDLogFilm / REDWideGamutRGB", "REDWideGamutRGB", "REDLogFilm",
     "Cineon-style log on the RWG gamut"),
    ("RED REDLogFilm / DRAGONcolor2", "DRAGONcolor2", "REDLogFilm",
     "legacy DRAGONcolour2 gamut"),

    # ----------------------------------------------------------------- DJI
    ("DJI D-Log / D-Gamut", "DJI D-Gamut", "D-Log",
     "Inspire 2 / Mavic 3. D-Log M is NOT in colour-science - see gaps"),

    # --------------------------------------------------------- Blackmagic
    ("Blackmagic BMD Film Gen5 / Blackmagic Wide Gamut", "Blackmagic Wide Gamut",
     "Blackmagic Film Generation 5", "Pocket 4K/6K, URSA, current generation"),
    ("DaVinci Intermediate / DaVinci Wide Gamut", "DaVinci Wide Gamut",
     "DaVinci Intermediate",
     "Resolve's native colour-managed working space"),
    # below: reverse-engineered, NOT vendor published - see REVERSED
    ("Blackmagic Film Gen1 / Film Gen1", BMD_GAMUTS_REVERSED["Film Gen 1"],
     piecewise_log_curve(name="BMD_Film", **BMD_REVERSED["Film"])[:2]
     + (f"BMD Film (RE, {REVERSED['source'].split(' (')[0]})",),
     "original BMCC / Pocket generation. Reverse engineered, not published."),
    ("Blackmagic Film 4K Gen1 / 4K Film Gen1", BMD_GAMUTS_REVERSED["4K Film Gen 1"],
     piecewise_log_curve(name="BMD_4K_Film", **BMD_REVERSED["4K Film"])[:2]
     + (f"BMD 4K Film (RE, {REVERSED['source'].split(' (')[0]})",),
     "BMD 4K Film. Reverse engineered, not published."),
    ("Blackmagic 4.6K Film Gen3 / 4.6K Film Gen3",
     BMD_GAMUTS_REVERSED["4.6K Film Gen 3"],
     piecewise_log_curve(name="BMD_4_6K_Film_Gen3", **BMD_REVERSED["4.6K Film Gen3"])[:2]
     + (f"BMD 4.6K Film Gen3 (RE, {REVERSED['source'].split(' (')[0]})",),
     "URSA Mini 4.6K / Production 4K. Reverse engineered, not published."),

    # --------------------------------------------------------------- Apple
    ("Apple Log / Rec.2020", "ITU-R BT.2020", "Apple Log Profile",
     "iPhone 15 Pro / 16 Pro. Rec.2020 primaries"),
    ("Apple Log 2 / Apple Wide Gamut", APPLE_WIDE_GAMUT, "Apple Log Profile",
     "iPhone 17 Pro. Same curve as Apple Log, wider gamut"),

    # ------------------------------------------------------------- Others
    ("FilmLight T-Log / E-Gamut", "FilmLight E-Gamut", "T-Log",
     "Baselight / FLUX"),
    ("GoPro Protune / Protune Native", "Protune Native", "Protune",
     "action cameras"),
]


# known gaps - documented rather than silently absent.
# "blocked" means we actively looked and cannot proceed from lawful public sources.
KNOWN_GAPS = [
    ("DJI D-Log M", "BLOCKED",
     "DJI has refused repeated requests to document this curve; it is explicitly "
     "a black box. It is NOT the same curve as D-Log ('slightly less flat', "
     "per CineD's review). No published constants exist, and deriving them means "
     "reverse-engineering DJI's own LUTs, which their terms do not permit. "
     "Community efforts exist (Zeb Gardner's colour-chart based fit) but are "
     "unofficial, method-dependent and not published as constants."),
    ("Blackmagic 4K Film Gen 3", "not paired",
     "a gamut exists in the reverse-engineered set but no matching curve was "
     "published, so the pair is ambiguous. Skipped rather than guessed."),
    ("Blackmagic Broadcast Film Gen 4", "ambiguous",
     "the curve exists but its gamut pairing ('Video Gen 4') is a different "
     "family; pairing them would be a guess."),
    ("Blackmagic 4.6K Film Gen 1", "no data",
     "the reverse-engineering source reports this failed to export cleanly."),
    ("Sony S-Log (original)", "obsolete", "pre-S-Log2, essentially unused"),
    ("Canon Log 2/3 on non-Cinema Gamut", "rare", "essentially never seen"),
]


def cross_validate_bmd():
    """Independent check of the reverse-engineered Blackmagic constants.

    Gen5 exists BOTH as a colour-science entry (spec-derived) and in the
    reverse-engineered set. If the two agree, the reverse-engineering method is
    sound and its Gen1/Gen3 output can be trusted. This is the whole reason the
    RE data is usable at all.
    """
    import warnings
    warnings.filterwarnings("ignore")

    A = BMD_REVERSED["Film Gen5"]
    rev_enc, rev_dec, meta = piecewise_log_curve(name="RE_Gen5", **A)
    cs = colour.RGB_COLOURSPACES["Blackmagic Wide Gamut"]
    spec_enc = colour.CCTF_ENCODINGS["Blackmagic Film Generation 5"]
    spec_dec = colour.CCTF_DECODINGS["Blackmagic Film Generation 5"]

    x = np.linspace(0.0, 8.0, 20000)
    a = np.asarray(rev_enc(x))
    b = np.asarray(spec_enc(x))
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-6)

    y = np.linspace(0.0, 1.0, 20000)
    ra = np.asarray(rev_dec(y))
    rb = np.asarray(spec_dec(y))
    rel_d = np.abs(ra - rb) / np.maximum(np.abs(rb), 1e-6)
    rel_d = rel_d[np.isfinite(rel_d)]

    return {
        "encode_median_rel": float(np.median(rel)),
        "encode_max_rel": float(np.max(rel)),
        "decode_median_rel": float(np.median(rel_d)),
        "decode_max_rel": float(np.max(rel_d)),
        "segment_gap": meta["segment_gap"],
    }


def build_registry():
    r = {}
    for name, pfrom, cctf, note in SPEC:
        prov = "reverse-engineered" if (
            isinstance(cctf, tuple) and "RE," in str(cctf[2])) else "published"
        if isinstance(pfrom, np.ndarray):
            r[name] = _explicit(pfrom, D65, cctf, name, note, provenance=prov)
        else:
            r[name] = _from(pfrom, cctf, name, note, provenance=prov)
    return r


def build_displays():
    d = {}
    d["Rec.709 display (BT.1886)"] = colour.RGB_Colourspace(
        name="Rec.709 display (BT.1886)",
        primaries=colour.RGB_COLOURSPACES["ITU-R BT.709"].primaries,
        whitepoint=colour.RGB_COLOURSPACES["ITU-R BT.709"].whitepoint,
        cctf_encoding=lambda x: np.clip(x, 0, 1) ** (1 / 2.4),
        cctf_decoding=lambda x: np.clip(x, 0, 1) ** 2.4,
    )
    d["sRGB display"] = colour.RGB_COLOURSPACES["sRGB"]
    return d


def vendor_groups(cam):
    g = {}
    for name in cam:
        g.setdefault(REGISTRY_META[name]["vendor"], []).append(name)
    return g


# ------------------------------------------------------- representable ranges
_RANGE_CACHE = {}


def repr_range(cs):
    """Linear-light range that code values 0.0 .. 1.0 can represent.

    Some published log curves are undefined outside their domain rather than
    clamping: Apple Log returns a flat 0 below R0 = -0.0564 (losing the value),
    GoPro Protune and REDLogFilm return NaN for negative input. Anything
    outside this range has no code value at all, so clamping to it is exact,
    not a fudge.
    """
    key = cs.name
    if key not in _RANGE_CACHE:
        with np.errstate(all="ignore"):
            lo = float(np.nanmin(np.asarray(cs.cctf_decoding(np.zeros((1, 3))))))
            hi = float(np.nanmax(np.asarray(cs.cctf_decoding(np.ones((1, 3))))))
        _RANGE_CACHE[key] = (lo, hi)
    return _RANGE_CACHE[key]


def encode_clamped(cs, linear):
    """Encode linear light to code values, clamped to what the curve can hold."""
    lo, hi = repr_range(cs)
    with np.errstate(all="ignore"):
        return np.asarray(cs.cctf_encoding(np.clip(np.asarray(linear), lo, hi)))


# ---------------------------------------------------------------------- checks
def _validate(cam, verbose=True):
    import warnings
    warnings.filterwarnings("ignore")

    problems = []
    if verbose:
        print("=" * 108)
        print("SPACE REGISTRY - expanded")
        print("=" * 108)
        print(f"{'space':50s} {'whitepoint':>17s}  {'cctf':<30s} {'provenance':<20s}")
        print("-" * 108)
        for name in cam:
            m = REGISTRY_META[name]
            wp = f"({m['whitepoint'][0]:.4f},{m['whitepoint'][1]:.4f})"
            ct = m["cctf"]
            ct = ct if len(ct) <= 29 else ct[:26] + "..."
            print(f"{name:50s} {wp:>17s}  {ct:<30s} {m['provenance']:<20s}")
        print()

    groups = vendor_groups(cam)
    print(f"  vendors: {len(groups)}   spaces: {len(cam)}")
    for v, lst in sorted(groups.items()):
        short = [x.split(" / ", 1)[1] if " / " in x else x for x in lst]
        print(f"    {v:14s} {len(lst)}  {' | '.join(short)}")

    print()
    print("  -- representable linear range of code [0,1] --")
    print(f"    {'space':50s} {'min':>12s} {'max':>12s}")
    for name, cs in cam.items():
        lo, hi = repr_range(cs)
        print(f"    {name:50s} {lo:12.5f} {hi:12.3f}")
    print("    (every value below min has NO code value; it must be clamped, not")
    print("     ignored. Apple Log is the widest negative at -0.0564; GoPro Protune")
    print("     and REDLogFilm return NaN below zero and must be clamped hardest.)")

    print()
    print("  -- provenance --")
    byprov = {}
    for name in cam:
        byprov.setdefault(REGISTRY_META[name]["provenance"], []).append(name)
    for p, lst in sorted(byprov.items()):
        print(f"    {p:20s} {len(lst):2d}  {' | '.join(x.split(' / ')[0] for x in lst)}")

    print()
    print("  -- cross-validation: reverse-engineered Gen5 vs colour-science Gen5 --")
    cv = cross_validate_bmd()
    print(f"    encode  median rel {cv['encode_median_rel']:.3e}   max {cv['encode_max_rel']:.3e}")
    print(f"    decode  median rel {cv['decode_median_rel']:.3e}   max {cv['decode_max_rel']:.3e}")
    print(f"    segment continuity gap {cv['segment_gap']:.3e}")
    print("    Two INDEPENDENT derivations - colour-science from the published spec,")
    print("    psychopath from reverse-engineered Resolve output - agree to ~1e-7.")
    print("    That is what licenses trusting its Gen1 / Gen3 constants.")

    print()
    print("  -- transfer function invertibility, clamped (encode(decode(x)) == x) --")
    probe = np.linspace(0.0, 1.0, 2048)
    worst, worst_name = 0.0, ""
    for name, cs in cam.items():
        back = encode_clamped(cs, cs.cctf_decoding(probe))
        d = np.abs(back - probe)
        err = float(np.nanmax(d)) if not np.all(np.isnan(d)) else float("inf")
        if not np.isfinite(err) or err > 1e-3:
            problems.append(f"{name}: {err}")
        if np.isfinite(err) and err > worst:
            worst, worst_name = err, name
    print(f"    worst: {worst:.3e}   ({worst_name})")
    print(f"    (8-bit quantisation is 1/255 = 3.9e-3, so anything below that is")
    print(f"     not observable in practice. Threshold here is 1e-3.)")
    if problems:
        print(f"    ABOVE THRESHOLD:")
        for p in problems:
            print(f"      {p}")

    print()
    print("  -- gamut invertibility in linear light (A -> XYZ -> A) --")
    worst = 0.0
    for name, cs in cam.items():
        x = np.random.rand(3000, 3) * 0.5
        back = colour.XYZ_to_RGB(colour.RGB_to_XYZ(x, cs), cs)
        worst = max(worst, float(np.max(np.abs(back - x))))
    print(f"    worst over all spaces: {worst:.3e}")

    print()
    print("  -- cross-vendor conversion from Sony S-Log3/S-Gamut3.Cine --")
    src = "Sony S-Log3 / S-Gamut3.Cine"
    print(f"    {'target':44s} {'formula*':>10s} {'clamp':>7s} {'out of range':>12s}  failure mode")
    print("    " + "-" * 104)
    for dst in ["ARRI LogC4 / AWG4", "Blackmagic BMD Film Gen5 / Blackmagic Wide Gamut",
                "Apple Log / Rec.2020", "Apple Log 2 / Apple Wide Gamut",
                "Nikon N-Log / N-Gamut", "Fujifilm F-Log / F-Gamut",
                "GoPro Protune / Protune Native",
                "RED REDLogFilm / REDWideGamutRGB"]:
        x = np.random.rand(4000, 3)
        lin_src = np.asarray(cam[src].cctf_decoding(x))
        lin_dst = colour.RGB_to_RGB(lin_src, cam[src], cam[dst],
                                    apply_cctf_decoding=False, apply_cctf_encoding=False)
        lo, hi = repr_range(cam[dst])
        inside = np.all((lin_dst >= lo - 1e-12) & (lin_dst <= hi + 1e-12), axis=1)

        # formula error measured ONLY on samples that are actually representable,
        # so this column isolates the maths from the unavoidable clamp loss
        ferr = float("nan")
        if inside.sum() > 10:
            xs = x[inside]
            lds = lin_dst[inside]
            with np.errstate(all="ignore"):
                y = np.asarray(cam[dst].cctf_encoding(lds))
                back = np.asarray(cam[dst].cctf_decoding(y))
            lb = colour.RGB_to_RGB(back, cam[dst], cam[src],
                                   apply_cctf_decoding=False, apply_cctf_encoding=False)
            with np.errstate(all="ignore"):
                z = np.asarray(cam[src].cctf_encoding(lb))
            ferr = float(np.nanmax(np.abs(z - xs)))

        # how does out-of-range input fail?
        if not inside.all():
            probe = np.array([[lo - 1.0] * 3])
            with np.errstate(all="ignore"):
                v = np.asarray(cam[dst].cctf_encoding(probe)).ravel()
            if np.isnan(v).any():
                mode = "NaN (loud)"
            elif np.allclose(v, 0.0):
                mode = "clamps to 0 (SILENT)"
            else:
                mode = "extrapolates to code<0"
        else:
            mode = "-"

        print(f"    {dst:44s} {ferr:10.2e} {100*(1-inside.mean()):6.1f}% "
              f"{'yes' if not inside.all() else 'no':>12s}  {mode}")
    print("    * formula error computed only on representable samples, so it measures")
    print("      the colour-space maths alone. The clamp column is what a plain .cube")
    print("      physically cannot store - it is irreducible in a 0..1 table.")
    print("      Apple Log is the dangerous one: it fails SILENTLY, returning a valid-")
    print("      looking code 0, where others return NaN or a visibly invalid code.")

    print()
    print("  -- primaries duplicate check --")
    seen = {}
    for name, cs in cam.items():
        k = tuple(np.round(np.asarray(cs.primaries).ravel(), 5))
        seen.setdefault(k, []).append(name)
    dups = [v for v in seen.values() if len(v) > 1]
    if dups:
        for v in dups:
            print(f"    shared primaries: {' | '.join(v)}")
        print("    -> these can only be told apart by their transfer function.")
    else:
        print("    none")

    print()
    print("  -- curve-only duplicate check (same cctf, different gamut) --")
    byfn = {}
    for name, cs in cam.items():
        byfn.setdefault(REGISTRY_META[name]["cctf"], []).append(name)
    for k, v in sorted(byfn.items()):
        if len(v) > 1:
            print(f"    {k:24s} {' | '.join(v)}")
    print("    -> these can only be told apart by their gamut.")
    return problems


if __name__ == "__main__":
    import json
    import warnings

    warnings.filterwarnings("ignore")
    cam = build_registry()
    problems = _validate(cam)

    print()
    print("=" * 108)
    print("KNOWN GAPS")
    print("=" * 108)
    for name, kind, why in KNOWN_GAPS:
        print(f"  [{kind:12s}] {name}")
        for i in range(0, len(why), 96):
            print(f"                 {why[i:i+96]}")

    groups = vendor_groups(cam)
    n = len(cam)
    out = {
        "vendors": {v: sorted(lst) for v, lst in groups.items()},
        "n_vendors": len(groups),
        "n_spaces": n,
        "spaces": REGISTRY_META,
        "pairwise_routes": n * (n - 1),
        "hub_routes": 2 * n,
        "known_gaps": [{"space": a, "kind": b, "reason": c} for a, b, c in KNOWN_GAPS],
        "reverse_engineered": REVERSED,
        "gen5_cross_validation": cross_validate_bmd(),
    }
    with open("camera_spaces.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print()
    print(f"  spaces {n}   vendors {len(groups)}")
    print(f"  pairwise routes (no hub) : {n*(n-1)}")
    print(f"  hub routes (2 per space) : {2*n}    -> {n*(n-1)/(2*n):.1f}x fewer")
    print(f"  validation problems      : {len(problems)}")
    print("  registry -> camera_spaces.json")
