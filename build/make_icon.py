"""
make_icon.py - generate the application icon
===========================================

    python build/make_icon.py            # writes app/icon.png and app/icon.ico

THE PICTURE
-----------
A white rounded tile. On it, an isometric RGB COLOUR GAMUT CUBE, and below it a
narrow saturated strip in the same colours.

The cube is a real gamut cube, not a decorative one: its three visible faces are
the R=1, G=1 and B=1 planes of the RGB unit cube, so the eight corners are the
eight gamut corners. The black corner ends up hidden behind the solid, which is
why the result reads as bright rather than muddy.

Corner assignment. The three 3D axes leave the front-top vertex B; call them
towards L (up-left), R (up-right) and B+d (straight down). Put white at B and the
rest follows from the parallelogram identity T = L + R - B:

    B   = white   (1,1,1)      L   = cyan   (0,1,1)
    R   = magenta (1,0,1)      B+d = yellow (1,1,0)
    T   = blue    (0,0,1)      L+d = green  (0,1,0)
    R+d = red     (1,0,0)      T+d = black  (0,0,0)  <- hidden

That gives the three faces:
    top    blue -> magenta -> white -> cyan      (the B=1 plane)
    right  white -> magenta -> red -> yellow     (the R=1 plane)
    left   white -> cyan -> green -> yellow      (the G=1 plane)

which is exactly the RGB cube's maximum planes, and the shared edges agree on both
faces so there is no seam.

Each face is filled by bilinear interpolation between its four corner colours,
solved by inverting the parallelogram's 2x2 linear map. Everything is drawn at 4x
and box-filtered down, because a hard-edged cube at 16 px is mostly aliasing.

THE PROPORTIONS
---------------
Golden ratio throughout, which is what keeps a stack of two shapes from looking
arbitrary:

    cube width : strip width        = phi   : 1        (the strip is the "shorter"
    strip width : strip height      = phi^3 : 1         part of each pair)
    strip height : gap above it     = phi   : 1
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_PNG = os.path.join(ROOT, "app", "icon.png")
OUT_ICO = os.path.join(ROOT, "app", "icon.ico")

PHI = (1 + math.sqrt(5)) / 2          # 1.6180339887

S = 512                                # final icon size
SS = 4                                 # supersampling factor
W = S * SS

WHITE = np.array([1.0, 1.0, 1.0])
BORDER = np.array([0.886, 0.886, 0.878])   # hairline, so a white tile is not
                                           # invisible on a white background
TILE_RADIUS = 0.21                     # of the tile, matches the old icon


# ------------------------------------------------------------------ the cube
def cube_faces(a):
    """Visible faces as (corners, colours) in screen space, y down.

    a = half-width. The top rhombus is 2a x a (2:1 isometric), the sides drop by a.
    """
    b, d = a / 2.0, a
    T = np.array([0.0, -b])
    R = np.array([a, 0.0])
    B = np.array([0.0, b])
    L = np.array([-a, 0.0])
    down = np.array([0.0, d])

    blue = np.array([0.0, 0.0, 1.0])
    magenta = np.array([1.0, 0.0, 1.0])
    cyan = np.array([0.0, 1.0, 1.0])
    white = np.array([1.0, 1.0, 1.0])
    red = np.array([1.0, 0.0, 0.0])
    yellow = np.array([1.0, 1.0, 0.0])
    green = np.array([0.0, 1.0, 0.0])

    return [
        # (P00,        P10,   P01,       P11,      C00,   C10,     C01,    C11)
        (T,           R,     L,         B,        blue,  magenta, cyan,   white),
        (B,           R,     B + down,  R + down, white, magenta, yellow, red),
        (B,           L,     B + down,  L + down, white, cyan,    yellow, green),
    ]


def paint_faces(canvas, faces, cx, cy):
    """Bilinear-fill each parallelogram. canvas is HxWx3 float, y down."""
    h, w, _ = canvas.shape
    for P00, P10, P01, P11, C00, C10, C01, C11 in faces:
        P00 = P00 + (cx, cy)
        P10 = P10 + (cx, cy)
        P01 = P01 + (cx, cy)
        P11 = P11 + (cx, cy)

        e1 = P10 - P00                      # u direction
        e2 = P01 - P00                      # v direction
        det = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(det) < 1e-9:
            continue

        x0 = int(math.floor(min(P00[0], P10[0], P01[0], P11[0]))) - 1
        x1 = int(math.ceil(max(P00[0], P10[0], P01[0], P11[0]))) + 1
        y0 = int(math.floor(min(P00[1], P10[1], P01[1], P11[1]))) - 1
        y1 = int(math.ceil(max(P00[1], P10[1], P01[1], P11[1]))) + 1
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue

        ys, xs = np.mgrid[y0:y1, x0:x1]
        px = xs + 0.5 - P00[0]
        py = ys + 0.5 - P00[1]
        u = (px * e2[1] - py * e2[0]) / det
        v = (e1[0] * py - e1[1] * px) / det

        inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        if not inside.any():
            continue

        uu = np.clip(u[inside], 0, 1)[:, None]
        vv = np.clip(v[inside], 0, 1)[:, None]
        col = ((1 - uu) * (1 - vv) * C00 + uu * (1 - vv) * C10 +
               (1 - uu) * vv * C01 + uu * vv * C11)
        canvas[y0:y1, x0:x1][inside] = col


# ----------------------------------------------------------------- the strip
def strip_colours(n):
    """Saturated hue sweep: the chromatic corners of the cube, in order.

    Full saturation and value on purpose - these ARE the colours on the cube's
    faces, so the strip is the same palette, laid out flat.

    endpoint=True so the sweep closes: with endpoint=False the last column is
    magenta and the first is red, which leaves a visible step because the two ends
    are meant to meet.
    """
    h = np.linspace(0.0, 1.0, n, endpoint=True)
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = np.zeros_like(f)
    q = 1.0 - f
    t = f
    table = np.stack([  # r, g, b per hue sector
        np.stack([np.ones_like(f), t, p], -1),
        np.stack([q, np.ones_like(f), p], -1),
        np.stack([p, np.ones_like(f), t], -1),
        np.stack([p, q, np.ones_like(f)], -1),
        np.stack([t, p, np.ones_like(f)], -1),
        np.stack([np.ones_like(f), p, q], -1),
    ])
    return np.take_along_axis(table, i[None, :, None], axis=0)[0]


# --------------------------------------------------------------------- build
def build(cube_frac=0.680, strip_aspect=PHI ** 2, gap_ratio=PHI, with_strip=True):
    """Render one variant.

    cube_frac    cube width as a fraction of the canvas
    strip_aspect strip width / strip height
    gap_ratio    gap / strip height
    with_strip   False renders the cube alone, centred - used for the smallest
                 sizes, where a strip a couple of pixels tall is just noise
    """
    cube_w = W * cube_frac                # cube bounding box is square
    a = cube_w / 2.0
    strip_w = cube_w / PHI                # narrower than the cube
    strip_h = (strip_w / strip_aspect) if with_strip else 0.0
    gap = strip_h * gap_ratio

    total_h = cube_w + gap + strip_h
    top = (W - total_h) / 2.0
    cy = top + a                          # cube centre
    cx = W / 2.0

    # --- the tile: white, rounded, with a hairline so it reads on white
    tile = Image.new("L", (W, W), 0)
    ImageDraw.Draw(tile).rounded_rectangle(
        [0, 0, W - 1, W - 1], radius=int(TILE_RADIUS * W), fill=255)
    tile = np.asarray(tile, dtype=np.float64)[:, :, None] / 255.0

    border = Image.new("L", (W, W), 0)
    ImageDraw.Draw(border).rounded_rectangle(
        [0, 0, W - 1, W - 1], radius=int(TILE_RADIUS * W),
        outline=255, width=max(1, int(W * 0.004)))
    border = np.asarray(border, dtype=np.float64)[:, :, None] / 255.0

    canvas = np.ones((W, W, 3), dtype=np.float64)
    canvas = canvas * (1 - border) + BORDER * border

    # --- the cube
    paint_faces(canvas, cube_faces(a), cx, cy)

    # --- the strip
    if with_strip:
        sy0 = int(round(top + cube_w + gap))
        sy1 = int(round(sy0 + strip_h))
        sx0 = int(round(cx - strip_w / 2.0))
        sx1 = int(round(cx + strip_w / 2.0))
        cols = strip_colours(sx1 - sx0)
        canvas[sy0:sy1, sx0:sx1] = cols[None, :, :]

    # --- compose: RGB content, masked by the tile
    rgb = np.clip(canvas, 0, 1)
    out = np.concatenate([rgb, np.ones((W, W, 1))], axis=2)
    out[:, :, 3:4] *= tile
    return out


ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def write_ico(path, images):
    """Assemble an .ico from a list of (size, PIL.Image), by hand.

    Pillow's own ICO writer takes ONE image and resizes it into every slot. That
    silently discards the size-specific artwork this file exists to produce - the
    first attempt came out with a single 128x128 entry and nothing else.

    The container is simple: a 6-byte header, a 16-byte directory entry per image,
    then the payloads. Payloads are PNG rather than the legacy BMP-with-AND-mask;
    Windows has accepted PNG entries since Vista and every size here is 32-bit.
    """
    import struct

    blobs = []
    for size, img in images:
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        blobs.append((size, buf.getvalue()))

    header = struct.pack("<HHH", 0, 1, len(blobs))
    offset = len(header) + 16 * len(blobs)
    entries, payload = b"", b""
    for size, data in blobs:
        dim = 0 if size >= 256 else size        # 0 means 256 in this format
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32,
                               len(data), offset)
        payload += data
        offset += len(data)
    with open(path, "wb") as fh:
        fh.write(header + entries + payload)
# Below this, the strip is under about two pixels tall and stops being a shape.
# Real icon sets redraw the small sizes rather than shrinking the big one; a cube
# alone, given more room, reads far better than a cube plus a speck.
SIMPLE_BELOW = 32


def render(size):
    """Render one raster size, redrawing rather than downscaling."""
    if size <= SIMPLE_BELOW:
        big = build(cube_frac=0.780, with_strip=False)
    else:
        big = build()
    img = Image.fromarray((big * 255 + 0.5).astype(np.uint8), "RGBA")
    return img.resize((size, size), Image.LANCZOS)


def main():
    render(S).save(OUT_PNG)

    write_ico(OUT_ICO, [(n, render(n)) for n in ICO_SIZES])

    print(f"  {OUT_PNG}   {os.path.getsize(OUT_PNG):,} bytes")
    print(f"  {OUT_ICO}   {os.path.getsize(OUT_ICO):,} bytes")
    print(f"  尺寸 {ICO_SIZES}，其中 <= {SIMPLE_BELOW}px 用简化版（只画方块）")
    print()
    print("  proportions:")
    print(f"    cube width : strip width  = {PHI:.4f} : 1")
    print(f"    strip w    : strip h      = {PHI**2:.4f} : 1")
    print(f"    strip h    : gap          = {PHI:.4f} : 1")


if __name__ == "__main__":
    main()
