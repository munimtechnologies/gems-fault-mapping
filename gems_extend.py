"""Fault-tip extension candidates for the DOE GEMS Prize.

The scoring masks the known USGS/INGENIOUS fault pixels exactly, and a "new
fault" is any fault pixel not already in that set, which includes extensions
past a mapped tip, splays and parallel strands (forum topics 11516, 11536).
So the pixels just beyond each known fault's ends are a natural place to
predict new fault, whatever the model says. This module finds those tips and
rasterises a short extension along each fault's local direction. The result
can be OR-ed into a submission (`gems_baseline.py extend`) or used as an
extra feature. How long the extension should be is a leaderboard question.

Method (pure numpy, works on the raw 100 m raster, no thinning needed):
  1. For every fault pixel, take the fault pixels in a (2r + 1)^2 window and
     compute their centroid offset from the pixel. Mid-line pixels have an
     offset near zero; a tip's offset is about r / 2 pointing into the line.
  2. A tip is a fault pixel that passes three checks: its 8 neighbours hold
     exactly one run of fault pixels (so the line leaves it on one side
     only; mid-line pixels have two runs, junctions three), the window holds
     at least r + 1 fault pixels (enough line for a direction), and the
     centroid offset is longer than `frac * r` (which rules out the edges
     of lines two pixels thick, whose offset points sideways by only 0.5).
  3. Step away from the tip, opposite to the offset, one pixel per step in
     the dominant axis (so steps are distinct and 8-connected), marking
     pixels that are inside the grid and not already known fault.

By MV2 of Munim, Inc. Made with AI agents. MIT licence.
"""

import argparse
import json
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

__version__ = "0.2.0"

DEFAULT_RADIUS = 3      # window half-size in pixels (3 px = 300 m)
DEFAULT_FRAC = 0.4      # tip if |centroid offset| > frac * radius
DEFAULT_LENGTH = 3      # extension length in pixels

# The 8 neighbours in circular order: N, NE, E, SE, S, SW, W, NW.
RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def centroid_offsets(known, radius=DEFAULT_RADIUS):
    """(count, dy, dx): fault pixels in the window and their mean offset.

    All three are float32 [H, W]; dy, dx are 0 where count is 0.
    """
    k = np.asarray(known, dtype=bool)
    h, w = k.shape
    pad = np.pad(k, radius).astype(np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    sy = np.zeros((h, w), dtype=np.float32)
    sx = np.zeros((h, w), dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            win = pad[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            count += win
            sy += win * np.float32(dy)
            sx += win * np.float32(dx)
    safe = np.where(count > 0, count, 1.0)
    return count, sy / safe, sx / safe


def ring_runs(known):
    """int32 [H, W]: runs of fault pixels around each pixel's 8 neighbours.

    0 for an isolated pixel (or one fully surrounded), 1 at a line end, 2 in
    the middle of a line, 3 or more at a junction. Outside the grid counts
    as not fault.
    """
    k = np.asarray(known, dtype=bool)
    h, w = k.shape
    pad = np.pad(k, 1)
    nb = [pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in RING]
    runs = np.zeros((h, w), dtype=np.int32)
    for i in range(len(RING)):
        runs += nb[i] & ~nb[i - 1]
    return runs


def _tip_mask(k, count, dy, dx, radius, frac):
    # Chessboard length of the offset: a diagonal line then behaves like a
    # straight one (Euclidean length would flag the pixel next to its tip).
    dist = np.maximum(np.abs(dy), np.abs(dx))
    return (k & (ring_runs(k) == 1) & (count >= radius + 1)
            & (dist > frac * radius))


def tips(known, radius=DEFAULT_RADIUS, frac=DEFAULT_FRAC):
    """Boolean [H, W]: fault pixels at the end of a line.

    See the module docstring for the three checks. Isolated pixels and very
    short segments (under radius + 1 pixels) do not qualify, which is
    intended: their direction is not reliable.
    """
    k = np.asarray(known, dtype=bool)
    count, dy, dx = centroid_offsets(k, radius)
    return _tip_mask(k, count, dy, dx, radius, frac)


def extension_steps(dy, dx, length):
    """Pixel offsets (y, x) walking `length` steps away from the line body.

    (dy, dx) is the centroid offset at the tip (pointing into the line); the
    walk goes the other way, scaled so the dominant axis moves one pixel per
    step, which keeps the steps distinct and 8-connected.
    """
    ay, ax = -float(dy), -float(dx)
    big = max(abs(ay), abs(ax))
    if big == 0:
        return []
    ay, ax = ay / big, ax / big
    out = []
    for s in range(1, int(length) + 1):
        out.append((int(round(ay * s)), int(round(ax * s))))
    return out


def extend_tips(known, length=DEFAULT_LENGTH, radius=DEFAULT_RADIUS,
                frac=DEFAULT_FRAC, valid=None):
    """Boolean [H, W] of extension pixels beyond every tip; never on known.

    `valid` (optional boolean [H, W]) limits extensions to the data area.
    """
    k = np.asarray(known, dtype=bool)
    h, w = k.shape
    count, oy, ox = centroid_offsets(k, radius)
    is_tip = _tip_mask(k, count, oy, ox, radius, frac)
    out = np.zeros((h, w), dtype=bool)
    for y, x in zip(*np.nonzero(is_tip)):
        for sy, sx in extension_steps(oy[y, x], ox[y, x], length):
            yy, xx = y + sy, x + sx
            if 0 <= yy < h and 0 <= xx < w and not k[yy, xx]:
                out[yy, xx] = True
    if valid is not None:
        out &= np.asarray(valid, dtype=bool)
    return out


def summarize(known, ext):
    k = np.asarray(known, dtype=bool)
    e = np.asarray(ext, dtype=bool)
    return {"version": __version__, "known_pixels": int(k.sum()),
            "tips": int(tips(k).sum()), "extension_pixels": int(e.sum()),
            "overlap_with_known": int((k & e).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Rasterise short extensions beyond known fault tips.")
    ap.add_argument("--labels", required=True, help="labels.tif (known faults)")
    ap.add_argument("--out", required=True, help="output GeoTIFF (float32 0/1)")
    ap.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    ap.add_argument("--frac", type=float, default=DEFAULT_FRAC)
    args = ap.parse_args(argv)

    import rasterio

    with rasterio.open(args.labels) as src:
        raw = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    valid = np.isfinite(raw) & (raw != -1)
    known = np.where(np.isfinite(raw), raw, 0.0) >= 1.0
    ext = extend_tips(known, args.length, args.radius, args.frac, valid)
    out = np.where(valid, ext.astype(np.float32), np.nan).astype(np.float32)
    profile.update(count=1, dtype="float32", nodata=float("nan"),
                   compress="deflate")
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(out, 1)
    print(json.dumps(summarize(known, ext), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
