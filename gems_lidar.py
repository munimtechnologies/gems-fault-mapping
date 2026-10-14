"""Fault-scarp features from the 1 m lidar DEM, aggregated to the 100 m grid.

    python gems_lidar.py --links data/1m_DEM_links.csv \
        --grid data/numeric_features.tif --out lidar_features.tif

The GEMS data comes with `1m_DEM_links.csv`, links to USGS 3DEP 1 m lidar
DEM tiles over the GeoDAWN area. Young faults in the Basin and Range leave
scarps: straight, steep steps in the ground a few metres high, much finer
than the 100 m feature grid can show. This module reads each tile at a
chosen resolution (`--factor` DEM pixels per 100 m cell side: 20 = 5 m,
100 = the full 1 m), computes slope, and summarises every 100 m cell:

    lidar_slope_mean   mean slope (degrees)
    lidar_slope_p95    95th percentile slope (lower nearest rank)
    lidar_steep_frac   share of pixels steeper than --steep degrees
    lidar_relief       highest minus lowest elevation (m)
    lidar_coherence    how much the slopes in the cell face one way, from
                       the structure tensor of the gradient: 1 for a
                       straight scarp or a plane, 0 for a bowl or for
                       roughness that faces every way

Cells only partly covered by a tile are left for a neighbouring tile, cells
with under half their pixels valid are NaN, and the first tile to fill a
cell wins (3DEP tiles overlap by a few metres). Tiles must be in UTM zone
11N metres (NAD83 or WGS84; the datum shift is about a metre, far below
100 m). Progress is saved to --state every --save-every tiles, so a long
run can be stopped and resumed. Feed the output to
`gems_baseline.py train --extra-features lidar_features.tif`.

By MV2 of Munim, Inc. Made with AI agents. MIT licence.
"""

import argparse
import csv
import json
import math
import os
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

__version__ = "0.2.0"

FEATURE_NAMES = ["lidar_slope_mean", "lidar_slope_p95", "lidar_steep_frac",
                 "lidar_relief", "lidar_coherence"]
DEFAULT_FACTOR = 20        # 5 m DEM pixels per 100 m cell
DEFAULT_STEEP_DEG = 20.0
MIN_VALID = 0.5
UTM11_EPSG = {26911, 32611, 6340}
NODATA_FLOOR = -1000.0     # DEM values below this are treated as nodata


def read_links(path):
    """Tile URLs or paths from a links CSV, in file order, without repeats.

    The first cell of each row that looks like a URL or a .tif path is
    taken, so the column order and header names do not matter.
    """
    out, seen = [], set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            for cell in row:
                c = cell.strip()
                low = c.lower()
                if (low.startswith(("http://", "https://", "s3://", "/vsi"))
                        or low.endswith((".tif", ".tiff"))):
                    if c not in seen:
                        seen.add(c)
                        out.append(c)
                    break
    return out


def is_utm11(crs):
    """True for UTM zone 11N in metres (EPSG 26911, 32611, 6340 or by name)."""
    if crs is None:
        return False
    try:
        epsg = crs.to_epsg()
    except Exception:  # compound or unusual CRS
        epsg = None
    if epsg in UTM11_EPSG:
        return True
    return "zone 11N" in crs.to_wkt()


def cell_window(tile_bounds, grid_origin, cell=100.0, shape=None):
    """(i0, i1, j0, j1): grid rows and columns, half-open, of the cells that
    lie fully inside tile_bounds (left, bottom, right, top).

    grid_origin is the grid's top-left corner (x0, y0); shape (h, w) clips
    the result to the grid. The window is empty when i1 <= i0 or j1 <= j0.
    """
    left, bottom, right, top = tile_bounds
    x0, y0 = grid_origin
    j0 = int(math.ceil((left - x0) / cell - 1e-9))
    j1 = int(math.floor((right - x0) / cell + 1e-9))
    i0 = int(math.ceil((y0 - top) / cell - 1e-9))
    i1 = int(math.floor((y0 - bottom) / cell + 1e-9))
    if shape is not None:
        h, w = shape
        i0, i1 = max(i0, 0), min(i1, h)
        j0, j1 = max(j0, 0), min(j1, w)
    return i0, i1, j0, j1


def _blocks(a, f):
    """[ny * f, nx * f] -> [ny, nx, f * f], one row of values per cell."""
    ny, nx = a.shape[0] // f, a.shape[1] // f
    a = a[:ny * f, :nx * f]
    return a.reshape(ny, f, nx, f).transpose(0, 2, 1, 3).reshape(ny, nx, f * f)


def cell_features(dem, res, factor, steep_deg=DEFAULT_STEEP_DEG,
                  min_valid=MIN_VALID):
    """float32 [len(FEATURE_NAMES), ny, nx] from a DEM strip of factor-sized cells.

    dem: elevations in metres, NaN for nodata, shape (ny * factor, nx * factor).
    res: DEM pixel size in metres.
    """
    if factor < 2:
        raise ValueError("factor must be at least 2")
    z = np.asarray(dem, dtype=np.float64)
    gy, gx = np.gradient(z, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    ok = np.isfinite(slope)

    s, v = _blocks(slope, factor), _blocks(ok, factor)
    gxb, gyb = _blocks(gx, factor), _blocks(gy, factor)
    zb = _blocks(z, factor)
    n = factor * factor
    cnt = v.sum(axis=2)
    keep = cnt >= min_valid * n
    safe = np.maximum(cnt, 1)

    mean = np.where(v, s, 0.0).sum(axis=2) / safe
    steep = (np.where(v, s, -1.0) > steep_deg).sum(axis=2) / safe
    ranked = np.sort(np.where(v, s, np.inf), axis=2)  # valid values first
    idx = np.clip((0.95 * (cnt - 1)).astype(np.int64), 0, n - 1)
    p95 = np.take_along_axis(ranked, idx[..., None], axis=2)[..., 0]

    vz = np.isfinite(zb)
    relief = (np.where(vz, zb, -np.inf).max(axis=2)
              - np.where(vz, zb, np.inf).min(axis=2))

    gx0, gy0 = np.where(v, gxb, 0.0), np.where(v, gyb, 0.0)
    jxx = (gx0 * gx0).sum(axis=2)
    jyy = (gy0 * gy0).sum(axis=2)
    jxy = (gx0 * gy0).sum(axis=2)
    tr = jxx + jyy
    coh = np.where(tr > 0,
                   np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
                   / np.where(tr > 0, tr, 1.0), 0.0)

    out = np.stack([mean, p95, steep, relief, coh]).astype(np.float32)
    out[:, ~keep] = np.nan
    return out


def empty_output(shape):
    """All-NaN feature array [len(FEATURE_NAMES), h, w] for the whole grid."""
    return np.full((len(FEATURE_NAMES),) + tuple(shape), np.nan,
                   dtype=np.float32)


def process_tile(src, grid_transform, grid_shape, factor, out,
                 steep_deg=DEFAULT_STEEP_DEG, strip_rows=10):
    """Fill `out` with features for the cells fully inside one open tile.

    Reads the tile in strips of `strip_rows` cell rows, resampled so each
    100 m cell is factor x factor pixels. Cells already filled are kept.
    Returns the number of cells newly filled.
    """
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    cell = float(grid_transform.a)
    x0, y0 = float(grid_transform.c), float(grid_transform.f)
    i0, i1, j0, j1 = cell_window(tuple(src.bounds), (x0, y0), cell,
                                 grid_shape)
    if i1 <= i0 or j1 <= j0:
        return 0
    res = cell / factor
    filled = 0
    for r0 in range(i0, i1, strip_rows):
        r1 = min(i1, r0 + strip_rows)
        win = from_bounds(x0 + j0 * cell, y0 - r1 * cell,
                          x0 + j1 * cell, y0 - r0 * cell,
                          transform=src.transform)
        dem = src.read(1, window=win,
                       out_shape=((r1 - r0) * factor, (j1 - j0) * factor),
                       resampling=Resampling.bilinear, masked=True)
        dem = np.ma.filled(dem.astype(np.float32), np.nan)
        dem[dem < NODATA_FLOOR] = np.nan
        feats = cell_features(dem, res, factor, steep_deg)
        target = out[:, r0:r1, j0:j1]
        new = ~np.isfinite(target[0]) & np.isfinite(feats[0])
        target[:, new] = feats[:, new]
        filled += int(new.sum())
    return filled


def write_features(path, out, profile):
    import rasterio

    prof = profile.copy()
    prof.update(count=out.shape[0], dtype="float32", nodata=float("nan"),
                compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out.astype(np.float32))
        for i, name in enumerate(FEATURE_NAMES, 1):
            dst.set_band_description(i, name)


def _resolve(link, links_path):
    if (link.lower().startswith(("http://", "https://", "s3://", "/vsi"))
            or os.path.isabs(link)):
        return link
    return os.path.join(os.path.dirname(os.path.abspath(links_path)), link)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Aggregate 1 m lidar DEM tiles into 100 m fault-scarp "
                    "features on the GEMS grid.")
    ap.add_argument("--links", required=True, help="1m_DEM_links.csv")
    ap.add_argument("--grid", required=True,
                    help="numeric_features.tif (defines the 100 m grid)")
    ap.add_argument("--out", required=True, help="output GeoTIFF")
    ap.add_argument("--factor", type=int, default=DEFAULT_FACTOR,
                    help="DEM pixels per cell side (20 = 5 m, 100 = 1 m)")
    ap.add_argument("--steep", type=float, default=DEFAULT_STEEP_DEG,
                    help="slope in degrees counted as steep")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N tiles this run (0 = all)")
    ap.add_argument("--state", default="lidar_state",
                    help="folder for resumable progress")
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args(argv)

    import rasterio

    links = [_resolve(u, args.links) for u in read_links(args.links)]
    with rasterio.open(args.grid) as g:
        transform, shape = g.transform, (g.height, g.width)
        profile = g.profile.copy()
    os.makedirs(args.state, exist_ok=True)
    part = os.path.join(args.state, "partial.npy")
    done_path = os.path.join(args.state, "done.txt")
    out = np.load(part) if os.path.exists(part) else empty_output(shape)
    done = set()
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as f:
            done = {line.strip() for line in f if line.strip()}
    todo = [u for u in links if u not in done]
    if args.limit:
        todo = todo[:args.limit]

    stats = {"version": __version__, "tiles_listed": len(links),
             "already_done": len(done), "processed": 0, "cells_filled": 0,
             "skipped": []}
    pending = []

    def save():
        np.save(part, out)
        with open(done_path, "a", encoding="utf-8") as f:
            for u in pending:
                f.write(u + "\n")
        del pending[:]

    for n, url in enumerate(todo, 1):
        try:
            with rasterio.open(url) as src:
                if is_utm11(src.crs):
                    filled = process_tile(src, transform, shape, args.factor,
                                          out, args.steep)
                else:
                    filled = 0
                    stats["skipped"].append({"tile": url,
                                             "reason": "not UTM zone 11N"})
        except rasterio.errors.RasterioIOError as exc:
            # not marked done, so the next run tries it again
            stats["skipped"].append({"tile": url, "reason": str(exc)[:200]})
            continue
        stats["processed"] += 1
        stats["cells_filled"] += filled
        pending.append(url)
        print("tile %d/%d: %d cells" % (n, len(todo), filled), file=sys.stderr)
        if len(pending) >= args.save_every:
            save()
    save()
    write_features(args.out, out, profile)
    stats["cells_with_lidar"] = int(np.isfinite(out[0]).sum())
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
