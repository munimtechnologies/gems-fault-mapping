"""Outside grids and point layers on the GEMS grid, as extra features.

    python gems_regrid.py list   --src USGScontributions_forNVML_dataRelease2021.zip
    python gems_regrid.py grid   --grid data/numeric_features.tif \
                                 --src USGScontributions_forNVML_dataRelease2021.zip::grv_filthg \
                                 --src hfnvml_est.asc --out usgs_grids.tif
    python gems_regrid.py points --grid data/numeric_features.tif \
                                 --points grav_intersections.geojson --out usgs_points.tif
    python gems_baseline.py train --features data/numeric_features.tif --labels data/labels.tif \
                                 --extra-features usgs_grids.tif --extra-features usgs_points.tif

`grid` samples rasters onto the competition grid (100 m, EPSG:32611), one
band per source, NaN outside the data or the source. Sources are ESRI ASCII
grids (.asc, with a .prj beside them), .asc files inside a zip (or a zip in
a zip) given as ZIP::NAME, or any GeoTIFF rasterio can open. They must be in
UTM zone 11N metres (NAD83 or WGS84; the difference is about a metre), as
the USGS Nevada geothermal data release and the NBMG structural-setting
rasters are; anything else is refused rather than guessed. Sampling is
bilinear between source cell centres (nodata neighbours are left out and the
rest re-weighted), or nearest with --method nearest, which suits class maps.

`points` turns point layers (GeoJSON from gems_shapes.py, for example
lineation intersections and end points, gravity maximum spots or heat-flow
wells) into two bands each: distance to the nearest point in px (Euclidean
between cell centres with scipy, chessboard without it), capped at --cap,
and the number of points in the (2 radius + 1) px square around each cell.
Points up to --cap px outside the grid still count, so the edges are right.

By MV2 of Munim, Inc. Made with AI agents. MIT licence.
"""

import argparse
import collections
import io
import json
import math
import os
import sys
import zipfile

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

import gems_external
import gems_metric
import gems_shapes

__version__ = gems_metric.__version__

ASC_KEYS = ("ncols", "nrows", "xllcorner", "yllcorner", "xllcenter",
            "yllcenter", "cellsize", "dx", "dy", "nodata_value")
METRE_EPSG = (32611, 26911, 6340)
NODATA_FLOOR = gems_external.NODATA_FLOOR
CHUNK_ROWS = 256
DEFAULT_CAP = 50     # px (5 km)
DEFAULT_RADIUS = 20  # px (a 4.1 km square)

Source = collections.namedtuple("Source", "name values left top dx dy")


def parse_asc_header(lines):
    """(header dict with lower-case keys, number of header lines)."""
    head = {}
    n = 0
    for line in lines:
        parts = line.split()
        if len(parts) != 2 or parts[0].lower() not in ASC_KEYS:
            break
        head[parts[0].lower()] = float(parts[1])
        n += 1
    return head, n


def asc_geometry(head):
    """(left, top, dx, dy, nrows, ncols, nodata or None) of an ASCII grid."""
    for key in ("ncols", "nrows"):
        if key not in head:
            raise ValueError("ASCII grid header lacks %s" % key)
    dx = head.get("dx", head.get("cellsize"))
    dy = head.get("dy", head.get("cellsize"))
    if dx is None or dy is None:
        raise ValueError("ASCII grid header lacks cellsize")
    nrows, ncols = int(head["nrows"]), int(head["ncols"])
    if "xllcorner" in head:
        left = head["xllcorner"]
    elif "xllcenter" in head:
        left = head["xllcenter"] - dx / 2.0
    else:
        raise ValueError("ASCII grid header lacks xllcorner")
    if "yllcorner" in head:
        bottom = head["yllcorner"]
    elif "yllcenter" in head:
        bottom = head["yllcenter"] - dy / 2.0
    else:
        raise ValueError("ASCII grid header lacks yllcorner")
    return (left, bottom + nrows * dy, dx, dy, nrows, ncols,
            head.get("nodata_value"))


def read_asc_text(text):
    """(values float64 [nrows, ncols] with NaN for nodata, (left, top, dx, dy)).

    Rows may be wrapped over several lines; only the total count must match.
    """
    head, pos = {}, 0
    while pos < len(text):
        end = text.find("\n", pos)
        line = text[pos:] if end < 0 else text[pos:end]
        got, n = parse_asc_header([line])
        if not n:
            break
        head.update(got)
        pos = len(text) if end < 0 else end + 1
    left, top, dx, dy, nrows, ncols, nodata = asc_geometry(head)
    chunks = [np.array(line.split(), dtype=np.float64)
              for line in text[pos:].splitlines() if line.strip()]
    values = np.concatenate(chunks) if chunks else np.zeros(0)
    if values.size != nrows * ncols:
        raise ValueError("ASCII grid has %d values, header says %d x %d"
                         % (values.size, nrows, ncols))
    if nodata is not None:
        values[np.abs(values - nodata) <= 1e-6 * max(1.0, abs(nodata))] = np.nan
    values[values < NODATA_FLOOR] = np.nan
    return values.reshape(nrows, ncols), (left, top, dx, dy)


def _zip_asc(path, name):
    """(text, prj text or None) of NAME.asc in a zip, one zip level deep."""
    want = os.path.splitext(os.path.basename(name))[0].lower()

    def search(zf, depth):
        names = zf.namelist()
        for n in names:
            stem = os.path.splitext(os.path.basename(n))[0].lower()
            if n.lower().endswith(".asc") and stem == want:
                prj = gems_shapes._sibling(zf, n, ".prj")
                return (zf.read(n).decode("latin-1"),
                        prj.decode("latin-1") if prj else None)
        if depth == 0:
            for n in names:
                if n.lower().endswith(".zip"):
                    inner = zipfile.ZipFile(io.BytesIO(zf.read(n)))
                    found = search(inner, 1)
                    if found:
                        return found
        return None

    with zipfile.ZipFile(path) as zf:
        found = search(zf, 0)
    if found is None:
        raise SystemExit("no %s.asc in %s" % (want, path))
    return found


def list_zip_grids(path):
    """[(member, ncols, nrows, dx, projection)] for every .asc in a zip."""
    out = []

    def walk(zf, prefix, depth):
        for n in zf.namelist():
            low = n.lower()
            if low.endswith(".asc"):
                with zf.open(n) as fh:
                    head = fh.read(1024).decode("latin-1").splitlines()
                h, _ = parse_asc_header(head)
                prj = gems_shapes._sibling(zf, n, ".prj")
                kind = gems_shapes.prj_kind(prj.decode("latin-1")) if prj else None
                out.append((prefix + n, int(h.get("ncols", 0)),
                            int(h.get("nrows", 0)),
                            h.get("cellsize", h.get("dx")), kind or "unknown"))
            elif low.endswith(".zip") and depth == 0:
                walk(zipfile.ZipFile(io.BytesIO(zf.read(n))), n + "/", 1)

    with zipfile.ZipFile(path) as zf:
        walk(zf, "", 0)
    return out


def _tif_source(path, assume_utm11):
    import rasterio

    with rasterio.open(path) as src:
        t = src.transform
        if t.b != 0 or t.d != 0 or t.e >= 0:
            raise SystemExit("%s: only north-up grids without rotation are "
                             "supported" % path)
        crs = src.crs
        epsg = crs.to_epsg() if crs else None
        utm = epsg in METRE_EPSG or (
            crs is not None and gems_shapes.prj_kind(crs.to_wkt()) == "utm11")
        if not utm and not assume_utm11:
            raise SystemExit("%s: CRS %s is not UTM zone 11N; pass "
                             "--assume-utm11 if you know it is" % (path, crs))
        values = src.read(1).astype(np.float64)
        if src.nodata is not None and np.isfinite(src.nodata):
            values[values == src.nodata] = np.nan
    values[values < NODATA_FLOOR] = np.nan
    name = os.path.splitext(os.path.basename(path))[0]
    return Source(name, values, float(t.c), float(t.f), float(t.a),
                  float(-t.e))


def load_source(spec, assume_utm11=False):
    """A Source from PATH.asc, ZIP::NAME (an .asc in a zip) or a GeoTIFF."""
    path, _, member = spec.partition("::")
    if not member and not path.lower().endswith(".asc"):
        return _tif_source(path, assume_utm11)
    if member:
        text, prj = _zip_asc(path, member)
        name = os.path.splitext(os.path.basename(member))[0]
    else:
        with open(path, encoding="latin-1") as fh:
            text = fh.read()
        prj_path = os.path.splitext(path)[0] + ".prj"
        prj = None
        if os.path.exists(prj_path):
            with open(prj_path, encoding="latin-1") as fh:
                prj = fh.read()
        name = os.path.splitext(os.path.basename(path))[0]
    if gems_shapes.prj_kind(prj) != "utm11" and not assume_utm11:
        raise SystemExit("%s: projection not recognised as UTM zone 11N; pass "
                         "--assume-utm11 if you know it is" % spec)
    values, (left, top, dx, dy) = read_asc_text(text)
    return Source(name, values, left, top, dx, dy)


def _axis(f, n):
    """(i0, i1, w1) for fractional cell-centre indices f along n cells.

    f is clamped to [0, n - 1], so the half cell beyond the outer centres
    takes the edge value.
    """
    g = np.clip(f, 0, n - 1)
    i0 = np.minimum(np.floor(g).astype(np.int64), max(n - 2, 0))
    i1 = np.minimum(i0 + 1, n - 1)
    return i0, i1, g - i0


def bilinear(values, v, u):
    """float32 [len(v), len(u)] sampled at fractional row v, column u.

    v and u count cells from the grid's top-left corner (so cell centres
    sit at i + 0.5); points outside [0, n) are NaN. NaN neighbours are left
    out and the remaining weights re-normalised.
    """
    n_r, n_c = values.shape
    r0, r1, wr = _axis(v - 0.5, n_r)
    c0, c1, wc = _axis(u - 0.5, n_c)
    rin = (v >= 0) & (v < n_r)
    cin = (u >= 0) & (u < n_c)
    out = np.full((len(v), len(u)), np.nan, dtype=np.float32)
    for start in range(0, len(v), CHUNK_ROWS):
        sl = slice(start, start + CHUNK_ROWS)
        acc = np.zeros((len(r0[sl]), len(u)))
        wsum = np.zeros_like(acc)
        for ri, wy in ((r0[sl], 1.0 - wr[sl]), (r1[sl], wr[sl])):
            for ci, wx in ((c0, 1.0 - wc), (c1, wc)):
                val = values[ri[:, None], ci[None, :]]
                w = wy[:, None] * wx[None, :]
                ok = np.isfinite(val) & (w > 0)
                acc += np.where(ok, val, 0.0) * np.where(ok, w, 0.0)
                wsum += np.where(ok, w, 0.0)
        block = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
        block[~rin[sl], :] = np.nan
        block[:, ~cin] = np.nan
        out[sl] = block
    return out


def nearest(values, v, u):
    """float32 [len(v), len(u)]: the source cell each point falls in."""
    n_r, n_c = values.shape
    ri = np.floor(v).astype(np.int64)
    ci = np.floor(u).astype(np.int64)
    out = values[np.clip(ri, 0, n_r - 1)[:, None],
                 np.clip(ci, 0, n_c - 1)[None, :]].astype(np.float32)
    out[(ri < 0) | (ri >= n_r), :] = np.nan
    out[:, (ci < 0) | (ci >= n_c)] = np.nan
    return out


def regrid(src, origin, cell, shape, method="bilinear"):
    """A Source sampled at the cell centres of the target grid."""
    x0, y0 = origin
    h, w = shape
    xs = x0 + (np.arange(w) + 0.5) * cell
    ys = y0 - (np.arange(h) + 0.5) * cell
    u = (xs - src.left) / src.dx
    v = (src.top - ys) / src.dy
    if method == "nearest":
        return nearest(src.values, v, u)
    return bilinear(src.values, v, u)


def parse_points(doc):
    """[(x, y)] for every Point and MultiPoint vertex, as stored."""
    out = []
    for geom, _ in gems_external._geometries(doc):
        kind = geom.get("type")
        if kind == "Point":
            pts = [geom.get("coordinates") or []]
        elif kind == "MultiPoint":
            pts = geom.get("coordinates") or []
        else:
            continue
        out += [(float(p[0]), float(p[1])) for p in pts if len(p) >= 2]
    return out


def points_utm11(doc):
    """Points of a GeoJSON document in UTM zone 11N metres (the same rule
    as gems_external.lines_utm11: a metre crs member, else lon/lat when
    every point fits)."""
    pts = parse_points(doc)
    name = gems_external._crs_name(doc)
    if (any(code in name for code in gems_external.METRE_CRS)
            or not all(abs(x) <= 180 and abs(y) <= 90 for x, y in pts)):
        return pts
    return [gems_external.utm_forward(lat, lon) for lon, lat in pts]


def point_counts(points, origin, cell, shape, pad=0):
    """int32 [H + 2 pad, W + 2 pad] points per cell on the padded grid."""
    x0, y0 = origin
    h, w = shape
    counts = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.int32)
    for x, y in points:
        col = int(math.floor((x - x0) / cell)) + pad
        row = int(math.floor((y0 - y) / cell)) + pad
        if 0 <= row < h + 2 * pad and 0 <= col < w + 2 * pad:
            counts[row, col] += 1
    return counts


def distance_px(mask, cap, exact=True):
    """float32 distance in px to the nearest True cell, capped at `cap`.

    Euclidean between cell centres with scipy, chessboard otherwise.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.full(m.shape, float(cap), dtype=np.float32)
    if exact:
        try:
            from scipy import ndimage as ndi
        except ImportError:
            exact = False
    if not exact:
        return gems_external.chessboard_distance(m, cap)
    d = ndi.distance_transform_edt(~m)
    return np.minimum(d, float(cap)).astype(np.float32)


def window_counts(counts, radius):
    """Sum of counts over the (2 radius + 1)^2 square around each cell."""
    k = 2 * radius + 1
    h, w = counts.shape
    p = np.pad(counts.astype(np.int64),
               ((radius + 1, radius), (radius + 1, radius)))
    s = p.cumsum(0).cumsum(1)
    return (s[k:k + h, k:k + w] - s[0:h, k:k + w]
            - s[k:k + h, 0:w] + s[0:h, 0:w])


def point_bands(points, origin, cell, shape, cap=DEFAULT_CAP,
                radius=DEFAULT_RADIUS, exact=True):
    """(distance px float32 [H, W], count in the square int64 [H, W])."""
    pad = max(int(math.ceil(cap)), int(radius))
    h, w = shape
    counts = point_counts(points, origin, cell, shape, pad)
    inner = (slice(pad, pad + h), slice(pad, pad + w))
    dist = distance_px(counts > 0, cap, exact)[inner]
    return dist, window_counts(counts, radius)[inner]


def _unique(names):
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else "%s_%d" % (n, seen[n]))
    return out


def write_bands(path, bands, names, profile):
    import rasterio

    prof = profile.copy()
    prof.update(count=len(bands), dtype="float32", nodata=float("nan"),
                compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        for i, (band, name) in enumerate(zip(bands, _unique(names)), 1):
            dst.write(np.asarray(band, dtype=np.float32), i)
            dst.set_band_description(i, name)


def cmd_list(args):
    for member, ncols, nrows, dx, kind in list_zip_grids(args.src):
        print("%-50s %6d x %-6d cell %-8s %s" % (member, ncols, nrows, dx, kind))
    return 0


def cmd_grid(args):
    origin, cell, shape, valid, profile = gems_external._grid(args.grid)
    bands, names, report = [], [], []
    for spec in args.src:
        src = load_source(spec, args.assume_utm11)
        out = regrid(src, origin, cell, shape, args.method)
        out[~valid] = np.nan
        finite = out[np.isfinite(out)]
        bands.append(out)
        names.append(src.name)
        report.append({"name": src.name, "rows_cols": list(src.values.shape),
                       "cell_m": [src.dx, src.dy],
                       "covered": round(float(finite.size) / max(int(valid.sum()), 1), 4),
                       "min": float(finite.min()) if finite.size else None,
                       "max": float(finite.max()) if finite.size else None})
    write_bands(args.out, bands, names, profile)
    print(json.dumps({"version": __version__, "out": args.out,
                      "method": args.method, "sources": report}, indent=2))
    return 0


def cmd_points(args):
    origin, cell, shape, valid, profile = gems_external._grid(args.grid)
    bands, names, report = [], [], []
    for path in args.points:
        with open(path, encoding="utf-8") as fh:
            pts = points_utm11(json.load(fh))
        dist, count = point_bands(pts, origin, cell, shape, args.cap,
                                  args.radius)
        stem = os.path.splitext(os.path.basename(path))[0]
        for band, name in ((dist, stem + "_dist_px"),
                           (count, "%s_n_r%d" % (stem, args.radius))):
            band = band.astype(np.float32)
            band[~valid] = np.nan
            bands.append(band)
            names.append(name)
        inside = point_counts(pts, origin, cell, shape)
        report.append({"name": stem, "points": len(pts),
                       "inside_grid": int(inside.sum()),
                       "cells_within_cap": int(((dist < args.cap) & valid).sum())})
    write_bands(args.out, bands, names, profile)
    print(json.dumps({"version": __version__, "out": args.out, "cap_px": args.cap,
                      "radius_px": args.radius, "layers": report}, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="list the ASCII grids in a zip")
    ls.add_argument("--src", required=True)
    g = sub.add_parser("grid", help="sample rasters onto the GEMS grid")
    g.add_argument("--grid", required=True, help="numeric_features.tif")
    g.add_argument("--src", required=True, action="append",
                   help="PATH.asc, ZIP::NAME or a GeoTIFF (repeatable)")
    g.add_argument("--out", required=True)
    g.add_argument("--method", choices=("bilinear", "nearest"),
                   default="bilinear")
    g.add_argument("--assume-utm11", action="store_true",
                   help="treat sources with no or an unknown projection as "
                        "UTM 11N")
    p = sub.add_parser("points", help="distance and count bands for points")
    p.add_argument("--grid", required=True, help="numeric_features.tif")
    p.add_argument("--points", required=True, action="append",
                   help="GeoJSON with Point/MultiPoint features (repeatable)")
    p.add_argument("--out", required=True)
    p.add_argument("--cap", type=int, default=DEFAULT_CAP,
                   help="largest distance written, in px")
    p.add_argument("--radius", type=int, default=DEFAULT_RADIUS,
                   help="half-width of the counting square, in px")
    args = ap.parse_args(argv)
    return {"list": cmd_list, "grid": cmd_grid,
            "points": cmd_points}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
