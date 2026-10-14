"""Public fault maps as a candidate layer on the GEMS grid.

    python gems_external.py fetch     --grid data/numeric_features.tif --out qfaults.geojson
    python gems_external.py rasterize --grid data/numeric_features.tif --lines qfaults.geojson \
                                      --labels data/labels.tif --out external.tif
    python gems_external.py merge     --pred submission.tif --external external.tif --out merged.tif

The rules allow outside data that is licensed for the challenge and can be
shared with the sponsor (forum topic 11528). The USGS Quaternary Fault and
Fold Database (Qfaults) is public domain and served as GeoJSON by the USGS
ArcGIS service, so `fetch` downloads every Qfaults line over the grid. Any
other GeoJSON of fault lines (for example USGS State Geologic Map
Compilation faults exported from a GIS) can be passed to `rasterize` too.

`rasterize` writes three float32 bands on the competition grid, NaN outside
the data:

    external_fault       1 where a mapped line crosses the 100 m cell
    external_distance    chessboard distance to the nearest such cell (px),
                         capped at --cap
    external_candidate   external_fault cells more than --buffer px from
                         any known (labels.tif) fault

and prints how far the two maps agree. If most known faults sit on
external lines, the training labels were probably drawn from that map, so
do not feed external_distance to the classifier (it would learn "fault =
Qfault" and miss the unmapped ones); use the candidate band at prediction
time only, which is what `merge` does. Whether it pays is a question for
the public leaderboard.

Lines in longitude/latitude (GeoJSON's default) are projected to UTM zone
11N with the transverse Mercator series of Snyder (USGS Professional Paper
1395), so neither pyproj nor GDAL is needed for that step. NAD83 and WGS84
differ by about a metre here, far below the 100 m cell.

By MV2 of Munim, Inc. Made with AI agents. Apache-2.0 licence.
"""

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

__version__ = "0.3.0"

WGS84_A = 6378137.0
WGS84_E2 = (1 / 298.257223563) * (2 - 1 / 298.257223563)
UTM_K0 = 0.9996
UTM_FALSE_EASTING = 500000.0
ZONE = 11

QFAULTS_URL = ("https://earthquake.usgs.gov/arcgis/rest/services/haz/"
               "Qfaults/MapServer/%d/query")
QFAULTS_LAYER = 21  # "National Database" (polylines)
PAGE_SIZE = 2000    # the service's maxRecordCount
USER_AGENT = "gems-fault-mapping/%s (MV2 of Munim, Inc.)" % __version__

BAND_NAMES = ["external_fault", "external_distance", "external_candidate"]
DEFAULT_BUFFER = 2
DEFAULT_CAP = 10
METRE_CRS = ("32611", "26911", "6340")
NODATA_FLOOR = -1e38


def central_meridian(zone=ZONE):
    return -183.0 + 6.0 * zone


def meridian_arc(phi, a=WGS84_A, e2=WGS84_E2):
    """Distance along the meridian from the equator to latitude phi (radians)."""
    e4, e6 = e2 * e2, e2 * e2 * e2
    return a * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
                - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * phi)
                + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * phi)
                - (35 * e6 / 3072) * math.sin(6 * phi))


def tm_forward(lat, lon, lon0, k0=UTM_K0, a=WGS84_A, e2=WGS84_E2):
    """(x, y) in metres from the central meridian and the equator.

    Snyder's equations 8-9 to 8-15 (with 3-21 for the meridian arc); his
    worked example on Clarke 1866 gives x = 127106.47, y = 4484124.43.
    """
    phi = math.radians(lat)
    ep2 = e2 / (1 - e2)
    s, c = math.sin(phi), math.cos(phi)
    n = a / math.sqrt(1 - e2 * s * s)
    t = math.tan(phi) ** 2
    cc = ep2 * c * c
    aa = math.radians(lon - lon0) * c
    x = k0 * n * (aa + (1 - t + cc) * aa ** 3 / 6
                  + (5 - 18 * t + t * t + 72 * cc - 58 * ep2) * aa ** 5 / 120)
    y = k0 * (meridian_arc(phi, a, e2) + n * math.tan(phi) * (
        aa ** 2 / 2 + (5 - t + 9 * cc + 4 * cc * cc) * aa ** 4 / 24
        + (61 - 58 * t + t * t + 600 * cc - 330 * ep2) * aa ** 6 / 720))
    return x, y


def tm_inverse(x, y, lon0, k0=UTM_K0, a=WGS84_A, e2=WGS84_E2):
    """(lat, lon) in degrees; Snyder's equations 8-17 to 8-25."""
    ep2 = e2 / (1 - e2)
    e4, e6 = e2 * e2, e2 * e2 * e2
    mu = (y / k0) / (a * (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256))
    r = math.sqrt(1 - e2)
    e1 = (1 - r) / (1 + r)
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    s, c = math.sin(phi1), math.cos(phi1)
    c1 = ep2 * c * c
    t1 = math.tan(phi1) ** 2
    n1 = a / math.sqrt(1 - e2 * s * s)
    r1 = a * (1 - e2) / (1 - e2 * s * s) ** 1.5
    d = x / (n1 * k0)
    phi = phi1 - (n1 * math.tan(phi1) / r1) * (
        d ** 2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * ep2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * ep2 - 3 * c1 * c1)
        * d ** 6 / 720)
    lam = (d - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * ep2 + 24 * t1 * t1)
           * d ** 5 / 120) / c
    return math.degrees(phi), lon0 + math.degrees(lam)


def utm_forward(lat, lon, zone=ZONE):
    """(easting, northing) in UTM zone `zone` north, metres."""
    x, y = tm_forward(lat, lon, central_meridian(zone))
    return x + UTM_FALSE_EASTING, y


def utm_inverse(easting, northing, zone=ZONE):
    """(lat, lon) in degrees from UTM zone `zone` north."""
    return tm_inverse(easting - UTM_FALSE_EASTING, northing,
                      central_meridian(zone))


def grid_bbox_lonlat(origin, cell, shape, margin=0.05):
    """(min_lon, min_lat, max_lon, max_lat) around a UTM 11N grid.

    origin is the grid's top-left corner (x0, y0); the corners and edge
    midpoints are converted and `margin` degrees added on every side.
    """
    x0, y0 = origin
    h, w = shape
    lats, lons = [], []
    for fx in (0.0, 0.5, 1.0):
        for fy in (0.0, 0.5, 1.0):
            lat, lon = utm_inverse(x0 + fx * w * cell, y0 - fy * h * cell)
            lats.append(lat)
            lons.append(lon)
    return (min(lons) - margin, min(lats) - margin,
            max(lons) + margin, max(lats) + margin)


def query_url(bbox, offset=0, count=PAGE_SIZE, layer=QFAULTS_LAYER):
    """ArcGIS REST query for every line crossing bbox, as GeoJSON in lon/lat."""
    params = [("where", "1=1"),
              ("geometry", "%.6f,%.6f,%.6f,%.6f" % tuple(bbox)),
              ("geometryType", "esriGeometryEnvelope"),
              ("inSR", "4326"),
              ("spatialRel", "esriSpatialRelIntersects"),
              ("outFields", "*"),
              ("outSR", "4326"),
              ("orderByFields", "OBJECTID"),
              ("resultOffset", str(int(offset))),
              ("resultRecordCount", str(int(count))),
              ("f", "geojson")]
    return QFAULTS_URL % layer + "?" + urllib.parse.urlencode(params)


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_features(bbox, layer=QFAULTS_LAYER, count=PAGE_SIZE, get_json=None,
                   max_pages=500):
    """Every feature in bbox, following the service's pages.

    Stops when a page says the transfer limit was not exceeded, comes back
    empty, or adds no new feature ids (a server that ignores resultOffset).
    """
    get_json = get_json or _get_json
    features, seen, offset = [], set(), 0
    for _ in range(max_pages):
        page = get_json(query_url(bbox, offset, count, layer))
        if "error" in page:
            raise RuntimeError("service error: %s" % (page["error"],))
        got = page.get("features") or []
        new = 0
        for f in got:
            key = f.get("id")
            if key is None:
                key = (f.get("properties") or {}).get("OBJECTID")
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            features.append(f)
            new += 1
        more = (page.get("exceededTransferLimit")
                or (page.get("properties") or {}).get("exceededTransferLimit"))
        if not got or not more or new == 0:
            break
        offset += len(got)
    return features


def _crs_name(doc):
    crs = doc.get("crs") if isinstance(doc, dict) else None
    if isinstance(crs, dict):
        return str((crs.get("properties") or {}).get("name", ""))
    return ""


def _geometries(doc):
    """(geometry, properties) pairs from a FeatureCollection, Feature or geometry."""
    kind = doc.get("type")
    if kind == "FeatureCollection":
        for f in doc.get("features") or []:
            for item in _geometries(f):
                yield item
    elif kind == "Feature":
        if doc.get("geometry"):
            yield doc["geometry"], doc.get("properties") or {}
    elif kind == "GeometryCollection":
        for g in doc.get("geometries") or []:
            yield g, {}
    elif kind:
        yield doc, {}


def parse_lines(doc):
    """[(coords, properties)] for every LineString / MultiLineString part.

    coords are [(x, y), ...] exactly as stored; points and polygons are
    skipped (Qfaults' "Fault Areas" are polygons and not used here).
    """
    out = []
    for geom, props in _geometries(doc):
        kind = geom.get("type")
        parts = []
        if kind == "LineString":
            parts = [geom.get("coordinates") or []]
        elif kind == "MultiLineString":
            parts = geom.get("coordinates") or []
        for part in parts:
            coords = [(float(p[0]), float(p[1])) for p in part if len(p) >= 2]
            if coords:
                out.append((coords, props))
    return out


def looks_lonlat(lines):
    """True when every vertex is a plausible longitude/latitude pair."""
    return all(abs(x) <= 180 and abs(y) <= 90
               for coords, _ in lines for x, y in coords)


def lines_utm11(doc):
    """Line parts of a GeoJSON document in UTM zone 11N metres.

    A `crs` member naming EPSG 32611, 26911 or 6340 means metres already;
    otherwise lon/lat (the GeoJSON default) is assumed when every vertex
    fits, and projected.
    """
    lines = parse_lines(doc)
    name = _crs_name(doc)
    metres = any(code in name for code in METRE_CRS)
    if metres or not looks_lonlat(lines):
        return lines
    return [([utm_forward(lat, lon) for lon, lat in coords], props)
            for coords, props in lines]


def rasterize_lines(lines, origin, cell, shape, step=0.25):
    """Set of (row, col) cells the lines pass through, sampled every
    `step` cells along each segment. Cells outside the grid are dropped."""
    x0, y0 = origin
    h, w = shape
    cells = set()

    def add(x, y):
        col = int(math.floor((x - x0) / cell))
        row = int(math.floor((y0 - y) / cell))
        if 0 <= row < h and 0 <= col < w:
            cells.add((row, col))

    for coords, _ in lines:
        if len(coords) == 1:
            add(*coords[0])
            continue
        for (xa, ya), (xb, yb) in zip(coords[:-1], coords[1:]):
            n = max(1, int(math.ceil(math.hypot(xb - xa, yb - ya)
                                     / (step * cell))))
            for i in range(n + 1):
                add(xa + (xb - xa) * i / n, ya + (yb - ya) * i / n)
    return cells


def cells_to_mask(cells, shape):
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def dilate(mask, radius):
    """Chessboard dilation: every pixel within `radius` px (8-connected)."""
    m = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return m.copy()
    h, w = m.shape
    pad = np.pad(m, radius)
    out = np.zeros((h, w), dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= pad[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
    return out


def chessboard_distance(mask, cap):
    """float32 distance in px to the nearest True pixel, capped at `cap`."""
    m = np.asarray(mask, dtype=bool)
    dist = np.full(m.shape, float(cap), dtype=np.float32)
    dist[m] = 0.0
    reached = m.copy()
    for d in range(1, int(cap)):
        grown = dilate(reached, 1)
        dist[grown & ~reached] = float(d)
        reached = grown
    return dist


def candidates(external, known, buffer=DEFAULT_BUFFER):
    """External fault cells more than `buffer` px from any known fault."""
    return (np.asarray(external, dtype=bool)
            & ~dilate(np.asarray(known, dtype=bool), buffer))


def agreement(external, known, buffer=DEFAULT_BUFFER):
    """How far an external map and the known faults overlap."""
    e = np.asarray(external, dtype=bool)
    k = np.asarray(known, dtype=bool)
    near_e = dilate(e, 1)
    n_k, n_e = int(k.sum()), int(e.sum())
    cand = candidates(e, k, buffer)
    return {"known_pixels": n_k, "external_pixels": n_e,
            "known_within_1px_of_external":
                round(float((k & near_e).sum()) / n_k, 4) if n_k else None,
            "external_within_buffer_of_known":
                round(float((e & ~cand).sum()) / n_e, 4) if n_e else None,
            "candidate_pixels": int(cand.sum()), "buffer_px": int(buffer)}


def external_bands(external, known, valid, buffer=DEFAULT_BUFFER,
                   cap=DEFAULT_CAP):
    """float32 [3, H, W] in BAND_NAMES order, NaN outside `valid`."""
    e = np.asarray(external, dtype=bool)
    out = np.stack([e.astype(np.float32),
                    chessboard_distance(e, cap),
                    candidates(e, known, buffer).astype(np.float32)])
    out[:, ~np.asarray(valid, dtype=bool)] = np.nan
    return out


def _grid(path):
    import rasterio

    with rasterio.open(path) as g:
        band = g.read(1).astype(np.float32)
        profile = g.profile.copy()
        t = g.transform
        if g.nodata is not None and np.isfinite(g.nodata):
            band[band == np.float32(g.nodata)] = np.nan
    valid = np.isfinite(band) & (band > NODATA_FLOOR)
    return (float(t.c), float(t.f)), float(t.a), band.shape, valid, profile


def cmd_fetch(args):
    import rasterio

    with rasterio.open(args.grid) as g:
        t, shape = g.transform, (g.height, g.width)
    bbox = grid_bbox_lonlat((float(t.c), float(t.f)), float(t.a), shape)
    feats = fetch_features(bbox, layer=args.layer)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    print(json.dumps({"bbox_lonlat": [round(v, 4) for v in bbox],
                      "layer": args.layer, "features": len(feats),
                      "out": args.out}, indent=2))
    return 0


def cmd_rasterize(args):
    import rasterio

    origin, cell, shape, valid, profile = _grid(args.grid)
    lines = []
    for path in args.lines:
        with open(path, encoding="utf-8") as f:
            lines += lines_utm11(json.load(f))
    ext = cells_to_mask(rasterize_lines(lines, origin, cell, shape), shape)
    if args.labels:
        with rasterio.open(args.labels) as src:
            raw = src.read(1).astype(np.float32)
        known = np.where(np.isfinite(raw), raw, 0.0) >= 1.0
    else:
        known = np.zeros(shape, dtype=bool)
    out = external_bands(ext, known, valid, args.buffer, args.cap)
    prof = profile.copy()
    prof.update(count=len(BAND_NAMES), dtype="float32", nodata=float("nan"),
                compress="deflate")
    with rasterio.open(args.out, "w", **prof) as dst:
        dst.write(out)
        for i, name in enumerate(BAND_NAMES, 1):
            dst.set_band_description(i, name)
    report = {"version": __version__, "lines": len(lines)}
    report.update(agreement(ext & valid, known, args.buffer))
    print(json.dumps(report, indent=2))
    return 0


def cmd_merge(args):
    """OR the external_candidate band into an existing 0/1 submission."""
    import rasterio

    with rasterio.open(args.pred) as src:
        pred = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    with rasterio.open(args.external) as src:
        cand = src.read(BAND_NAMES.index("external_candidate") + 1)
    valid = np.isfinite(pred)
    binary = np.where(valid, pred, 0.0) >= 0.5
    add = (np.where(np.isfinite(cand), cand, 0.0) >= 0.5) & valid
    out = np.where(valid, (binary | add).astype(np.float32), np.nan)
    profile.update(count=1, dtype="float32", nodata=float("nan"),
                   compress="deflate")
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(out.astype(np.float32), 1)
    print(json.dumps({"input_pixels": int(binary.sum()),
                      "added": int((add & ~binary).sum())}, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Public fault maps (USGS Qfaults or any GeoJSON) as a "
                    "candidate layer on the GEMS grid.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download Qfaults lines over the grid")
    f.add_argument("--grid", required=True, help="numeric_features.tif")
    f.add_argument("--out", required=True, help="output GeoJSON")
    f.add_argument("--layer", type=int, default=QFAULTS_LAYER,
                   help="Qfaults MapServer layer (21 = national database)")
    r = sub.add_parser("rasterize", help="write the three external bands")
    r.add_argument("--grid", required=True, help="numeric_features.tif")
    r.add_argument("--lines", required=True, action="append",
                   help="GeoJSON of fault lines (repeatable)")
    r.add_argument("--labels", help="labels.tif (known faults)")
    r.add_argument("--out", required=True)
    r.add_argument("--buffer", type=int, default=DEFAULT_BUFFER,
                   help="px from known faults inside which lines are not "
                        "candidates")
    r.add_argument("--cap", type=int, default=DEFAULT_CAP,
                   help="largest distance written, in px")
    m = sub.add_parser("merge", help="OR the candidate band into a submission")
    m.add_argument("--pred", required=True)
    m.add_argument("--external", required=True, help="rasterize output")
    m.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"fetch": cmd_fetch, "rasterize": cmd_rasterize,
            "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
