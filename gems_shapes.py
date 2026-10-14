"""Shapefiles (plain or inside zips) to GeoJSON for gems_external.py.

    python gems_shapes.py list    --src USGScontributions_forNVML_dataRelease2021.zip
    python gems_shapes.py geojson --src USGScontributions_forNVML_dataRelease2021.zip \
                                  --layer magneticLineations_nvML --out mag_lineations.geojson
    python gems_external.py rasterize --grid data/numeric_features.tif \
                                  --lines mag_lineations.geojson --labels data/labels.tif \
                                  --out mag_external.tif

Many public structure maps of Nevada come as shapefiles, for example the
USGS data release for the Nevada Geothermal Machine Learning project
(DeAngelo and others, 2021, https://doi.org/10.5066/P9V5SQRD; no use
constraints): gravity and magnetic lineations traced from the maxima of
horizontal gradients, their end points and intersections, Quaternary
faults, and fault lines with slip and dilation tendency. Lineations in the
potential fields are a classic sign of buried faults, which is what GEMS
asks for.

This module reads .shp/.dbf/.prj with the standard library only (no GDAL,
fiona or pyshp) and writes GeoJSON. It looks one level into zips nested in
zips. Coordinates are written as stored; a .prj naming UTM zone 11N (NAD83
or WGS84) adds a `crs` member that gems_external.py reads as metres, and a
geographic .prj leaves the GeoJSON default (longitude/latitude). Any other
projection is refused rather than guessed.

By MV2 of Munim, Inc. Made with AI agents. Apache-2.0 licence.
"""

import argparse
import io
import json
import os
import re
import struct
import sys
import zipfile

import gems_metric

__version__ = gems_metric.__version__

UTM11_CRS = {"type": "name",
             "properties": {"name": "urn:ogc:def:crs:EPSG::26911"}}

POINT_TYPES = (1, 11, 21)
MULTIPOINT_TYPES = (8, 18, 28)
LINE_TYPES = (3, 13, 23)
POLYGON_TYPES = (5, 15, 25)
TYPE_NAMES = {0: "null", 1: "point", 3: "polyline", 5: "polygon",
              8: "multipoint", 11: "pointZ", 13: "polylineZ", 15: "polygonZ",
              18: "multipointZ", 21: "pointM", 23: "polylineM",
              25: "polygonM", 28: "multipointM"}


def shp_header(data):
    """(shape type, (xmin, ymin, xmax, ymax)) from a .shp main header."""
    if len(data) < 100 or struct.unpack(">i", data[0:4])[0] != 9994:
        raise ValueError("not a shapefile (file code is not 9994)")
    shape_type = struct.unpack("<i", data[32:36])[0]
    bbox = struct.unpack("<4d", data[36:68])
    return shape_type, bbox


def read_shp(data):
    """[(shape type, parts)] for every record of a .shp file (bytes).

    parts is a list of [(x, y), ...]: one part per point, one per line or
    ring. Z and M values are skipped. Null shapes give an empty list.
    """
    shp_header(data)
    out = []
    pos = 100
    while pos + 8 <= len(data):
        length = struct.unpack(">i", data[pos + 4:pos + 8])[0] * 2
        content = data[pos + 8:pos + 8 + length]
        pos += 8 + length
        if len(content) < 4:
            break
        kind = struct.unpack("<i", content[0:4])[0]
        parts = []
        if kind in POINT_TYPES:
            parts = [[struct.unpack("<2d", content[4:20])]]
        elif kind in MULTIPOINT_TYPES:
            n = struct.unpack("<i", content[36:40])[0]
            pts = struct.unpack("<%dd" % (2 * n), content[40:40 + 16 * n])
            parts = [[(pts[2 * i], pts[2 * i + 1])] for i in range(n)]
        elif kind in LINE_TYPES + POLYGON_TYPES:
            n_parts, n_points = struct.unpack("<2i", content[36:44])
            starts = list(struct.unpack("<%di" % n_parts,
                                        content[44:44 + 4 * n_parts]))
            p0 = 44 + 4 * n_parts
            pts = struct.unpack("<%dd" % (2 * n_points),
                                content[p0:p0 + 16 * n_points])
            xy = [(pts[2 * i], pts[2 * i + 1]) for i in range(n_points)]
            for i, s in enumerate(starts):
                e = starts[i + 1] if i + 1 < n_parts else n_points
                parts.append(xy[s:e])
        out.append((kind, parts))
    return out


def _dbf_value(raw, kind):
    text = raw.decode("latin-1").strip()
    if kind in ("N", "F"):
        if not text or set(text) <= set("*?"):
            return None
        try:
            return int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
        except ValueError:
            return None
    if kind == "L":
        return {"T": True, "Y": True, "F": False, "N": False}.get(text[:1].upper())
    return text


def read_dbf(data):
    """[dict] attribute rows of a dBASE (.dbf) file, in shape order.

    Deleted rows are None, so row i still belongs to shape i.
    """
    n_rec = struct.unpack("<I", data[4:8])[0]
    header_len, rec_len = struct.unpack("<2H", data[8:12])
    fields = []
    pos = 32
    while pos + 32 <= header_len and data[pos] != 0x0D:
        name = data[pos:pos + 11].split(b"\x00")[0].decode("latin-1").strip()
        kind = chr(data[pos + 11])
        size = data[pos + 16]
        fields.append((name, kind, size))
        pos += 32
    rows = []
    for i in range(n_rec):
        start = header_len + i * rec_len
        rec = data[start:start + rec_len]
        if len(rec) < rec_len:
            break
        if rec[0:1] == b"*":
            rows.append(None)
            continue
        off = 1
        row = {}
        for name, kind, size in fields:
            row[name] = _dbf_value(rec[off:off + size], kind)
            off += size
        rows.append(row)
    return rows


def _prj_params(wkt):
    return {k.lower(): float(v) for k, v in re.findall(
        r'PARAMETER\[\s*"([^"]+)"\s*,\s*([-+0-9.eE]+)\s*\]', wkt)}


def prj_kind(wkt):
    """'utm11' (UTM zone 11N metres), 'lonlat' (geographic) or None."""
    if not wkt or not wkt.strip():
        return None
    w = wkt.upper()
    if not w.lstrip().startswith("PROJCS"):
        return "lonlat" if w.lstrip().startswith("GEOGCS") else None
    if re.search(r"UTM[_ ]ZONE[_ ]11N", w):
        return "utm11"
    p = _prj_params(wkt)
    if ("TRANSVERSE_MERCATOR" in w
            and abs(p.get("central_meridian", 0) + 117.0) < 1e-6
            and abs(p.get("false_easting", 0) - 500000.0) < 1e-3
            and abs(p.get("false_northing", 1)) < 1e-3
            and abs(p.get("scale_factor", 0) - 0.9996) < 1e-9
            and abs(p.get("latitude_of_origin", 1)) < 1e-9
            and re.search(r'UNIT\[\s*"METERS?"', w)):
        return "utm11"
    return None


def _geometry(kind, parts):
    if not parts:
        return None
    if kind in POINT_TYPES:
        return {"type": "Point", "coordinates": list(parts[0][0])}
    if kind in MULTIPOINT_TYPES:
        return {"type": "MultiPoint",
                "coordinates": [list(p[0]) for p in parts]}
    coords = [[list(xy) for xy in part] for part in parts]
    if kind in LINE_TYPES:
        if len(coords) == 1:
            return {"type": "LineString", "coordinates": coords[0]}
        return {"type": "MultiLineString", "coordinates": coords}
    # Rings are kept as one polygon; GeoJSON readers that need holes nested
    # by winding order should use the original file.
    return {"type": "Polygon", "coordinates": coords}


def to_geojson(records, rows=None, kind=None, boundaries=False):
    """FeatureCollection from read_shp() records and read_dbf() rows.

    kind is prj_kind(): 'utm11' adds a crs member, 'lonlat' does not.
    boundaries=True writes polygons as MultiLineStrings of their rings, so
    gems_external.py rasterises their outlines.
    """
    feats = []
    for i, (shape, parts) in enumerate(records):
        props = rows[i] if rows and i < len(rows) else {}
        if props is None:  # deleted in the .dbf
            continue
        geom = _geometry(shape, parts)
        if geom is None:
            continue
        if boundaries and geom["type"] == "Polygon":
            geom = {"type": "MultiLineString", "coordinates": geom["coordinates"]}
        feats.append({"type": "Feature", "geometry": geom, "properties": props})
    doc = {"type": "FeatureCollection", "features": feats}
    if kind == "utm11":
        doc["crs"] = UTM11_CRS
    return doc


def _zip_layers(zf, prefix=""):
    """{layer name: (zip file, member path)} for every .shp, one level deep."""
    found = {}
    for name in zf.namelist():
        low = name.lower()
        if low.endswith(".shp"):
            base = os.path.splitext(os.path.basename(name))[0]
            found.setdefault(base, (zf, name, prefix + name))
        elif low.endswith(".zip") and not prefix:
            inner = zipfile.ZipFile(io.BytesIO(zf.read(name)))
            for k, v in _zip_layers(inner, name + "/").items():
                found.setdefault(k, v)
    return found


def _sibling(zf, member, ext):
    stem = os.path.splitext(member)[0]
    names = {n.lower(): n for n in zf.namelist()}
    real = names.get((stem + ext).lower())
    return zf.read(real) if real else None


def list_layers(src):
    """[(layer, shape type name, records, projection)] in a .shp or a zip."""
    out = []
    for name, (shp, _dbf, prj) in sorted(_layers(src).items()):
        kind = shp_header(shp)[0]
        out.append((name, TYPE_NAMES.get(kind, str(kind)),
                    len(read_shp(shp)), prj_kind(prj) or "unknown"))
    return out


def _layers(src):
    """{layer: (shp bytes, dbf bytes or None, prj text or None)}."""
    if src.lower().endswith(".zip"):
        out = {}
        with zipfile.ZipFile(src) as zf:
            for name, (z, member, _) in _zip_layers(zf).items():
                prj = _sibling(z, member, ".prj")
                out[name] = (z.read(member), _sibling(z, member, ".dbf"),
                             prj.decode("latin-1") if prj else None)
        return out
    stem = os.path.splitext(src)[0]

    def read(ext):
        path = stem + ext
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read()
        return None

    if read(".shp") is None:
        raise SystemExit("no such shapefile: %s" % src)
    prj = read(".prj")
    return {os.path.basename(stem): (read(".shp"), read(".dbf"),
                                     prj.decode("latin-1") if prj else None)}


def load_layer(src, layer=None, assume_utm11=False, boundaries=False):
    """GeoJSON dict for one layer of a .shp file or a zip."""
    layers = _layers(src)
    if layer is None:
        if len(layers) != 1:
            raise SystemExit("choose --layer: %s" % ", ".join(sorted(layers)))
        layer = next(iter(layers))
    match = {k.lower(): k for k in layers}.get(layer.lower())
    if match is None:
        raise SystemExit("no layer %r; have: %s" % (layer, ", ".join(sorted(layers))))
    shp, dbf, prj = layers[match]
    kind = prj_kind(prj)
    if kind is None:
        if not assume_utm11:
            raise SystemExit("%s: projection not recognised as UTM zone 11N or "
                             "longitude/latitude; pass --assume-utm11 if you "
                             "know it is" % match)
        kind = "utm11"
    rows = read_dbf(dbf) if dbf else None
    return to_geojson(read_shp(shp), rows, kind, boundaries)


def cmd_list(args):
    for name, kind, n, proj in list_layers(args.src):
        print("%-40s %-12s %8d  %s" % (name, kind, n, proj))
    return 0


def cmd_geojson(args):
    doc = load_layer(args.src, args.layer, args.assume_utm11, args.boundaries)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    kinds = {}
    for f in doc["features"]:
        kinds[f["geometry"]["type"]] = kinds.get(f["geometry"]["type"], 0) + 1
    print(json.dumps({"version": __version__, "out": args.out,
                      "crs": "utm11" if "crs" in doc else "lonlat",
                      "features": kinds}, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="list the shapefile layers in a zip")
    ls.add_argument("--src", required=True, help=".zip or .shp")
    g = sub.add_parser("geojson", help="write one layer as GeoJSON")
    g.add_argument("--src", required=True, help=".zip or .shp")
    g.add_argument("--layer", help="layer name (the .shp file name without .shp)")
    g.add_argument("--out", required=True)
    g.add_argument("--assume-utm11", action="store_true",
                   help="treat a layer with no or an unknown .prj as UTM 11N")
    g.add_argument("--boundaries", action="store_true",
                   help="write polygons as their outlines (lines)")
    args = ap.parse_args(argv)
    return {"list": cmd_list, "geojson": cmd_geojson}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
