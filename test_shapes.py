"""Offline tests for gems_shapes (shapefile and dBASE reading, GeoJSON output).

Run with `python -m unittest -v`. The files are built byte by byte from the
ESRI Shapefile Technical Description (1998) and the dBASE III layout, so
these tests use the standard library only.
"""

import contextlib
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gems_external  # noqa: E402
import gems_shapes  # noqa: E402

UTM11_PRJ = (
    'PROJCS["NAD_1983_UTM_Zone_11N",GEOGCS["GCS_North_American_1983",'
    'DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,'
    '298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",'
    '0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
    'PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",-117.0],PARAMETER["Scale_Factor",0.9996],'
    'PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
CUSTOM_TM_PRJ = UTM11_PRJ.replace("NAD_1983_UTM_Zone_11N",
                                  "NAD_1983_Transverse_Mercator")
GEO_PRJ = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",'
           '6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
           'UNIT["Degree",0.0174532925199433]]')


def shp_bytes(shape_type, contents):
    """A .shp file: 100-byte header, then (number, length) + content each."""
    body = b""
    for i, c in enumerate(contents, 1):
        body += struct.pack(">2i", i, len(c) // 2) + c
    total_words = (100 + len(body)) // 2
    header = (struct.pack(">7i", 9994, 0, 0, 0, 0, 0, total_words)
              + struct.pack("<2i", 1000, shape_type)
              + struct.pack("<8d", 0, 0, 0, 0, 0, 0, 0, 0))
    assert len(header) == 100
    return header + body


def polyline(parts, kind=3, z=False):
    pts = [p for part in parts for p in part]
    starts, n = [], 0
    for part in parts:
        starts.append(n)
        n += len(part)
    c = (struct.pack("<i", kind) + struct.pack("<4d", 0, 0, 0, 0)
         + struct.pack("<2i", len(parts), len(pts))
         + struct.pack("<%di" % len(parts), *starts)
         + struct.pack("<%dd" % (2 * len(pts)),
                       *[v for p in pts for v in p]))
    if z:
        c += struct.pack("<2d", 0, 9) + struct.pack("<%dd" % len(pts),
                                                    *[9.0] * len(pts))
    return c


def point(x, y):
    return struct.pack("<i2d", 1, x, y)


def multipoint(pts):
    return (struct.pack("<i", 8) + struct.pack("<4d", 0, 0, 0, 0)
            + struct.pack("<i", len(pts))
            + struct.pack("<%dd" % (2 * len(pts)), *[v for p in pts for v in p]))


NULL = struct.pack("<i", 0)


def dbf_bytes(fields, rows, deleted=()):
    """dBASE III: 32-byte header, 32-byte field descriptors, 0x0D, records."""
    header_len = 32 + 32 * len(fields) + 1
    rec_len = 1 + sum(f[2] for f in fields)
    out = struct.pack("<B3BIHH20x", 3, 126, 10, 14, len(rows), header_len,
                      rec_len)
    for name, kind, size, dec in fields:
        out += (name.encode("latin-1").ljust(11, b"\x00") + kind.encode()
                + b"\x00" * 4 + bytes([size, dec]) + b"\x00" * 14)
    out += b"\r"
    for i, row in enumerate(rows):
        rec = b"*" if i in deleted else b" "
        for (name, kind, size, dec), value in zip(fields, row):
            text = str(value)
            rec += (text.rjust(size) if kind in "NF" else text.ljust(size)
                    ).encode("latin-1")
        out += rec
    return out + b"\x1a"


FIELDS = [("NAME", "C", 10, 0), ("SLIP", "N", 8, 3), ("RANK", "N", 4, 0)]


class ShpTests(unittest.TestCase):
    def test_header_and_bad_file(self):
        data = shp_bytes(3, [])
        self.assertEqual(gems_shapes.shp_header(data)[0], 3)
        self.assertEqual(gems_shapes.read_shp(data), [])
        with self.assertRaises(ValueError):
            gems_shapes.shp_header(b"\x00" * 100)

    def test_records_of_every_kind(self):
        data = shp_bytes(3, [
            polyline([[(0, 0), (1, 1)], [(5, 5), (6, 5), (7, 5)]]),
            point(2.5, -3.0),
            NULL,
            polyline([[(10, 20), (30, 40)]], kind=13, z=True),
            multipoint([(1, 2), (3, 4)]),
            polyline([[(0, 0), (4, 0), (4, 4), (0, 0)]], kind=5),
        ])
        recs = gems_shapes.read_shp(data)
        self.assertEqual([k for k, _ in recs], [3, 1, 0, 13, 8, 5])
        self.assertEqual(recs[0][1], [[(0.0, 0.0), (1.0, 1.0)],
                                      [(5.0, 5.0), (6.0, 5.0), (7.0, 5.0)]])
        self.assertEqual(recs[1][1], [[(2.5, -3.0)]])
        self.assertEqual(recs[2][1], [])
        self.assertEqual(recs[3][1], [[(10.0, 20.0), (30.0, 40.0)]])
        self.assertEqual(recs[4][1], [[(1.0, 2.0)], [(3.0, 4.0)]])
        self.assertEqual(recs[5][1], [[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0),
                                       (0.0, 0.0)]])


class DbfTests(unittest.TestCase):
    def test_rows_types_and_deleted(self):
        data = dbf_bytes(FIELDS, [("fault A", "0.625", "12"),
                                  ("gone", "1.000", "3"),
                                  ("blank", "", "")], deleted=(1,))
        rows = gems_shapes.read_dbf(data)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], {"NAME": "fault A", "SLIP": 0.625, "RANK": 12})
        self.assertIsNone(rows[1])
        self.assertEqual(rows[2], {"NAME": "blank", "SLIP": None, "RANK": None})
        self.assertIsInstance(rows[0]["RANK"], int)


class PrjTests(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(gems_shapes.prj_kind(UTM11_PRJ), "utm11")
        self.assertEqual(gems_shapes.prj_kind(CUSTOM_TM_PRJ), "utm11")
        self.assertEqual(gems_shapes.prj_kind(
            'PROJCS["WGS 84 / UTM zone 11N",GEOGCS["WGS 84"]]'), "utm11")
        self.assertEqual(gems_shapes.prj_kind(GEO_PRJ), "lonlat")
        self.assertIsNone(gems_shapes.prj_kind(
            CUSTOM_TM_PRJ.replace("-117.0", "-111.0")))
        self.assertIsNone(gems_shapes.prj_kind(
            UTM11_PRJ.replace("NAD_1983_UTM_Zone_11N", "NAD_1983_UTM_Zone_12N")
            .replace("-117.0", "-111.0")))
        self.assertIsNone(gems_shapes.prj_kind(""))
        self.assertIsNone(gems_shapes.prj_kind(None))


class GeojsonTests(unittest.TestCase):
    def test_geometries_properties_and_crs(self):
        recs = gems_shapes.read_shp(shp_bytes(3, [
            polyline([[(0, 0), (1, 1)]]),
            polyline([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
            NULL,
            point(7, 8),
        ]))
        rows = [{"n": 1}, None, {"n": 3}, {"n": 4}]
        doc = gems_shapes.to_geojson(recs, rows, "utm11")
        self.assertEqual(doc["crs"]["properties"]["name"],
                         "urn:ogc:def:crs:EPSG::26911")
        kinds = [f["geometry"]["type"] for f in doc["features"]]
        # the deleted row (1) and the null shape (2) are left out
        self.assertEqual(kinds, ["LineString", "Point"])
        self.assertEqual(doc["features"][0]["properties"], {"n": 1})
        self.assertEqual(doc["features"][1]["geometry"]["coordinates"],
                         [7.0, 8.0])
        both = gems_shapes.to_geojson(recs, None, "lonlat")
        self.assertNotIn("crs", both)
        self.assertEqual([f["geometry"]["type"] for f in both["features"]],
                         ["LineString", "MultiLineString", "Point"])

    def test_polygon_boundaries(self):
        recs = gems_shapes.read_shp(shp_bytes(5, [
            polyline([[(0, 0), (4, 0), (4, 4), (0, 0)]], kind=5)]))
        poly = gems_shapes.to_geojson(recs)["features"][0]["geometry"]
        self.assertEqual(poly["type"], "Polygon")
        line = gems_shapes.to_geojson(recs, boundaries=True)["features"][0]
        self.assertEqual(line["geometry"]["type"], "MultiLineString")

    def test_output_feeds_gems_external_in_metres(self):
        recs = gems_shapes.read_shp(shp_bytes(3, [
            polyline([[(500050.0, 4399950.0), (500450.0, 4399950.0)]])]))
        doc = json.loads(json.dumps(gems_shapes.to_geojson(recs, None, "utm11")))
        lines = gems_external.lines_utm11(doc)
        self.assertEqual(lines[0][0], [(500050.0, 4399950.0),
                                       (500450.0, 4399950.0)])
        cells = gems_external.rasterize_lines(lines, (500000.0, 4400000.0),
                                              100.0, (5, 5))
        self.assertEqual(cells, {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)})


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lines_shp = shp_bytes(3, [polyline([[(1, 2), (3, 4)]]),
                                       polyline([[(5, 6), (7, 8)]])])
        self.lines_dbf = dbf_bytes(FIELDS, [("a", "0.5", "1"),
                                            ("b", "0.9", "2")])
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            z.writestr("pts.shp", shp_bytes(1, [point(1, 1)]))
        self.zip_path = os.path.join(self.tmp, "release.zip")
        with zipfile.ZipFile(self.zip_path, "w") as z:
            z.writestr("data/Lines.shp", self.lines_shp)
            z.writestr("data/Lines.dbf", self.lines_dbf)
            z.writestr("data/Lines.prj", UTM11_PRJ)
            z.writestr("inner.zip", inner.getvalue())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_layers_in_nested_zip(self):
        got = gems_shapes.list_layers(self.zip_path)
        self.assertEqual(got, [("Lines", "polyline", 2, "utm11"),
                               ("pts", "point", 1, "unknown")])

    def test_load_layer(self):
        doc = gems_shapes.load_layer(self.zip_path, "lines")
        self.assertIn("crs", doc)
        self.assertEqual([f["properties"]["NAME"] for f in doc["features"]],
                         ["a", "b"])
        with self.assertRaises(SystemExit):
            gems_shapes.load_layer(self.zip_path, "pts")  # no .prj
        pts = gems_shapes.load_layer(self.zip_path, "pts", assume_utm11=True)
        self.assertEqual(pts["features"][0]["geometry"]["coordinates"],
                         [1.0, 1.0])
        with self.assertRaises(SystemExit):
            gems_shapes.load_layer(self.zip_path)  # two layers: must choose
        with self.assertRaises(SystemExit):
            gems_shapes.load_layer(self.zip_path, "nothing")

    def test_plain_shapefile_and_cli(self):
        stem = os.path.join(self.tmp, "faults")
        for ext, data in ((".shp", self.lines_shp), (".dbf", self.lines_dbf),
                          (".prj", UTM11_PRJ.encode())):
            with open(stem + ext, "wb") as fh:
                fh.write(data)
        doc = gems_shapes.load_layer(stem + ".shp")
        self.assertEqual(len(doc["features"]), 2)
        out = os.path.join(self.tmp, "out.geojson")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = gems_shapes.main(["geojson", "--src", self.zip_path,
                                     "--layer", "Lines", "--out", out])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buf.getvalue())["features"],
                         {"LineString": 2})
        with open(out, encoding="utf-8") as fh:
            self.assertEqual(len(json.load(fh)["features"]), 2)


if __name__ == "__main__":
    unittest.main()
