"""Offline tests for gems_external (projection, GeoJSON, rasterising, paging).

Run with `python -m unittest -v`. The numpy tests are skipped when numpy is
not installed; nothing here touches the network.
"""

import os
import sys
import unittest
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gems_external as gx  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

CLARKE_1866_A = 6378206.4
CLARKE_1866_E2 = 0.00676866


class ProjectionTests(unittest.TestCase):
    def test_snyder_worked_example(self):
        # Snyder, Map Projections - A Working Manual, transverse Mercator
        # numerical example: Clarke 1866, 40.5 N 73.5 W, central meridian
        # 75 W, k0 0.9996 -> x 127106.47 m, y 4484124.43 m.
        x, y = gx.tm_forward(40.5, -73.5, -75.0, k0=0.9996,
                             a=CLARKE_1866_A, e2=CLARKE_1866_E2)
        self.assertAlmostEqual(x, 127106.47, delta=0.06)
        self.assertAlmostEqual(y, 4484124.43, delta=0.06)

    def test_central_meridian_on_the_equator(self):
        self.assertEqual(gx.central_meridian(11), -117.0)
        e, n = gx.utm_forward(0.0, -117.0)
        self.assertAlmostEqual(e, 500000.0, places=6)
        self.assertAlmostEqual(n, 0.0, places=6)

    def test_symmetric_about_the_central_meridian(self):
        e1, n1 = gx.utm_forward(39.5, -118.0)
        e2, n2 = gx.utm_forward(39.5, -116.0)
        self.assertAlmostEqual(e1 + e2, 1000000.0, places=6)
        self.assertAlmostEqual(n1, n2, places=6)
        self.assertLess(e1, 500000.0)

    def test_round_trip_across_the_zone(self):
        for lat, lon in [(39.5, -118.5), (41.0, -116.2), (38.2, -119.9),
                         (40.0, -117.0)]:
            e, n = gx.utm_forward(lat, lon)
            lat2, lon2 = gx.utm_inverse(e, n)
            self.assertAlmostEqual(lat2, lat, delta=2e-6)
            self.assertAlmostEqual(lon2, lon, delta=2e-6)

    def test_grid_bbox(self):
        # 100 km x 100 km centred on the central meridian
        lo_lon, lo_lat, hi_lon, hi_lat = gx.grid_bbox_lonlat(
            (450000.0, 4400000.0), 100.0, (1000, 1000))
        self.assertAlmostEqual(lo_lon + hi_lon, -234.0, places=6)
        self.assertTrue(-118.0 < lo_lon < -117.5)
        self.assertTrue(-116.5 < hi_lon < -116.0)
        self.assertTrue(38.0 < lo_lat < 39.5 < hi_lat < 40.5)


class GeoJSONTests(unittest.TestCase):
    DOC = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"fault_name": "A"},
         "geometry": {"type": "LineString",
                      "coordinates": [[-117.0, 0.0], [-117.0, 1.0]]}},
        {"type": "Feature", "properties": {},
         "geometry": {"type": "MultiLineString",
                      "coordinates": [[[-118.0, 39.5], [-118.0, 39.6]],
                                      [[-116.0, 39.5]]]}},
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Polygon",
                      "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Point", "coordinates": [-117.0, 39.0]}},
        {"type": "Feature", "properties": {}, "geometry": None}]}

    def test_parse_lines_keeps_line_parts_only(self):
        lines = gx.parse_lines(self.DOC)
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0][1], {"fault_name": "A"})
        self.assertEqual(lines[2][0], [(-116.0, 39.5)])

    def test_lonlat_is_projected(self):
        lines = gx.lines_utm11(self.DOC)
        e, n = lines[0][0][0]
        self.assertAlmostEqual(e, 500000.0, places=6)
        self.assertAlmostEqual(n, 0.0, places=6)
        self.assertEqual(lines[0][1]["fault_name"], "A")
        west = lines[1][0][0]
        east = lines[2][0][0]
        self.assertAlmostEqual(west[0] + east[0], 1000000.0, places=6)

    def test_metres_are_kept(self):
        line = {"type": "Feature", "properties": {},
                "geometry": {"type": "LineString",
                             "coordinates": [[300000.0, 4400000.0],
                                             [300100.0, 4400000.0]]}}
        named = {"type": "FeatureCollection", "features": [line],
                 "crs": {"type": "name", "properties":
                         {"name": "urn:ogc:def:crs:EPSG::32611"}}}
        bare = {"type": "FeatureCollection", "features": [line]}
        for doc in (named, bare):
            self.assertEqual(gx.lines_utm11(doc)[0][0],
                             [(300000.0, 4400000.0), (300100.0, 4400000.0)])
        # CRS84 names lon/lat, so small numbers are still projected
        crs84 = {"type": "FeatureCollection", "features": self.DOC["features"],
                 "crs": {"type": "name", "properties":
                         {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}}
        self.assertAlmostEqual(gx.lines_utm11(crs84)[0][0][0][0], 500000.0,
                               places=6)


class RasterizeTests(unittest.TestCase):
    ORIGIN = (0.0, 1000.0)  # top-left corner; 100 m cells, 10 x 10

    def cells(self, coords):
        return gx.rasterize_lines([(coords, {})], self.ORIGIN, 100.0, (10, 10))

    def test_horizontal_line(self):
        self.assertEqual(self.cells([(50.0, 950.0), (450.0, 950.0)]),
                         {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)})

    def test_diagonal_line_stays_on_the_diagonal(self):
        self.assertEqual(self.cells([(50.0, 950.0), (350.0, 650.0)]),
                         {(0, 0), (1, 1), (2, 2), (3, 3)})

    def test_clipped_to_the_grid(self):
        self.assertEqual(self.cells([(-500.0, 950.0), (150.0, 950.0)]),
                         {(0, 0), (0, 1)})

    def test_single_vertex_and_several_lines(self):
        self.assertEqual(self.cells([(250.0, 750.0)]), {(2, 2)})
        both = gx.rasterize_lines(
            [([(250.0, 750.0)], {}), ([(50.0, 950.0), (150.0, 950.0)], {})],
            self.ORIGIN, 100.0, (10, 10))
        self.assertEqual(both, {(2, 2), (0, 0), (0, 1)})


class FetchTests(unittest.TestCase):
    BBOX = (-120.5, 38.0, -117.0, 41.0)

    def test_query_url(self):
        url = gx.query_url(self.BBOX, offset=2000)
        self.assertTrue(url.startswith(gx.QFAULTS_URL % 21 + "?"))
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertEqual(q["geometry"],
                         ["-120.500000,38.000000,-117.000000,41.000000"])
        self.assertEqual(q["where"], ["1=1"])
        self.assertEqual(q["outSR"], ["4326"])
        self.assertEqual(q["f"], ["geojson"])
        self.assertEqual(q["resultOffset"], ["2000"])

    def fake(self, pages):
        calls = []

        def get_json(url):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            offset = int(q["resultOffset"][0])
            calls.append(offset)
            return pages(offset)
        return get_json, calls

    def test_follows_pages_until_the_limit_clears(self):
        def pages(offset):
            if offset == 0:
                return {"features": [{"id": 1}, {"id": 2}],
                        "exceededTransferLimit": True}
            return {"features": [{"id": 3}],
                    "properties": {"exceededTransferLimit": False}}
        get_json, calls = self.fake(pages)
        feats = gx.fetch_features(self.BBOX, count=2, get_json=get_json)
        self.assertEqual([f["id"] for f in feats], [1, 2, 3])
        self.assertEqual(calls, [0, 2])

    def test_stops_when_the_server_ignores_the_offset(self):
        def pages(offset):
            return {"features": [{"id": 1}, {"id": 2}],
                    "exceededTransferLimit": True}
        get_json, calls = self.fake(pages)
        feats = gx.fetch_features(self.BBOX, count=2, get_json=get_json)
        self.assertEqual(len(feats), 2)
        self.assertEqual(calls, [0, 2])

    def test_service_error_raises(self):
        get_json, _ = self.fake(lambda offset: {"error": {"code": 400}})
        with self.assertRaises(RuntimeError):
            gx.fetch_features(self.BBOX, get_json=get_json)

    def test_cli_parses(self):
        with self.assertRaises(SystemExit):
            gx.main(["--help"])


@unittest.skipIf(np is None, "numpy not installed")
class MaskTests(unittest.TestCase):
    def layout(self):
        known = np.zeros((9, 9), dtype=bool)
        known[:, 2] = True
        ext = np.zeros((9, 9), dtype=bool)
        ext[:, 2] = True
        ext[:, 6] = True
        return known, ext

    def test_dilate(self):
        m = np.zeros((7, 7), dtype=bool)
        m[3, 3] = True
        d = gx.dilate(m, 2)
        self.assertEqual(int(d.sum()), 25)
        self.assertTrue(d[1, 1] and d[5, 5])
        self.assertFalse(d[0, 3])
        self.assertTrue((gx.dilate(m, 0) == m).all())

    def test_chessboard_distance_is_capped(self):
        m = np.zeros((9, 9), dtype=bool)
        m[0, 0] = True
        d = gx.chessboard_distance(m, 4)
        self.assertEqual(float(d[0, 0]), 0.0)
        self.assertEqual(float(d[1, 0]), 1.0)
        self.assertEqual(float(d[2, 3]), 3.0)
        self.assertEqual(float(d[3, 0]), 3.0)
        self.assertEqual(float(d[4, 0]), 4.0)
        self.assertEqual(float(d[8, 8]), 4.0)

    def test_candidates_skip_lines_near_known_faults(self):
        known, ext = self.layout()
        cand = gx.candidates(ext, known, 1)
        self.assertEqual(int(cand.sum()), 9)
        self.assertTrue(cand[:, 6].all())
        self.assertFalse(gx.candidates(ext, known, 4).any())

    def test_agreement(self):
        known, ext = self.layout()
        a = gx.agreement(ext, known, 1)
        self.assertEqual(a["known_pixels"], 9)
        self.assertEqual(a["external_pixels"], 18)
        self.assertAlmostEqual(a["known_within_1px_of_external"], 1.0)
        self.assertAlmostEqual(a["external_within_buffer_of_known"], 0.5)
        self.assertEqual(a["candidate_pixels"], 9)

    def test_external_bands(self):
        known, ext = self.layout()
        valid = np.ones((9, 9), dtype=bool)
        valid[0, :] = False
        out = gx.external_bands(ext, known, valid)
        self.assertEqual(out.shape, (3, 9, 9))
        self.assertTrue(np.isnan(out[:, 0, :]).all())
        self.assertEqual(float(out[0, 4, 6]), 1.0)
        self.assertEqual(float(out[1, 4, 4]), 2.0)
        self.assertEqual(float(out[2, 4, 2]), 0.0)
        self.assertEqual(float(out[2, 4, 6]), 1.0)


if __name__ == "__main__":
    unittest.main()
