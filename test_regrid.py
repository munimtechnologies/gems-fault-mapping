"""Offline tests for gems_regrid (ASCII grids, resampling, point bands) and
for how the baseline treats a band with a gap.

Run with `python -m unittest -v`. numpy is needed; the scipy and rasterio
tests are skipped when those are missing. Every expected value below was
worked out by hand from the grid geometry.
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False
try:
    import rasterio
    from rasterio.transform import from_origin
    HAVE_RASTERIO = True
except ImportError:  # pragma: no cover
    HAVE_RASTERIO = False

import gems_baseline  # noqa: E402
import gems_external  # noqa: E402
import gems_regrid  # noqa: E402

UTM11_PRJ = (
    'PROJCS["NAD_1983_UTM_Zone_11N",GEOGCS["GCS_North_American_1983",'
    'DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,'
    '298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",'
    '0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
    'PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",-117.0],PARAMETER["Scale_Factor",0.9996],'
    'PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')

# 2 rows x 3 columns, the second row wrapped over two lines, one nodata.
ASC = ("ncols 3\nnrows 2\nxllcorner 1000\nyllcorner 2000\ncellsize 50\n"
       "NODATA_value -9999\n1 2 -9999\n4 5\n6\n")


def plane():
    """4 x 5 source grid, value 10 * row + col, 100 m cells, top-left (0, 400)."""
    values = np.add.outer(10.0 * np.arange(4), np.arange(5.0))
    return gems_regrid.Source("s", values, 0.0, 400.0, 100.0, 100.0)


@unittest.skipIf(np is None, "numpy not installed")
class AscTests(unittest.TestCase):
    def test_header_variants(self):
        head, n = gems_regrid.parse_asc_header(ASC.splitlines())
        self.assertEqual(n, 6)
        self.assertEqual(gems_regrid.asc_geometry(head),
                         (1000.0, 2100.0, 50.0, 50.0, 2, 3, -9999.0))
        centre, _ = gems_regrid.parse_asc_header(
            ["ncols 3", "nrows 2", "xllcenter 1025", "yllcenter 2025",
             "cellsize 50"])
        self.assertEqual(gems_regrid.asc_geometry(centre),
                         (1000.0, 2100.0, 50.0, 50.0, 2, 3, None))
        dxdy, _ = gems_regrid.parse_asc_header(
            ["NCOLS 3", "NROWS 2", "XLLCORNER 0", "YLLCORNER 0", "DX 50",
             "DY 25"])
        self.assertEqual(gems_regrid.asc_geometry(dxdy),
                         (0.0, 50.0, 50.0, 25.0, 2, 3, None))
        with self.assertRaises(ValueError):
            gems_regrid.asc_geometry({"ncols": 3, "nrows": 2,
                                      "xllcorner": 0, "yllcorner": 0})

    def test_wrapped_rows_nodata_and_crlf(self):
        for text in (ASC, ASC.replace("\n", "\r\n")):
            values, geo = gems_regrid.read_asc_text(text)
            self.assertEqual(geo, (1000.0, 2100.0, 50.0, 50.0))
            self.assertEqual(values.shape, (2, 3))
            self.assertEqual(values[0, :2].tolist(), [1.0, 2.0])
            self.assertTrue(np.isnan(values[0, 2]))
            self.assertEqual(values[1].tolist(), [4.0, 5.0, 6.0])
        with self.assertRaises(ValueError):
            gems_regrid.read_asc_text(ASC.replace("\n6\n", "\n"))


@unittest.skipIf(np is None, "numpy not installed")
class ResampleTests(unittest.TestCase):
    def test_bilinear_reproduces_a_plane(self):
        # 50 m target cells: centre (r, c) sits at fractional source index
        # 0.5 r - 0.25 (rows) and 0.5 c - 0.25 (cols); the half cell beyond
        # the outer source centres takes the edge value.
        out = gems_regrid.regrid(plane(), (0.0, 400.0), 50.0, (8, 10))
        r = np.arange(8)[:, None]
        c = np.arange(10)[None, :]
        want = (10 * np.clip(0.5 * r - 0.25, 0, 3)
                + np.clip(0.5 * c - 0.25, 0, 4))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.allclose(out, want, atol=1e-5))

    def test_outside_is_nan_and_aligned_cells_copy(self):
        src = plane()
        for method in ("bilinear", "nearest"):
            out = gems_regrid.regrid(src, (-100.0, 500.0), 100.0, (6, 7),
                                     method)
            self.assertTrue(np.isnan(out[0]).all() and np.isnan(out[5]).all())
            self.assertTrue(np.isnan(out[:, 0]).all()
                            and np.isnan(out[:, 6]).all())
            self.assertTrue(np.array_equal(out[1:5, 1:6],
                                           src.values.astype(np.float32)))

    def test_nan_neighbours_are_left_out(self):
        values = plane().values.copy()
        values[1, 1] = np.nan
        src = gems_regrid.Source("s", values, 0.0, 400.0, 100.0, 100.0)
        aligned = gems_regrid.regrid(src, (-100.0, 500.0), 100.0, (6, 7))
        self.assertTrue(np.isnan(aligned[2, 2]))
        self.assertEqual(float(aligned[2, 3]), 12.0)
        # halfway between source cells (1, 1) and (1, 2): only 12 is left
        mid = gems_regrid.regrid(src, (150.0, 300.0), 100.0, (1, 1))
        self.assertAlmostEqual(float(mid[0, 0]), 12.0, places=5)
        values[1, 2] = np.nan
        mid = gems_regrid.regrid(src, (150.0, 300.0), 100.0, (1, 1))
        self.assertTrue(np.isnan(mid[0, 0]))

    def test_nearest_takes_the_containing_cell(self):
        src = plane()
        out = gems_regrid.regrid(src, (0.0, 400.0), 50.0, (8, 10), "nearest")
        for r in range(8):
            for c in range(10):
                self.assertEqual(float(out[r, c]), src.values[r // 2, c // 2])


@unittest.skipIf(np is None, "numpy not installed")
class PointTests(unittest.TestCase):
    def test_projection_rule(self):
        doc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {},
             "geometry": {"type": "Point", "coordinates": [-117.0, 40.0]}},
            {"type": "Feature", "properties": {},
             "geometry": {"type": "MultiPoint",
                          "coordinates": [[-118.0, 39.5], [-116.5, 40.2]]}},
            {"type": "Feature", "properties": {},
             "geometry": {"type": "LineString",
                          "coordinates": [[0, 0], [1, 1]]}}]}
        pts = gems_regrid.points_utm11(doc)
        self.assertEqual(len(pts), 3)
        self.assertAlmostEqual(pts[0][0], 500000.0, places=6)
        self.assertEqual(pts[1], gems_external.utm_forward(39.5, -118.0))
        metres = {"type": "FeatureCollection",
                  "crs": {"type": "name", "properties":
                          {"name": "urn:ogc:def:crs:EPSG::26911"}},
                  "features": [{"type": "Feature", "properties": {},
                                "geometry": {"type": "Point",
                                             "coordinates": [50, 60]}}]}
        self.assertEqual(gems_regrid.points_utm11(metres), [(50.0, 60.0)])
        big = {"type": "Point", "coordinates": [500050.0, 4399950.0]}
        self.assertEqual(gems_regrid.points_utm11(big),
                         [(500050.0, 4399950.0)])

    def test_window_counts(self):
        counts = np.zeros((5, 5), dtype=np.int32)
        counts[0, 0] = 1
        counts[2, 3] = 2
        got = gems_regrid.window_counts(counts, 1)
        for r in range(5):
            for c in range(5):
                want = int(counts[max(0, r - 1):r + 2, max(0, c - 1):c + 2].sum())
                self.assertEqual(int(got[r, c]), want)
        self.assertEqual([int(got[0, 0]), int(got[1, 1]), int(got[1, 2]),
                          int(got[3, 4]), int(got[4, 0])], [1, 1, 2, 2, 0])

    def test_distance_chessboard_fallback(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True
        d = gems_regrid.distance_px(mask, 4, exact=False)
        self.assertEqual([float(d[3, 3]), float(d[3, 6]), float(d[0, 0])],
                         [0.0, 3.0, 3.0])
        empty = gems_regrid.distance_px(np.zeros((3, 3), dtype=bool), 4)
        self.assertTrue((empty == 4.0).all())

    @unittest.skipIf(not HAVE_SCIPY, "scipy not installed")
    def test_distance_euclidean(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True
        d = gems_regrid.distance_px(mask, 4)
        self.assertEqual(float(d[3, 6]), 3.0)
        self.assertEqual(float(d[0, 0]), 4.0)  # sqrt(18) capped
        self.assertAlmostEqual(float(d[1, 2]), 5 ** 0.5, places=5)

    def test_point_bands_see_points_beyond_the_edge(self):
        # a point half a cell left of the grid, in row 2 (column -1)
        for exact in (False, True):
            dist, count = gems_regrid.point_bands(
                [(-50.0, 250.0)], (0.0, 500.0), 100.0, (5, 5), cap=3,
                radius=1, exact=exact)
            self.assertEqual([float(dist[2, 0]), float(dist[2, 1]),
                              float(dist[2, 2])], [1.0, 2.0, 3.0])
            self.assertEqual([int(count[1, 0]), int(count[2, 0]),
                              int(count[3, 0]), int(count[2, 1]),
                              int(count[0, 0])], [1, 1, 1, 0, 0])
        dist, count = gems_regrid.point_bands(
            [(10000.0, 10000.0)], (0.0, 500.0), 100.0, (5, 5), cap=3, radius=1)
        self.assertTrue((dist == 3.0).all() and (count == 0).all())


@unittest.skipIf(np is None, "numpy not installed")
class SourceFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            z.writestr("grids/grv_filthg.asc", ASC)
            z.writestr("grids/grv_filthg.prj", UTM11_PRJ)
        self.zip_path = os.path.join(self.tmp, "release.zip")
        with zipfile.ZipFile(self.zip_path, "w") as z:
            z.writestr("noprj.asc", ASC)
            z.writestr("inner.zip", inner.getvalue())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_nested_zip_member(self):
        src = gems_regrid.load_source(self.zip_path + "::grv_filthg")
        self.assertEqual(src.name, "grv_filthg")
        self.assertEqual(src.values.shape, (2, 3))
        self.assertEqual((src.left, src.top, src.dx, src.dy),
                         (1000.0, 2100.0, 50.0, 50.0))
        with self.assertRaises(SystemExit):
            gems_regrid.load_source(self.zip_path + "::nothing")

    def test_projection_is_required(self):
        with self.assertRaises(SystemExit):
            gems_regrid.load_source(self.zip_path + "::noprj")
        src = gems_regrid.load_source(self.zip_path + "::noprj",
                                      assume_utm11=True)
        self.assertEqual(src.values.shape, (2, 3))

    def test_plain_asc_with_prj(self):
        path = os.path.join(self.tmp, "heat.asc")
        with open(path, "w", encoding="latin-1") as fh:
            fh.write(ASC)
        with self.assertRaises(SystemExit):
            gems_regrid.load_source(path)
        with open(os.path.join(self.tmp, "heat.prj"), "w",
                  encoding="latin-1") as fh:
            fh.write(UTM11_PRJ)
        self.assertEqual(gems_regrid.load_source(path).name, "heat")

    def test_list_zip_grids(self):
        self.assertEqual(gems_regrid.list_zip_grids(self.zip_path), [
            ("noprj.asc", 3, 2, 50.0, "unknown"),
            ("inner.zip/grids/grv_filthg.asc", 3, 2, 50.0, "utm11")])


@unittest.skipIf(np is None or not HAVE_SCIPY, "numpy and scipy needed")
class BaselineGapTests(unittest.TestCase):
    def test_gap_and_its_edge_stay_missing(self):
        band = np.add.outer(np.arange(20.0), np.arange(30.0)).astype(np.float32)
        valid = np.ones((20, 30), dtype=bool)
        full = gems_baseline.band_features(band, valid)
        self.assertTrue(all(np.isfinite(f).all() for f in full))
        band[:, :10] = np.nan
        feats = gems_baseline.band_features(band, valid)
        self.assertTrue(np.isnan(feats[0][:, :10]).all())
        self.assertTrue(np.isfinite(feats[0][:, 10:]).all())
        for f in feats[1:]:
            # within EDGE_PX = 6 px of the gap (columns 10-15) the filters
            # are missing; from column 16 on they are computed
            self.assertTrue(np.isnan(f[:, :16]).all())
            self.assertTrue(np.isfinite(f[:, 16:]).all())


@unittest.skipIf(np is None or not HAVE_RASTERIO, "numpy and rasterio needed")
class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.grid = os.path.join(self.tmp, "grid.tif")
        g = np.ones((4, 5), dtype=np.float32)
        g[0, 0] = np.nan  # outside the data
        with rasterio.open(self.grid, "w", driver="GTiff", width=5, height=4,
                           count=1, dtype="float32", crs="EPSG:32611",
                           transform=from_origin(0.0, 400.0, 100.0, 100.0)) as dst:
            dst.write(g, 1)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_main(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = gems_regrid.main(argv)
        self.assertEqual(code, 0)
        return json.loads(buf.getvalue())

    def test_grid_command(self):
        asc = os.path.join(self.tmp, "src.asc")
        with open(asc, "w", encoding="latin-1") as fh:
            fh.write("ncols 5\nnrows 4\nxllcorner 0\nyllcorner 0\n"
                     "cellsize 100\nNODATA_value -9999\n")
            for r in range(4):
                fh.write(" ".join(str(10 * r + c) for c in range(5)) + "\n")
        with open(os.path.join(self.tmp, "src.prj"), "w") as fh:
            fh.write(UTM11_PRJ)
        out = os.path.join(self.tmp, "grids.tif")
        report = self.run_main(["grid", "--grid", self.grid, "--src", asc,
                                "--out", out])
        self.assertEqual(report["sources"][0]["name"], "src")
        self.assertEqual(report["sources"][0]["covered"], 1.0)
        with rasterio.open(out) as src:
            band = src.read(1)
            self.assertEqual(src.descriptions[0], "src")
        self.assertTrue(np.isnan(band[0, 0]))
        self.assertEqual(float(band[2, 3]), 23.0)
        self.assertEqual(float(band[3, 4]), 34.0)

    def test_points_command(self):
        pts = os.path.join(self.tmp, "pts.geojson")
        with open(pts, "w", encoding="utf-8") as fh:
            json.dump({"type": "FeatureCollection",
                       "crs": {"type": "name", "properties":
                               {"name": "urn:ogc:def:crs:EPSG::26911"}},
                       "features": [{"type": "Feature", "properties": {},
                                     "geometry": {"type": "Point",
                                                  "coordinates": [250, 150]}}]},
                      fh)
        out = os.path.join(self.tmp, "points.tif")
        report = self.run_main(["points", "--grid", self.grid, "--points", pts,
                                "--out", out, "--cap", "3", "--radius", "1"])
        self.assertEqual(report["layers"][0]["inside_grid"], 1)
        with rasterio.open(out) as src:
            dist, count = src.read(1), src.read(2)
            self.assertEqual(list(src.descriptions), ["pts_dist_px", "pts_n_r1"])
        self.assertEqual(float(dist[2, 2]), 0.0)
        self.assertTrue(np.isnan(dist[0, 0]))
        self.assertEqual([float(count[2, 2]), float(count[1, 1]),
                          float(count[0, 4])], [1.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
