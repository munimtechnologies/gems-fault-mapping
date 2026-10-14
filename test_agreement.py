"""Tests for gems_agreement.py. Run: python -m unittest -v test_agreement"""

import json
import os
import shutil
import struct
import tempfile
import unittest
import zipfile

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

import gems_agreement
import gems_external

GEOGCS = ('GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",'
          'SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",'
          '0.0],UNIT["Degree",0.0174532925199433]]')
UTM11 = ('PROJCS["NAD_1983_UTM_Zone_11N",GEOGCS["GCS_North_American_1983",'
         'DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,'
         '298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",'
         '0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
         'UNIT["Meter",1.0]]')


def polyline_shp(lines):
    """Bytes of a polyline (type 3) shapefile, one single-part record per line."""
    recs = b""
    for i, pts in enumerate(lines, 1):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        content = struct.pack("<i4d2i", 3, min(xs), min(ys), max(xs), max(ys),
                              1, len(pts))
        content += struct.pack("<i", 0)
        content += b"".join(struct.pack("<2d", x, y) for x, y in pts)
        recs += struct.pack(">2i", i, len(content) // 2) + content
    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(recs)) // 2)
    header += struct.pack("<2i", 1000, 3) + struct.pack("<8d", *([0.0] * 8))
    return header + recs


class PickLayerTests(unittest.TestCase):
    TYPES = {"gravityLineations_nvML": 3, "gravityEndpoints_nvML": 1,
             "magneticLineations_nvML": 13, "tdts2020nvML_faultLines": 3,
             "tdts2020nvML_faultVertices": 1, "quaternaryFaults_nvML": 3}

    def test_lines_only(self):
        pick = gems_agreement.pick_layer
        self.assertEqual(pick(self.TYPES, ("grav", "lineation")),
                         "gravityLineations_nvML")
        self.assertEqual(pick(self.TYPES, ("mag", "lineation")),
                         "magneticLineations_nvML")
        self.assertEqual(pick(self.TYPES, ("tdts", "faultline")),
                         "tdts2020nvML_faultLines")
        self.assertEqual(pick(self.TYPES, ("quaternary", "fault")),
                         "quaternaryFaults_nvML")
        self.assertIsNone(pick(self.TYPES, ("grav", "endpoint")))
        self.assertIsNone(pick(self.TYPES, ("heat", "flow")))

    def test_related(self):
        self.assertTrue(gems_agreement.related("potential_field",
                                               "gravity_lineations"))
        self.assertTrue(gems_agreement.related("magnetic_lineations",
                                               "potential_field"))
        self.assertFalse(gems_agreement.related("quaternary_faults",
                                                "potential_field"))


class LayerLinesTests(unittest.TestCase):
    def test_lonlat_is_projected(self):
        shp = polyline_shp([[(-117.1, 39.05), (-117.1, 39.15)]])
        lines, how = gems_agreement.layer_lines((shp, None, GEOGCS))
        self.assertEqual(how, "lon/lat, projected")
        self.assertEqual(len(lines), 1)
        x, y = gems_external.utm_forward(39.05, -117.1)
        self.assertAlmostEqual(lines[0][0][0][0], x, places=3)
        self.assertAlmostEqual(lines[0][0][0][1], y, places=3)

    def test_utm_prj_kept(self):
        pts = [(400000.0, 4300000.0), (400500.0, 4300000.0)]
        lines, how = gems_agreement.layer_lines((polyline_shp([pts]), None,
                                                 UTM11))
        self.assertEqual(how, "UTM 11N (.prj)")
        self.assertEqual(lines[0][0], pts)

    def test_no_prj_metres_assumed(self):
        pts = [(400000.0, 4300000.0), (400500.0, 4300000.0)]
        lines, how = gems_agreement.layer_lines((polyline_shp([pts]), None,
                                                 None))
        self.assertTrue(how.startswith("assumed UTM 11N"))
        self.assertEqual(lines[0][0], pts)


@unittest.skipIf(np is None, "numpy not installed")
class GridTests(unittest.TestCase):
    BBOX = (-117.5, 39.0, -117.0, 39.5)

    def test_grid_covers_box(self):
        (x0, y0), (h, w) = gems_agreement.study_grid(self.BBOX)
        self.assertEqual(x0 % 100, 0)
        self.assertEqual(y0 % 100, 0)
        for lat in (39.0, 39.5):
            for lon in (-117.5, -117.0):
                x, y = gems_external.utm_forward(lat, lon)
                self.assertTrue(x0 <= x <= x0 + w * 100.0)
                self.assertTrue(y0 - h * 100.0 <= y <= y0)
        # about 43 km wide and 55 km tall
        self.assertTrue(420 <= w <= 450, w)
        self.assertTrue(545 <= h <= 565, h)

    def test_mask_fills_box(self):
        origin, shape = gems_agreement.study_grid(self.BBOX)
        mask = gems_agreement.study_mask(origin, shape, self.BBOX)
        self.assertEqual(mask.shape, shape)
        self.assertTrue(mask[shape[0] // 2, shape[1] // 2])
        self.assertGreater(mask.mean(), 0.9)


@unittest.skipIf(np is None, "numpy not installed")
class CompareTests(unittest.TestCase):
    def lines(self):
        truth = np.zeros((20, 40), dtype=bool)
        truth[:, 10] = True
        pred = np.zeros((20, 40), dtype=bool)
        pred[:, 12] = True
        return pred, truth, np.ones((20, 40), dtype=bool)

    def test_identical(self):
        _, truth, mask = self.lines()
        r = gems_agreement.compare(truth, truth, mask, shifts=())
        self.assertEqual(r["score"], 1.0)
        self.assertEqual(r["truth_cells"], 20)
        self.assertEqual(r["truth_near_pred"], 1.0)
        self.assertIsNone(r["chance_score"])
        self.assertIsNone(r["lift"])

    def test_two_pixels_apart(self):
        # Every truth cell's best match is 2 px away: k = 1/3, so
        # TP = 20/3, FN = 40/3, FP = 20 * 2/3 = 40/3 and
        # DTI = (20/3) / (20/3 + 0.2 * 40/3 + 0.8 * 40/3) = 1/3.
        pred, truth, mask = self.lines()
        r = gems_agreement.compare(pred, truth, mask, shifts=((0, 20),))
        self.assertAlmostEqual(r["score"], 0.3333, places=4)
        self.assertAlmostEqual(r["tp"], 6.7, places=6)
        self.assertAlmostEqual(r["fp"], 13.3, places=6)
        self.assertEqual(r["truth_near_pred"], 1.0)
        self.assertEqual(r["pred_near_truth"], 1.0)
        # shifted 20 columns the prediction is 22 px from the truth
        self.assertEqual(r["chance_scores"], [0.0])
        self.assertIsNone(r["lift"])

    def test_mask_drops_cells(self):
        pred, truth, mask = self.lines()
        mask[10:, :] = False
        r = gems_agreement.compare(pred, truth, mask, shifts=())
        self.assertEqual(r["truth_cells"], 10)
        self.assertEqual(r["pred_cells"], 10)
        self.assertAlmostEqual(r["score"], 0.3333, places=4)


@unittest.skipIf(np is None, "numpy not installed")
class ProfileTests(unittest.TestCase):
    def lines(self):
        truth = np.zeros((20, 40), dtype=bool)
        truth[:, 10] = True
        pred = np.zeros((20, 40), dtype=bool)
        pred[:, 12] = True
        return pred, truth, np.ones((20, 40), dtype=bool)

    def test_two_pixels_apart(self):
        # The truth column is 2 px from the prediction (Euclidean and
        # chessboard alike) and 22 px from its copy shifted 20 columns.
        pred, truth, mask = self.lines()
        prof = gems_agreement.near_profile(pred, truth, mask, radii=(1, 2, 5),
                                           shifts=((0, 20),))
        self.assertEqual([p["radius_px"] for p in prof], [1, 2, 5])
        self.assertEqual([p["radius_m"] for p in prof], [100.0, 200.0, 500.0])
        self.assertEqual([p["share"] for p in prof], [0.0, 1.0, 1.0])
        self.assertEqual([p["chance_share"] for p in prof], [0.0, 0.0, 0.0])
        self.assertEqual([p["lift"] for p in prof], [None, None, None])

    def test_one_pixel_shift_as_chance(self):
        # Identical maps; the "chance" copy sits 1 px over, so it reaches
        # the truth only from radius 1 on and the lift there is 1.
        _, truth, mask = self.lines()
        prof = gems_agreement.near_profile(truth, truth, mask, radii=(0, 1, 3),
                                           shifts=((0, 1),))
        self.assertEqual([p["share"] for p in prof], [1.0, 1.0, 1.0])
        self.assertEqual([p["chance_share"] for p in prof], [0.0, 1.0, 1.0])
        self.assertEqual([p["lift"] for p in prof], [None, 1.0, 1.0])

    def test_empty(self):
        pred, truth, mask = self.lines()
        self.assertEqual(gems_agreement.near_profile(
            pred, np.zeros_like(truth), mask), [])
        self.assertEqual(gems_agreement.near_profile(pred, truth, mask,
                                                     radii=()), [])


@unittest.skipIf(np is None, "numpy not installed")
class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        line = [(-117.1, 39.05), (-117.1, 39.15)]
        self.src = os.path.join(self.tmp, "release.zip")
        with zipfile.ZipFile(self.src, "w") as zf:
            zf.writestr("faults/quaternaryFaults_nvML.shp", polyline_shp([line]))
            zf.writestr("faults/quaternaryFaults_nvML.prj", GEOGCS)
            zf.writestr("geophys/gravityLineations_nvML.shp", polyline_shp([line]))
            zf.writestr("geophys/gravityLineations_nvML.prj", GEOGCS)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_report(self, *extra):
        out = os.path.join(self.tmp, "out")
        # "--bbox=..." because argparse reads a lone "-117.2,..." as an option
        code = gems_agreement.main(["run", "--src", self.src, "--out-dir", out,
                                    "--bbox=-117.2,39.0,-117.0,39.2"]
                                   + list(extra))
        self.assertEqual(code, 0)
        with open(os.path.join(out, "agreement.json"), encoding="utf-8") as fh:
            report = json.load(fh)
        with open(os.path.join(out, "agreement.md"), encoding="utf-8") as fh:
            return report, fh.read()

    def test_end_to_end_on_a_small_zip(self):
        report, md = self.run_report()
        self.assertEqual(sorted(report["maps"]),
                         ["gravity_lineations", "quaternary_faults"])
        self.assertEqual(report["maps"]["quaternary_faults"]["layer"],
                         "quaternaryFaults_nvML")
        self.assertIn("tdts_fault_lines", report["missing"])
        self.assertIn("not used", report["grid"]["footprint"])
        self.assertEqual(report["profile_radii_px"], [1, 3, 5, 10, 20, 50])
        self.assertEqual(len(report["pairs"]), 2)
        for row in report["pairs"]:
            self.assertEqual(row["score"], 1.0)
            self.assertGreater(row["truth_cells"], 100)
            # both maps are the same line, so every truth cell is on it
            self.assertEqual([p["share"] for p in row["profile"]], [1.0] * 6)
        self.assertIn("| 5000 m |", md)

    def test_profile_off(self):
        report, md = self.run_report("--radii=")
        self.assertEqual(report["profile_radii_px"], [])
        for row in report["pairs"]:
            self.assertEqual(row["profile"], [])
        self.assertNotIn("5000 m", md)


if __name__ == "__main__":
    unittest.main()
