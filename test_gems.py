"""Offline tests for gems_metric, gems_baseline, gems_extend and gems_lidar.

Run with `python -m unittest -v`. The numpy tests are skipped when numpy is
not installed (and the raster test when rasterio is not); the rest use the
standard library only.
"""

import math
import os
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gems_baseline  # noqa: E402
import gems_extend  # noqa: E402
import gems_lidar  # noqa: E402
import gems_metric  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None


def grid(h, w, cells=(), value=1.0):
    g = [[0.0] * w for _ in range(h)]
    for y, x in cells:
        g[y][x] = value
    return g


def column(h, x):
    return [(y, x) for y in range(h)]


class KernelTests(unittest.TestCase):
    def test_offsets_for_three_pixel_radius(self):
        offs = gems_metric.kernel_offsets(3.0)
        self.assertEqual(len(offs), 25)  # |dy|, |dx| <= 2; (0, 3) weighs 0
        self.assertEqual(offs[0], (0, 0, 1.0))
        weights = {(dy, dx): w for dy, dx, w in offs}
        self.assertAlmostEqual(weights[(0, 1)], 2.0 / 3.0)
        self.assertAlmostEqual(weights[(1, 1)], 1.0 - math.sqrt(2) / 3.0)
        self.assertAlmostEqual(weights[(2, 0)], 1.0 / 3.0)
        self.assertNotIn((0, 3), weights)

    def test_worked_example_from_problem_page(self):
        self.assertAlmostEqual(gems_metric.score_from_counts(3.0, 1.89, 2.0),
                               3.0 / 4.978, places=6)

    def test_threshold_rule(self):
        self.assertAlmostEqual(gems_metric.threshold_rule(0.5), 0.1)
        self.assertAlmostEqual(gems_metric.threshold_rule(0.0), 0.0)
        # general alpha/beta: lambda a / (1 - lambda (1 - b) + lambda a)
        self.assertAlmostEqual(gems_metric.threshold_rule(0.5, 0.5, 0.5),
                               0.25 / (1 - 0.25 + 0.25))


class ReferenceTests(unittest.TestCase):
    def test_perfect_prediction(self):
        t = grid(7, 7, column(7, 3))
        r = gems_metric.dti_reference(t, t)
        self.assertEqual(r["n"], 7)
        self.assertAlmostEqual(r["tp"], 7.0)
        self.assertAlmostEqual(r["fp"], 0.0)
        self.assertGreater(r["score"], 0.999)

    def test_one_pixel_shift_scores_two_thirds(self):
        t = grid(7, 7, column(7, 2))
        p = grid(7, 7, column(7, 3))
        r = gems_metric.dti_reference(p, t)
        self.assertAlmostEqual(r["tp"], 7 * 2.0 / 3.0)
        self.assertAlmostEqual(r["fn"], 7 / 3.0)
        self.assertAlmostEqual(r["fp"], 7 / 3.0)
        self.assertAlmostEqual(r["score"], 2.0 / 3.0, places=6)

    def test_empty_prediction_scores_zero(self):
        t = grid(5, 5, [(2, 2)])
        r = gems_metric.dti_reference(grid(5, 5), t)
        self.assertEqual(r["score"], 0.0)
        self.assertAlmostEqual(r["fn"], 1.0)

    def test_far_prediction_is_all_false_positive(self):
        t = grid(9, 9, [(0, 0)])
        p = grid(9, 9, [(8, 8)], value=0.5)
        r = gems_metric.dti_reference(p, t)
        self.assertAlmostEqual(r["tp"], 0.0)
        self.assertAlmostEqual(r["fp"], 0.5)

    def test_half_confidence_on_truth(self):
        t = grid(5, 5, [(2, 2)])
        p = grid(5, 5, [(2, 2)], value=0.5)
        r = gems_metric.dti_reference(p, t)
        self.assertAlmostEqual(r["tp"], 0.5)
        self.assertAlmostEqual(r["fp"], 0.0)
        self.assertAlmostEqual(r["score"], 0.5 / (0.5 + 0.8 * 0.5), places=6)

    def test_nan_none_and_clipping(self):
        t = grid(3, 3, [(1, 1)])
        p = [[float("nan"), None, 0.0], [0.0, 7.0, 0.0], [-1.0, 0.0, 0.0]]
        r = gems_metric.dti_reference(p, t)
        self.assertAlmostEqual(r["tp"], 1.0)
        self.assertAlmostEqual(r["fp"], 0.0)

    def test_mask_ignores_outside_pixels(self):
        t = grid(5, 5, [(0, 0), (4, 4)])
        p = grid(5, 5, [(4, 4), (0, 4)])
        mask = [[x >= 3 for x in range(5)] for _ in range(5)]
        r = gems_metric.dti_reference(p, t, mask=mask)
        self.assertEqual(r["n"], 1)  # (0, 0) is outside the mask
        self.assertAlmostEqual(r["tp"], 1.0)
        self.assertAlmostEqual(r["fp"], 1.0)  # (0, 4) is far from (4, 4)

    def test_partial_credit_takes_best_match(self):
        t = grid(7, 7, [(3, 3)])
        p = grid(7, 7, [(3, 4), (3, 5)])  # 1 px away (2/3) and 2 px (1/3)
        r = gems_metric.dti_reference(p, t)
        self.assertAlmostEqual(r["tp"], 2.0 / 3.0)
        self.assertAlmostEqual(r["fp"], 1.0 / 3.0 + 2.0 / 3.0)


class SubmissionMetaTests(unittest.TestCase):
    REF = {"epsg": 32611, "res": (100.0, 100.0), "width": 3292,
           "height": 3730, "bounds": (0.0, 0.0, 329200.0, 373000.0),
           "count": 19, "dtype": "float32"}

    def good(self):
        meta = dict(self.REF, count=1, vmin=0.0, vmax=1.0)
        return meta

    def test_good_submission_has_no_problems(self):
        self.assertEqual(gems_baseline.check_submission_meta(self.good(),
                                                             self.REF), [])

    def test_each_problem_is_reported(self):
        bad = dict(self.good(), epsg=4326, res=(30.0, 30.0), width=10,
                   count=2, dtype="float64", vmin=-0.5, vmax=2.0,
                   bounds=(1.0, 0.0, 329200.0, 373000.0))
        problems = gems_baseline.check_submission_meta(bad, self.REF)
        text = " ".join(problems)
        for word in ("EPSG:32611", "100 m", "width", "bounds", "1 band",
                     "float32", ">= 0", "<= 1"):
            self.assertIn(word, text)

    def test_cli_parses(self):
        with self.assertRaises(SystemExit):
            gems_baseline.main(["--help"])

    def test_band_name_falls_back_to_tag_then_index(self):
        self.assertEqual(gems_baseline.band_name("TMI", {}, 1), "TMI")
        # numeric_features.tif keeps its names in a 'description' tag
        self.assertEqual(gems_baseline.band_name(
            None, {"description": "Depth to basement"}, 15),
            "Depth to basement")
        self.assertEqual(gems_baseline.band_name(None, None, 3), "band3")


@unittest.skipIf(np is None, "numpy not installed")
class NumpyTests(unittest.TestCase):
    def test_matches_reference_on_random_grids(self):
        rng = random.Random(7)
        for trial in range(25):
            h, w = rng.randint(1, 12), rng.randint(1, 12)
            t = [[1.0 if rng.random() < 0.15 else 0.0 for _ in range(w)]
                 for _ in range(h)]
            p = [[rng.choice([0.0, 0.0, 1.0, rng.random()]) for _ in range(w)]
                 for _ in range(h)]
            mask = None
            if trial % 3 == 0:
                mask = [[rng.random() < 0.8 for _ in range(w)] for _ in range(h)]
            ref = gems_metric.dti_reference(p, t, mask=mask)
            fast = gems_metric.dti(np.array(p), np.array(t),
                                   mask=None if mask is None else np.array(mask))
            for key in ("tp", "fp", "fn", "score"):
                self.assertAlmostEqual(ref[key], fast[key], places=4,
                                       msg="%s trial %d" % (key, trial))
            self.assertEqual(ref["n"], fast["n"])

    def test_sweep_thresholds(self):
        t = np.zeros((9, 9))
        t[:, 4] = 1
        prob = np.zeros((9, 9))
        prob[:, 4] = 0.3
        prob[0, 0] = 0.9  # a confident false positive
        table = dict(gems_metric.sweep_thresholds(prob, t, [0.2, 0.5]))
        self.assertGreater(table[0.2], table[0.5])
        self.assertAlmostEqual(table[0.2], 9 / (9 + 0.2 * 1), places=5)

    def test_fold_map_and_sampling(self):
        folds = gems_baseline.fold_map((10, 7), block=4, folds=3, seed=1)
        self.assertEqual(folds.shape, (10, 7))
        self.assertTrue(((folds >= 0) & (folds < 3)).all())
        self.assertTrue((folds[:4, :4] == folds[0, 0]).all())
        labels = np.zeros((10, 7), dtype=bool)
        labels[2, :] = True
        valid = np.ones((10, 7), dtype=bool)
        idx = gems_baseline.sample_pixels(labels, valid, 2.0, seed=0)
        self.assertEqual(int(labels.ravel()[idx].sum()), 7)
        self.assertEqual(idx.size, 21)
        self.assertTrue((np.diff(idx) > 0).all())

    def test_exclusion_mask_ignores_known_faults(self):
        new = np.zeros((9, 9))
        new[:, 1] = 1
        known = np.zeros((9, 9), dtype=bool)
        known[:, 6] = True
        pred = np.zeros((9, 9))
        pred[:, 1] = 1
        pred[:, 6] = 1  # on the known fault: masked, costs nothing
        r = gems_metric.dti(pred, new, mask=~known)
        self.assertAlmostEqual(r["fp"], 0.0)
        pred[:, 7] = 1  # next to the known fault: a full false positive
        r = gems_metric.dti(pred, new, mask=~known)
        self.assertAlmostEqual(r["fp"], 9.0)


@unittest.skipIf(np is None, "numpy not installed")
class ExtendTests(unittest.TestCase):
    def test_extension_steps_are_distinct_and_connected(self):
        self.assertEqual(gems_extend.extension_steps(1.5, 0.0, 2),
                         [(-1, 0), (-2, 0)])
        self.assertEqual(gems_extend.extension_steps(-1.5, -1.5, 2),
                         [(1, 1), (2, 2)])
        steps = gems_extend.extension_steps(-0.5, -0.866, 3)
        self.assertEqual(len(set(steps)), 3)
        prev = (0, 0)
        for s in steps:
            self.assertLessEqual(max(abs(s[0] - prev[0]), abs(s[1] - prev[1])), 1)
            prev = s
        self.assertEqual(gems_extend.extension_steps(0.0, 0.0, 3), [])

    def test_vertical_line_tips_and_extension(self):
        known = np.zeros((15, 15), dtype=bool)
        known[2:9, 5] = True  # rows 2..8
        t = gems_extend.tips(known)
        self.assertEqual(int(t.sum()), 2)
        self.assertTrue(t[2, 5] and t[8, 5])
        ext = gems_extend.extend_tips(known, length=2)
        expect = {(1, 5), (0, 5), (9, 5), (10, 5)}
        self.assertEqual(set(zip(*map(list, np.nonzero(ext)))), expect)
        self.assertFalse((ext & known).any())

    def test_diagonal_line_extends_diagonally(self):
        known = np.zeros((14, 14), dtype=bool)
        for i in range(3, 10):
            known[i, i] = True
        ext = gems_extend.extend_tips(known, length=2)
        got = set(zip(*map(list, np.nonzero(ext))))
        self.assertEqual(got, {(2, 2), (1, 1), (10, 10), (11, 11)})

    def test_edge_isolated_and_short_segments(self):
        known = np.zeros((12, 12), dtype=bool)
        known[0:7, 0] = True  # touches the top edge
        ext = gems_extend.extend_tips(known, length=3)
        got = set(zip(*map(list, np.nonzero(ext))))
        self.assertEqual(got, {(7, 0), (8, 0), (9, 0)})
        lone = np.zeros((9, 9), dtype=bool)
        lone[4, 4] = True
        lone[7, 1] = True
        lone[7, 2] = True  # a 2-pixel stub: too short for a direction
        self.assertFalse(gems_extend.tips(lone).any())
        self.assertFalse(gems_extend.extend_tips(lone).any())

    def test_t_junction_has_three_tips(self):
        known = np.zeros((20, 20), dtype=bool)
        known[7, 2:13] = True   # bar, cols 2..12
        known[8:15, 7] = True   # stem, rows 8..14
        t = gems_extend.tips(known)
        self.assertEqual(int(t.sum()), 3)
        self.assertTrue(t[7, 2] and t[7, 12] and t[14, 7])
        self.assertFalse(t[7, 7])
        ext = gems_extend.extend_tips(known, length=1)
        self.assertTrue(ext[15, 7] and ext[7, 1] and ext[7, 13])
        self.assertEqual(int(ext.sum()), 3)

    def test_ring_runs_counts_line_ends_middles_and_junctions(self):
        known = np.zeros((20, 20), dtype=bool)
        known[7, 2:13] = True
        known[8:15, 7] = True
        runs = gems_extend.ring_runs(known)
        self.assertEqual(runs[7, 2], 1)    # bar end
        self.assertEqual(runs[7, 4], 2)    # bar middle
        self.assertEqual(runs[7, 7], 3)    # junction
        self.assertEqual(runs[10, 7], 2)   # stem middle
        self.assertEqual(runs[14, 7], 1)   # stem end
        lone = np.zeros((5, 5), dtype=bool)
        lone[2, 2] = True
        self.assertEqual(gems_extend.ring_runs(lone)[2, 2], 0)

    def test_thick_line_has_tips_only_at_its_ends(self):
        known = np.zeros((12, 15), dtype=bool)
        known[5:7, 2:13] = True  # two pixels thick, cols 2..12
        t = gems_extend.tips(known)
        self.assertTrue(t.any())
        self.assertEqual(set(np.nonzero(t)[1].tolist()), {2, 12})

    def test_valid_mask_and_summary(self):
        known = np.zeros((15, 15), dtype=bool)
        known[2:9, 5] = True
        valid = np.ones((15, 15), dtype=bool)
        valid[:2, :] = False
        ext = gems_extend.extend_tips(known, length=2, valid=valid)
        self.assertFalse(ext[0, 5] or ext[1, 5])
        self.assertTrue(ext[9, 5] and ext[10, 5])
        s = gems_extend.summarize(known, ext)
        self.assertEqual(s["known_pixels"], 7)
        self.assertEqual(s["tips"], 2)
        self.assertEqual(s["extension_pixels"], 2)
        self.assertEqual(s["overlap_with_known"], 0)

    def test_extension_scores_as_partial_credit_when_fault_continues(self):
        # If the real fault runs on past the mapped tip, extending scores.
        known = np.zeros((20, 20), dtype=bool)
        known[2:9, 5] = True
        truth = np.zeros((20, 20))
        truth[9:12, 5] = 1  # new pixels continuing the line at both ends
        truth[0:2, 5] = 1
        ext = gems_extend.extend_tips(known, length=3)
        self.assertEqual(int(ext.sum()), 5)  # 3 below, 2 above (edge)
        r = gems_metric.dti(ext.astype(np.float32), truth, mask=~known)
        self.assertGreater(r["score"], 0.99)
        # a fault that stops where mapped: the extension is a false alarm,
        # which the metric weighs at a fifth of a miss
        r = gems_metric.dti(ext.astype(np.float32), np.zeros((20, 20)),
                            mask=~known)
        self.assertAlmostEqual(r["fp"], 5.0)

    def test_diagonal_pixel_next_to_tip_is_not_a_tip(self):
        known = np.zeros((14, 14), dtype=bool)
        for i in range(3, 10):
            known[i, i] = True
        t = gems_extend.tips(known)
        self.assertEqual(int(t.sum()), 2)
        self.assertTrue(t[3, 3] and t[9, 9])


class LidarPlainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_read_links_takes_first_url_per_row_once(self):
        path = os.path.join(self.tmp, "links.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("tile_id,url,note\n")
            f.write("a,https://example.org/USGS_1M_11_x30y440.tif,ok\n")
            f.write("b,https://example.org/USGS_1M_11_x31y440.tif,\n")
            f.write("a2,https://example.org/USGS_1M_11_x30y440.tif,dup\n")
            f.write("c,no link here,\n")
            f.write("d,tiles/local.tif,\n")
        self.assertEqual(gems_lidar.read_links(path), [
            "https://example.org/USGS_1M_11_x30y440.tif",
            "https://example.org/USGS_1M_11_x31y440.tif",
            "tiles/local.tif"])

    def test_cell_window_keeps_only_fully_covered_cells(self):
        # grid origin (0, 1000), 100 m cells
        self.assertEqual(gems_lidar.cell_window((150, 350, 560, 980),
                                                (0.0, 1000.0)),
                         (1, 6, 2, 5))
        # exact cell edges are kept; shape clips to the grid
        self.assertEqual(gems_lidar.cell_window((0, 0, 300, 1000),
                                                (0.0, 1000.0), shape=(4, 2)),
                         (0, 4, 0, 2))
        i0, i1, j0, j1 = gems_lidar.cell_window((5000, 0, 6000, 1000),
                                                (0.0, 1000.0), shape=(10, 10))
        self.assertLessEqual(j1, j0)  # tile east of the grid: empty

    def test_is_utm11(self):
        class FakeCRS(object):
            def __init__(self, epsg, wkt):
                self.epsg, self.wkt = epsg, wkt

            def to_epsg(self):
                if self.epsg == "raise":
                    raise ValueError("compound")
                return self.epsg

            def to_wkt(self):
                return self.wkt

        self.assertTrue(gems_lidar.is_utm11(FakeCRS(26911, "")))
        self.assertTrue(gems_lidar.is_utm11(FakeCRS(32611, "")))
        self.assertTrue(gems_lidar.is_utm11(FakeCRS(
            "raise", 'COMPD_CS["NAD83 / UTM zone 11N + NAVD88 height"]')))
        self.assertFalse(gems_lidar.is_utm11(FakeCRS(
            None, 'PROJCS["NAD83 / UTM zone 12N"]')))
        self.assertFalse(gems_lidar.is_utm11(None))

    def test_cli_parses(self):
        with self.assertRaises(SystemExit):
            gems_lidar.main(["--help"])


@unittest.skipIf(np is None, "numpy not installed")
class LidarFeatureTests(unittest.TestCase):
    def feats(self, dem, res, factor):
        out = gems_lidar.cell_features(dem, res, factor)
        return dict(zip(gems_lidar.FEATURE_NAMES, out))

    def test_plane(self):
        # z rises 0.5 m per metre eastward, 2 m pixels: slope atan(0.5)
        dem = np.tile(np.arange(10, dtype=np.float64), (10, 1))
        f = self.feats(dem, 2.0, 10)
        slope = math.degrees(math.atan(0.5))
        self.assertAlmostEqual(float(f["lidar_slope_mean"][0, 0]), slope, places=4)
        self.assertAlmostEqual(float(f["lidar_slope_p95"][0, 0]), slope, places=4)
        self.assertAlmostEqual(float(f["lidar_steep_frac"][0, 0]), 1.0)
        self.assertAlmostEqual(float(f["lidar_relief"][0, 0]), 9.0)
        self.assertAlmostEqual(float(f["lidar_coherence"][0, 0]), 1.0, places=5)

    def test_scarp(self):
        # a 10 m north-south step between columns 4 and 5, 1 m pixels
        dem = np.zeros((10, 10))
        dem[:, 5:] = 10.0
        f = self.feats(dem, 1.0, 10)
        steep = math.degrees(math.atan(5.0))  # central difference 10 / 2
        self.assertAlmostEqual(float(f["lidar_steep_frac"][0, 0]), 0.2)
        self.assertAlmostEqual(float(f["lidar_slope_mean"][0, 0]), 0.2 * steep,
                               places=4)
        self.assertAlmostEqual(float(f["lidar_slope_p95"][0, 0]), steep, places=4)
        self.assertAlmostEqual(float(f["lidar_relief"][0, 0]), 10.0)
        self.assertAlmostEqual(float(f["lidar_coherence"][0, 0]), 1.0, places=5)

    def test_bowl_has_no_preferred_direction(self):
        c = np.arange(10, dtype=np.float64) - 4.5
        dem = c[None, :] ** 2 + c[:, None] ** 2
        f = self.feats(dem, 1.0, 10)
        self.assertLess(float(f["lidar_coherence"][0, 0]), 1e-6)
        self.assertGreater(float(f["lidar_slope_mean"][0, 0]), 0.0)

    def test_mostly_missing_cell_is_nan(self):
        dem = np.tile(np.arange(20, dtype=np.float64), (10, 1))
        dem[:, 10:] = np.nan  # second cell has no data
        out = gems_lidar.cell_features(dem, 1.0, 10)
        self.assertEqual(out.shape, (len(gems_lidar.FEATURE_NAMES), 1, 2))
        self.assertTrue(np.isfinite(out[:, 0, 0]).all())
        self.assertTrue(np.isnan(out[:, 0, 1]).all())


@unittest.skipIf(np is None or rasterio is None, "numpy/rasterio not installed")
class LidarRasterTests(unittest.TestCase):
    def test_tile_fills_the_cells_it_covers(self):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin

        grid_t = from_origin(1000.0, 5000.0, 100.0, 100.0)  # 6 x 6 cells
        tile_t = from_origin(1150.0, 4950.0, 10.0, 10.0)    # 31 x 31 px
        dem = np.zeros((31, 31), dtype=np.float32)
        dem[:, 16:] = 20.0  # a step at x = 1310 m, inside grid column 3
        out = gems_lidar.empty_output((6, 6))
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", height=31, width=31, count=1,
                          dtype="float32", crs="EPSG:32611",
                          transform=tile_t, nodata=-9999.0) as dst:
                dst.write(dem, 1)
            with mem.open() as src:
                self.assertTrue(gems_lidar.is_utm11(src.crs))
                n = gems_lidar.process_tile(src, grid_t, (6, 6), 10, out)
                again = gems_lidar.process_tile(src, grid_t, (6, 6), 10, out)
        self.assertEqual(again, 0)  # the first tile to fill a cell wins
        # tile spans x 1150..1460, y 4640..4950: rows 1-2, columns 2-3
        self.assertEqual(n, 4)
        filled = np.isfinite(out[0])
        expect = np.zeros((6, 6), dtype=bool)
        expect[1:3, 2:4] = True
        self.assertTrue((filled == expect).all())
        steep = out[gems_lidar.FEATURE_NAMES.index("lidar_steep_frac")]
        relief = out[gems_lidar.FEATURE_NAMES.index("lidar_relief")]
        for row in (1, 2):
            self.assertAlmostEqual(float(steep[row, 3]), 0.2, places=5)
            self.assertAlmostEqual(float(steep[row, 2]), 0.0, places=5)
            self.assertGreater(float(relief[row, 3]), 19.0)
            self.assertLess(float(relief[row, 2]), 1.0)


try:
    import scipy  # noqa: F401
except ImportError:  # pragma: no cover
    scipy = None


@unittest.skipIf(np is None or scipy is None, "numpy/scipy not installed")
class ScipyTests(unittest.TestCase):
    def test_suppress_near_known_clears_the_ring_only(self):
        known = np.zeros((7, 7), dtype=bool)
        known[:, 3] = True
        binary = np.ones((7, 7), dtype=bool)
        out = gems_baseline.suppress_near_known(binary, known, 1)
        self.assertTrue(out[:, [0, 1, 3, 5, 6]].all())
        self.assertFalse(out[:, [2, 4]].any())
        self.assertTrue(gems_baseline.suppress_near_known(binary, known, 0).all())


if __name__ == "__main__":
    unittest.main()
