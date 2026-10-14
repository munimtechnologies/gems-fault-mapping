"""Tests for gems_gridskill.py. Run: python -m unittest -v test_gridskill"""

import struct
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

import gems_gridskill
import gems_regrid


def point_shp(points):
    """Bytes of a point (type 1) shapefile, one record per point."""
    recs = b""
    for i, (x, y) in enumerate(points, 1):
        content = struct.pack("<i2d", 1, x, y)
        recs += struct.pack(">2i", i, len(content) // 2) + content
    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(recs)) // 2)
    header += struct.pack("<2i", 1000, 1) + struct.pack("<8d", *([0.0] * 8))
    return header + recs


@unittest.skipIf(np is None, "numpy missing")
class RankTests(unittest.TestCase):
    def test_average_ranks_with_ties(self):
        self.assertEqual(gems_gridskill.average_ranks([3, 1, 2, 2]).tolist(),
                         [4.0, 1.0, 2.5, 2.5])

    def test_auc(self):
        ranks = gems_gridskill.average_ranks([0.1, 0.4, 0.35, 0.8])
        auc = gems_gridskill.auc_from_ranks
        self.assertEqual(auc(ranks, [False, True, False, True]), 1.0)
        self.assertEqual(auc(ranks, [True, False, True, False]), 0.0)
        self.assertIsNone(auc(ranks, [False] * 4))
        self.assertIsNone(auc(ranks, [True] * 4))
        tie = gems_gridskill.average_ranks([1.0, 1.0])
        self.assertEqual(auc(tie, [True, False]), 0.5)

    def test_score_feature_columns(self):
        values = np.tile(np.arange(4, dtype=float), (4, 1))
        cols = np.tile(np.arange(4), (4, 1))
        valid = np.ones((4, 4), dtype=bool)
        got = gems_gridskill.score_feature(values, valid,
                                           [cols == 3, cols == 0, cols == 1])
        self.assertAlmostEqual(got[0], 1.0)
        self.assertAlmostEqual(got[1], 0.0)
        self.assertAlmostEqual(got[2], 1.0 / 3.0)
        self.assertAlmostEqual(gems_gridskill.score_feature(
            values, cols != 2, [cols == 1])[0], 0.5)


@unittest.skipIf(np is None, "numpy missing")
class BlockTests(unittest.TestCase):
    def test_block_ids(self):
        valid = np.ones((3, 5), dtype=bool)
        bid = gems_gridskill.block_ids(valid, 2)
        # 3 blocks per row of blocks; cell (2, 4) is block row 1, column 2.
        self.assertEqual(bid.reshape(3, 5).tolist(),
                         [[0, 0, 1, 1, 2], [0, 0, 1, 1, 2], [3, 3, 4, 4, 5]])
        valid[0, 0] = False
        self.assertEqual(gems_gridskill.block_ids(valid, 2)[:4].tolist(),
                         [0, 1, 1, 2])

    def test_block_ranks(self):
        v = np.array([1, 2, 100, 101, 3, 10, 102, 103], dtype=float)
        bid = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        self.assertEqual(gems_gridskill.block_ranks(v, bid).tolist(),
                         [1, 2, 1, 2, 3, 4, 3, 4])
        tie = gems_gridskill.block_ranks(np.array([5.0, 5.0, 1.0]),
                                         np.array([0, 0, 1]))
        self.assertEqual(tie.tolist(), [1.5, 1.5, 1.0])
        self.assertEqual(len(gems_gridskill.block_ranks(np.array([]),
                                                        np.array([], int))), 0)

    def test_within_block_beats_pooled(self):
        # Each block's positive is its highest cell, but block 0 is low
        # overall: pooled AUC 9/12, within blocks 1.
        values = np.array([[1, 2, 100, 101], [3, 10, 102, 103]], dtype=float)
        pos = np.zeros((2, 4), dtype=bool)
        pos[1, 1] = pos[1, 3] = True
        valid = np.ones((2, 4), dtype=bool)
        self.assertAlmostEqual(
            gems_gridskill.score_feature(values, valid, [pos])[0], 0.75)
        bid = gems_gridskill.block_ids(valid, 2)
        ranks = gems_gridskill.block_ranks(values[valid], bid)
        auc, n = gems_gridskill.stratified_auc(ranks, bid, pos[valid])
        self.assertEqual((auc, n), (1.0, 2))

    def test_regional_layer_is_half_within_blocks(self):
        # Block-constant values: 1 in block 0, 2 in block 1. One positive in
        # block 0, two in block 1. Pooled U = 1.5 + 4 + 4 = 9.5 of 15 pairs;
        # within blocks U = 1.5 + 2 = 3.5 of 3 + 4 = 7 pairs.
        values = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=float)
        pos = np.zeros((2, 4), dtype=bool)
        pos[0, 0] = pos[0, 2] = pos[1, 3] = True
        valid = np.ones((2, 4), dtype=bool)
        self.assertAlmostEqual(
            gems_gridskill.score_feature(values, valid, [pos])[0], 9.5 / 15)
        bid = gems_gridskill.block_ids(valid, 2)
        ranks = gems_gridskill.block_ranks(values[valid], bid)
        auc, n = gems_gridskill.stratified_auc(ranks, bid, pos[valid])
        self.assertAlmostEqual(auc, 0.5)
        self.assertEqual(n, 2)

    def test_no_block_with_both_classes(self):
        valid = np.ones((2, 4), dtype=bool)
        pos = np.zeros((2, 4), dtype=bool)
        pos[:, :2] = True
        bid = gems_gridskill.block_ids(valid, 2)
        ranks = gems_gridskill.block_ranks(np.arange(8.0), bid)
        self.assertEqual(gems_gridskill.stratified_auc(ranks, bid, pos[valid]),
                         (None, 0))


@unittest.skipIf(np is None, "numpy missing")
class HelperTests(unittest.TestCase):
    def test_gradient_per_km(self):
        src = gems_regrid.Source("g", np.array([[0.0, 250.0, 500.0],
                                                [0.0, 250.0, 500.0]]),
                                 0.0, 500.0, 250.0, 250.0)
        g = gems_gridskill.gradient_per_km(src)
        self.assertEqual(g.name, "g_grad")
        self.assertTrue(np.allclose(g.values, 1000.0))
        self.assertEqual((g.left, g.top, g.dx, g.dy), (0.0, 500.0, 250.0, 250.0))

    def test_top_k_map(self):
        v = np.array([[0.2, 0.9], [0.5, 0.1]])
        ok = np.ones((2, 2), dtype=bool)
        top = gems_gridskill.top_k_map
        self.assertEqual(top(v, ok, 2).tolist(), [[False, True], [True, False]])
        self.assertEqual(top(v, ok, 2, -1.0).tolist(), [[True, False], [False, True]])
        ok[0, 1] = False
        self.assertEqual(top(v, ok, 2).tolist(), [[True, False], [True, False]])
        self.assertFalse(top(v, ok, 0).any())

    def test_is_derived(self):
        self.assertTrue(gems_gridskill.is_derived("tdts_nvml"))
        self.assertTrue(gems_gridskill.is_derived("dist_tdts2020nvML_faultVertices"))
        self.assertFalse(gems_gridskill.is_derived("grv_filthg_grad"))
        self.assertFalse(gems_gridskill.is_derived("dist_heatFlowWells_nvML"))

    def test_layer_points(self):
        shp = point_shp([(1.0, 2.0), (3.0, 4.0)])
        self.assertEqual(gems_gridskill.layer_points((shp, None, None)),
                         [(1.0, 2.0), (3.0, 4.0)])


@unittest.skipIf(np is None, "numpy missing")
class EvaluateTests(unittest.TestCase):
    def setUp(self):
        cols = np.tile(np.arange(40), (40, 1))
        self.cols = cols
        self.fault = cols == 20
        self.near = np.abs(cols - 20) <= 3
        self.mask = np.ones((40, 40), dtype=bool)
        self.dist = -np.abs(cols - 20).astype(np.float32)

    def test_closer_is_better(self):
        row = gems_gridskill.evaluate("dist_x", self.dist, self.mask, self.near,
                                      ((0, 15),), self.fault, "line distance",
                                      gems=False)
        self.assertEqual(row["auc"], 1.0)
        self.assertEqual(row["cells"], 1600)
        self.assertEqual(row["near_cells"], 280)
        self.assertGreater(row["edge"], 0)
        self.assertFalse(row["derived"])
        self.assertNotIn("gems", row)
        self.assertNotIn("block", row)

    def test_nan_cells_left_out(self):
        values = self.dist.copy()
        values[:, :5] = np.nan
        row = gems_gridskill.evaluate("v", values, self.mask, self.near,
                                      ((0, 15),), self.fault, "grid", gems=False)
        self.assertEqual(row["cells"], 1400)

    def test_gems_of_exact_map(self):
        row = gems_gridskill.evaluate("tdts_copy", self.fault.astype(np.float32),
                                      self.mask, self.near, ((0, 15),),
                                      self.fault, "grid", gems=True)
        self.assertTrue(row["derived"])
        self.assertEqual(row["gems"]["direction"], "high")
        self.assertEqual(row["gems"]["score"], 1.0)

    def test_blocks_on_distance(self):
        # 10-cell blocks: near cells (columns 17-23) sit in the blocks of
        # columns 10-19 and 20-29, 4 rows of blocks each, and are closer than
        # every other cell there. Shifted 15 columns, the near cells are
        # columns 32-38 of block 30-39 and beat only column 39: U = 7 of 21.
        row = gems_gridskill.evaluate("dist_x", self.dist, self.mask, self.near,
                                      ((0, 15),), self.fault, "line distance",
                                      gems=False, blocks=(10,))
        b = row["block"][0]
        self.assertEqual(b["px"], 10)
        self.assertEqual(b["auc"], 1.0)
        self.assertEqual(b["blocks"], 8)
        self.assertEqual(b["chance_auc"], [0.3333])
        self.assertEqual(b["edge"], 0.3333)

    def test_blocks_remove_a_regional_step(self):
        # A step at column 20 lines up with the 10-cell blocks: pooled it
        # favours the near cells (U = 4 * 25 + 3 * 8.5 of 7 * 33 per row),
        # within blocks it is flat.
        values = (self.cols >= 20).astype(np.float32)
        row = gems_gridskill.evaluate("step", values, self.mask, self.near,
                                      ((0, 15),), self.fault, "grid",
                                      gems=False, blocks=(10,))
        self.assertEqual(row["auc"], round(125.5 / 231, 4))
        self.assertEqual(row["block"][0]["auc"], 0.5)

    def test_markdown_block_columns(self):
        row = gems_gridskill.evaluate("dist_x", self.dist, self.mask, self.near,
                                      ((0, 15),), self.fault, "line distance",
                                      gems=False, blocks=(10,))
        report = {"version": "x", "release": "r", "features": [row],
                  "grid": {"scored_cells": 1600, "near_cells": 280,
                           "cell_m": 100.0}}
        text = gems_gridskill.markdown(report)
        self.assertIn("within 1 km (chance) | edge |", text)
        self.assertIn("| 1.000 (0.333-0.333) | +0.333 |", text)
        old = dict(row)
        del old["block"]
        report["features"] = [old]
        text = gems_gridskill.markdown(report)
        self.assertNotIn("within 1 km", text)
        self.assertNotIn("Within B km", text)


if __name__ == "__main__":
    unittest.main()
