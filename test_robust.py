"""Offline tests for gems_robust (band table, label perturbation, stress runs).

Run with `python -m unittest -v`. The closed-form tests use the standard
library only; the raster tests are skipped without numpy and scipy.
"""

import contextlib
import io
import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gems_metric  # noqa: E402
import gems_robust  # noqa: E402

try:
    import numpy as np
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover
    np = None
    ndi = None


class BandTests(unittest.TestCase):
    def test_band_rows(self):
        self.assertEqual(gems_robust.band_rows(1), (0,))
        self.assertEqual(gems_robust.band_rows(3), (-1, 0, 1))
        self.assertEqual(gems_robust.band_rows(5), (-2, -1, 0, 1, 2))
        self.assertEqual(gems_robust.band_rows(2), (0, 1))

    def test_band_counts_by_hand(self):
        cases = [
            ((0,), 0, (1.0, 0.0, 0.0)),
            ((0,), 1, (2 / 3, 1 / 3, 1 / 3)),
            ((0,), 3, (0.0, 1.0, 1.0)),
            ((-1, 0, 1), 0, (1.0, 2 / 3, 0.0)),
            # distances 3, 2, 1: matched at 1 px; costs 1 + 2/3 + 1/3
            ((-1, 0, 1), 2, (2 / 3, 2.0, 1 / 3)),
            # distances 2, 1, 0, 1, 2
            ((-2, -1, 0, 1, 2), 1, (1.0, 1 / 3 + 0 + 1 / 3 + 2 / 3 + 1, 0.0)),
        ]
        for rows, shift, want in cases:
            got = gems_robust.band_counts(rows, shift)
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, msg="%s shift %d" % (rows, shift))

    def test_band_counts_symmetric_in_shift(self):
        for s in range(4):
            a = gems_robust.band_counts((-1, 0, 1), s)
            b = gems_robust.band_counts((-1, 0, 1), -s)
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y)

    def test_single_offset_scores(self):
        score = gems_robust.pooled_score
        self.assertAlmostEqual(score((0,), {1: 1.0}), 2 / 3, places=6)
        self.assertAlmostEqual(score((0,), {2: 1.0}), 1 / 3, places=6)
        self.assertAlmostEqual(score((-1, 0, 1), {0: 1.0}), 15 / 17, places=6)
        self.assertAlmostEqual(score((-1, 0, 1), {1: 1.0}), 1 / 1.2, places=6)
        self.assertAlmostEqual(score((-1, 0, 1), {2: 1.0}), 0.5, places=6)
        # tp 1/3, fp 8/3, fn 2/3 -> 1 / 4.2
        self.assertAlmostEqual(score((-1, 0, 1), {3: 1.0}), 1 / 4.2, places=6)
        self.assertAlmostEqual(score(gems_robust.band_rows(5), {0: 1.0}),
                               1 / 1.4, places=6)

    def test_uniform_shifts(self):
        probs = gems_robust.uniform_shifts(2)
        self.assertEqual(sorted(probs), [-2, -1, 0, 1, 2])
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertEqual(gems_robust.uniform_shifts(0), {0: 1.0})

    def test_pooled_scores_under_uncertain_position(self):
        score = gems_robust.pooled_score
        u1, u2 = gems_robust.uniform_shifts(1), gems_robust.uniform_shifts(2)
        w1, w3, w5 = (gems_robust.band_rows(w) for w in (1, 3, 5))
        # +-1 px: tp 7/9, fp 2/9, fn 2/9 -> 7/9
        self.assertAlmostEqual(score(w1, u1), 7 / 9, places=6)
        # tp 1, fp 8/9 -> 1 / (1 + 0.2 * 8/9)
        self.assertAlmostEqual(score(w3, u1), 9 / 10.6, places=6)
        # tp 1, fp 20/9 -> 9 / 13
        self.assertAlmostEqual(score(w5, u1), 9 / 13, places=6)
        # +-2 px: tp 9/15, fp 6/15, fn 6/15 -> 0.6
        self.assertAlmostEqual(score(w1, u2), 0.6, places=6)
        # tp 13/15, fp 20/15, fn 2/15 -> 13 / 18.6
        self.assertAlmostEqual(score(w3, u2), 13 / 18.6, places=6)
        # tp 1, fp 38/15 -> 15 / 22.6
        self.assertAlmostEqual(score(w5, u2), 15 / 22.6, places=6)

    def test_break_even_matches_closed_form(self):
        # a 1 px line scores 1 - p/3 and a 3 px band 15 / (17 + p) when a
        # share p of the fault is 1 px off; equal where p^2 + 14p - 6 = 0
        want = (-14 + math.sqrt(220)) / 2
        got = gems_robust.break_even(gems_robust.band_rows(1),
                                     gems_robust.band_rows(3))
        self.assertAlmostEqual(got, want, places=6)
        self.assertAlmostEqual(1 - got / 3, 15 / (17 + got), places=6)

    def test_break_even_none_when_one_band_always_wins(self):
        # a 5 px band never beats a 3 px band against 0-1 px offsets
        self.assertIsNone(gems_robust.break_even(gems_robust.band_rows(3),
                                                 gems_robust.band_rows(5)))

    def test_table_and_cli(self):
        rows = gems_robust.table()
        self.assertEqual([r["width"] for r in rows], [1, 3, 5])
        self.assertEqual(len(rows[0]["by_shift"]), 4)
        self.assertAlmostEqual(rows[1]["within_1"], 9 / 10.6, places=6)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(gems_robust.main(["table"]), 0)
        self.assertIn("41.6%", buf.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(gems_robust.main(["table", "--json"]), 0)
        self.assertEqual(len(json.loads(buf.getvalue())), 3)


@unittest.skipIf(np is None, "numpy/scipy not installed")
class RasterTests(unittest.TestCase):
    def test_band_counts_match_the_metric_on_a_grid(self):
        for width in (1, 3, 5):
            rows = gems_robust.band_rows(width)
            for shift in range(4):
                truth = np.zeros((15, 12))
                truth[7 + shift, :] = 1
                pred = np.zeros((15, 12))
                for r in rows:
                    pred[7 + r, :] = 1
                got = gems_metric.dti(pred, truth)
                tp, fp, fn = gems_robust.band_counts(rows, shift)
                msg = "width %d shift %d" % (width, shift)
                self.assertAlmostEqual(got["tp"] / 12, tp, places=4, msg=msg)
                self.assertAlmostEqual(got["fp"] / 12, fp, places=4, msg=msg)
                self.assertAlmostEqual(got["fn"] / 12, fn, places=4, msg=msg)

    def test_draw_line(self):
        out = np.zeros((8, 8), dtype=bool)
        gems_robust.draw_line(out, 3.5, 2.25, 0.0, 4)
        self.assertEqual(sorted(zip(*np.nonzero(out))),
                         [(3, 2), (3, 3), (3, 4), (3, 5)])
        out[:] = False
        gems_robust.draw_line(out, 3.5, 2.25, math.pi / 2, 4)
        self.assertEqual(sorted(zip(*np.nonzero(out))),
                         [(3, 2), (4, 2), (5, 2), (6, 2)])
        out[:] = False
        gems_robust.draw_line(out, 3.5, 2.25, math.pi, 4)  # leaves the grid
        self.assertEqual(sorted(zip(*np.nonzero(out))),
                         [(3, 0), (3, 1), (3, 2)])

    def test_perturb_without_changes_is_identity(self):
        truth = np.zeros((10, 10), dtype=bool)
        truth[2:8, 4] = True
        truth[9, 0:3] = True
        out = gems_robust.perturb(truth, np.random.default_rng(0))
        self.assertTrue((out == truth).all())

    def test_perturb_drop_all(self):
        truth = np.zeros((10, 10), dtype=bool)
        truth[2:8, 4] = True
        out = gems_robust.perturb(truth, np.random.default_rng(0), drop=1.0)
        self.assertFalse(out.any())

    def test_perturb_moves_each_fault_whole(self):
        truth = np.zeros((20, 20), dtype=bool)
        truth[3:9, 2] = True     # 6 px vertical
        truth[12, 8:15] = True   # 7 px horizontal
        for seed in range(10):
            out = gems_robust.perturb(truth, np.random.default_rng(seed),
                                      shift=2)
            self.assertEqual(int(out.sum()), 13)
            lab, n = ndi.label(out, structure=np.ones((3, 3), dtype=bool))
            self.assertEqual(n, 2)
            shapes = sorted((s[0].stop - s[0].start, s[1].stop - s[1].start)
                            for s in ndi.find_objects(lab))
            self.assertEqual(shapes, [(1, 7), (6, 1)])
            ys, xs = np.nonzero(out[:, :5])
            self.assertTrue(((ys >= 1) & (ys <= 10)).all())

    def test_perturb_respects_valid(self):
        truth = np.zeros((6, 6), dtype=bool)
        truth[3, :] = True
        valid = np.zeros((6, 6), dtype=bool)
        valid[:, :3] = True
        out = gems_robust.perturb(truth, np.random.default_rng(0), valid=valid)
        self.assertTrue((out == (truth & valid)).all())

    def test_perturb_adds_new_faults(self):
        truth = np.zeros((30, 30), dtype=bool)
        truth[15, 15] = True
        out = gems_robust.perturb(truth, np.random.default_rng(3), add=1.0,
                                  add_length=5)
        self.assertTrue(out[15, 15])
        self.assertGreaterEqual(int(out.sum()), 2)

    def test_dilate(self):
        line = np.zeros((11, 11), dtype=bool)
        line[5, :] = True
        self.assertTrue((gems_robust.dilate(line, 0) == line).all())
        one = gems_robust.dilate(line, 1)
        self.assertTrue(one[4:7].all())
        self.assertFalse(one[:4].any() or one[7:].any())
        two = gems_robust.dilate(line, 2)
        self.assertTrue(two[3:8].all())
        self.assertFalse(two[:3].any() or two[8:].any())

    def test_stress_shares_truths_and_scores_scenarios(self):
        truth = np.zeros((21, 21), dtype=bool)
        truth[:, 10] = True
        line = truth.astype(np.float32)
        band = np.zeros((21, 21), dtype=np.float32)
        band[:, 9:12] = 1
        preds = {"line": line, "band": band, "line copy": line.copy()}
        scenarios = {"exact": {}, "moved": {"shift": 1}, "gone": {"drop": 1.0}}
        rows = gems_robust.stress(preds, truth, scenarios=scenarios, trials=5)
        self.assertEqual(len(rows), 9)
        got = {(r["pred"], r["scenario"]): r for r in rows}
        self.assertEqual(got[("line", "exact")]["trials"], 1)
        self.assertAlmostEqual(got[("line", "exact")]["mean"], 1.0, places=5)
        self.assertAlmostEqual(got[("band", "exact")]["mean"], 15 / 17,
                               places=4)
        self.assertEqual(got[("band", "moved")]["trials"], 5)
        self.assertAlmostEqual(got[("line", "moved")]["mean"],
                               got[("line copy", "moved")]["mean"], places=12)
        for name in preds:
            self.assertEqual(got[(name, "gone")]["mean"], 0.0)

    def test_stress_cli_parses(self):
        with self.assertRaises(SystemExit):
            gems_robust.main(["stress"])  # --labels is required


if __name__ == "__main__":
    unittest.main()
