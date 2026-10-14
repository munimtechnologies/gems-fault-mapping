"""How a GEMS submission holds up when the Final Round labels change.

The Final Round ($250k) re-scores the whole GeoDAWN area against labels that
experts revise after reviewing Phase 1 submissions: they may add, remove or
move faults (forum topics 11527, 11556). A submission tuned to the exact
pixels of today's labels can lose most of that edge. Two tools:

  table   Closed form for a straight fault: what the metric pays for a
          predicted band of width w when the true trace is s pixels away,
          and the pooled score when s is uncertain (`band_counts`,
          `pooled_score`, `break_even`).
  stress  Simulation on real rasters: perturb a truth raster many times
          (move each fault, drop some, add new ones) and report how each
          prediction's score moves (`perturb`, `stress`).

    python gems_robust.py table
    python gems_robust.py stress --labels data/labels.tif --pred a.tif --pred b.tif
    python gems_robust.py stress --labels data/labels.tif --prob work/oof.npy \
                                 --threshold 0.05 --threshold 0.1 --dilate 0 --dilate 1

`table` uses the standard library only; `stress` needs numpy and scipy (and
rasterio to read GeoTIFFs).

By MV2 of Munim, Inc. Made with AI agents. Apache-2.0 licence.
"""

import argparse
import json
import math
import sys

import gems_metric

__version__ = gems_metric.__version__

WIDTHS = (1, 3, 5)
MAX_SHIFT = 3

# Perturbations for `stress`: keyword arguments of perturb().
SCENARIOS = {
    "as labelled": {},
    "moved up to 1 px": {"shift": 1},
    "moved up to 2 px": {"shift": 2},
    "20% removed": {"drop": 0.2},
    "20% more faults": {"add": 0.2},
    "revised (1 px, 10% removed, 10% added)": {"shift": 1, "drop": 0.1,
                                                "add": 0.1},
}


def band_rows(width):
    """Row offsets of a band `width` px wide around the predicted trace.

    Odd widths are centred; even widths lean to the + side.
    """
    half = (width - 1) // 2
    return tuple(range(-half, width - half))


def band_counts(rows, shift, radius_px=gems_metric.RADIUS_PX):
    """Weighted (tp, fp, fn) per pixel of length along a straight fault.

    The prediction marks the rows in `rows` (offsets across the fault from
    the predicted trace, e.g. (-1, 0, 1) for a 3 px band) and the true trace
    lies `shift` rows away. Along an axis-aligned fault every column is the
    same, so these are exact per-column counts of the metric: the true pixel
    is matched by the nearest marked row, and each marked row costs one
    minus the kernel weight of its distance to the true trace.
    """
    def k(d):
        return max(1.0 - d / radius_px, 0.0)

    dists = [abs(r - shift) for r in rows]
    tp = k(min(dists)) if dists else 0.0
    fp = sum(1.0 - k(d) for d in dists)
    return tp, fp, 1.0 - tp


def uniform_shifts(max_shift):
    """{shift: probability}, uniform over -max_shift .. max_shift."""
    n = 2 * max_shift + 1
    return {s: 1.0 / n for s in range(-max_shift, max_shift + 1)}


def pooled_score(rows, shift_probs, alpha=gems_metric.ALPHA,
                 beta=gems_metric.BETA):
    """Expected DTI over a long fault whose offset follows `shift_probs`.

    The metric pools counts over all pixels, so the counts (not the scores)
    are averaged with the given probabilities.
    """
    tp = fp = fn = 0.0
    for s, prob in shift_probs.items():
        a, b, c = band_counts(rows, s)
        tp += prob * a
        fp += prob * b
        fn += prob * c
    return gems_metric.score_from_counts(tp, fp, fn, alpha, beta)


def break_even(rows_a, rows_b, shift=1, iterations=60):
    """Share p of fault length lying `shift` px off at which b equals a.

    The rest of the fault (1 - p) lies exactly on the predicted trace.
    Returns None when one band wins for every p in [0, 1].
    """
    def gain(p):
        probs = {0: 1.0 - p, shift: p}
        return pooled_score(rows_b, probs) - pooled_score(rows_a, probs)

    lo, hi = 0.0, 1.0
    g_lo, g_hi = gain(lo), gain(hi)
    if g_lo == 0:
        return lo
    if g_lo * g_hi > 0:
        return None
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        g = gain(mid)
        if (g < 0) == (g_lo < 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def table(widths=WIDTHS, max_shift=MAX_SHIFT):
    """Rows of {width, by_shift: [...], exact, within_1, within_2}."""
    out = []
    for w in widths:
        rows = band_rows(w)
        out.append({
            "width": w,
            "by_shift": [pooled_score(rows, {s: 1.0})
                         for s in range(max_shift + 1)],
            "exact": pooled_score(rows, uniform_shifts(0)),
            "within_1": pooled_score(rows, uniform_shifts(1)),
            "within_2": pooled_score(rows, uniform_shifts(2)),
        })
    return out


def format_table(rows, max_shift=MAX_SHIFT):
    head = "width  " + "  ".join("s=%d  " % s for s in range(max_shift + 1))
    head += "|  exact  +-1 px  +-2 px"
    lines = [head]
    for r in rows:
        cells = "  ".join("%.3f" % v for v in r["by_shift"])
        lines.append("%-5d  %s  |  %.3f  %.3f   %.3f"
                     % (r["width"], cells, r["exact"], r["within_1"],
                        r["within_2"]))
    p = break_even(band_rows(1), band_rows(3))
    if p is not None:
        lines.append("A 3 px band beats a 1 px line once more than %.1f%% of "
                     "the true fault length lies 1 px off the predicted trace."
                     % (100 * p))
    return "\n".join(lines)


def draw_line(out, y0, x0, angle, length):
    """Mark `length` steps of 1 px from (y0, x0) along `angle` (radians).

    Pixels are floor()ed, so consecutive pixels are 8-connected; pixels off
    the grid are skipped.
    """
    h, w = out.shape
    sy, sx = math.sin(angle), math.cos(angle)
    for i in range(int(length)):
        y = math.floor(y0 + i * sy)
        x = math.floor(x0 + i * sx)
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = True
    return out


def perturb(truth, rng, shift=0, drop=0.0, add=0.0, add_length=20,
            valid=None):
    """One plausible expert revision of a fault raster (bool array).

    Each connected fault (8-connected) is removed with probability `drop`,
    otherwise moved by a whole-pixel offset drawn uniformly from
    [-shift, shift] in each direction. Then round(add x number of faults)
    new straight faults of `add_length` px are drawn at random positions and
    directions. Pixels moved off the grid, or outside `valid`, are lost.
    """
    import numpy as np
    from scipy import ndimage as ndi

    t = np.asarray(truth, dtype=bool)
    h, w = t.shape
    lab, n = ndi.label(t, structure=np.ones((3, 3), dtype=bool))
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs] - 1
    keep = rng.random(n) >= drop
    dy = rng.integers(-shift, shift + 1, size=n)
    dx = rng.integers(-shift, shift + 1, size=n)
    ny, nx = ys + dy[ids], xs + dx[ids]
    ok = keep[ids] & (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
    out = np.zeros((h, w), dtype=bool)
    out[ny[ok], nx[ok]] = True
    for _ in range(int(round(add * n))):
        draw_line(out, rng.uniform(0, h), rng.uniform(0, w),
                  rng.uniform(0, 2 * math.pi), add_length)
    if valid is not None:
        out &= np.asarray(valid, dtype=bool)
    return out


def dilate(binary, px):
    """Grow a binary map by `px` pixels (4-neighbour, as gems_baseline)."""
    if px <= 0:
        return binary
    from scipy import ndimage as ndi

    return ndi.binary_dilation(binary, iterations=px)


def stress(preds, truth, scenarios=None, trials=20, seed=0, mask=None):
    """Score each prediction against perturbed copies of the truth.

    preds: dict name -> 2-D array. scenarios: dict name -> perturb()
    keyword arguments ({} scores the truth as it is, once). Every
    prediction sees the same perturbed truths, so their means compare
    fairly. Returns rows {pred, scenario, trials, mean, std, min, max}.
    """
    import numpy as np

    scenarios = SCENARIOS if scenarios is None else scenarios
    rows = []
    for sname, kwargs in scenarios.items():
        rng = np.random.default_rng(seed)
        n = trials if kwargs else 1
        scores = {p: [] for p in preds}
        for _ in range(n):
            t = perturb(truth, rng, valid=mask, **kwargs) if kwargs else truth
            for pname, pred in preds.items():
                scores[pname].append(
                    gems_metric.dti(pred, t, mask=mask)["score"])
        for pname, vals in scores.items():
            v = np.asarray(vals, dtype=np.float64)
            rows.append({"pred": pname, "scenario": sname, "trials": n,
                         "mean": float(v.mean()), "std": float(v.std()),
                         "min": float(v.min()), "max": float(v.max())})
    return rows


def _read_raster(path):
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def cmd_table(args):
    rows = table()
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(format_table(rows))
    return 0


def cmd_stress(args):
    import numpy as np

    import gems_baseline

    truth = gems_baseline.load_labels(args.labels)
    mask = np.ones(truth.shape, dtype=bool)
    if args.exclude:
        mask &= ~gems_baseline.load_labels(args.exclude)
    maps = {}
    for path in args.pred:
        p = _read_raster(path)
        mask &= np.isfinite(p)
        maps[path] = np.where(np.isfinite(p), p, 0.0) >= 0.5
    if args.prob:
        prob = np.load(args.prob)
        for thr in args.threshold or [0.1]:
            maps["%s>=%g" % (args.prob, thr)] = prob >= thr
    if not maps:
        raise SystemExit("give at least one --pred or --prob")
    preds = {}
    for name, binary in maps.items():
        for px in args.dilate or [0]:
            label = name if px == 0 else "%s+dilate%d" % (name, px)
            preds[label] = (dilate(binary, px) & mask).astype(np.float32)
    rows = stress(preds, truth & mask, trials=args.trials, seed=args.seed,
                  mask=mask)
    print(json.dumps({"version": __version__, "rows": rows}, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="closed-form band width vs. offset table")
    t.add_argument("--json", action="store_true")
    s = sub.add_parser("stress", help="score predictions on perturbed labels")
    s.add_argument("--labels", required=True,
                   help="truth raster to perturb (e.g. labels.tif)")
    s.add_argument("--pred", action="append", default=[],
                   help="submission GeoTIFF (repeatable; >= 0.5 is a fault)")
    s.add_argument("--prob", help="probability .npy, e.g. work/oof.npy")
    s.add_argument("--threshold", type=float, action="append",
                   help="threshold for --prob (repeatable, default 0.1)")
    s.add_argument("--dilate", type=int, action="append",
                   help="also score each map dilated by N px (repeatable)")
    s.add_argument("--exclude", help="fault raster to mask out (known faults)")
    s.add_argument("--trials", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return {"table": cmd_table, "stress": cmd_stress}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
