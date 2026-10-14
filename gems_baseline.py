"""Baseline pipeline for the DOE GEMS Prize: features -> spatial CV -> submission.

    python gems_baseline.py train    --features data/numeric_features.tif \
                                     --labels data/labels.tif --out submission.tif
    python gems_baseline.py score    --pred submission.tif --labels data/labels.tif
    python gems_baseline.py validate --pred submission.tif --features data/numeric_features.tif

Method (a pixel classifier, kept simple so it runs on a CPU):
  1. Each of the 19 feature bands (plus any --extra-features rasters on the
     same grid, such as gems_lidar.py or gems_regrid.py output) is
     standardised and expanded with edge and ridge filters (Gaussian
     gradient magnitude at 1 and 3 px, the two Hessian eigenvalues at 2 px),
     which respond to the linear breaks faults leave in magnetics, gravity
     and topography. Where a band has no data inside the area (an outside
     grid that covers only part of it), the band is left missing there and
     its filters within EDGE_PX px of the gap too, so the edge of its
     coverage is not mistaken for a break; the classifier handles missing
     values natively.
  2. A HistGradientBoostingClassifier learns fault vs. not-fault from all
     fault pixels plus a random sample of the rest.
  3. Spatial block cross-validation (128 x 128 px blocks, 5 folds) gives
     out-of-fold probabilities; the threshold (and an optional 1 px dilation)
     is picked by the competition metric on those, not by guesswork. They
     are saved as <workdir>/oof.npy for `gems_robust.py stress`.
  4. A model on all samples maps the region; the binary map is written as a
     float32 GeoTIFF on the feature grid, NaN outside the data.

numpy, scipy, rasterio and scikit-learn are imported only where needed, so
the metadata checks below can be tested with the standard library alone.

By MV2 of Munim, Inc. Made with AI agents. MIT licence.
"""

import argparse
import json
import sys

import gems_metric

__version__ = gems_metric.__version__

NODATA_FLOOR = -1e38  # numeric_features.tif uses -3.4e38 as nodata
THRESHOLDS = [0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.17, 0.22, 0.3, 0.4, 0.5]
EDGE_PX = 6  # twice the widest filter's sigma


def check_submission_meta(meta, ref):
    """Problems with a submission's metadata, compared with the feature grid.

    Both are dicts with: epsg, res (x, y), width, height, bounds (l, b, r, t),
    count, dtype; meta also has vmin, vmax (over finite pixels).
    """
    problems = []
    if meta.get("epsg") != 32611:
        problems.append("CRS must be EPSG:32611, got %s" % meta.get("epsg"))
    res = meta.get("res") or (None, None)
    if any(r is None or abs(abs(r) - 100.0) > 1e-6 for r in res):
        problems.append("resolution must be 100 m, got %s" % (res,))
    for key in ("width", "height"):
        if meta.get(key) != ref.get(key):
            problems.append("%s %s differs from the features' %s"
                            % (key, meta.get(key), ref.get(key)))
    mb, rb = meta.get("bounds"), ref.get("bounds")
    if mb is None or rb is None or any(abs(a - b) > 1e-3 for a, b in zip(mb, rb)):
        problems.append("bounds %s differ from the features' %s" % (mb, rb))
    if meta.get("count") != 1:
        problems.append("must have exactly 1 band, got %s" % meta.get("count"))
    if meta.get("dtype") != "float32":
        problems.append("dtype must be float32, got %s" % meta.get("dtype"))
    vmin, vmax = meta.get("vmin"), meta.get("vmax")
    if vmin is not None and vmin < 0:
        problems.append("values must be >= 0, min is %s" % vmin)
    if vmax is not None and vmax > 1:
        problems.append("values must be <= 1, max is %s" % vmax)
    return problems


def band_name(description, tags, index):
    """A band's name: its description, else its 'description' tag (where
    numeric_features.tif keeps the names), else band<N>."""
    return description or (tags or {}).get("description") or "band%d" % index


def _raster_meta(path):
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
        finite = band[np.isfinite(band)]
        epsg = src.crs.to_epsg() if src.crs else None
        return {"epsg": epsg, "res": tuple(src.res), "width": src.width,
                "height": src.height, "bounds": tuple(src.bounds),
                "count": src.count, "dtype": src.dtypes[0],
                "vmin": float(finite.min()) if finite.size else None,
                "vmax": float(finite.max()) if finite.size else None}


def load_features(path):
    """(bands float32 [B, H, W] with NaN for nodata, names, rasterio profile)."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        bands = src.read().astype(np.float32)
        if src.nodata is not None and np.isfinite(src.nodata):
            bands[bands == np.float32(src.nodata)] = np.nan
        names = [band_name(d, src.tags(i + 1), i + 1)
                 for i, d in enumerate(src.descriptions)]
        profile = src.profile.copy()
    bands[bands < NODATA_FLOOR] = np.nan
    return bands, names, profile


def load_labels(path):
    """Binary fault mask (labels.tif: >= 1 is a fault, -1 is nodata)."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        y = src.read(1).astype(np.float32)
    return np.where(np.isfinite(y), y, 0.0) >= 1.0


def band_features(band, valid):
    """Standardised band plus edge and ridge responses, all float32 [H, W].

    Inside `valid`, cells where the band itself is missing stay NaN, and so
    do its four filter responses within EDGE_PX px of them.
    """
    import numpy as np
    from scipy import ndimage as ndi

    vals = band[valid & np.isfinite(band)]
    if vals.size == 0:
        vals = np.zeros(1, dtype=np.float32)
    mu = float(vals.mean())
    sd = float(vals.std()) or 1.0
    z = np.where(np.isfinite(band), (band - mu) / sd, 0.0).astype(np.float32)

    hxx = ndi.gaussian_filter(z, 2.0, order=(0, 2))
    hyy = ndi.gaussian_filter(z, 2.0, order=(2, 0))
    hxy = ndi.gaussian_filter(z, 2.0, order=(1, 1))
    half_diff = np.sqrt(((hxx - hyy) / 2.0) ** 2 + hxy ** 2)
    mean = (hxx + hyy) / 2.0

    feats = [z,
             ndi.gaussian_gradient_magnitude(z, 1.0),
             ndi.gaussian_gradient_magnitude(z, 3.0),
             mean + half_diff,
             mean - half_diff]
    gap = valid & ~np.isfinite(band)
    near_gap = ndi.binary_dilation(gap, iterations=EDGE_PX) if gap.any() else None
    out = []
    for j, f in enumerate(feats):
        f = f.astype(np.float32)
        f[~valid] = np.nan
        if near_gap is not None:
            f[gap if j == 0 else near_gap] = np.nan
        out.append(f)
    return out


def build_feature_stack(bands, valid, path):
    """Write every band's features into an on-disk [H, W, F] float32 array."""
    import numpy as np

    n_per_band = 5
    h, w = valid.shape
    stack = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                      shape=(h, w, bands.shape[0] * n_per_band))
    for b in range(bands.shape[0]):
        for j, f in enumerate(band_features(bands[b], valid)):
            stack[:, :, b * n_per_band + j] = f
    stack.flush()
    return stack


def fold_map(shape, block, folds, seed):
    """Assign each block x block tile of the grid to one of `folds` folds."""
    import numpy as np

    h, w = shape
    nby, nbx = -(-h // block), -(-w // block)
    tiles = np.random.default_rng(seed).integers(0, folds, size=(nby, nbx))
    return np.kron(tiles, np.ones((block, block), dtype=tiles.dtype))[:h, :w]


def sample_pixels(labels, valid, neg_ratio, seed):
    """Sorted flat indices: every fault pixel plus neg_ratio x as many others."""
    import numpy as np

    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels & valid)
    neg_all = np.flatnonzero(~labels & valid)
    n_neg = min(neg_all.size, int(neg_ratio * max(pos.size, 1)))
    neg = rng.choice(neg_all, size=n_neg, replace=False)
    return np.sort(np.concatenate([pos, neg]))


def fit_model(x, y, max_iter, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=False, random_state=seed)
    model.fit(x, y)
    return model


def predict_map(model, stack, where, rows=256):
    """Fault probability for pixels in `where`, 0 elsewhere."""
    import numpy as np

    h, w, f = stack.shape
    prob = np.zeros((h, w), dtype=np.float32)
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        sel = where[r0:r1].ravel()
        if not sel.any():
            continue
        x = np.asarray(stack[r0:r1]).reshape(-1, f)[sel]
        chunk = np.zeros(sel.size, dtype=np.float32)
        chunk[sel] = model.predict_proba(x)[:, 1]
        prob[r0:r1] = chunk.reshape(r1 - r0, w)
    return prob


def choose_postprocess(oof, labels, valid):
    """Best (threshold, dilate) on out-of-fold probabilities by the metric."""
    import numpy as np
    from scipy import ndimage as ndi

    results = []
    for thr in THRESHOLDS:
        binary = (oof >= thr) & valid
        for dilate in (0, 1):
            pred = ndi.binary_dilation(binary) & valid if dilate else binary
            s = gems_metric.dti(pred.astype(np.float32), labels, mask=valid)
            results.append({"threshold": thr, "dilate": dilate,
                            "score": s["score"]})
    results.sort(key=lambda r: -r["score"])
    return results[0], results


def suppress_near_known(binary, known, radius):
    """Clear predictions within `radius` px of known faults, but not on them.

    Scoring masks the known USGS/INGENIOUS fault pixels exactly; pixels next
    to them are scored like any other (forum topic 11516), so a model trained
    on known faults pays false positives along their edges.
    """
    if radius <= 0:
        return binary
    from scipy import ndimage as ndi

    ring = ndi.binary_dilation(known, iterations=radius) & ~known
    return binary & ~ring


def add_tip_extensions(binary, known, valid, length):
    """OR short extensions beyond known fault tips into a binary map."""
    if length <= 0:
        return binary
    import gems_extend

    return binary | gems_extend.extend_tips(known, length=length, valid=valid)


def write_submission(path, binary, valid, profile):
    import numpy as np
    import rasterio

    out = np.where(valid, binary.astype(np.float32), np.nan).astype(np.float32)
    prof = profile.copy()
    prof.update(count=1, dtype="float32", nodata=float("nan"), compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def add_extra_bands(bands, names, paths):
    """Append the bands of each raster in `paths`; they must share the grid."""
    import numpy as np

    for path in paths or []:
        extra, extra_names, _ = load_features(path)
        if extra.shape[1:] != bands.shape[1:]:
            raise SystemExit("%s is %s pixels, the features are %s"
                             % (path, extra.shape[1:], bands.shape[1:]))
        bands = np.concatenate([bands, extra])
        names = names + extra_names
    return bands, names


def cmd_train(args):
    import os

    import numpy as np
    from scipy import ndimage as ndi

    bands, names, profile = load_features(args.features)
    labels = load_labels(args.labels)
    valid = np.isfinite(bands).any(axis=0)
    bands, names = add_extra_bands(bands, names, args.extra_features)
    os.makedirs(args.workdir, exist_ok=True)
    stack = build_feature_stack(bands, valid,
                                os.path.join(args.workdir, "features.npy"))
    del bands
    f = stack.shape[2]
    flat = stack.reshape(-1, f)
    idx = sample_pixels(labels, valid, args.neg_ratio, args.seed)
    x_all = np.asarray(flat[idx])
    y_all = labels.ravel()[idx].astype(np.int8)
    folds = fold_map(valid.shape, args.block, args.folds, args.seed)
    sample_fold = folds.ravel()[idx]

    oof = np.zeros(valid.shape, dtype=np.float32)
    for k in range(args.folds):
        train = sample_fold != k
        model = fit_model(x_all[train], y_all[train], args.max_iter, args.seed)
        held = valid & (folds == k)
        oof[held] = predict_map(model, stack, held)[held]
        print("fold %d done" % k, file=sys.stderr)

    oof_path = os.path.join(args.workdir, "oof.npy")
    np.save(oof_path, oof)
    best, table = choose_postprocess(oof, labels, valid)
    model = fit_model(x_all, y_all, args.max_iter, args.seed)
    prob = predict_map(model, stack, valid)
    binary = (prob >= best["threshold"]) & valid
    if best["dilate"]:
        binary = ndi.binary_dilation(binary) & valid
    binary = suppress_near_known(binary, labels, args.suppress_near_known)
    binary = add_tip_extensions(binary, labels, valid, args.extend_tips)
    write_submission(args.out, binary, valid, profile)
    np.save(os.path.join(args.workdir, "probability.npy"), prob)
    summary = {"version": __version__, "bands": names, "features": f,
               "samples": int(idx.size), "positives": int(y_all.sum()),
               "chosen": best, "oof_table": table, "oof": oof_path,
               "out": args.out,
               "threshold_rule_at_chosen_score":
                   gems_metric.threshold_rule(best["score"])}
    print(json.dumps(summary, indent=2))
    return 0


def cmd_score(args):
    import numpy as np
    import rasterio

    with rasterio.open(args.pred) as src:
        pred = src.read(1).astype(np.float32)
    labels = load_labels(args.labels)
    mask = np.isfinite(pred)
    if args.exclude:
        mask &= ~load_labels(args.exclude)  # as the official scoring does
    result = gems_metric.dti(pred, labels, mask=mask)
    print(json.dumps(result, indent=2))
    return 0


def cmd_extend(args):
    """Add tip extensions to an existing submission without retraining."""
    import numpy as np
    import rasterio

    with rasterio.open(args.pred) as src:
        pred = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    valid = np.isfinite(pred)
    known = load_labels(args.labels)
    binary = (np.where(valid, pred, 0.0) >= 0.5)
    out = add_tip_extensions(binary, known, valid, args.length)
    write_submission(args.out, out, valid, profile)
    print(json.dumps({"input_pixels": int(binary.sum()),
                      "output_pixels": int(out.sum()),
                      "added": int((out & ~binary).sum())}, indent=2))
    return 0


def cmd_validate(args):
    problems = check_submission_meta(_raster_meta(args.pred),
                                     _raster_meta(args.features))
    for p in problems:
        print("PROBLEM: " + p)
    if not problems:
        print("OK: submission matches the feature grid")
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="fit, cross-validate and write a submission")
    t.add_argument("--features", required=True)
    t.add_argument("--labels", required=True)
    t.add_argument("--extra-features", action="append", default=[],
                   help="another raster on the same grid (repeatable), "
                        "e.g. gems_lidar.py or gems_regrid.py output")
    t.add_argument("--out", default="submission.tif")
    t.add_argument("--workdir", default="work")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--block", type=int, default=128)
    t.add_argument("--neg-ratio", type=float, default=3.0)
    t.add_argument("--max-iter", type=int, default=400)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--suppress-near-known", type=int, default=0,
                   help="clear predictions within N px of known faults")
    t.add_argument("--extend-tips", type=int, default=0,
                   help="mark N px beyond every known fault tip")
    e = sub.add_parser("extend", help="add tip extensions to a submission")
    e.add_argument("--pred", required=True)
    e.add_argument("--labels", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--length", type=int, default=3)
    s = sub.add_parser("score", help="score a prediction raster against labels")
    s.add_argument("--pred", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--exclude", help="fault raster to mask out (known faults)")
    v = sub.add_parser("validate", help="check a submission's grid and values")
    v.add_argument("--pred", required=True)
    v.add_argument("--features", required=True)
    args = ap.parse_args(argv)
    return {"train": cmd_train, "score": cmd_score, "extend": cmd_extend,
            "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
