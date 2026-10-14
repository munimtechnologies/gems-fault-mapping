"""Local scorer for the DOE GEMS Prize (geothermal fault mapping, DrivenData).

Implements the distance-weighted Tversky index (DTI) as described on the
competition's public problem-description page:

    k(d)  = max(1 - d / R, 0)          R = 300 m = 3 pixels at 100 m
    TP_w  = sum over truth pixels g of  max_x p(x) * k(d(x, g))
    FN_w  = sum over truth pixels g of  1 - (that best match)
    FP_w  = sum over all pixels x   of  p(x) * (1 - max_g k(d(x, g)))
    DTI   = TP_w / (TP_w + alpha * FP_w + beta * FN_w + eps)
    alpha = 0.2 (false positives), beta = 0.8 (false negatives)

This is unofficial: it is written from the published description, not from
the organisers' code, so treat small differences (eps, edge handling) as
possible. `dti_reference` is plain Python for checking; `dti` is the fast
numpy version for full 3730 x 3292 rasters.

By MV2 of Munim, Inc. Made with AI agents. MIT licence.
"""

import math

__version__ = "0.4.0"

ALPHA = 0.2
BETA = 0.8
RADIUS_M = 300.0
PIXEL_M = 100.0
RADIUS_PX = RADIUS_M / PIXEL_M
EPS = 1e-9


def kernel_offsets(radius_px=RADIUS_PX):
    """Every (dy, dx, weight) with weight k(d) > 0, heaviest first."""
    r = int(math.floor(radius_px))
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            w = 1.0 - math.hypot(dy, dx) / radius_px
            if w > 0:
                out.append((dy, dx, w))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    return out


def score_from_counts(tp, fp, fn, alpha=ALPHA, beta=BETA, eps=EPS):
    """DTI from weighted counts (the page's worked example: 3, 1.89, 2 -> 0.60)."""
    return tp / (tp + alpha * fp + beta * fn + eps)


def threshold_rule(expected_score, alpha=ALPHA, beta=BETA):
    """Probability above which adding a pixel raises the expected DTI.

    For a pixel that is a fault with probability q, adding it at p = 1 gains
    q of TP_w and costs (1 - q) of FP_w. Maximising TP / ((1 - beta) TP +
    alpha FP + beta N) (Dinkelbach, at the optimum score lambda) says: add it
    when q > lambda * alpha / (1 - lambda * (1 - beta) + lambda * alpha).
    With alpha = 0.2 and beta = 0.8 this is simply 0.2 * lambda, so a model
    that can reach DTI 0.4 should mark every pixel with q above about 0.08.
    """
    lam = float(expected_score)
    return lam * alpha / (1.0 - lam * (1.0 - beta) + lam * alpha)


def _value(v):
    if v is None:
        return 0.0
    v = float(v)
    if v != v:  # NaN
        return 0.0
    return v


def dti_reference(pred, truth, radius_px=RADIUS_PX, alpha=ALPHA, beta=BETA,
                  eps=EPS, mask=None):
    """Plain-Python DTI on lists of rows. Returns dict(tp, fp, fn, n, score).

    pred: probabilities in [0, 1] (NaN / None count as 0, values are clipped).
    truth: a pixel is a fault when its value is > 0.5 (labels.tif uses 1).
    mask: optional rows of booleans; pixels outside it are ignored.
    """
    h = len(truth)
    w = len(truth[0]) if h else 0
    offs = kernel_offsets(radius_px)

    def inside(y, x):
        return mask is None or bool(mask[y][x])

    p = [[min(max(_value(pred[y][x]), 0.0), 1.0) if inside(y, x) else 0.0
          for x in range(w)] for y in range(h)]
    t = [[_value(truth[y][x]) > 0.5 and inside(y, x) for x in range(w)]
         for y in range(h)]

    tp = 0.0
    n = 0
    for y in range(h):
        for x in range(w):
            if not t[y][x]:
                continue
            n += 1
            best = 0.0
            for dy, dx, k in offs:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    best = max(best, p[yy][xx] * k)
            tp += best

    fp = 0.0
    for y in range(h):
        for x in range(w):
            if p[y][x] <= 0:
                continue
            near = 0.0
            for dy, dx, k in offs:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and t[yy][xx]:
                    near = max(near, k)
            fp += p[y][x] * (1.0 - near)

    fn = n - tp
    return {"tp": tp, "fp": fp, "fn": fn, "n": n,
            "score": score_from_counts(tp, fp, fn, alpha, beta, eps)}


def dti(pred, truth, radius_px=RADIUS_PX, alpha=ALPHA, beta=BETA, eps=EPS,
        mask=None):
    """Fast numpy DTI on 2-D arrays; same contract as dti_reference."""
    import numpy as np

    p = np.asarray(pred, dtype=np.float32)
    p = np.clip(np.where(np.isfinite(p), p, 0.0), 0.0, 1.0).astype(np.float32)
    tv = np.asarray(truth, dtype=np.float32)
    t = np.where(np.isfinite(tv), tv, 0.0) > 0.5
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        p = np.where(m, p, 0.0).astype(np.float32)
        t = t & m

    offs = kernel_offsets(radius_px)
    r = int(math.floor(radius_px))
    h, w = p.shape
    p_pad = np.pad(p, r)
    t_pad = np.pad(t.astype(np.float32), r)
    best = np.zeros((h, w), dtype=np.float32)
    near = np.zeros((h, w), dtype=np.float32)
    for dy, dx, k in offs:
        window = (slice(r + dy, r + dy + h), slice(r + dx, r + dx + w))
        np.maximum(best, p_pad[window] * np.float32(k), out=best)
        np.maximum(near, t_pad[window] * np.float32(k), out=near)

    n = int(t.sum())
    tp = float(best[t].sum(dtype=np.float64))
    fp = float((p * (1.0 - near)).sum(dtype=np.float64))
    fn = n - tp
    return {"tp": tp, "fp": fp, "fn": fn, "n": n,
            "score": score_from_counts(tp, fp, fn, alpha, beta, eps)}


def sweep_thresholds(prob, truth, thresholds, mask=None):
    """Score the binary map prob >= t for each t; returns [(t, score)]."""
    import numpy as np

    prob = np.asarray(prob, dtype=np.float32)
    prob = np.where(np.isfinite(prob), prob, 0.0)
    out = []
    for thr in thresholds:
        binary = (prob >= thr).astype(np.float32)
        out.append((float(thr), dti(binary, truth, mask=mask)["score"]))
    return out
