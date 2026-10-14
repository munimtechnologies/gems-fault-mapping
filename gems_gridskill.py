"""Which public layers single out cells near mapped young faults, at the GEMS scale.

    python gems_gridskill.py run --data-dir data/ --out-dir gridskill/ --download
    python gems_gridskill.py run --src USGScontributions_forNVML_dataRelease2021.zip \
                                 --out-dir gridskill/ --block-km 10 --block-km 25

gems_agreement.py found that lineations traced along gravity and magnetic
gradient maxima land near mapped Quaternary faults hardly more often than the
same lines moved 14 to 21 km. A drawn line is a hard yes or no, though. This
module asks the softer question of every layer in the USGS Nevada geothermal
machine-learning data release (DeAngelo and others, 2021,
https://doi.org/10.5066/P9V5SQRD): does it rank the cells within 300 m of a
mapped Quaternary fault (the GEMS metric's reach) above the other cells?

Features, all on the 100 m UTM 11N study grid of gems_agreement.py:

    <grid>          each 250 m ASCII grid, sampled bilinearly
    <grid>_grad     its horizontal gradient magnitude per km, taken on the
                    source grid and then sampled
    dist_<layer>    minus the distance to the nearest point or line of each
                    point and line layer (capped at 10 km), so that "higher"
                    always means "closer"

For each feature the area under the ROC curve (AUC) is the chance that a
random near-fault cell outranks a random other cell (ties count half). A
null model scores the same feature against the near-fault map shifted 14 to
21 km in five directions, which keeps how smooth both maps are and breaks
only their alignment. "edge" is |AUC - 0.5| minus the largest |chance AUC -
0.5|: above zero, the feature beats every shifted placement. Each feature's
top cells (as many as there are fault cells, on the side the AUC points to)
are also scored as a fault map with the GEMS metric next to the same
shifted-map chance.

With --block-km B (new in 0.8.0) the AUC is also taken within square blocks
of B km only: a near-fault cell is compared only with other cells of its own
block, and the pairs of all blocks are pooled (a stratified Mann-Whitney).
A layer that is merely high over the faulted part of the region, and not
along the faults within it, scores 0.5 there.

Layers computed from the fault traces themselves (slip and dilation tendency,
the fault vertices) are flagged "derived": they are a positive control, not
independent evidence.

By MV2 of Munim, Inc. Made with AI agents. Apache-2.0 licence.
"""

import argparse
import json
import os
import struct
import sys
import time
import zipfile

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

import gems_agreement
import gems_metric
import gems_regrid
import gems_shapes

__version__ = gems_metric.__version__

DIST_CAP_PX = 100  # 10 km
DERIVED = ("tdts", "dilation", "slip", "fault")


def is_derived(name):
    """True for layers computed from the fault traces (a positive control)."""
    low = name.lower()
    return any(k in low for k in DERIVED)


def average_ranks(x):
    """1-based ranks of a 1-D array, ties given their mean rank."""
    x = np.asarray(x)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    ends = np.r_[starts[1:], n]
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return ranks


def auc_from_ranks(ranks, pos):
    """AUC (Mann-Whitney) of the positives from precomputed ranks; None
    when either class is empty."""
    pos = np.asarray(pos, dtype=bool)
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if not n1 or not n0:
        return None
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * float(n0)))


def score_feature(values, valid, targets):
    """[AUC of `values` for each bool target map] over the `valid` cells."""
    v = np.asarray(values)[valid]
    ranks = average_ranks(v)
    return [auc_from_ranks(ranks, np.asarray(t, dtype=bool)[valid])
            for t in targets]


def block_ids(valid, block_px):
    """Block number of each valid cell, in the order of values[valid], for
    square blocks of block_px cells counted from the grid's top left."""
    rows, cols = np.nonzero(np.asarray(valid, dtype=bool))
    per_row = -(-np.shape(valid)[1] // block_px)
    return (rows // block_px) * per_row + cols // block_px


def block_ranks(v, bid):
    """1-based ranks of v within each block id, ties given their mean rank."""
    v = np.asarray(v)
    bid = np.asarray(bid)
    n = len(v)
    if not n:
        return np.empty(0, dtype=np.float64)
    order = np.lexsort((v, bid))
    bs, vs = bid[order], v[order]
    new_block = np.r_[True, bs[1:] != bs[:-1]]
    new_group = new_block | np.r_[True, vs[1:] != vs[:-1]]
    starts = np.flatnonzero(new_group)
    ends = np.r_[starts[1:], n]
    rank = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    block_start = np.flatnonzero(new_block)[np.cumsum(new_block) - 1]
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = rank - block_start
    return ranks


def stratified_auc(ranks, bid, pos):
    """(AUC over same-block pairs only, number of blocks holding both
    classes) from block_ranks; AUC None when no block holds both."""
    pos = np.asarray(pos, dtype=bool)
    bid = np.asarray(bid)
    n = np.bincount(bid).astype(np.float64)
    n1 = np.bincount(bid, weights=pos.astype(np.float64), minlength=len(n))
    r1 = np.bincount(bid, weights=np.where(pos, ranks, 0.0), minlength=len(n))
    pairs = n1 * (n - n1)
    ok = pairs > 0
    if not ok.any():
        return None, 0
    u = r1[ok] - n1[ok] * (n1[ok] + 1) / 2.0
    return float(u.sum() / pairs[ok].sum()), int(ok.sum())


def gradient_per_km(src):
    """Source of the horizontal gradient magnitude of a Source, per km."""
    v = np.asarray(src.values, dtype=np.float64)
    if min(v.shape) < 2:
        g = np.full(v.shape, np.nan)
    else:
        gy, gx = np.gradient(v, src.dy, src.dx)
        g = np.hypot(gy, gx) * 1000.0
    return gems_regrid.Source(src.name + "_grad", g, src.left, src.top,
                              src.dx, src.dy)


def top_k_map(values, valid, k, sign=1.0):
    """bool map of the k valid cells with the highest sign * value."""
    out = np.zeros(np.shape(values), dtype=bool)
    idx = np.flatnonzero(np.asarray(valid).ravel())
    k = int(min(k, len(idx)))
    if k <= 0:
        return out
    v = sign * np.asarray(values, dtype=np.float64).ravel()[idx]
    pick = idx[np.argpartition(-v, k - 1)[:k]]
    out.ravel()[pick] = True
    return out


def layer_points(layer):
    """[(x, y)] of a point layer (shp bytes, dbf, prj), as stored (metres)."""
    shp = layer[0]
    return [p for _kind, parts in gems_shapes.read_shp(shp)
            for part in parts for p in part]


def evaluate(name, values, mask, near, shifts, fault, kind, gems=True,
             blocks=()):
    """One feature's row of the report. blocks: block sizes in cells for
    the within-block AUC."""
    valid = mask & np.isfinite(values)
    targets = [near] + [np.roll(near, sh, axis=(0, 1)) for sh in shifts]
    aucs = score_feature(values, valid, targets)
    row = {"feature": name, "kind": kind, "derived": is_derived(name),
           "cells": int(valid.sum()), "near_cells": int((near & valid).sum())}
    if aucs[0] is None:
        row["auc"] = None
        return row
    chance = [a for a in aucs[1:] if a is not None]
    row["auc"] = round(aucs[0], 4)
    row["chance_auc"] = [round(a, 4) for a in chance]
    worst = max(abs(a - 0.5) for a in chance) if chance else 0.0
    row["edge"] = round(abs(aucs[0] - 0.5) - worst, 4)
    if blocks:
        v = np.asarray(values)[valid]
        row["block"] = []
        for px in blocks:
            bid = block_ids(valid, px)
            ranks = block_ranks(v, bid)
            got = [stratified_auc(ranks, bid, np.asarray(t, dtype=bool)[valid])
                   for t in targets]
            if got[0][0] is None:
                row["block"].append({"px": int(px), "auc": None})
                continue
            b_chance = [a for a, _n in got[1:] if a is not None]
            b_worst = max(abs(a - 0.5) for a in b_chance) if b_chance else 0.0
            row["block"].append({
                "px": int(px), "auc": round(got[0][0], 4),
                "blocks": got[0][1],
                "chance_auc": [round(a, 4) for a in b_chance],
                "edge": round(abs(got[0][0] - 0.5) - b_worst, 4)})
    if gems:
        sign = 1.0 if aucs[0] >= 0.5 else -1.0
        k = int((fault & valid).sum())
        pred = top_k_map(values, valid, k, sign)
        cmp_ = gems_agreement.compare(pred, fault, valid, shifts)
        row["gems"] = {"direction": "high" if sign > 0 else "low",
                       "score": cmp_["score"], "chance_score": cmp_["chance_score"],
                       "lift": cmp_["lift"]}
    return row


def markdown(report):
    rows = sorted([r for r in report["features"] if r.get("auc") is not None],
                  key=lambda r: -r["edge"])
    cell = report["grid"].get("cell_m", gems_agreement.CELL_M)
    sizes = []
    for r in rows:
        for b in r.get("block") or []:
            if b["px"] not in sizes:
                sizes.append(b["px"])
    out = ["# Which public layers single out cells near mapped Quaternary faults",
           "",
           "gems_gridskill.py %s, data: %s, %d scored cells, %d within 300 m of "
           "a mapped Quaternary fault. AUC: chance a near-fault cell outranks "
           "another cell (0.5 = no signal; distance features are negated, so "
           "above 0.5 means closer). Chance: the same feature against the "
           "near-fault map shifted 14-21 km five ways (min-max). Edge: |AUC - "
           "0.5| minus the largest shifted |AUC - 0.5|. GEMS: the feature's top "
           "cells (as many as fault cells) scored as a fault map, with the "
           "shifted-map chance in brackets. * = derived from the fault traces "
           "(positive control)." % (
               report["version"], report["release"],
               report["grid"]["scored_cells"], report["grid"]["near_cells"])]
    if sizes:
        out.append("")
        out.append("Within B km: the AUC over pairs of cells in the same B km "
                   "square block only, pooled over blocks, with the same "
                   "shifted-map chance (min-max) and edge.")
    head = "| feature | AUC | chance AUC | edge | GEMS (chance) |"
    rule = "|---|---|---|---|---|"
    for px in sizes:
        head += " within %g km (chance) | edge |" % (px * cell / 1000.0)
        rule += "---|---|"
    out += ["", head, rule]
    for r in rows:
        ch = r.get("chance_auc") or []
        g = r.get("gems")
        line = "| %s%s | %.3f | %s | %+.3f | %s |" % (
            r["feature"], " *" if r["derived"] else "", r["auc"],
            "%.3f-%.3f" % (min(ch), max(ch)) if ch else "-", r["edge"],
            "%.3f (%.3f)" % (g["score"], g["chance_score"] or 0.0) if g else "-")
        by_px = {b["px"]: b for b in r.get("block") or []}
        for px in sizes:
            b = by_px.get(px)
            if not b or b.get("auc") is None:
                line += " - | - |"
                continue
            bc = b.get("chance_auc") or []
            line += " %.3f (%s) | %+.3f |" % (
                b["auc"], "%.3f-%.3f" % (min(bc), max(bc)) if bc else "-",
                b["edge"])
        out.append(line)
    if report.get("skipped"):
        out += ["", "Not scored: " + "; ".join(
            "%s (%s)" % (k, v) for k, v in sorted(report["skipped"].items()))]
    return "\n".join(out) + "\n"


def cmd_run(args):
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    zip_path = args.src or os.path.join(args.data_dir, gems_agreement.RELEASE_ZIP)
    report = {"version": __version__, "release": gems_agreement.RELEASE_DOI}
    if args.download and not args.src:
        os.makedirs(args.data_dir, exist_ok=True)
        report["download"] = gems_agreement.download(
            gems_agreement.RELEASE_URL, zip_path, gems_agreement.RELEASE_BYTES)

    origin, shape = gems_agreement.study_grid()
    mask = gems_agreement.study_mask(origin, shape)
    foot, note = gems_agreement.grid_footprint(zip_path, origin, shape)
    if foot is not None and foot.any():
        mask &= foot
    cell = gems_agreement.CELL_M
    shifts = gems_agreement.SHIFTS_PX
    blocks = tuple(sorted({max(1, int(round(km * 1000.0 / cell)))
                           for km in args.block_km or ()}))

    layers = gems_shapes._layers(zip_path)
    types = {}
    for name in layers:
        try:
            types[name] = gems_shapes.shp_header(layers[name][0])[0]
        except (ValueError, struct.error):
            continue
    qname = gems_agreement.pick_layer(types, ("quaternary", "fault"))
    if qname is None:
        raise SystemExit("no Quaternary fault line layer in %s" % zip_path)
    qlines, _how = gems_agreement.layer_lines(layers[qname])
    fault = gems_agreement.rasterize(qlines, origin, shape) & mask
    radius = gems_agreement.NEAR_PX
    near = (gems_regrid.distance_px(fault, int(radius) + 1) <= radius + 1e-9) & mask
    report["grid"] = {"origin": list(origin), "shape": list(shape),
                      "cell_m": cell, "scored_cells": int(mask.sum()),
                      "fault_cells": int(fault.sum()),
                      "near_cells": int(near.sum()), "footprint": note,
                      "fault_layer": qname, "block_px": list(blocks)}
    print(json.dumps(report["grid"]))
    sys.stdout.flush()

    rows, skipped = [], {}

    def add(name, values, kind):
        row = evaluate(name, values, mask, near, shifts, fault, kind,
                       gems=not args.no_gems, blocks=blocks)
        rows.append(row)
        print("%-34s auc %s  edge %s  gems %s  block %s" % (
            name, row.get("auc"), row.get("edge"),
            (row.get("gems") or {}).get("score"),
            [(b.get("auc"), b.get("edge")) for b in row.get("block") or []]))
        sys.stdout.flush()

    try:
        grids = gems_regrid.list_zip_grids(zip_path)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        grids = []
        skipped["ascii grids"] = str(exc)
    for member, _w, _h, _dx, _kind in grids:
        stem = os.path.splitext(os.path.basename(member))[0]
        try:
            src = gems_regrid.load_source("%s::%s" % (zip_path, stem),
                                          assume_utm11=True)
        except (SystemExit, ValueError, KeyError, OSError) as exc:
            skipped[stem] = str(exc)
            continue
        add(stem, gems_regrid.regrid(src, origin, cell, shape), "grid")
        add(stem + "_grad", gems_regrid.regrid(gradient_per_km(src), origin,
                                               cell, shape), "gradient")
        del src

    for name in sorted(layers):
        kind = types.get(name)
        if name == qname or kind is None:
            continue
        if kind in gems_shapes.LINE_TYPES:
            try:
                lines, _how = gems_agreement.layer_lines(layers[name])
            except (ValueError, struct.error) as exc:
                skipped[name] = str(exc)
                continue
            hit = gems_agreement.rasterize(lines, origin, shape)
            dist = gems_regrid.distance_px(hit, DIST_CAP_PX)
            add("dist_" + name, -dist, "line distance")
        elif kind in gems_shapes.POINT_TYPES or kind in gems_shapes.MULTIPOINT_TYPES:
            try:
                pts = layer_points(layers[name])
            except (ValueError, struct.error) as exc:
                skipped[name] = str(exc)
                continue
            dist, _counts = gems_regrid.point_bands(pts, origin, cell, shape,
                                                    cap=DIST_CAP_PX, radius=1)
            add("dist_" + name, -dist, "point distance")
        else:
            skipped[name] = "shape type %s not used" % kind

    report["features"] = rows
    report["skipped"] = skipped
    report["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out_dir, "gridskill.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(args.out_dir, "gridskill.md"), "w",
              encoding="utf-8") as fh:
        fh.write(markdown(report))
    print(json.dumps({"version": __version__, "features": len(rows),
                      "seconds": report["seconds"]}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score every layer and write the report")
    r.add_argument("--src", help="the USGS release zip (else DATA_DIR/%s)"
                                 % gems_agreement.RELEASE_ZIP)
    r.add_argument("--data-dir", default="data")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--download", action="store_true",
                   help="download the 256 MB release zip into --data-dir")
    r.add_argument("--no-gems", action="store_true",
                   help="skip the GEMS-metric score of each feature's top cells")
    r.add_argument("--block-km", type=float, action="append",
                   help="also score the AUC within square blocks of this many "
                        "km (repeatable)")
    args = ap.parse_args(argv)
    return {"run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
