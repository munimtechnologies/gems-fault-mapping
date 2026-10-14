"""How far public fault maps agree with each other, scored with the GEMS metric.

    python gems_agreement.py run --data-dir data/ --out-dir agreement/ --download --qfaults
    python gems_agreement.py run --src USGScontributions_forNVML_dataRelease2021.zip \
                                 --out-dir agreement/

The GEMS Prize scores a fault map against hidden labels with a
distance-weighted Tversky index (gems_metric.py). Before training anything it
helps to know what that number looks like between maps made independently by
geologists and by instruments: it is a yardstick for what a model's score
means. This module measures it on public data around the competition area:

    quaternary_faults     Quaternary faults in the USGS Nevada geothermal
                          machine-learning data release (DeAngelo and others,
                          2021, https://doi.org/10.5066/P9V5SQRD)
    tdts_fault_lines      the fault lines of the same release that carry slip
                          and dilation tendency
    gravity_lineations    lineations traced along maxima of the horizontal
    magnetic_lineations   gravity and magnetic gradient (no surface mapping)
    potential_field       gravity and magnetic lineations together
    qfaults_national      the USGS Quaternary Fault and Fold Database,
                          downloaded live with --qfaults

Every map is drawn onto one 100 m grid in UTM zone 11N over the release's
bounding box, and every ordered pair is scored as (prediction, truth) with
the GEMS metric, next to the share of each map's cells within 300 m of the
other's. A null model shifts the prediction map by 14 to 21 km in five
directions (wrapping round the grid) and scores it again, so "lift" says how
much better than chance placement the agreement is.

The metric only looks 300 m out. A distance profile (--radii, default 100 m
to 5 km) gives the share of the truth map's cells within each distance of the
prediction map, for the map and for its shifted copies, so it shows at what
scale two maps are related even when they do not coincide.

Only cells inside the bounding box and, when the zip holds it, inside the
footprint of the release's filtered gravity grid are scored.

By MV2 of Munim, Inc. Made with AI agents. Apache-2.0 licence.
"""

import argparse
import json
import math
import os
import struct
import sys
import time
import urllib.request
import zipfile

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

import gems_external
import gems_metric
import gems_regrid
import gems_shapes

__version__ = gems_metric.__version__

RELEASE_DOI = "https://doi.org/10.5066/P9V5SQRD"
RELEASE_URL = ("https://www.sciencebase.gov/catalog/file/get/"
               "60be89e3d34e86b938912329?f=__disk__a7%2F98%2F1e%2F"
               "a7981e73b299f58eaf574cd0fa093dcf641c5d53")
RELEASE_ZIP = "USGScontributions_forNVML_dataRelease2021.zip"
RELEASE_BYTES = 256302969
# ScienceBase item 60be89e3d34e86b938912329: west, south, east, north.
STUDY_BBOX = (-119.5306, 38.448, -114.417, 40.7282)
CELL_M = 100.0
BLOCK_PX = 10
NEAR_PX = gems_metric.RADIUS_PX
SHIFTS_PX = ((0, 157), (0, -211), (173, 0), (-149, 0), (101, 97))
PROFILE_PX = (1, 3, 5, 10, 20, 50)
FOOTPRINT_GRID = "grv_filthg"
QFAULTS_MARGIN = 0.05

MAPS = (("quaternary_faults", ("quaternary", "fault")),
        ("tdts_fault_lines", ("tdts", "faultline")),
        ("gravity_lineations", ("grav", "lineation")),
        ("magnetic_lineations", ("mag", "lineation")))
UNIONS = (("potential_field", ("gravity_lineations", "magnetic_lineations")),)


def pick_layer(types, needles):
    """First layer (by name) whose lower-case name holds every needle and
    whose shapes are lines; None when there is none.

    types maps layer name to shapefile shape type.
    """
    for name in sorted(types):
        low = name.lower()
        if (all(n in low for n in needles)
                and types[name] in gems_shapes.LINE_TYPES):
            return name
    return None


def layer_lines(layer):
    """(line parts in UTM 11N metres, how the projection was decided) for
    one (shp bytes, dbf bytes, prj text) layer.

    A .prj naming UTM zone 11N is taken as metres, lon/lat vertices are
    projected, and anything else is assumed to be the release's own NAD83
    transverse Mercator on -117, which is UTM zone 11N.
    """
    shp, _dbf, prj = layer
    kind = gems_shapes.prj_kind(prj)
    doc = gems_shapes.to_geojson(gems_shapes.read_shp(shp), None, kind)
    raw = gems_external.parse_lines(doc)
    if kind == "utm11":
        how = "UTM 11N (.prj)"
    elif raw and gems_external.looks_lonlat(raw):
        how = "lon/lat, projected"
    else:
        how = "assumed UTM 11N (release metadata: NAD83 transverse Mercator, CM -117)"
    return gems_external.lines_utm11(doc), how


def study_grid(bbox=STUDY_BBOX, cell=CELL_M, samples=50):
    """(origin (x0, y0), shape (h, w)) of the UTM 11N grid around a lon/lat
    box (west, south, east, north), from `samples` points along each edge."""
    west, south, east, north = bbox
    xs, ys = [], []
    for i in range(samples + 1):
        f = i / float(samples)
        lon = west + f * (east - west)
        lat = south + f * (north - south)
        for la, lo in ((south, lon), (north, lon), (lat, west), (lat, east)):
            x, y = gems_external.utm_forward(la, lo)
            xs.append(x)
            ys.append(y)
    x0 = math.floor(min(xs) / cell) * cell
    y0 = math.ceil(max(ys) / cell) * cell
    w = int(math.ceil((max(xs) - x0) / cell))
    h = int(math.ceil((y0 - min(ys)) / cell))
    return (x0, y0), (h, w)


def study_mask(origin, shape, bbox=STUDY_BBOX, cell=CELL_M, block=BLOCK_PX):
    """bool [h, w]: cells whose block's centre lies inside the lon/lat box.

    The test is made once per block x block cells, which is plenty for a
    mask whose edges are kilometres from the faults being compared.
    """
    x0, y0 = origin
    h, w = shape
    west, south, east, north = bbox
    bh, bw = -(-h // block), -(-w // block)
    coarse = np.zeros((bh, bw), dtype=bool)
    for i in range(bh):
        y = y0 - (i * block + block / 2.0) * cell
        for j in range(bw):
            lat, lon = gems_external.utm_inverse(
                x0 + (j * block + block / 2.0) * cell, y)
            coarse[i, j] = south <= lat <= north and west <= lon <= east
    return np.repeat(np.repeat(coarse, block, 0), block, 1)[:h, :w]


def grid_footprint(zip_path, origin, shape, name=FOOTPRINT_GRID, cell=CELL_M):
    """(bool [h, w] where the release's grid `name` has data, note), or
    (None, why not) when the zip does not hold it."""
    try:
        src = gems_regrid.load_source("%s::%s" % (zip_path, name),
                                      assume_utm11=True)
    except (SystemExit, ValueError, KeyError, OSError,
            zipfile.BadZipFile) as exc:
        return None, "%s not used: %s" % (name, exc)
    vals = gems_regrid.regrid(src, origin, cell, shape, method="nearest")
    return np.isfinite(vals), "%s: %d x %d cells of %g m" % (
        name, src.values.shape[0], src.values.shape[1], src.dx)


def rasterize(lines, origin, shape, cell=CELL_M):
    """bool [h, w]: cells the lines pass through."""
    return gems_external.cells_to_mask(
        gems_external.rasterize_lines(lines, origin, cell, shape), shape)


def near_share(a, b, radius=NEAR_PX):
    """Share of a's cells within `radius` px of a cell of b (Euclidean
    between centres with scipy, chessboard without it); None if a is empty."""
    a = np.asarray(a, dtype=bool)
    n = int(a.sum())
    if not n:
        return None
    d = gems_regrid.distance_px(np.asarray(b, dtype=bool),
                                int(math.ceil(radius)) + 1)
    return round(float((d[a] <= radius + 1e-9).sum()) / n, 4)


def near_profile(pred, truth, mask, radii=PROFILE_PX, shifts=SHIFTS_PX):
    """Share of truth's cells within each radius (px) of pred, for pred as
    it is and for its shifted copies (masked like pred), with their ratio.

    One distance transform per placement serves every radius. [] when the
    truth is empty or no radius is asked for.
    """
    m = np.asarray(mask, dtype=bool)
    p = np.asarray(pred, dtype=bool) & m
    t = np.asarray(truth, dtype=bool) & m
    n = int(t.sum())
    if not n or not radii:
        return []
    cap = int(math.ceil(max(radii))) + 1

    def shares(q):
        d = gems_regrid.distance_px(q, cap)[t]
        return [float((d <= r + 1e-9).sum()) / n for r in radii]

    real = shares(p)
    null = [shares(np.roll(p, sh, axis=(0, 1)) & m) for sh in shifts]
    out = []
    for i, r in enumerate(radii):
        ch = sum(s[i] for s in null) / len(null) if null else None
        out.append({"radius_px": r, "radius_m": r * CELL_M,
                    "share": round(real[i], 4),
                    "chance_share": None if ch is None else round(ch, 4),
                    "lift": round(real[i] / ch, 2) if ch else None})
    return out


def compare(pred, truth, mask, shifts=SHIFTS_PX):
    """GEMS score of map `pred` against map `truth` inside `mask`, with the
    shifted-map null model."""
    m = np.asarray(mask, dtype=bool)
    p = np.asarray(pred, dtype=bool) & m
    t = np.asarray(truth, dtype=bool) & m
    s = gems_metric.dti(p.astype(np.float32), t, mask=m)
    chance = [gems_metric.dti(np.roll(p, sh, axis=(0, 1)).astype(np.float32),
                              t, mask=m)["score"] for sh in shifts]
    mean = sum(chance) / len(chance) if chance else None
    return {"score": round(s["score"], 4), "tp": round(s["tp"], 1),
            "fp": round(s["fp"], 1), "fn": round(s["fn"], 1),
            "truth_cells": s["n"], "pred_cells": int(p.sum()),
            "truth_near_pred": near_share(t, p),
            "pred_near_truth": near_share(p, t),
            "chance_score": None if mean is None else round(mean, 4),
            "chance_scores": [round(c, 4) for c in chance],
            "lift": round(s["score"] / mean, 2) if mean else None}


def related(a, b):
    """True when one map is a union that contains the other."""
    for key, parts in UNIONS:
        if (a == key and b in parts) or (b == key and a in parts):
            return True
    return False


def download(url, path, expect_bytes=None):
    """Fetch url to path unless a file of the expected size is there."""
    if os.path.exists(path) and (expect_bytes is None
                                 or os.path.getsize(path) == expect_bytes):
        return "cached"
    part = path + ".part"
    req = urllib.request.Request(
        url, headers={"User-Agent": gems_external.USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp, \
            open(part, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    size = os.path.getsize(part)
    os.replace(part, path)
    if expect_bytes is not None and size != expect_bytes:
        return "downloaded %d bytes, expected %d" % (size, expect_bytes)
    return "downloaded"


def markdown(report):
    """The report as a short Markdown page."""
    keys = list(report["maps"])
    got = {(r["pred"], r["truth"]): r for r in report["pairs"]}
    out = ["# How far public fault maps agree, scored with the GEMS metric", "",
           "gems_agreement.py %s, data: %s. Each cell is the GEMS score of "
           "the row map as the prediction against the column map as the "
           "truth; in brackets, the mean score of the row map shifted 14-21 "
           "km (chance)." % (report["version"], report["release"]), "",
           "| prediction \\ truth | " + " | ".join(keys) + " |",
           "|---|" + "---|" * len(keys)]
    for a in keys:
        cells = []
        for b in keys:
            r = got.get((a, b))
            if r is None:
                cells.append("-")
            else:
                ch = r["chance_score"]
                cells.append("%.3f (%s)" % (
                    r["score"], "-" if ch is None else "%.3f" % ch))
        out.append("| %s | %s |" % (a, " | ".join(cells)))
    out += ["", "| map | source | cells scored |", "|---|---|---|"]
    for key, info in report["maps"].items():
        if "layer" in info:
            src = "%s (%s)" % (info["layer"], info["projection"])
        elif "union_of" in info:
            src = "union of " + " + ".join(info["union_of"])
        else:
            src = "%s, %s features" % (info.get("service"), info.get("features"))
        out.append("| %s | %s | %s |" % (key, src, info.get("cells_in_mask")))
    out += ["", "| prediction | truth | truth within 300 m | prediction within "
            "300 m | lift over chance |", "|---|---|---|---|---|"]
    for r in report["pairs"]:
        out.append("| %s | %s | %s | %s | %s |" % (
            r["pred"], r["truth"], r["truth_near_pred"], r["pred_near_truth"],
            r["lift"]))
    prof = [r for r in report["pairs"] if r.get("profile")]
    if prof:
        heads = ["%g m" % p["radius_m"] for p in prof[0]["profile"]]
        out += ["", "Share of the truth map's cells within each distance of "
                "the prediction map; in brackets, the share divided by the "
                "mean share for the shifted prediction maps.", "",
                "| prediction | truth | " + " | ".join(heads) + " |",
                "|---|---|" + "---|" * len(heads)]
        for r in prof:
            out.append("| %s | %s | %s |" % (r["pred"], r["truth"], " | ".join(
                "%.3f (%s)" % (p["share"], "-" if p["lift"] is None
                               else p["lift"]) for p in r["profile"])))
    if report.get("missing"):
        out += ["", "Not compared: " + "; ".join(
            "%s (%s)" % (k, v) for k, v in sorted(report["missing"].items()))]
    return "\n".join(out) + "\n"


def cmd_run(args):
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    bbox = tuple(float(v) for v in args.bbox.split(",")) if args.bbox \
        else STUDY_BBOX
    radii = tuple(int(v) for v in args.radii.split(",") if v.strip())
    zip_path = args.src or os.path.join(args.data_dir, RELEASE_ZIP)
    report = {"version": __version__, "release": RELEASE_DOI,
              "bbox_lonlat": list(bbox), "profile_radii_px": list(radii)}
    if args.download and not args.src:
        os.makedirs(args.data_dir, exist_ok=True)
        report["download"] = download(RELEASE_URL, zip_path, RELEASE_BYTES)

    layers = gems_shapes._layers(zip_path)
    types, listing = {}, []
    for name in sorted(layers):
        shp, _dbf, prj = layers[name]
        try:
            kind = gems_shapes.shp_header(shp)[0]
            n = len(gems_shapes.read_shp(shp))
        except (ValueError, struct.error) as exc:
            listing.append([name, "unreadable: %s" % exc, None, None])
            continue
        types[name] = kind
        listing.append([name, gems_shapes.TYPE_NAMES.get(kind, str(kind)), n,
                        gems_shapes.prj_kind(prj) or "unknown"])
    report["shapefile_layers"] = listing
    try:
        report["ascii_grids"] = [list(g) for g in
                                 gems_regrid.list_zip_grids(zip_path)]
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        report["ascii_grids"] = "not listed: %s" % exc

    origin, shape = study_grid(bbox)
    mask = study_mask(origin, shape, bbox)
    foot, note = grid_footprint(zip_path, origin, shape)
    if foot is not None and foot.any():
        mask &= foot
    report["grid"] = {"crs": "EPSG:26911 / 32611 (UTM 11N)", "cell_m": CELL_M,
                      "origin": list(origin), "shape": list(shape),
                      "scored_cells": int(mask.sum()), "footprint": note}

    maps, info, missing = {}, {}, {}
    for key, needles in MAPS:
        name = pick_layer(types, needles)
        if name is None:
            missing[key] = "no line layer named like %s" % "+".join(needles)
            continue
        try:
            lines, how = layer_lines(layers[name])
        except (ValueError, struct.error) as exc:
            missing[key] = "%s unreadable: %s" % (name, exc)
            continue
        maps[key] = rasterize(lines, origin, shape)
        info[key] = {"layer": name, "lines": len(lines), "projection": how}
    for key, parts in UNIONS:
        if all(p in maps for p in parts):
            maps[key] = np.logical_or.reduce([maps[p] for p in parts])
            info[key] = {"union_of": list(parts)}
    del layers
    if args.qfaults:
        west, south, east, north = bbox
        try:
            feats = gems_external.fetch_features(
                (west - QFAULTS_MARGIN, south - QFAULTS_MARGIN,
                 east + QFAULTS_MARGIN, north + QFAULTS_MARGIN))
            lines = gems_external.lines_utm11(
                {"type": "FeatureCollection", "features": feats})
            maps["qfaults_national"] = rasterize(lines, origin, shape)
            info["qfaults_national"] = {
                "service": gems_external.QFAULTS_URL % gems_external.QFAULTS_LAYER,
                "features": len(feats), "lines": len(lines)}
        except Exception as exc:  # network trouble is reported, not fatal
            missing["qfaults_national"] = "fetch failed: %s" % exc
    for key in maps:
        info[key]["cells_in_mask"] = int((maps[key] & mask).sum())
    report["maps"] = info
    report["missing"] = missing

    pairs = []
    for a in maps:
        for b in maps:
            if a == b or related(a, b):
                continue
            row = {"pred": a, "truth": b}
            row.update(compare(maps[a], maps[b], mask))
            row["profile"] = near_profile(maps[a], maps[b], mask, radii)
            pairs.append(row)
            print("%-20s vs %-20s score %.4f  chance %s  lift %s"
                  % (a, b, row["score"], row["chance_score"], row["lift"]))
            sys.stdout.flush()
    report["pairs"] = pairs
    report["seconds"] = round(time.time() - t0, 1)

    with open(os.path.join(args.out_dir, "agreement.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(args.out_dir, "agreement.md"), "w",
              encoding="utf-8") as fh:
        fh.write(markdown(report))
    print(json.dumps({"version": __version__, "out_dir": args.out_dir,
                      "maps": list(maps), "pairs": len(pairs),
                      "scored_cells": report["grid"]["scored_cells"],
                      "seconds": report["seconds"]}, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="compare the maps and write the report")
    r.add_argument("--src", help="the USGS release zip (else DATA_DIR/%s)"
                                 % RELEASE_ZIP)
    r.add_argument("--data-dir", default="data")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--download", action="store_true",
                   help="download the 256 MB release zip into --data-dir")
    r.add_argument("--qfaults", action="store_true",
                   help="also fetch the national Qfaults lines (network)")
    r.add_argument("--bbox", help="--bbox=west,south,east,north in degrees "
                                  "(default: the release's bounding box)")
    r.add_argument("--radii", default=",".join(str(v) for v in PROFILE_PX),
                   help="--radii=1,3,10 distances in px for the distance "
                        "profile; --radii= turns it off")
    args = ap.parse_args(argv)
    return {"run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
