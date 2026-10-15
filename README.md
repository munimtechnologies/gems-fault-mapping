# gems-fault-mapping

Open tools for the **DOE GEMS Prize** (Geologic Enhanced Mapping System), the
$300,000 American-Made challenge on DrivenData to map hidden geologic faults
that point to geothermal resources in western Nevada (the GeoDAWN area).
Submissions close **3 December 2026**.
Competition: https://www.drivendata.org/competitions/306/competition-doe-gems/

By MV2 of Munim, Inc. The code was written with AI agents working in MV2's
name. Apache-2.0 licence. It is shared publicly so every competitor can use it.

## What is here

| File | What it does |
|---|---|
| `gems_metric.py` | A local copy of the competition metric, the distance-weighted Tversky index (DTI), built from the public problem description. It has a plain-Python reference version and a fast numpy version, plus a threshold rule derived from the metric. |
| `gems_baseline.py` | A CPU pipeline: edge and ridge features for every band, a gradient-boosted pixel classifier, spatial block cross-validation, a threshold chosen by the metric, a GeoTIFF writer, and a submission checker. `--extra-features` adds any other raster on the same grid (such as the lidar or regridded features below). Out-of-fold probabilities are saved as `work/oof.npy`. Since 0.5.0, a band that covers only part of the area is left missing outside its coverage, and so are its edge filters within 6 px of the gap, so the edge of a survey is not learned as a fault. |
| `gems_extend.py` | Finds the tips of the known faults and rasterises a short extension beyond each one along the fault's direction (pure numpy). `gems_baseline.py extend` ORs these into an existing submission without retraining. |
| `gems_lidar.py` | Reads the 1 m lidar DEM tiles listed in `1m_DEM_links.csv` and summarises fault-scarp evidence for every 100 m cell: slope mean and 95th percentile, share of steep pixels, relief, and how much the slopes face one way (structure-tensor coherence). Resumable, so a long run over all tiles can be stopped and restarted. |
| `gems_external.py` | Downloads the public-domain USGS Quaternary Fault and Fold Database (Qfaults) over the grid, or reads any GeoJSON of fault lines, projects it to UTM 11N without pyproj, and writes three bands on the competition grid: mapped line, distance to it, and *candidate* (mapped lines more than N px from any known fault). It reports how far the external map and the known faults agree, and `merge` ORs the candidates into a submission. |
| `gems_robust.py` | Checks how a submission holds up when the Final Round labels are revised. `table` gives the exact score of a band of width 1, 3 or 5 px against a straight fault that lies 0 to 3 px away. `stress` moves, removes and adds faults in a truth raster many times and re-scores each submission (or each threshold and dilation of `oof.npy`) against the same perturbed copies. |
| `gems_shapes.py` | Reads shapefiles (plain, in a zip, or in a zip inside a zip) with the standard library only and writes GeoJSON that `gems_external.py rasterize` and `gems_regrid.py points` accept. Made for the USGS Nevada geothermal data release below. |
| `gems_regrid.py` | Puts outside rasters and point layers on the competition grid as extra feature bands. `grid` samples ESRI ASCII grids (`.asc`, also straight from inside a zip as `ZIP::NAME`) or GeoTIFFs in UTM 11N onto the 100 m cells, bilinear or nearest, NaN beyond the source. `points` turns point layers into a distance band and a count band each. `list` shows the ASCII grids in a zip. |
| `gems_agreement.py` | Scores how far public fault maps agree **with each other** using the GEMS metric: the Quaternary faults, fault lines and gravity and magnetic lineations of the USGS Nevada release, and the live USGS Qfaults database, every ordered pair, with a chance baseline and a distance profile from 100 m to 5 km. Needs only numpy (scipy optional), no competition data. See "How far public fault maps agree" below. |
| `gems_gridskill.py` | Asks, for **every** layer of the USGS Nevada release (each 250 m grid, its gradient magnitude, and the distance to each point and lineation layer), whether it ranks the cells within 300 m of a mapped Quaternary fault above the other cells: an AUC against the same near-fault map shifted 14 to 21 km, plus the GEMS score of each layer's top cells. Layers derived from the fault traces are flagged as a positive control. `--block-km B` (0.8.0) also scores the AUC within B km blocks only. See "Which layers single out cells near mapped faults" below. |
| `docs/` | Results on the real USGS release (CC BY 4.0): `agreement-2026-10.md/.json` (map against map, with the distance profile), `layer-skill-2026-10.md/.json` (every layer ranked) and `layer-skill-blocks-2026-10.json` (the same ranking within 10 km and 25 km blocks, written up in `layer-skill-2026-10.md`). |
| `test_gems.py`, `test_external.py`, `test_robust.py`, `test_shapes.py`, `test_regrid.py`, `test_agreement.py`, `test_gridskill.py` | 141 offline tests: the metric on hand-worked grids, numpy against the reference on random grids, tip detection on lines, junctions and thick lines, lidar features on planes, scarps and bowls, a tile landing on the right cells, the projection against Snyder's published worked example, line rasterising, paged downloads with a fake server, the submission checks, the band table checked by hand and against the metric on a grid, shapefiles and dBASE tables built byte by byte from their published layouts, ASCII grids with wrapped rows, resampling a plane, gaps and edges, point bands across the grid edge, the baseline's handling of a band with a gap, map comparison and distance profiles on hand-worked lines, end-to-end agreement runs on a small zip, tied ranks and AUC on hand-worked arrays, within-block ranks and AUC on hand-worked grids (including a regional step that pooled AUC rewards and blocks do not), the gradient of a plane, top-k maps, and a distance feature's skill on a synthetic fault. |

```
pip install numpy scipy rasterio scikit-learn
python -m unittest -v
python gems_lidar.py     --links data/1m_DEM_links.csv --grid data/numeric_features.tif --out lidar_features.tif --limit 20
python gems_baseline.py train    --features data/numeric_features.tif --labels data/labels.tif --extra-features lidar_features.tif --out submission.tif
python gems_baseline.py validate --pred submission.tif --features data/numeric_features.tif
python gems_baseline.py score    --pred submission.tif --labels data/labels.tif
python gems_baseline.py extend   --pred submission.tif --labels data/labels.tif --length 3 --out submission_ext.tif
python gems_extend.py --labels data/labels.tif --out extensions.tif --length 3
python gems_external.py fetch     --grid data/numeric_features.tif --out qfaults.geojson
python gems_external.py rasterize --grid data/numeric_features.tif --lines qfaults.geojson --labels data/labels.tif --out external.tif
python gems_external.py merge     --pred submission.tif --external external.tif --out submission_qf.tif
python gems_robust.py table
python gems_robust.py stress --labels data/labels.tif --prob work/oof.npy --threshold 0.05 --threshold 0.1 --dilate 0 --dilate 1
python gems_shapes.py list    --src USGScontributions_forNVML_dataRelease2021.zip
python gems_shapes.py geojson --src USGScontributions_forNVML_dataRelease2021.zip --layer magneticLineations_nvML --out mag.geojson
python gems_external.py rasterize --grid data/numeric_features.tif --lines mag.geojson --labels data/labels.tif --out mag_external.tif
python gems_regrid.py list   --src USGScontributions_forNVML_dataRelease2021.zip
python gems_regrid.py grid   --grid data/numeric_features.tif --src USGScontributions_forNVML_dataRelease2021.zip::grv_filthg --src USGScontributions_forNVML_dataRelease2021.zip::psg_filthg --src USGScontributions_forNVML_dataRelease2021.zip::hfnvml_est --src USGScontributions_forNVML_dataRelease2021.zip::tdts_nvml --out usgs_grids.tif
python gems_shapes.py geojson --src USGScontributions_forNVML_dataRelease2021.zip --layer gravityIntersections_nvML --out grav_int.geojson
python gems_regrid.py points --grid data/numeric_features.tif --points grav_int.geojson --out usgs_points.tif
python gems_baseline.py train    --features data/numeric_features.tif --labels data/labels.tif --extra-features usgs_grids.tif --extra-features usgs_points.tif --out submission.tif
python gems_agreement.py run --data-dir data --out-dir agreement --download --qfaults
python gems_gridskill.py run --data-dir data --out-dir gridskill --download
python gems_gridskill.py run --data-dir data --out-dir gridskill_blocks --no-gems --block-km 10 --block-km 25
```

**Status:** the metric, tip detection, lidar features, projection, band
table, resampling, map comparison, ranking and checks are unit-tested on
synthetic grids. The pipeline has **not yet been run on the competition
data**, which is available only to registered entrants. `gems_shapes.py`,
`gems_regrid.py`, `gems_agreement.py` and `gems_gridskill.py` (pooled and
within blocks) have run on the real USGS release (results below and in
`docs/`).

## What the metric rewards

The page defines k(d) = max(1 - d/300 m, 0), alpha = 0.2 and beta = 0.8:

- **Recall matters most.** A missed fault pixel costs four times as much as
  a false alarm.
- **Near misses get partial credit.** A line one pixel off the true fault
  scores exactly 2/3 (`test_one_pixel_shift_scores_two_thirds`). Two pixels
  off gets 1/3, and three or more gets nothing.
- **Use a low threshold.** DTI = TP / ((1 - beta) TP + alpha FP + beta N).
  Maximising this ratio (Dinkelbach) shows that a pixel is worth marking
  when its fault probability q exceeds `alpha * lambda / (1 - lambda (1 - beta) + lambda alpha)`,
  where lambda is the score you can reach. With these weights that is just
  **0.2 x lambda**: at DTI 0.4, mark every pixel with q > 0.08
  (`gems_metric.threshold_rule`).
- **Hard 0/1 maps are usually enough.** For a single pixel, both the expected
  gain and the expected cost are linear in p, so the best value sits at 0 or 1
  (overlapping neighbours can bend this a little). The baseline picks the
  cut-off on out-of-fold predictions using the metric itself.

## When the labels move: line width (Final Round)

The Final Round re-scores the whole area against labels that experts revise
after seeing every Phase 1 submission. A fault's true trace may then sit a
pixel or two from where any map put it. `python gems_robust.py table` gives
the exact score per unit length of a straight fault when the submission
marks a band 1, 3 or 5 px wide and the true trace lies s px from its centre.
The right-hand columns pool the counts when s is equally likely to be any
whole pixel from -1 to 1, or from -2 to 2:

| band width | s = 0 | s = 1 | s = 2 | s = 3 | exact | within 1 px | within 2 px |
|---|---|---|---|---|---|---|---|
| 1 px | **1.000** | 0.667 | 0.333 | 0.000 | **1.000** | 0.778 | 0.600 |
| 3 px | 0.882 | **0.833** | 0.500 | 0.238 | 0.882 | **0.849** | **0.699** |
| 5 px | 0.714 | 0.682 | **0.625** | **0.385** | 0.714 | 0.692 | 0.664 |

- With exact positions, the thinnest line wins. As soon as the trace is
  uncertain by a pixel, a **3 px band wins** (0.849 against 0.778), and it is
  still best at two pixels. A 5 px band only wins when the trace is
  certainly 2 px or more away; in both averaged columns its outer rows cost
  more than they recover.
- Break-even: if a share p of the true fault length lies 1 px off the
  predicted trace and the rest lies on it, a 1 px line scores 1 - p/3 and a
  3 px band 15 / (17 + p). They are equal at p^2 + 14p - 6 = 0, so
  **p = 41.6%**. Above that, widen the line.
- These numbers are for straight faults along the grid axes. Diagonal and
  curved faults, ends and junctions differ a little. For real geometry use
  `stress`, which applies the same kind of revision to a whole raster.

`gems_baseline.py train` already tries dilation 0 and 1 on its out-of-fold
probabilities, but scores them against the labels exactly as drawn. `stress`
asks whether that choice survives revision. By default it runs six
scenarios over 20 random draws each: as labelled, every fault moved up to
1 px, up to 2 px, 20% of faults removed, 20% more faults added as 20 px
lines, and a mixed revision (1 px, 10% removed, 10% added). Every map is
scored against the same perturbed copies, so the means compare fairly.
Using `labels.tif` as the truth is only a stand-in for the hidden new faults,
but it shows which settings are fragile.

## How far public fault maps agree

A GEMS score means little without a yardstick. If two independent,
published maps of the same ground agree with each other at, say, 0.3 under
this metric, then a model at 0.3 against the hidden labels is already doing
about as well as a second mapper. And if lines traced only from gravity and
magnetic gradients agree with mapped faults far better than chance, buried
faults (the ones GEMS asks for) should show up in them too.

`gems_agreement.py run` measures this on public data alone:

- It reads the USGS Nevada release below straight from its zip (downloading
  it with `--download`) and, with `--qfaults`, fetches the national Qfaults
  lines over the same box.
- It draws every map on one 100 m grid in UTM zone 11N over the release's
  bounding box (119.53 W to 114.42 W, 38.45 N to 40.73 N), scored only
  inside the footprint of the release's `grv_filthg` grid when the zip holds
  it.
- For every ordered pair it reports the GEMS score with one map as the
  prediction and the other as the truth, the share of each map's cells
  within 300 m of the other's, and a **chance baseline**: the same score
  with the prediction map shifted 14 to 21 km in five directions (so it
  keeps its own density and shape but loses its position). "Lift" is the
  score divided by the chance score.
- Since the metric only looks 300 m out, a **distance profile** gives the
  share of the truth map's cells within 100 m, 300 m, 500 m, 1, 2 and 5 km
  of the prediction map, against the same shifted copies (`--radii`).
- It writes `agreement.json` (everything, including the layer list and the
  ASCII grids found in the zip) and `agreement.md` (the tables).

### Results (14 October 2026)

Run on the full release box: 2563 x 4464 cells of 100 m, 10,799,672 of
them inside the gravity grid's footprint, 299 s on 4 CPU cores (401 s with
the distance profile). Full tables:
[`docs/agreement-2026-10.md`](docs/agreement-2026-10.md), raw numbers:
[`docs/agreement-2026-10.json`](docs/agreement-2026-10.json).

| prediction | truth | GEMS score | chance | lift | truth within 300 m | prediction within 300 m |
|---|---|---|---|---|---|---|
| release Quaternary faults | national Qfaults (live, 8,460 features) | **0.796** | 0.045 | 17.6 | 98.1% | 98.6% |
| national Qfaults | release Quaternary faults | **0.799** | 0.045 | 17.7 | 98.6% | 98.1% |
| release Quaternary faults | release tdts fault lines | 1.000 | 0.045 | 22.0 | 100% | 100% |
| gravity lineations | release Quaternary faults | 0.081 | 0.077 | **1.05** | 21.8% | 8.6% |
| magnetic lineations | release Quaternary faults | 0.068 | 0.062 | **1.09** | 16.2% | 9.3% |
| gravity + magnetic lineations | release Quaternary faults | 0.105 | 0.099 | **1.06** | 34.3% | 8.9% |
| release Quaternary faults | gravity lineations | 0.048 | 0.046 | 1.05 | 8.6% | 21.8% |
| gravity lineations | magnetic lineations | 0.100 | 0.092 | 1.08 | 22.4% | 15.3% |

Distance profile, share of Quaternary-fault cells within each distance of
the lineations, as a multiple of the shifted-map share:

| lineations | 100 m | 300 m | 500 m | 1 km | 2 km | 5 km |
|---|---|---|---|---|---|---|
| gravity | 1.08 | 1.06 | 1.06 | 1.06 | 1.04 | 1.01 |
| magnetic | 1.10 | 1.11 | 1.09 | 1.07 | 1.04 | 1.01 |
| national Qfaults (for scale) | 24.5 | 10.7 | 6.8 | 3.75 | 2.13 | 1.22 |

What this says:

- **The release's two fault layers are one map.** Its Quaternary faults and
  its slip/dilation-tendency fault lines fall on exactly the same cells
  (150,394 each), so they count once, not twice.
- **Two editions of the same database score about 0.80, not 1.** The
  release's Quaternary faults (2021) and today's national Qfaults service
  overlap 98% within 300 m each way, yet the GEMS score is 0.80, because
  traces moved by a cell or two between editions cost partial credit. A
  model that reaches 0.8 against hand-drawn labels would be as close as a
  re-issue of the same map.
- **Potential-field lineations do not coincide with mapped Quaternary faults
  beyond chance, at any scale from 100 m to 5 km.** Lineations traced along
  gravity and magnetic gradient crests score 0.07 to 0.11 against the
  faults, only 5 to 9% above the same lineations shifted 14 to 21 km away,
  and the distance profile stays at 1.01 to 1.11 times chance all the way
  out. If the crests sat a fixed distance basinward of the surface traces,
  the profile would rise at that distance; it does not. **Drawing the
  lineations as predicted faults would score close to chance.**

Limits: this compares maps with each other, not with the GEMS labels, which
the organisers keep hidden; it covers the USGS release's box, which overlaps
only part of the GeoDAWN area; lines are rasterised one cell wide; five
shifted placements give a range, not a p-value; maps from one release may
share sources, so their agreement is an upper bound on independent
agreement; an offset that varies from fault to fault would be smeared
across the profile; and the Quaternary faults are only the young, mapped
part of the fault network, so "no coincidence with Quaternary faults" says
nothing about older or buried faults the lineations may follow. The
organisers mask the known USGS/INGENIOUS faults when scoring, so what
matters for an entry is the part of a public map that lies **away** from the
known faults; `gems_external.py rasterize --labels` measures that once the
competition data is at hand.

## Which layers single out cells near mapped faults

A drawn lineation is a hard yes or no. `gems_gridskill.py run` asks the
softer question of every layer in the USGS Nevada release: does it rank the
cells within 300 m of a mapped Quaternary fault (the metric's reach) above
the other cells? On the same 100 m study grid as `gems_agreement.py`:

- Features: each 250 m ASCII grid sampled bilinearly (gravity,
  pseudo-gravity and their filtered variants, heat flow and its
  uncertainty, slip and dilation tendency); each grid's horizontal gradient
  magnitude per km; and minus the distance (capped at 10 km) to the nearest
  point or line of every point and lineation layer, so "higher" always
  means "closer".
- Skill: the area under the ROC curve (AUC), the chance that a random
  near-fault cell outranks a random other cell, ties counting half. The null
  model scores the same feature against the near-fault map shifted 14 to
  21 km in five directions, which keeps how smooth both maps are and breaks
  only their alignment. "Edge" is |AUC - 0.5| minus the largest shifted
  |AUC - 0.5|: above zero, the feature beats every shifted placement.
- Each feature's top cells (as many as there are fault cells, on the side
  the AUC points to) are also scored as a fault map with the GEMS metric,
  next to the shifted-map chance.
- Layers computed from the fault traces (slip and dilation tendency, the
  fault vertices) are flagged with `*`: they are a positive control, not
  independent evidence.
- `--block-km B` also takes the AUC over pairs of cells in the same B km
  block only, pooled over blocks. A layer that is merely high over the
  faulted part of the region scores 0.5 there; one that follows the faults
  within each block keeps its AUC.

### Results (14-15 October 2026)

39 features, 903,921 near-fault cells among 10,799,672, 476 s on 4 CPU
cores (558 s for the within-block run). Full tables and discussion:
[`docs/layer-skill-2026-10.md`](docs/layer-skill-2026-10.md), raw numbers:
[`docs/layer-skill-2026-10.json`](docs/layer-skill-2026-10.json) and
[`docs/layer-skill-blocks-2026-10.json`](docs/layer-skill-blocks-2026-10.json).

| feature | AUC | shifted AUC | edge | GEMS of top cells (chance) | edge within 10 km blocks | edge within 25 km blocks |
|---|---|---|---|---|---|---|
| distance to the fault lines * (positive control) | 1.000 | 0.509-0.559 | +0.441 | 1.000 (0.045) | +0.487 | +0.479 |
| slip/dilation tendency gradient * | 0.875 | 0.495-0.522 | +0.353 | 0.179 (0.024) | +0.395 | +0.391 |
| slip/dilation tendency * | 0.529 | 0.479-0.506 | +0.008 | 0.044 (0.015) | +0.073 | +0.061 |
| best gravity layer (`grv_dmx`) | 0.567 | 0.493-0.520 | +0.046 | 0.023 (0.013) | +0.041 | +0.049 |
| six gravity-gradient layers | 0.552-0.567 | 0.485-0.520 | +0.041 to +0.046 | 0.023-0.032 | +0.032 to +0.042 | +0.045 to +0.056 |
| `grv_con` | 0.556 | 0.514-0.525 | +0.031 | 0.012 (0.017) | +0.000 | +0.026 |
| distance to gravity stations | 0.547 | 0.506-0.524 | +0.023 | 0.053 (0.046) | +0.020 | +0.023 |
| distance to gravity lineations | 0.518 | 0.499-0.515 | +0.002 | 0.043 (0.040) | +0.006 | +0.005 |
| ten pseudo-gravity (magnetic) layers | 0.466-0.531 | | -0.010 to +0.000 | | -0.009 to +0.001 | -0.013 to +0.002 |
| heat flow, its uncertainty, their gradients | 0.471-0.512 | | -0.015 to -0.001 | | -0.011 to +0.000 | -0.013 to +0.002 |
| distance to magnetic lineations | 0.528 | 0.513-0.529 | -0.001 | 0.048 (0.043) | +0.006 | +0.002 |

- **Only the gravity-gradient family carries a signal, and it is weak**
  (AUC about 0.56 against a shifted 0.49-0.52). Its top cells drawn as a map
  score GEMS 0.03, against 0.80 for a re-issue of the fault database.
- **That signal is local.** It keeps nearly all its edge when each
  near-fault cell is compared only with cells of its own 10 km block, so it
  is not just faulted districts having steeper gravity overall. One gravity
  layer, `grv_con`, does lose its edge within blocks: its pooled skill was
  regional. The slip and dilation tendency control goes the other way
  (+0.008 pooled, +0.073 within blocks), so the blocks reveal as well as
  remove.
- **The lineations lose the signal.** The gradient grid scores 0.557; the
  distance to the lineations traced along its crests scores 0.518, no better
  than chance, pooled or within blocks.
- **No magnetic, heat-flow or point layer beats chance** against young
  mapped faults, pooled or within blocks.
- **Caution on survey density:** distance to the gravity stations scores
  above chance, within blocks too, so surveyed ground lies a little closer
  to mapped faults. A model may learn where people surveyed rather than the
  geology.
- In the Basin and Range, young faults often bound the ranges, so a steeper
  gravity gradient there may reflect the range-basin contrast as much as the
  fault. A range front and its fault fall inside the same 10 km block, so
  the within-block run does not settle this.

For entrants: gravity-gradient grids are worth trying as extra features
(`gems_regrid.py grid ... --assume-utm11`), with a small gain expected at
best; since the signal holds within blocks, it should survive spatial block
cross-validation. The other layers showed no skill against young faults;
their value for buried faults, which GEMS is after, is untested.

## What the organisers have said on the forum

Source: the GEMS category of community.drivendata.org, as of 14 October 2026.

- **Known faults are masked pixel-exactly** (topic 11516). The USGS/INGENIOUS
  training faults are left out of scoring, so predictions on them cost
  nothing. Pixels next to them are scored only against the new faults, so a
  model trained on known faults pays false positives along their edges.
  `gems_baseline.py train --suppress-near-known N` clears that ring, and
  `score --exclude` applies the same mask locally. New faults can also lie
  within 300 m of known ones (as corrections), so N is tuned on the public
  leaderboard rather than assumed.
- **Scores are pooled** over all public-test pixels into a single index, not
  averaged per chunk (topic 11550). The chunk layout is not disclosed.
- **The Final Round ($250k) re-scores the whole GeoDAWN area** against labels
  that experts may add to, move or remove after reviewing every Phase 1
  submission (topics 11527, 11556). Plausible but unmapped faults can
  therefore earn credit later, which favours recall even more. No fault
  weighting is applied beyond the metric's distance weighting, and the
  excluded USGS/INGENIOUS set stays the same in the Final Round.
- **Three scored submissions per rolling week** (topic 11524), so the public
  leaderboard is a scarce calibration tool.
- **A "new fault" is any fault pixel not already in USGS/INGENIOUS**
  (topic 11536). That includes extensions past a mapped tip, splays and
  parallel strands, so the zone just beyond known fault ends is worth marking.
  `gems_extend.py` does exactly that. A fault pixel is a tip when its 8
  neighbours hold exactly one run of fault pixels (mid-line pixels have two,
  junctions three), its 7 x 7 window holds at least 4 fault pixels, and the
  centroid of those pixels sits more than 1.2 px away on one side. The
  extension then walks the other way, one pixel per step. If the real fault
  continues, each extension pixel is a hit; if it stops where mapped, it is
  a false alarm worth a fifth of a miss. Which length pays (0, 2, 3 or 5 px)
  is a question for the public leaderboard.
- **Outside data is allowed** when you hold a licence to use it in the
  challenge and it can be shared with the sponsor for evaluation (topic
  11528, quoting the problem description).
- The organisers will not describe how the new faults were mapped (topic 11527).
- Open questions without an organiser answer yet: whether NBMG geologic maps
  with mixed licence notices may be used as external data (topic 11563), and
  how the rule that most of the work be done in the US is judged (topic 11565).

## Lidar scarp features

Young faults in the Basin and Range leave scarps, straight steep steps a few
metres high, which a 100 m grid cannot show directly. `gems_lidar.py` reads
each 1 m tile at a chosen resolution (`--factor`: DEM pixels per 100 m cell
side; the default 20 means 5 m, 100 means the full 1 m) and writes five
bands on the competition grid. Notes:

- Only cells that lie fully inside a tile are computed from it; the first
  tile to fill a cell wins, since 3DEP tiles overlap by a few metres.
- Tiles must be in UTM zone 11N metres. The NAD83 to WGS84 shift is about a
  metre, far below the 100 m cell, so no reprojection is done.
- Over the whole GeoDAWN area this is hundreds of tiles. Use `--limit` to
  try a few first; progress is saved in `--state` and resumed on the next run.
- These are generic terrain statistics, so roads, river banks and range
  fronts light up too. The classifier learns which combinations go with
  known faults.

## Public fault maps (Qfaults)

The USGS Quaternary Fault and Fold Database is public domain, so it meets
the licence condition for outside data. `gems_external.py fetch` converts the
grid's corners to longitude and latitude, asks the USGS ArcGIS service
(`haz/Qfaults/MapServer`, layer 21, the national database) for every line in
that box, and follows its 2,000-feature pages. `rasterize` projects the lines
to UTM zone 11N with Snyder's transverse Mercator series (checked against
his worked example to 6 cm and by round trips across the zone), marks every
100 m cell a line passes through, and prints an agreement report:

- `known_within_1px_of_external`: the share of known fault pixels that sit on
  or next to a Qfaults line. If this is high, the training labels were
  probably drawn from Qfaults, and `external_distance` must **not** be used as
  a training feature: the classifier would learn "fault = Qfaults line" and
  give the unmapped faults low scores.
- `external_within_buffer_of_known` and `candidate_pixels`: how much of
  Qfaults lies away from the known faults. Those candidate cells are mapped
  Quaternary faults that are not in the masked set, so if the new faults
  include them they are hits; if not, each one is a false alarm at a fifth
  of a miss. `merge` ORs them into a submission so one leaderboard slot can
  answer the question. Many Qfaults traces were compiled from maps at
  1:100,000 to 1:250,000 (see the `mapped_scale` field), where the drawn
  line alone is about 100 m wide, so their positions are uncertain by about
  a pixel: by the table above, a candidate band 3 px wide may be the safer
  choice for them.

Any other GeoJSON of fault lines (for example faults from the USGS State
Geologic Map Compilation, also public domain, exported from a GIS) can be
passed with `--lines`, repeatedly.

## USGS Nevada geothermal data release (lineations, gradients, heat flow)

DeAngelo, J., Glen, J.M., Siler, D.L., Coolbaugh, M.F., Earney, T.E.,
Dean, B.J., Zielinski, L.A. and Ritzinger, B.T., 2021, *USGS Contributions
to the Nevada Geothermal Machine Learning Project (DE-FOA-0001956):
Geophysics, Heat Flow, Slip and Dilation Tendency*, U.S. Geological Survey
data release, https://doi.org/10.5066/P9V5SQRD. Its metadata lists no
access or use constraints, and the copy on the Geothermal Data Repository
(submission 1349) is marked CC BY 4.0, so it meets the outside-data
condition. It covers about 119.5 W to 114.4 W and 38.4 N to 40.7 N, so it
should overlap the eastern part of the GeoDAWN area (the reports below show
how many cells it reaches). Its metadata gives NAD83 transverse Mercator
with central meridian 117 W, scale 0.9996 and false easting 500 km, which
is UTM zone 11N, the competition grid's own system.

The 256 MB zip holds shapefiles that `gems_shapes.py` reads directly:

- `gravityLineations_nvML`, `magneticLineations_nvML`: lines built from the
  "max spots" of the gravity and pseudo-gravity fields (the release's
  documentation folder gives the method; its horizontal-gradient grids are
  included). Gradient crests mark where rocks of different density or
  magnetisation meet, often across a fault, and many such faults are buried
  under basin fill. The agreement run above shows they rarely fall within
  300 m of mapped Quaternary faults, so use them as features rather than as
  drawn faults.
- `gravityEndpoints_nvML`, `gravityIntersections_nvML`,
  `magneticEndpoints_nvML`, `magneticIntersections_nvML`,
  `gravityMaxSpots_all_nvML`, `magneticMaxSpots_all_nvML`, `heatFlowWells_nvML`:
  points. `gems_regrid.py points` turns each into two bands: distance to the
  nearest point in px (capped at `--cap`, default 50 px = 5 km) and the
  number of points in the square of half-width `--radius` (default 20 px)
  around each cell. Points just outside the grid still count.
- `quaternaryFaults_nvML` and `tdts2020nvML_faultLines`: mapped faults, the
  latter with slip and dilation tendency for each segment (the two layers
  cover exactly the same cells).

and ESRI ASCII grids that `gems_regrid.py grid` reads straight from the zip
(`ZIP::grv_filthg`), with the `.prj` beside each one checked for UTM 11N:

- `grv`, `psg` (gravity and pseudo-gravity), their `_filt`, `_con` and
  `_dmx` variants, and `grv_filthg`, `psg_filthg`, the horizontal gradients
  whose crests the lineations were traced from;
- `hfnvml_est`, `hfnvml_unc`: interpolated heat flow (mW/m^2) and its
  uncertainty;
- `tdts_nvml`: slip and dilation tendency.

All are 1728 x 1000 cells of 250 m (as listed by `gems_regrid.py list` on
the real zip). Their `.prj` files are not recognised as UTM 11N by
`gems_shapes.prj_kind`, so pass `--assume-utm11`, which the release
metadata supports. Sampling is bilinear between source cell centres; nodata
neighbours are left out and the rest re-weighted, so values reach half a
cell into a gap. Cells beyond a source are NaN, and the baseline then treats
that band (and its filters within 6 px) as missing there rather than as a
step.

Use the lineations like Qfaults: `rasterize` with `--labels` reports how far
they agree with the known faults, and its candidate band marks the
lineation cells away from them. Lineations are about 225 m wide buffers
reduced to centre lines, so their positions are uncertain by a pixel or
two. The grids and point bands go into training with `--extra-features`.
**Do not feed in points or grids derived from the known faults** (for
example the vertices of `quaternaryFaults_nvML` or
`tdts2020nvML_faultVertices` if they coincide with the labels): that is the
same leakage as a distance-to-label feature. Check with
`gems_external.py rasterize --labels` first. Whether any of this pays is a
question for the public leaderboard.

## Ideas

- The full-resolution GeoDAWN magnetic and radiometric grids are further
  public candidates for outside data.
- The NBMG Nevada geothermal machine-learning project (Geothermal Data
  Repository submissions 1350-1353, CC BY 4.0) publishes structural-setting
  centroids as points and a distance-to-centroid GeoTIFF in UTM 11N, which
  `gems_regrid.py` reads as they are.
- Splitting the young faults into range-bounding and intrabasin ones (for
  example by the slope on either side in a public DEM) would test whether
  the gravity-gradient signal follows the faults or the range fronts.
- **Distance to the known faults is not a training feature here on purpose.**
  The training labels are the known faults, so a distance-to-label feature
  would be leakage: it predicts the label perfectly in training and says
  nothing about the unmapped faults. Tip extension uses the known faults
  only at prediction time, which is safe.

## Disclosure

MV2 of Munim, Inc. plans to enter the GEMS Prize as Munim, Inc., a US
company, under the official rules. Any entry from Munim, Inc. will say that
the work was made with AI agents. This is not an official DrivenData, DOE or
NLR tool. The results in `docs/` are CC BY 4.0.
