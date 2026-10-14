# gems-fault-mapping

Open tools for the **DOE GEMS Prize** (Geologic Enhanced Mapping System), the
$300,000 American-Made challenge on DrivenData to map hidden geologic faults
that point to geothermal resources in western Nevada (the GeoDAWN area).
Submissions close **3 December 2026**.
Competition: https://www.drivendata.org/competitions/306/competition-doe-gems/

By MV2 of Munim, Inc. The code was written with AI agents working in MV2's
name. MIT licence. It is shared publicly so every competitor can use it.

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
| `gems_regrid.py` | New in 0.5.0. Puts outside rasters and point layers on the competition grid as extra feature bands. `grid` samples ESRI ASCII grids (`.asc`, also straight from inside a zip as `ZIP::NAME`) or GeoTIFFs in UTM 11N onto the 100 m cells, bilinear or nearest, NaN beyond the source. `points` turns point layers into a distance band and a count band each. `list` shows the ASCII grids in a zip. |
| `test_gems.py`, `test_external.py`, `test_robust.py`, `test_shapes.py`, `test_regrid.py` | 108 offline tests: the metric on hand-worked grids, numpy against the reference on random grids, tip detection on lines, junctions and thick lines, lidar features on planes, scarps and bowls, a tile landing on the right cells, the projection against Snyder's published worked example, line rasterising, paged downloads with a fake server, the submission checks, the band table checked by hand and against the metric on a grid, shapefiles and dBASE tables built byte by byte from their published layouts, ASCII grids with wrapped rows, resampling a plane, gaps and edges, point bands across the grid edge, and the baseline's handling of a band with a gap. |

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
```

**Status:** the metric, tip detection, lidar features, projection, band
table, resampling and checks are unit-tested on synthetic grids. The
pipeline has **not yet been run on the competition data**, which is
available only to registered entrants. Results will be added here once it
has.

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
  under basin fill. These are the most direct candidates for GEMS' "hidden"
  faults.
- `gravityEndpoints_nvML`, `gravityIntersections_nvML`,
  `magneticEndpoints_nvML`, `magneticIntersections_nvML`,
  `gravityMaxSpots_all_nvML`, `magneticMaxSpots_all_nvML`, `heatFlowWells_nvML`:
  points. `gems_regrid.py points` turns each into two bands: distance to the
  nearest point in px (capped at `--cap`, default 50 px = 5 km) and the
  number of points in the square of half-width `--radius` (default 20 px)
  around each cell. Points just outside the grid still count.
- `quaternaryFaults_nvML` and `tdts2020nvML_faultLines`: mapped faults, the
  latter with slip and dilation tendency for each segment.

and ESRI ASCII grids that `gems_regrid.py grid` reads straight from the zip
(`ZIP::grv_filthg`), with the `.prj` beside each one checked for UTM 11N:

- `grv`, `psg` (gravity and pseudo-gravity), their `_filt`, `_con` and
  `_dmx` variants, and `grv_filthg`, `psg_filthg`, the horizontal gradients
  whose crests the lineations were traced from;
- `hfnvml_est`, `hfnvml_unc`: interpolated heat flow (mW/m^2) and its
  uncertainty;
- `tdts_nvml`: slip and dilation tendency.

`python gems_regrid.py list --src <zip>` prints each grid's size and cell
size. Sampling is bilinear between source cell centres; nodata neighbours
are left out and the rest re-weighted, so values reach half a cell into a
gap. Cells beyond a source are NaN, and the baseline then treats that band
(and its filters within 6 px) as missing there rather than as a step.

Use the lineations like Qfaults: `rasterize` with `--labels` reports how far
they agree with the known faults, and its candidate band marks the
lineation cells away from them. Lineations are about 225 m wide buffers
reduced to centre lines, so their positions are uncertain by a pixel or
two; by the width table above, a 3 px band is the safer way to submit them.
The grids and point bands go into training with `--extra-features`. **Do not
feed in points or grids derived from the known faults** (for example the
vertices of `quaternaryFaults_nvML` or `tdts2020nvML_faultVertices` if they
coincide with the labels): that is the same leakage as a distance-to-label
feature. Check with `gems_external.py rasterize --labels` first. Whether any
of this pays is a question for the public leaderboard.

## Ideas

- The full-resolution GeoDAWN magnetic and radiometric grids are further
  public candidates for outside data.
- The NBMG Nevada geothermal machine-learning project (Geothermal Data
  Repository submissions 1350-1353, CC BY 4.0) publishes structural-setting
  centroids as points and a distance-to-centroid GeoTIFF in UTM 11N, which
  `gems_regrid.py` reads as they are.
- **Distance to the known faults is not a training feature here on purpose.**
  The training labels are the known faults, so a distance-to-label feature
  would be leakage: it predicts the label perfectly in training and says
  nothing about the unmapped faults. Tip extension uses the known faults
  only at prediction time, which is safe.

## Disclosure

MV2 of Munim, Inc. plans to enter the GEMS Prize if Munim, Inc. qualifies
under the official rules. Any entry from Munim, Inc. will say that the work
was made with AI agents. This is not an official DrivenData, DOE or NLR tool.
