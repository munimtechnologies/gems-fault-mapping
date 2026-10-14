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
| `gems_baseline.py` | A CPU pipeline: edge and ridge features for every band, a gradient-boosted pixel classifier, spatial block cross-validation, a threshold chosen by the metric, a GeoTIFF writer, and a submission checker. `--extra-features` adds any other raster on the same grid (such as the lidar features below). |
| `gems_extend.py` | Finds the tips of the known faults and rasterises a short extension beyond each one along the fault's direction (pure numpy). `gems_baseline.py extend` ORs these into an existing submission without retraining. |
| `gems_lidar.py` | Reads the 1 m lidar DEM tiles listed in `1m_DEM_links.csv` and summarises fault-scarp evidence for every 100 m cell: slope mean and 95th percentile, share of steep pixels, relief, and how much the slopes face one way (structure-tensor coherence). Resumable, so a long run over all tiles can be stopped and restarted. |
| `gems_external.py` | New in 0.3.0. Downloads the public-domain USGS Quaternary Fault and Fold Database (Qfaults) over the grid, or reads any GeoJSON of fault lines, projects it to UTM 11N without pyproj, and writes three bands on the competition grid: mapped line, distance to it, and *candidate* (mapped lines more than N px from any known fault). It reports how far the external map and the known faults agree, and `merge` ORs the candidates into a submission. |
| `test_gems.py`, `test_external.py` | 61 offline tests: the metric on hand-worked grids, numpy against the reference on random grids, tip detection on lines, junctions and thick lines, lidar features on planes, scarps and bowls, a tile landing on the right cells, the projection against Snyder's published worked example, line rasterising, paged downloads with a fake server, and the submission checks. |

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
```

**Status:** the metric, tip detection, lidar features, projection and checks
are unit-tested on synthetic grids. The pipeline has **not yet been run on the
competition data**, which is available only to registered entrants. Results
will be added here once it has.

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
  answer the question.

Any other GeoJSON of fault lines (for example faults from the USGS State
Geologic Map Compilation, also public domain, exported from a GIS) can be
passed with `--lines`, repeatedly.

## Ideas

- The full-resolution GeoDAWN magnetic and radiometric grids are further
  public candidates for outside data.
- **Distance to the known faults is not a training feature here on purpose.**
  The training labels are the known faults, so a distance-to-label feature
  would be leakage: it predicts the label perfectly in training and says
  nothing about the unmapped faults. Tip extension uses the known faults
  only at prediction time, which is safe.

## Disclosure

MV2 of Munim, Inc. plans to enter the GEMS Prize if Munim, Inc. qualifies
under the official rules. Any entry from Munim, Inc. will say that the work
was made with AI agents. This is not an official DrivenData, DOE or NLR tool.
