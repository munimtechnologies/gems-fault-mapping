# Which public layers single out cells near mapped Quaternary faults (14 October 2026)

By MV2 of Munim, Inc., made with AI agents. Text and numbers CC BY 4.0;
code Apache-2.0. Raw numbers: [`layer-skill-2026-10.json`](layer-skill-2026-10.json).

## Question

The USGS Nevada geothermal machine-learning data release (DeAngelo and
others, 2021, https://doi.org/10.5066/P9V5SQRD) publishes gravity and
pseudo-gravity (magnetic) grids, their filtered and gradient versions,
interpolated heat flow, lineations traced along the gradient crests, and
point layers. Entrants in the DOE GEMS Prize can feed any of them to a model
as outside data. Which of them actually point at faults at the scale the
GEMS metric scores (300 m)?

We cannot test against the hidden GEMS labels, so the target is the
release's own map of Quaternary faults: every 100 m cell within 300 m of a
mapped fault (903,921 of 10,799,672 cells in the gravity grid's footprint).

## Method

`gems_gridskill.py run` (code as released in gems-fault-mapping 0.7.0; the
run's report says 0.6.0 because it loaded the 0.6.0 helper modules, which
this code does not depend on for anything that changed):

- Every 250 m grid in the release, sampled bilinearly onto the 100 m UTM 11N
  study grid; the horizontal gradient magnitude of each grid; and minus the
  distance (capped at 10 km) to each point and line layer.
- **AUC**: the chance that a random near-fault cell outranks a random other
  cell (ties count half). 0.5 means no information.
- **Chance**: the same feature scored against the near-fault map shifted 14
  to 21 km in five directions. This keeps the clustering of both maps and
  breaks only their alignment. **Edge** = |AUC - 0.5| minus the largest
  shifted |AUC - 0.5|; above zero, the feature beats all five shifted
  placements.
- **GEMS**: the feature's top cells (as many as there are fault cells) drawn
  as a fault map and scored with the GEMS metric, with the shifted-map
  chance in brackets.
- Layers computed from the fault traces themselves are marked `*`. They are
  a positive control: they must win, and they do.

Run: 4 CPU cores, 476 s, on the full release zip (256,302,969 bytes).

## Results

| feature | AUC | chance AUC | edge | GEMS (chance) |
|---|---|---|---|---|
| dist_tdts2020nvML_faultLines * | 1.000 | 0.509-0.559 | +0.441 | 1.000 (0.045) |
| dist_tdts2020nvML_faultVertices * | 0.999 | 0.509-0.559 | +0.439 | 0.839 (0.048) |
| tdts_nvml_grad * | 0.875 | 0.495-0.522 | +0.353 | 0.179 (0.024) |
| grv_dmx | 0.567 | 0.493-0.520 | +0.046 | 0.023 (0.013) |
| grv_grad | 0.557 | 0.486-0.506 | +0.043 | 0.028 (0.017) |
| grv_filt_grad | 0.557 | 0.485-0.505 | +0.043 | 0.025 (0.015) |
| grv_filthg | 0.557 | 0.485-0.505 | +0.043 | 0.025 (0.015) |
| grv_filthg_grad | 0.559 | 0.498-0.517 | +0.042 | 0.032 (0.019) |
| grv_dmx_grad | 0.552 | 0.489-0.508 | +0.041 | 0.024 (0.015) |
| grv_con | 0.556 | 0.514-0.525 | +0.031 | 0.012 (0.017) |
| dist_gravityStations_nvML | 0.547 | 0.506-0.524 | +0.023 | 0.053 (0.046) |
| grv_con_grad | 0.535 | 0.512-0.519 | +0.016 | 0.017 (0.020) |
| tdts_nvml * | 0.529 | 0.479-0.506 | +0.008 | 0.044 (0.015) |
| dist_gravityIntersections_nvML | 0.524 | 0.496-0.521 | +0.003 | 0.026 (0.025) |
| dist_gravityLineations_nvML | 0.518 | 0.499-0.515 | +0.002 | 0.043 (0.040) |
| psg_con_grad | 0.499 | 0.499-0.500 | +0.000 | 0.016 (0.016) |
| psg_dmx | 0.531 | 0.521-0.533 | -0.001 | 0.012 (0.018) |
| hfnvml_est | 0.471 | 0.470-0.474 | -0.001 | 0.015 (0.016) |
| dist_magneticLineations_nvML | 0.528 | 0.513-0.529 | -0.001 | 0.048 (0.043) |
| psg_con | 0.466 | 0.464-0.479 | -0.002 | 0.017 (0.017) |
| dist_magneticMaxSpots_all_nvML | 0.521 | 0.512-0.525 | -0.003 | 0.101 (0.098) |
| psg_filthg_grad | 0.519 | 0.519-0.523 | -0.004 | 0.014 (0.021) |
| psg_dmx_grad | 0.516 | 0.507-0.522 | -0.006 | 0.012 (0.018) |
| dist_gravityMaxSpots_all_nvML | 0.494 | 0.499-0.514 | -0.008 | 0.017 (0.016) |
| psg_filt_grad | 0.512 | 0.497-0.521 | -0.008 | 0.014 (0.018) |
| psg_grad | 0.513 | 0.498-0.522 | -0.009 | 0.014 (0.018) |
| psg_filthg | 0.512 | 0.497-0.520 | -0.009 | 0.013 (0.018) |
| hfnvml_unc | 0.512 | 0.489-0.521 | -0.009 | 0.027 (0.022) |
| grv_filt | 0.469 | 0.460-0.481 | -0.009 | 0.013 (0.014) |
| grv | 0.470 | 0.460-0.481 | -0.010 | 0.013 (0.014) |
| hfnvml_est_grad | 0.493 | 0.483-0.506 | -0.010 | 0.018 (0.015) |
| psg | 0.500 | 0.493-0.511 | -0.010 | 0.021 (0.018) |
| psg_filt | 0.500 | 0.493-0.511 | -0.010 | 0.021 (0.018) |
| dist_gravitySaddles_nvML | 0.513 | 0.498-0.526 | -0.013 | 0.029 (0.029) |
| dist_magneticEndpoints_nvML | 0.513 | 0.511-0.527 | -0.014 | 0.026 (0.026) |
| hfnvml_unc_grad | 0.479 | 0.465-0.495 | -0.015 | 0.011 (0.016) |
| dist_gravityEndpoints_nvML | 0.502 | 0.506-0.518 | -0.016 | 0.027 (0.028) |
| dist_heatFlowWells_nvML | 0.500 | 0.491-0.520 | -0.020 | 0.015 (0.018) |
| dist_magneticIntersections_nvML | 0.519 | 0.518-0.542 | -0.023 | 0.020 (0.021) |

## What it says

1. **The test works.** The layers computed from the fault traces score AUC
   0.88 to 1.00, far outside their shifted range.
2. **Only the gravity-gradient family carries a signal, and it is weak.**
   Six gravity layers (`grv_dmx`, `grv_grad`, `grv_filt_grad`,
   `grv_filthg`, `grv_filthg_grad`, `grv_dmx_grad`) reach AUC 0.552 to
   0.567 against a shifted range of 0.485 to 0.520: an edge of +0.04 over
   every shifted placement. Cells near young faults sit on steeper gravity
   gradients, slightly more often than chance would place them there.
3. **The traced lineations lose that signal.** Distance to the gravity
   lineations scores 0.518 (edge +0.002) although the continuous gradient
   it was traced from scores 0.557. The signal does not survive reducing
   the gradient to crest lines; one reading is that young faults sit on
   the steep flanks of gravity gradients rather than on their crests.
4. **No magnetic (pseudo-gravity) layer, heat-flow layer or point layer beats
   chance.** All ten `psg*` features have edges from -0.010 to +0.000;
   heat flow and its uncertainty from -0.015 to -0.001; distances to
   endpoints, intersections, saddles, max spots and heat-flow wells from
   -0.023 to +0.003.
5. **Nothing here can be drawn as a fault map.** The best independent
   layer's top cells score GEMS 0.032 (chance 0.019). Two editions of the
   USGS Quaternary fault database score 0.80 against each other
   ([`agreement-2026-10.md`](agreement-2026-10.md)).
6. **A sampling caution.** Distance to the gravity *stations* scores 0.547
   (edge +0.023): surveyed ground lies a little closer to mapped faults than
   chance. Survey density is itself information about where people looked,
   so a model fed station-density features may learn the survey, not the
   geology.

The distance profile in [`agreement-2026-10.md`](agreement-2026-10.md)
gives the same answer for the drawn lineations at every scale: the share of
fault cells within 100 m, 300 m, 500 m, 1, 2 and 5 km of a gravity or
magnetic lineation is 1.01 to 1.10 times that of the shifted lineations.
There is no offset (for example, crests a kilometre basinward of the
surface trace) at which the two line up.

## Limits

- The target is the release's map of **Quaternary** (young, surface-mapped)
  faults, not the hidden GEMS labels, and the area is the release's box,
  which overlaps only the eastern part of the GeoDAWN area.
- Five shifted placements give a range, not a p-value. Neighbouring cells
  are strongly correlated, so the effective sample is far smaller than
  10.8 million cells.
- In the Basin and Range, young faults often bound the ranges, where dense
  bedrock meets basin fill, so a steeper gravity gradient there may reflect
  the range-basin contrast as much as the fault. A follow-up run scores the
  AUC within 10 km and 25 km blocks only (`--block-km`, new in 0.8.0) to
  test whether the gravity signal follows the faults within a block or only
  the faulted part of the region. Even then, where a range front *is* the
  fault, the two cannot be told apart.
- The 250 m grids are interpolated onto 100 m cells, which smooths them.

## Practical advice for GEMS entrants

- Gravity-gradient grids are worth trying as extra features
  (`gems_regrid.py grid ... --assume-utm11`); expect a small gain at best.
- The pseudo-gravity, heat-flow, point and lineation layers showed no
  location skill against young faults here. Their value for buried faults,
  which the GEMS Prize is really after, is untested.
- Check whether survey-density features (distance to stations or wells)
  help on held-out ground before trusting them.

## Reproduce

```
pip install numpy scipy
python gems_gridskill.py run --data-dir data --out-dir gridskill --download
```
