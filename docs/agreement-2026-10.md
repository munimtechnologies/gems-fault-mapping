# How far public fault maps agree, scored with the GEMS metric (October 2026)

By MV2 of Munim, Inc., made with AI agents. Text and tables: CC BY 4.0.
Code: `gems_agreement.py` in this repository (Apache-2.0).

Runs of 14 October 2026 (`gems_agreement.py run --download --qfaults`, 4 CPU
cores; first run 299 s, second run with the distance profile 401 s; the
scores are identical). Data: the USGS Nevada geothermal machine-learning
data release, https://doi.org/10.5066/P9V5SQRD, and the national USGS
Quaternary Fault and Fold Database fetched live from
`earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer/21` (8,460
features over the box). Grid: UTM zone 11N, 100 m cells, 2563 x 4464 over
the release's bounding box (119.53 W to 114.42 W, 38.45 N to 40.73 N);
10,799,672 cells inside the footprint of the release's `grv_filthg` grid are
scored. Raw numbers: `agreement-2026-10.json`.

Each cell below is the GEMS score (distance-weighted Tversky index, alpha
0.2, beta 0.8, 300 m) of the row map as the prediction against the column
map as the truth; in brackets, the mean score of the row map shifted 14-21
km in five directions (chance).

| prediction \ truth | quaternary_faults | tdts_fault_lines | gravity_lineations | magnetic_lineations | potential_field | qfaults_national |
|---|---|---|---|---|---|---|
| quaternary_faults | - | 1.000 (0.045) | 0.048 (0.046) | 0.049 (0.045) | 0.051 (0.048) | 0.796 (0.045) |
| tdts_fault_lines | 1.000 (0.045) | - | 0.048 (0.046) | 0.049 (0.045) | 0.051 (0.048) | 0.796 (0.045) |
| gravity_lineations | 0.081 (0.077) | 0.081 (0.077) | - | 0.100 (0.092) | - | 0.081 (0.077) |
| magnetic_lineations | 0.068 (0.062) | 0.068 (0.062) | 0.079 (0.073) | - | - | 0.068 (0.062) |
| potential_field | 0.105 (0.099) | 0.105 (0.099) | - | - | - | 0.105 (0.099) |
| qfaults_national | 0.799 (0.045) | 0.799 (0.045) | 0.048 (0.046) | 0.050 (0.046) | 0.052 (0.048) | - |

| map | source | cells scored |
|---|---|---|
| quaternary_faults | quaternaryFaults_nvML (UTM 11N (.prj)) | 150394 |
| tdts_fault_lines | tdts2020nvML_faultLines (UTM 11N (.prj)) | 150394 |
| gravity_lineations | gravityLineations_nvML (UTM 11N (.prj)) | 401684 |
| magnetic_lineations | magneticLineations_nvML (UTM 11N (.prj)) | 264549 |
| potential_field | union of gravity_lineations + magnetic_lineations | 655942 |
| qfaults_national | USGS Qfaults MapServer layer 21, 8460 features | 151157 |

| prediction | truth | truth within 300 m | prediction within 300 m | lift over chance |
|---|---|---|---|---|
| quaternary_faults | tdts_fault_lines | 1.0 | 1.0 | 22.02 |
| quaternary_faults | gravity_lineations | 0.0861 | 0.2177 | 1.05 |
| quaternary_faults | magnetic_lineations | 0.0932 | 0.1615 | 1.09 |
| quaternary_faults | potential_field | 0.0889 | 0.3426 | 1.07 |
| quaternary_faults | qfaults_national | 0.9812 | 0.9858 | 17.61 |
| tdts_fault_lines | quaternary_faults | 1.0 | 1.0 | 22.02 |
| tdts_fault_lines | gravity_lineations | 0.0861 | 0.2177 | 1.05 |
| tdts_fault_lines | magnetic_lineations | 0.0932 | 0.1615 | 1.09 |
| tdts_fault_lines | potential_field | 0.0889 | 0.3426 | 1.07 |
| tdts_fault_lines | qfaults_national | 0.9812 | 0.9858 | 17.61 |
| gravity_lineations | quaternary_faults | 0.2177 | 0.0861 | 1.05 |
| gravity_lineations | tdts_fault_lines | 0.2177 | 0.0861 | 1.05 |
| gravity_lineations | magnetic_lineations | 0.224 | 0.1527 | 1.08 |
| gravity_lineations | qfaults_national | 0.2181 | 0.0866 | 1.05 |
| magnetic_lineations | quaternary_faults | 0.1615 | 0.0932 | 1.09 |
| magnetic_lineations | tdts_fault_lines | 0.1615 | 0.0932 | 1.09 |
| magnetic_lineations | gravity_lineations | 0.1527 | 0.224 | 1.08 |
| magnetic_lineations | qfaults_national | 0.1623 | 0.0943 | 1.1 |
| potential_field | quaternary_faults | 0.3426 | 0.0889 | 1.06 |
| potential_field | tdts_fault_lines | 0.3426 | 0.0889 | 1.06 |
| potential_field | qfaults_national | 0.3438 | 0.0896 | 1.07 |
| qfaults_national | quaternary_faults | 0.9858 | 0.9812 | 17.72 |
| qfaults_national | tdts_fault_lines | 0.9858 | 0.9812 | 17.72 |
| qfaults_national | gravity_lineations | 0.0866 | 0.2181 | 1.05 |
| qfaults_national | magnetic_lineations | 0.0943 | 0.1623 | 1.1 |
| qfaults_national | potential_field | 0.0896 | 0.3438 | 1.07 |

## Distance profile

Share of the truth map's cells within each distance of the prediction map;
in brackets, that share divided by the mean share for the prediction map
shifted 14-21 km (1.00 = chance). Rows for the duplicate `tdts_fault_lines`
map and the `qfaults_national` rows that repeat the `quaternary_faults` ones
to within 0.01 are left out here; all rows are in the JSON.

| prediction | truth | 100 m | 300 m | 500 m | 1 km | 2 km | 5 km |
|---|---|---|---|---|---|---|---|
| quaternary_faults | qfaults_national | 0.969 (24.3) | 0.981 (10.68) | 0.985 (6.79) | 0.989 (3.77) | 0.993 (2.14) | 0.998 (1.22) |
| qfaults_national | quaternary_faults | 0.975 (24.51) | 0.986 (10.73) | 0.988 (6.8) | 0.989 (3.75) | 0.992 (2.13) | 0.997 (1.22) |
| gravity_lineations | quaternary_faults | 0.098 (1.08) | 0.218 (1.06) | 0.346 (1.06) | 0.631 (1.06) | 0.924 (1.04) | 1.000 (1.01) |
| magnetic_lineations | quaternary_faults | 0.069 (1.1) | 0.162 (1.11) | 0.258 (1.09) | 0.484 (1.07) | 0.796 (1.04) | 0.989 (1.01) |
| potential_field | quaternary_faults | 0.160 (1.08) | 0.343 (1.07) | 0.510 (1.06) | 0.799 (1.04) | 0.981 (1.03) | 1.000 (1.01) |
| quaternary_faults | gravity_lineations | 0.038 (1.06) | 0.086 (1.04) | 0.138 (1.05) | 0.255 (1.06) | 0.453 (1.05) | 0.820 (1.03) |
| quaternary_faults | magnetic_lineations | 0.040 (1.09) | 0.093 (1.1) | 0.147 (1.09) | 0.262 (1.06) | 0.451 (1.03) | 0.816 (1.02) |
| gravity_lineations | magnetic_lineations | 0.099 (1.09) | 0.224 (1.1) | 0.351 (1.08) | 0.626 (1.06) | 0.914 (1.03) | 1.000 (1.01) |
| magnetic_lineations | gravity_lineations | 0.066 (1.09) | 0.153 (1.09) | 0.245 (1.08) | 0.462 (1.06) | 0.778 (1.04) | 0.985 (1.01) |

## Findings

1. `quaternaryFaults_nvML` and `tdts2020nvML_faultLines` cover exactly the
   same 150,394 cells: one map, not two.
2. The release's Quaternary faults and today's national Qfaults agree at
   GEMS 0.80 (98% of cells within 300 m each way, lift 17.6 over chance).
   That is the score of a re-issue of the same map, a useful ceiling when
   reading leaderboard numbers.
3. Gravity and magnetic lineations (traced along horizontal-gradient
   crests) score 0.07 to 0.11 against the Quaternary faults, only 1.05 to
   1.09 times their score when shifted 14 to 21 km. At the metric's 300 m
   scale they do not coincide with mapped young faults beyond chance, so
   drawing them as predicted faults would score near chance; use them as
   features. Gravity and magnetic lineations also barely coincide with each
   other (lift 1.08).
4. **No scale rescues them.** The distance profile shows the same 1.01 to
   1.11 times chance at every distance from 100 m to 5 km. If the gradient
   crests sat a fixed distance basinward of the surface traces (as a
   dipping range-front fault would suggest), the share would rise above
   chance at that distance; it does not at any of the six. The two
   editions of the fault database, by contrast, are 24 times chance at
   100 m and still 2 times chance at 2 km.

## Limits

Maps are compared with each other, not with the hidden GEMS labels; the box
overlaps only part of the GeoDAWN area; lines are one cell wide; the
shifted-map null keeps each map's density and shape but wraps round the
grid edge, and five shifts give a range rather than a p-value; maps from
one release may share sources; and Quaternary faults are only the young,
mapped part of the fault network, so the lineation result says nothing
about older or buried faults. A crest offset that varies from fault to
fault (some 500 m, some 3 km) would be smeared across the profile and could
hide in it. For the continuous grids behind the lineations, see
[`layer-skill-2026-10.md`](layer-skill-2026-10.md).
