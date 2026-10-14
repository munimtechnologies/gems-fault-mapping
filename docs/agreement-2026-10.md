# How far public fault maps agree, scored with the GEMS metric (October 2026)

By MV2 of Munim, Inc., made with AI agents. Text and tables: CC BY 4.0.
Code: `gems_agreement.py` in this repository (Apache-2.0).

Run of 14 October 2026 (`gems_agreement.py run --download --qfaults`, 4 CPU
cores, 299 s). Data: the USGS Nevada geothermal machine-learning data
release, https://doi.org/10.5066/P9V5SQRD, and the national USGS Quaternary
Fault and Fold Database fetched live from
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

## Limits

Maps are compared with each other, not with the hidden GEMS labels; the box
overlaps only part of the GeoDAWN area; lines are one cell wide; the
shifted-map null keeps each map's density and shape but wraps round the
grid edge; maps from one release may share sources; and Quaternary faults
are only the young, mapped part of the fault network, so the lineation
result says nothing about older or buried faults. The chance baseline was
computed for the GEMS score only; the distance profile added in
`gems_agreement.py` 0.6.0 compares the 300 m shares (and shares out to
5 km) with chance too, and will be added here after its run.
