# Table 10: rz_augmentation

Same strict-LOSO protocol as `../PROTOCOL.md`.

## CHSZ

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 27/27 | 90.88 +/- 7.36 | 80.54 +/- 17.46 | 75.42 +/- 17.68 |
| `no_rz_augmentation` | RZ noise = 0. | 27/27 | 90.90 +/- 7.42 | 82.10 +/- 16.30 | 76.60 +/- 16.77 |
| `high_rz_augmentation` | RZ noise = 0.20 rad. | 27/27 | 91.31 +/- 7.11 | 81.79 +/- 16.97 | 76.43 +/- 17.04 |

## Sleep-EDF-20

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 20/20 | 91.08 +/- 3.21 | 76.58 +/- 5.04 | 73.06 +/- 5.43 |
| `no_rz_augmentation` | RZ noise = 0. | 20/20 | 90.79 +/- 3.81 | 76.92 +/- 5.16 | 73.19 +/- 6.04 |
| `high_rz_augmentation` | RZ noise = 0.20 rad. | 20/20 | 91.08 +/- 3.63 | 76.44 +/- 5.64 | 73.34 +/- 5.65 |

