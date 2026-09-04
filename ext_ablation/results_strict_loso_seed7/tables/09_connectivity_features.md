# Table 9: connectivity_features

Same strict-LOSO protocol as `../PROTOCOL.md`.

## CHSZ

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 27/27 | 90.88 +/- 7.36 | 80.54 +/- 17.46 | 75.42 +/- 17.68 |
| `no_connectivity` | Remove both correlation sketches. | 27/27 | 92.36 +/- 7.25 | 81.67 +/- 17.95 | 77.37 +/- 18.50 |
| `no_mean_connectivity` | Remove mean correlation. | 27/27 | 90.05 +/- 9.51 | 76.95 +/- 17.50 | 73.18 +/- 17.92 |
| `no_derivative_connectivity` | Remove derivative correlation. | 27/27 | 91.62 +/- 6.55 | 80.69 +/- 17.33 | 76.19 +/- 17.10 |

## Sleep-EDF-20

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 20/20 | 91.08 +/- 3.21 | 76.58 +/- 5.04 | 73.06 +/- 5.43 |
| `no_connectivity` | Remove both correlation sketches. | 20/20 | 89.82 +/- 4.93 | 74.89 +/- 6.53 | 71.68 +/- 6.42 |
| `no_mean_connectivity` | Remove mean correlation. | 20/20 | 90.81 +/- 3.91 | 75.97 +/- 5.76 | 72.79 +/- 5.95 |
| `no_derivative_connectivity` | Remove derivative correlation. | 20/20 | 90.39 +/- 3.92 | 74.35 +/- 6.42 | 71.51 +/- 6.14 |

