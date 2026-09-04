# Table 8: hjorth_features

Same strict-LOSO protocol as `../PROTOCOL.md`.

## CHSZ

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 27/27 | 90.88 +/- 7.36 | 80.54 +/- 17.46 | 75.42 +/- 17.68 |
| `no_hjorth` | Remove both Hjorth features. | 27/27 | 90.74 +/- 8.87 | 79.15 +/- 17.28 | 75.39 +/- 18.09 |
| `no_hjorth_mobility` | Remove Hjorth mobility. | 27/27 | 89.27 +/- 9.21 | 79.18 +/- 17.12 | 74.18 +/- 18.04 |
| `no_hjorth_complexity` | Remove Hjorth complexity. | 27/27 | 90.44 +/- 9.62 | 79.97 +/- 17.75 | 75.30 +/- 18.93 |

## Sleep-EDF-20

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 20/20 | 91.08 +/- 3.21 | 76.58 +/- 5.04 | 73.06 +/- 5.43 |
| `no_hjorth` | Remove both Hjorth features. | 20/20 | 90.55 +/- 4.18 | 75.62 +/- 5.76 | 72.35 +/- 6.04 |
| `no_hjorth_mobility` | Remove Hjorth mobility. | 20/20 | 91.06 +/- 3.49 | 76.59 +/- 5.84 | 73.32 +/- 5.93 |
| `no_hjorth_complexity` | Remove Hjorth complexity. | 20/20 | 90.78 +/- 4.22 | 75.68 +/- 5.76 | 72.69 +/- 6.04 |

