# Table 3: reuploading

Same strict-LOSO protocol as `../PROTOCOL.md`.

## CHSZ

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 27/27 | 90.88 +/- 7.36 | 80.54 +/- 17.46 | 75.42 +/- 17.68 |
| `no_reupload` | Encode only in layer 1. | 27/27 | 90.61 +/- 10.28 | 81.23 +/- 18.48 | 75.79 +/- 19.12 |
| `frozen_reupload_scale` | Freeze encoding scales at one. | 27/27 | 90.92 +/- 7.81 | 81.05 +/- 17.54 | 75.69 +/- 18.76 |

## Sleep-EDF-20

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 20/20 | 91.08 +/- 3.21 | 76.58 +/- 5.04 | 73.06 +/- 5.43 |
| `no_reupload` | Encode only in layer 1. | 20/20 | 90.28 +/- 3.99 | 75.34 +/- 5.95 | 71.81 +/- 6.32 |
| `frozen_reupload_scale` | Freeze encoding scales at one. | 20/20 | 91.02 +/- 3.30 | 76.54 +/- 5.61 | 73.19 +/- 5.67 |

