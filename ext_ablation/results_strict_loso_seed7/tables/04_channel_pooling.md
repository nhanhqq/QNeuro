# Table 4: channel_pooling

Same strict-LOSO protocol as `../PROTOCOL.md`.

## CHSZ

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 27/27 | 90.88 +/- 7.36 | 80.54 +/- 17.46 | 75.42 +/- 17.68 |
| `uniform_channel_pool` | Uniform channel mean. | 27/27 | 90.22 +/- 8.01 | 80.06 +/- 17.53 | 74.84 +/- 17.22 |
| `max_channel_pool` | Maximum channel pool. | 27/27 | 89.25 +/- 13.49 | 79.97 +/- 16.65 | 74.59 +/- 18.73 |

## Sleep-EDF-20

| Variant | Intervention | Folds | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| `full` | Independent matched reference. | 20/20 | 91.08 +/- 3.21 | 76.58 +/- 5.04 | 73.06 +/- 5.43 |
| `uniform_channel_pool` | Uniform channel mean. | 20/20 | 91.27 +/- 3.20 | 76.43 +/- 4.91 | 73.45 +/- 4.63 |
| `max_channel_pool` | Maximum channel pool. | 20/20 | 90.73 +/- 3.48 | 75.21 +/- 5.79 | 72.53 +/- 5.11 |

