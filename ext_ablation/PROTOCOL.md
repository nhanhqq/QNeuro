# Extended strict HybridNode11 ablation protocol

`ext_ablation` is isolated: it never modifies or reads checkpoints, runtime
status, reports, or outputs in `ablation/`. Its matched `full` reference is
trained anew for each held-out fold.

Scope is `CHSZ` and `Sleep-EDF-20`, with seed 7 per `(dataset, variant, fold)`.
The outer split is strict LOSO. N-1 source samples are deterministically split
stratified 80/20 into source train/validation. Scaling and class weights fit
source training only; maximum source-validation macro-F1 selects the checkpoint;
the held-out subject is evaluated once afterward. There is no target selection.

The 26 unique configurations make ten tables: quantum latent, entanglement,
re-uploading, channel pooling, temporal encoder, classifier, spectral, Hjorth,
connectivity features, and RZ augmentation. Every alternative replaces exactly one named mechanism
from the completed ten-ablation baseline. `full` is the independently rerun
matched reference in each table.

The `hybridnode11_v2` cache is read-only; no DataLoader is used (`num_workers=0`).
A result is complete only with exit code 0, a source-selected checkpoint, and
`strict_test_once.json`. Partial folds are excluded. The scheduler manages only
its own `extab_` screens and self-backfills up to 12 jobs; admission keeps cgroup RAM
below 30,000 MiB and RTX-3090 allocated VRAM below 22,000 MiB.
