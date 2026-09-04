# HybridNode QNeuro (<1000 trainable parameters)

This variant is implemented separately from QNeurov2. Existing V2 source and
caches are not modified.

## Why it is not a copy of the paper

The paper exhaustively trains 48 feature/graph combinations, builds weighted
`C x C` connectivity matrices (PLV, coherence, correlation, MI, PLI, AEC), and
uses a three-layer GCN plus post-hoc explainers. HybridNode uses none of those
graph builders, GCNs, explainers, wavelets, or Rashomon searches.

It combines the useful ideas as follows:

1. QNeurov2 temporal framing and five-band FFT are retained.
2. Paper-inspired spectral entropy, differential entropy, Hjorth mobility,
   Hjorth complexity, and connectivity information are added per frame.
3. Full connectivity is replaced by two exact algebraic sketches (signal and
   first-derivative mean off-diagonal correlation). They cost `O(C*T)` and do
   not materialize `C x C` matrices.
4. Per-channel identity is retained in a cached `[N,L,C,11]` tensor. PCA is not
   used. A learned `C`-scalar channel attention pool replaces both PCA and GNN.

All numerical preprocessing is implemented with PyTorch CUDA. CPU activity is
limited to mmap transfer, artifact serialization, and metric/log output.

## Model

```text
[B,L,C,11] HybridNode features
 -> source-only GPU standardization
 -> learned channel attention (C scalars)
 -> [B,L,11]
 -> BiLSTM(input=11, hidden=5, one layer, bidirectional)
 -> softmax fusion(final forward/reverse states, mean sequence state)
 -> LayerNorm(10)
 -> Linear(10,4), tanh, x pi
 -> four-qubit, two-block VQC with trainable data re-upload scales
 -> 4-D Pauli-Z expectations
 -> spline-only KAN classifier
```

Trainable parameters range from 854 (CHSZ) to 959 (FACED), always strictly
below 1000 for the ten installed datasets.

## Fixed evaluation protocol

Each LOSO fold trains on all `N-1` source subjects. There is no validation set.
After every epoch, the model is tested on the single held-out target subject.
`best.pt` is selected by maximum target-subject test accuracy. This matches the
requested target-selected protocol and is intentionally not presented as an
unbiased final-test estimate.

Default campaign settings: one seed (`7`), 100 epochs, batch size 512,
`num_workers=0`, AdamW, cosine learning-rate schedule, and train-only quantum
RZ noise sigma 0.10 radians.
