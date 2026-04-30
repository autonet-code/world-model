# Implementation notes (deferred work)

Items we discussed and decided to defer. Each section is self-contained
so anyone returning to this can pick up the work without re-deriving the
context.

## Batched NumPy calibration

**Status:** deferred. Decision date: 2026-05-01.

**What:** A NumPy-accelerated implementation of `_calibrate` (in
`world_model/dynamics/reseed.py`) that processes many test cases in
parallel via matrix-matrix multiplication, replacing the per-case
Python-loop path.

**Why deferred:** the unbatched version is fast enough for the demos
we have. The Digits demo (`demo_digits.py`) runs 200 test cases in
~30 seconds; not worth optimizing yet.

**When to revisit:** when we want to run the engine on datasets with
thousands of test cases, or when we start the cosmological novelty
curve experiment (which involves many reseed calls per epoch).

### Sketch of the approach

The graph-propagation calibration computes, each iteration, a target
for every non-anchored tendency as a weighted average of its neighbors'
allocations. In math:

```
targets = (W @ allocations) / row_sums
allocations = allocations + lr * (targets - allocations)
allocations = allocations / sum(allocations)    # normalize
```

where `W` is the (n_tendencies x n_tendencies) edge-weight matrix.
This is already a tight inner loop in NumPy terms.

For multiple test cases at once, stack the allocation vectors as
columns: `allocations` becomes (n_tendencies x n_cases). Every step
remains a single matrix-matrix multiply. The classifier's outer loop
collapses to one batched call.

### Estimated effort

3-5 hours. Steps:

1. Build a `_GraphMatrix` helper that converts `StakeWeightGraph` to a
   numpy array once, with an `id -> row_index` map.
2. Add a `_calibrate_numpy` that takes (n_tendencies x n_cases)
   allocation matrices and returns the same shape after convergence.
3. Add a wrapper at the demo level (`classify_batch(cases, ...)`) that
   builds the substitution matrix, calls `_calibrate_numpy`, and reads
   out winners per column.
4. Write a test that pure-Python and NumPy paths produce identical
   results on the same input (within numerical tolerance, say 1e-9).
5. Bench script confirming the speedup.

### What NOT to do

- **No Keras / PyTorch.** Those are tools for training parametric
  models via gradient descent. We don't have learnable parameters.
  Adopting them would be cargo culting.
- **No GPU acceleration yet.** Premature; we are nowhere near the
  scale (10^5+ tendencies) where GPU matters.
- **No closed-form eigenvector solution.** The graph propagation
  fixed-point IS the dominant eigenvector of the transition matrix
  modified by the anchored boundary conditions. `scipy.sparse.linalg.eigs`
  could replace the iteration loop entirely. Real optimization, but
  premature -- it would obscure what the engine is doing for marginal
  gain at our current scale.

### Where Keras / PyTorch might legitimately enter later

If we decide to *learn* graph edge weights from data instead of taking
them from domain priors -- i.e., the engine learns its own wiring --
that's a different architectural decision, not a library swap. At that
point an autograd library is the right tool. Until then, we are doing
relaxation on a weighted graph; NumPy is sufficient.
