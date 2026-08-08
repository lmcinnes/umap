# Optimizer rollout and documentation plan

## Status

The optimizer refactor is implemented and verified, but release rollout policy is intentionally deferred. This document records the decisions and work to address in a later session; it does not define release gates yet.

Current implementation state:

- `adam` is the default optimizer.
- `momentum` is the modern momentum update policy.
- `compatibility` preserves the legacy immediate-update objective.
- DensMAP is selected independently with `densmap=True`.
- The removed composite and historical optimizer names are rejected.
- Modern Euclidean and generic layouts support the applicable hard-negative behavior.
- Inverse transform and aligned UMAP retain their specialized objectives.

## Rollout decisions to make

### Compatibility and migration

Decide which changes require warnings or migration guidance:

- Whether rejected historical names should remain hard errors or receive a time-limited deprecation path.
- Whether serialized estimators using historical names require explicit pickle migration support.
- How long `compatibility` remains public and what behavior it promises to preserve.
- Whether any renamed parameters or changed defaults need compatibility shims.

No alias or warning period is implemented at present.

### Quality evidence

Choose datasets, metrics, repetition counts, and acceptable variance before setting thresholds. Candidate evidence includes:

- Trustworthiness and continuity.
- Multiscale neighbor overlap.
- Fragmentation and class contraction.
- Local-radius and graph-edge distributions.
- Density preservation for DensMAP.
- Fit and transform behavior for dense, sparse, Euclidean, and generic metrics.

Thresholds should distinguish expected optimizer-dependent embedding changes from regressions. A single snapshot or one random seed should not become a release gate.

### Performance evidence

Decide how to measure and gate:

- Cold process time, including Numba compilation.
- Warm fit and transform time.
- Peak resident memory.
- Small-input and large-input paths separately.
- Serial deterministic and unseeded parallel modes separately.
- Global negative sampling and hard-negative mining separately.

Performance gates should specify the machine class, thread limits, warm-up policy, repetitions, and tolerated noise.

### Feature-by-feature rollout

Consider independent acceptance decisions for:

1. Adam as the public default.
2. Momentum modernization.
3. Modern DensMAP kernels.
4. Generic metric optimizers and force-ranked hard negatives.
5. Sparse and asymmetric transform paths.
6. Hard-negative scaling and graph-neighbor exclusion.

Do not combine unrelated behavior changes into one acceptance decision.

## DensMAP verification follow-up

Permanent tests should cover both objective plumbing and user-visible behavior:

- Membership weights passed to modern DensMAP kernels must use the same CSR order as kernel traversal.
- Adam bias correction must use epoch index `n + 1`.
- Adam, momentum, and compatibility embeddings and density radii must remain finite.
- Existing trustworthiness checks must continue to pass.
- A future density-quality benchmark should use a dataset and metric that directly represent the DensMAP objective. Raw global radius correlation on separated synthetic clusters is not sufficient by itself because between-cluster geometry can dominate local-density ranks.

Before defining a DensMAP rollout gate, select a connected variable-density dataset, compare against ordinary UMAP and compatibility mode, and establish expected statistical ranges across seeds.

## Documentation project

Documentation should be handled as a dedicated project rather than folded into kernel refactoring.

### User guide

Add an optimizer section explaining:

- When to use `adam`, `momentum`, or `compatibility`.
- Reproducibility and parallelism implications.
- Cold compilation versus warm runtime.
- Which options affect only modern kernels.
- The distinction between optimizer policy and objectives such as DensMAP.

### Parameter reference

Document:

- `optimizer` and its accepted values.
- `negative_selection_range`.
- `negative_sample_scale`.
- `negative_sample_scale_adaptation_samples`.
- `exclude_graph_neighbors`.
- Applicable fit, transform, input-size, and output-metric boundaries.

### Migration guide

Provide a concise mapping from historical names and behavior to the current API, including examples and serialized-model considerations. State clearly whether changes are errors, warnings, or automatic migrations.

### DensMAP and specialized layouts

Clarify that:

- DensMAP is orthogonal to optimizer selection.
- Generic-output DensMAP is unsupported.
- Inverse transform uses a dedicated objective.
- Aligned UMAP currently uses its established immediate-update kernel; modern aligned kernels remain experimental and undispatched.

### Release notes and examples

Once policy is agreed:

- Update release notes with final compatibility and deprecation decisions.
- Add focused optimizer-selection and migration examples.
- Keep benchmark methodology separate from normative user guidance.

## Proposed next-session sequence

1. Agree on compatibility, warning, and migration policy.
2. Select representative quality and performance datasets.
3. Define measurement methodology and repeatability requirements.
4. Investigate and select a meaningful DensMAP density-quality metric.
5. Draft user documentation and migration guidance.
6. Only then define rollout gates and automate stable checks.
