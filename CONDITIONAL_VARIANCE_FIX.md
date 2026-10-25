# Conditional-variance repair of sequential S-SG-MLE-PL

This change **preserves the estimator**, its publicly exposed positional arguments,
and its original two-value return signature. It does not change the offline
phase-linking implementation or modify the original paper.

## Mathematical issue

In El Hajjar et al. (IEEE Signal Processing Letters 32, 811–815, 2025),
the conditional negative log likelihood contains

```text
    J = sum_i [(p+1) log(tau_i) + log(v)
               + abs(y_i)**2 / (tau_i * v) + q_i / tau_i]
    v = gamma_l - gamma @ A @ gamma
    y_i = x_new_i - w * (B_i @ gamma).
```

The published weighted least-squares update for real `gamma` solves the
`gamma` block **when `v` is held fixed**, not when `gamma_l` is held fixed
as the published explanation suggests.

We therefore optimize using the independent variables `(gamma, v)`.
For fixed `tau_i` and `w`, `gamma` is an ordinary real weighted
least-squares problem. Following the projected phase update, we set

```text
    v       = mean_i(abs(y_i)**2 / tau_i)
    gamma_l = v + gamma @ A @ gamma
```

and maintain `v > 0`. This reconstructs a positive-definite augmented
covariance whenever the past covariance `C` is positive definite (subject
to finite precision). The original update with the published **plus sign**
is recovered mathematically; the GitHub implementation already uses it.

## Implementation details

- Uses a real stacked weighted least-squares solve via `np.linalg.lstsq`,
  avoiding explicit inversion of normal equations.
- Uses the original Toeplitz initialization when its Schur complement is
  positive; otherwise initializes directly with a positive conditional variance.
- Keeps the original scale update, projected unit-magnitude phase update,
  and phase-extraction choices 3 and 4.
- Monitors changes in phase, coherence vector, **and** variance, not only
  the last variance.
- Adds optional `diagnostics={}` recording objective values after each
  block and Schur complements; existing callers do not need to change.
- Leaves `gamma` unconstrained and signed. Requiring all elements of gamma
  to be nonnegative would change the model.
- Raises clear errors for invalid inputs or non-positive-definite past
  covariance instead of continuing silently with an invalid likelihood.

## Test

From the repository root, with Python, NumPy, SciPy and pytest installed:

```bash
python -m pytest -q tests/test_optimization.py
```

The regression suite includes Gaussian and scaled-Gaussian simulated SLCs,
both phase-correction options, a check against a dependency-light transcription
of upstream at its pinned commit, objective monotonicity, feasible
initialization, low-sample least squares, and input validation. The reference
transcription is only for testing, not production execution.

## Limits

This is an **optimization/proof and numerical-stability fix**, not evidence
of improved phase RMSE. When initialized feasibly and solving nonsingular
blocks, the final estimates closely reproduce the released algorithm.
Blockwise non-increasing likelihood does not imply global optimality, nor a
general theorem for limit-point convergence without additional assumptions.
The variance floor is a numerical safeguard and can slightly perturb degenerate
problems. Rank-deficient cases may select different minimizers.

The regression tests do **not** replace an end-to-end benchmark on the
authors' Mexico City dataset or larger SAR stacks. Such a benchmark is
recommended before treating the patch as fully validated in production.
