"""Sequential scaled-Gaussian maximum-likelihood phase linking.

Mathematical repair of Dana El Hajjar et al., IEEE SPL (2025), Eq. (8)--(16).
Original upstream: https://github.com/DanaElhajjar/S-SG-MLE-PL
Baseline commit: 3871e3a8c1215016c2d7e696a2df96e28ff7afd3

Only S_SG_MLE_PL_BCD is changed. Its call signature and default two-item
return value are preserved. Existing src.estimation/src.utility remain intact.
"""

import warnings
import numpy as np

from .utility import add_one_obs, ToeplitzMatrix
from .estimation import phasecorrection3, phasecorrection4, SCM


def S_SG_MLE_PL_BCD(X, X_past, x_newdata, C, diag_w_past,
                    iter_max_BCD, phasecorrectionchoice, tol=0.001, diagnostics=None):
    """Estimate the newly acquired SAR-image phase by sequential ML phase linking.

    Repair: keep the Schur-complement variance v independent of the real
    coherence vector gamma while updating gamma. With fixed positive scales,
    gamma is an ordinary real weighted least-squares problem; the unit-phase
    and v updates are exact minimizers of the reparameterized blocks.

    The original positional call signature and two-item return value are
    preserved; optional diagnostics is new. gamma remains SIGNED and
    UNCONSTRAINED; imposing gamma >= 0 would alter the statistical model.
    Completed cycles agree with the released implementation in feasible,
    nonsingular cases (up to floating-point and stopping differences).

    Positive definiteness follows algebraically for C >> 0 and v > 0.
    No claim of global optimality or unconditional convergence is made.
    """
    X = np.asarray(X, dtype=np.complex128)
    X_past = np.asarray(X_past, dtype=np.complex128)
    x_newdata = np.asarray(x_newdata, dtype=np.complex128).reshape(-1)
    C = np.asarray(C, dtype=np.complex128)
    diag_w_past = np.asarray(diag_w_past, dtype=np.complex128)

    if X_past.ndim != 2:
        raise ValueError("X_past must have shape (p, n)")
    p, n = X_past.shape
    if p < 1 or n < 1 or X.shape != (p + 1, n) or x_newdata.shape != (n,):
        raise ValueError("Expected X:(p+1,n), X_past:(p,n), x_newdata:(n,)")
    if C.shape != (p, p) or diag_w_past.shape != (p, p):
        raise ValueError("C and diag_w_past must be square (p,p) matrices")
    if phasecorrectionchoice not in (3, 4):
        raise ValueError("phasecorrectionchoice must be 3 or 4")
    if iter_max_BCD < 1 or tol < 0:
        raise ValueError("iter_max_BCD >= 1 and tol >= 0 are required")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(C))
            and np.all(np.isfinite(diag_w_past))):
        raise ValueError("Inputs contain NaN or Inf")
    if not np.allclose(C, C.conj().T, rtol=1e-8, atol=1e-10):
        raise ValueError("Past covariance C must be Hermitian")
    try:
        np.linalg.cholesky(C)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Past covariance C must be positive definite") from exc

    # Past covariance is frozen in the sequential conditional likelihood.
    inv_C = np.linalg.solve(C, np.eye(p, dtype=complex))
    inv_C = (inv_C + inv_C.conj().T) / 2
    w_past = np.diag(diag_w_past)
    A = np.real(diag_w_past.conj().T @ inv_C @ diag_w_past)
    A = (A + A.T) / 2

    # y_i = x_l^i - w * gamma @ B_i, with real gamma.
    B = (diag_w_past.conj().T @ inv_C @ X_past).T  # (n, p)
    q = np.real(np.sum(X_past.conj() * (inv_C @ X_past), axis=0))
    if np.any(q < -1e-9 * max(1.0, np.max(np.abs(q)))):
        raise ValueError("Negative quadratic form: check C's conditioning")
    q = np.maximum(q, 0.0)

    # Retain the upstream Toeplitz initialization when it is feasible.
    init = ToeplitzMatrix(0.5, p + 1)
    gamma = np.asarray(init[p, :p], dtype=float).copy()
    w = 1.0 + 0.0j
    k = float(gamma @ A @ gamma)
    floor = 1e-12 * max(1.0, k, float(np.mean(np.abs(x_newdata) ** 2)))
    v = 1.0 - k
    if not np.isfinite(v) or v <= floor:
        # Initialize in the feasible (gamma, v) coordinates.
        resid0 = x_newdata - w * (B @ gamma)
        v = max(float(np.mean(np.abs(resid0) ** 2)), floor)

    gamma_l = k + v
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics['initial_schur_v'] = float(v)
        diagnostics['repaired_initialization'] = bool(1.0 - k <= floor)
        diagnostics['iterations'] = []

        def objective(scales, cohs, phase, cond_var):
            res = x_newdata - phase * (B @ cohs)
            return float(np.sum((p + 1) * np.log(scales) + np.log(cond_var)
                                + (q + np.abs(res) ** 2 / cond_var) / scales))

        tau_previous = np.ones(n)

    scm_bis = None
    for iteration in range(iter_max_BCD):
        if diagnostics is not None:
            obj_before = objective(tau_previous, gamma, w, v)

        # Block 1: Eq. (10); scales must be positive.
        y = x_newdata - w * (B @ gamma)
        tau = (q + np.abs(y) ** 2 / v) / (p + 1)
        if np.any(~np.isfinite(tau)) or np.any(tau <= 0):
            raise ValueError("Nonpositive/invalid scale tau: degenerate SAR samples")
        inv_sqrt_tau = 1.0 / np.sqrt(tau)
        if phasecorrectionchoice == 4:
            # Match the original final covariance-subdiagonal phase convention.
            scm_bis = SCM(X * inv_sqrt_tau[None, :])
        if diagnostics is not None:
            obj_tau = objective(tau, gamma, w, v)

        # Block 2: exact real weighted least squares at fixed v.
        # QR/SVD avoids explicit inversion; works also when n < p.
        design = (w * B) * inv_sqrt_tau[:, None]
        target = x_newdata * inv_sqrt_tau
        gamma_new = np.linalg.lstsq(
            np.vstack((design.real, design.imag)),
            np.concatenate((target.real, target.imag)),
            rcond=None,
        )[0]
        if diagnostics is not None:
            obj_gamma = objective(tau, gamma_new, w, v)

        # Block 3: Eq. (16), projected onto the complex unit circle.
        predictor = B @ gamma_new
        cross = np.vdot(predictor * inv_sqrt_tau, target)
        w_new = cross / abs(cross) if abs(cross) > 0 else w
        if diagnostics is not None:
            obj_phase = objective(tau, gamma_new, w_new, v)

        # Block 4: exact v minimizer, up to numerical floor.
        residual = x_newdata - w_new * predictor
        v_new = max(float(np.mean(np.abs(residual) ** 2 / tau)), floor)
        k_new = float(gamma_new @ A @ gamma_new)
        variance_new = k_new + v_new
        if not np.isfinite(variance_new):
            raise FloatingPointError("Updated new-image variance is not finite")

        # Check every updated parameter; upstream only monitored gamma_l.
        rel_variance = abs(variance_new - gamma_l) / max(abs(gamma_l), floor)
        rel_gamma = np.linalg.norm(gamma_new - gamma) / max(1.0, np.linalg.norm(gamma))
        phase_delta = abs(w_new - w)
        stop = max(rel_variance, rel_gamma, phase_delta)
        if diagnostics is not None:
            obj_v = objective(tau, gamma_new, w_new, v_new)
            diagnostics['iterations'].append({
                'objective_before': obj_before, 'after_tau': obj_tau,
                'after_gamma': obj_gamma, 'after_phase': obj_phase,
                'after_v': obj_v, 'schur_v': float(v_new),
                'relative_change': float(stop),
            })
            tau_previous = tau
        gamma, w, v, gamma_l = gamma_new, w_new, v_new, variance_new
        if stop <= tol:
            break
    else:
        warnings.warn("Sequential ML phase linking reached iter_max_BCD without "
                      "satisfying the multi-parameter stopping criterion",
                      RuntimeWarning, stacklevel=2)

    line_vector = w * gamma @ diag_w_past.conj().T
    C_tilde = add_one_obs(C, line_vector, gamma_l).astype(np.complex128)
    C_tilde = (C_tilde + C_tilde.conj().T) / 2
    if phasecorrectionchoice == 3:
        new_phase = phasecorrection3(C_tilde)
    else:
        diag_w = np.diag(np.append(w_past, w))
        Sigma = (diag_w.conj().T @ scm_bis @ diag_w).real
        new_phase = phasecorrection4(C_tilde, Sigma)
    return C_tilde, float(new_phase[-1])
