"""Numerical regression tests for the conditional-variance repair.

Run from the repository root:
    python -m pytest -q tests/test_optimization.py

The dependency-light shims remove unrelated GDAL requirements while keeping
the original optimization math and phase-extraction functions unchanged.
"""
from pathlib import Path
import importlib.util
import sys
import types
import warnings

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import reference_upstream as baseline

pkg = types.ModuleType("src")
pkg.__path__ = []
utils = types.ModuleType("src.utility")
utils.add_one_obs = baseline.add_one_obs
utils.ToeplitzMatrix = baseline.ToeplitzMatrix
estim = types.ModuleType("src.estimation")
estim.phasecorrection3 = baseline.phasecorrection3
estim.phasecorrection4 = baseline.phasecorrection4
estim.SCM = baseline.SCM
sys.modules.update({"src": pkg, "src.utility": utils, "src.estimation": estim})
spec = importlib.util.spec_from_file_location(
    "src.optimization", ROOT / "code" / "src" / "optimization.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fix = module.S_SG_MLE_PL_BCD


def sample(p=6, n=50, nu=np.inf, seed=87):
    rng = np.random.default_rng(seed)
    t = np.arange(p + 1)
    cor = 0.7 ** np.abs(t[:, None] - t[None, :])
    theta = np.linspace(0, 2, p + 1)
    D = np.diag(np.exp(1j * theta))
    true_cov = D @ cor @ D.conj().T
    x = (rng.standard_normal((p + 1, n))
         + 1j * rng.standard_normal((p + 1, n))) / np.sqrt(2)
    X = np.linalg.cholesky(true_cov) @ x
    if np.isfinite(nu):
        X *= np.sqrt(rng.gamma(nu, 1 / nu, n))[None, :]
    return X, true_cov[:p, :p], D[:p, :p]


@pytest.mark.parametrize("nu", [np.inf, 0.1])
@pytest.mark.parametrize("choice", [3, 4])
def test_matches_original_from_feasible_initialization(nu, choice):
    X, C, D = sample(p=6, n=50, nu=nu, seed=43)
    # Both algorithms run exactly 8 complete cycles at tol=0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        newcov, newphase = fix(X, X[:-1], X[-1], C, D, 8, choice, tol=0)
    oldcov, oldphase = baseline.upstream_bcd(
        X, X[:-1], X[-1], C, D, 8, choice, tol=0)
    assert np.allclose(newcov, oldcov, rtol=1e-8, atol=1e-9)
    assert abs(np.angle(np.exp(1j * (newphase - oldphase)))) < 1e-7


@pytest.mark.parametrize("nu", [np.inf, 0.1])
def test_each_new_block_decreases_objective_and_stays_feasible(nu):
    X, C, D = sample(p=6, n=40, nu=nu, seed=97)
    diag = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        newcov, _ = fix(X, X[:-1], X[-1], C, D, 15, 3, tol=0, diagnostics=diag)
    for stage in diag['iterations']:
        vals = [stage[key] for key in (
            'objective_before', 'after_tau', 'after_gamma', 'after_phase', 'after_v')]
        assert max(np.diff(vals)) <= 1e-8, vals
        assert stage['schur_v'] > 0
    assert np.linalg.eigvalsh(newcov).min() > 0


def test_repairs_infeasible_toeplitz_initialization():
    X, _, D = sample(p=10, n=40, seed=801)
    C = 0.2 * np.eye(10)  # valid C, but upstream v0 < 0
    diag = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        newcov, phase = fix(X, X[:-1], X[-1], C, D, 10, 3, diagnostics=diag)
    assert diag['repaired_initialization']
    assert diag['initial_schur_v'] > 0
    assert np.isfinite(phase)
    assert np.linalg.eigvalsh(newcov).min() > 0


def test_rank_deficient_regression_is_solved_without_matrix_inverse():
    X, _, D = sample(p=13, n=4, seed=18)
    C = np.eye(13)
    diag = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        newcov, phase = fix(X, X[:-1], X[-1], C, D, 9, 3, tol=0, diagnostics=diag)
    assert np.linalg.eigvalsh(newcov).min() > 0
    assert np.isfinite(phase)
    assert all(z['schur_v'] > 0 for z in diag['iterations'])


def test_rejects_nonpositive_past_covariance():
    X, _, D = sample(p=6, n=30)
    with pytest.raises(ValueError, match="positive definite"):
        fix(X, X[:-1], X[-1], np.diag([1, 1, 1, 1, 1, -1]), D, 5, 3)


def test_invalid_phase_correction_selector():
    X, C, D = sample(p=6, n=30)
    with pytest.raises(ValueError, match="phasecorrectionchoice"):
        fix(X, X[:-1], X[-1], C, D, 5, 5)
