import csv
import os
import subprocess
import sys

import numpy as np
import pytest

import kepler

E_ECC = 0.5  # test eccentricity


def rel_drift(series):
    return np.abs((series - series[0]) / series[0])


def final_error(method, spo, e=E_ECC):
    t, s = kepler.integrate(method, e, spo, 1.0)
    return np.linalg.norm(s[-1, :2] - kepler.exact_position(t[-1:], e)[0])


@pytest.mark.parametrize("e", [0.0, 0.3, 0.9])
def test_initial_conditions_match_analytic_invariants(e):
    s0 = kepler.initial_state(e)
    assert kepler.energy(s0) == pytest.approx(-0.5, abs=1e-14)  # -GM/(2a)
    assert kepler.ang_mom(s0) == pytest.approx(np.sqrt(1 - e**2), abs=1e-14)


def test_period_obeys_keplers_third_law():
    assert kepler.PERIOD == pytest.approx(2 * np.pi, abs=1e-15)  # a = GM = 1


def test_exact_solution_closes_after_one_period_and_hits_apoapsis():
    r = kepler.exact_position(np.array([0.0, kepler.PERIOD / 2, kepler.PERIOD]), E_ECC)
    assert r[0] == pytest.approx([1 - E_ECC, 0.0], abs=1e-12)
    assert r[1] == pytest.approx([-(1 + E_ECC), 0.0], abs=1e-12)  # apoapsis at a(1+e)
    assert r[2] == pytest.approx(r[0], abs=1e-12)


def test_rk4_with_tiny_step_reproduces_analytic_orbit():
    assert final_error("rk4", 2000) < 1e-8  # tolerance: ~10x observed 1.9e-9


def test_euler_energy_drift_is_large():
    _, s = kepler.integrate("euler", E_ECC, 200, 10)
    assert rel_drift(kepler.energy(s)).max() > 0.1  # observed ~0.5-0.9


def test_verlet_energy_error_is_bounded_not_secular():
    _, s = kepler.integrate("verlet", E_ECC, 200, 50)
    d = rel_drift(kepler.energy(s))
    half = len(d) // 2
    assert d.max() < 1e-2  # observed 2.7e-3
    assert d[half:].max() < 1.2 * d[:half].max()  # no growth over time


@pytest.mark.parametrize("method", ["verlet"])
def test_verlet_conserves_angular_momentum_to_roundoff(method):
    _, s = kepler.integrate(method, E_ECC, 200, 50)
    assert rel_drift(kepler.ang_mom(s)).max() < 1e-12  # central force, observed 2e-14


@pytest.mark.parametrize(
    "method,spo,order",
    [("euler", 4000, 1), ("verlet", 200, 2), ("rk4", 200, 4)],
)
def test_convergence_order(method, spo, order):
    measured = np.log2(final_error(method, spo) / final_error(method, 2 * spo))
    assert measured == pytest.approx(order, abs=0.3)


def test_accuracy_ranking_at_fixed_step():
    err = {m: final_error(m, 400) for m in kepler.METHODS}
    assert err["rk4"] < err["verlet"] < err["euler"]


@pytest.mark.parametrize("bad_e", [-0.1, 1.0, 1.5])
def test_unbound_or_invalid_eccentricity_rejected(bad_e):
    with pytest.raises(ValueError):
        kepler.initial_state(bad_e)


def test_cli_writes_expected_csv(tmp_path):
    out = tmp_path / "o.csv"
    subprocess.run(
        [sys.executable, "simulate.py", "--method", "verlet", "--e", "0.5",
         "--steps-per-orbit", "100", "--orbits", "2", "--out", str(out)],
        check=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 201
    assert float(rows[0]["energy"]) == pytest.approx(-0.5, abs=1e-12)
