"""Compare all integrators: energy drift, angular-momentum drift, closure error.

Example: python analyze.py --e 0.5 --steps-per-orbit 200 --orbits 50
"""
import argparse

import numpy as np

import kepler


def metrics(method, e, spo, orbits):
    t, s = kepler.integrate(method, e, spo, orbits)
    e0, l0 = kepler.energy(s[0]), kepler.ang_mom(s[0])
    dE = np.abs((kepler.energy(s) - e0) / e0)
    dL = np.abs((kepler.ang_mom(s) - l0) / l0)
    err = np.linalg.norm(s[:, :2] - kepler.exact_position(t, e), axis=1)
    return {"max_dE": dE.max(), "max_dL": dL.max(), "final_pos_err": err[-1]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e", type=float, default=0.5)
    p.add_argument("--steps-per-orbit", type=int, default=200)
    p.add_argument("--orbits", type=float, default=50)
    a = p.parse_args()
    print(f"e={a.e}  steps/orbit={a.steps_per_orbit}  orbits={a.orbits}  (normalized units, GM=a=1)")
    print(f"{'method':8s} {'max |dE/E|':>12s} {'max |dL/L|':>12s} {'final |dr|':>12s}")
    for m in kepler.METHODS:
        r = metrics(m, a.e, a.steps_per_orbit, a.orbits)
        print(f"{m:8s} {r['max_dE']:12.3e} {r['max_dL']:12.3e} {r['final_pos_err']:12.3e}")


if __name__ == "__main__":
    main()
