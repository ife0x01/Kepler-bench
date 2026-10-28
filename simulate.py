"""Run one integrator and write time, state, energy and angular momentum to CSV.

Example: python simulate.py --method verlet --e 0.5 --steps-per-orbit 200 --orbits 20 --out out.csv
"""
import argparse
import csv

import kepler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=kepler.METHODS, required=True)
    p.add_argument("--e", type=float, default=0.5, help="eccentricity, 0<=e<1")
    p.add_argument("--steps-per-orbit", type=int, default=200)
    p.add_argument("--orbits", type=float, default=10)
    p.add_argument("--out", default="out.csv")
    a = p.parse_args()

    t, s = kepler.integrate(a.method, a.e, a.steps_per_orbit, a.orbits)
    en, lz = kepler.energy(s), kepler.ang_mom(s)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "vx", "vy", "energy", "ang_mom"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.12e}", *(f"{v:.12e}" for v in s[i]), f"{en[i]:.12e}", f"{lz[i]:.12e}"])
    print(f"wrote {len(t)} rows to {a.out}")


if __name__ == "__main__":
    main()
