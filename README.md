# kepler-bench: integrator accuracy benchmark for the two-body problem

Simulates one body orbiting a point mass with three integrators (explicit Euler,
velocity-Verlet, RK4) and checks each against the analytic Kepler solution and
the conservation laws. Everything is checked by automated tests.

## Run

    pip install -r requirements.txt
    pytest -q
    python analyze.py --e 0.5 --steps-per-orbit 200 --orbits 50
    python simulate.py --method verlet --e 0.5 --steps-per-orbit 200 --orbits 20 --out out.csv

Or, reproducibly: `docker build -t kepler-bench . && docker run --rm kepler-bench`

## Units and model

- Normalized units: GM = 1, semi-major axis a = 1, so length = a, time = sqrt(a^3/GM),
  specific energy = GM/a. Orbital period = 2*pi (Kepler's third law).
- Point masses, Newtonian gravity, 2D, bound orbits only (0 <= e < 1), no relativity,
  no perturbations. The orbit starts at periapsis, counter-clockwise.
- Analytic references: energy = -GM/(2a) = -0.5; angular momentum = sqrt(GM*a*(1-e^2));
  position from Kepler's equation M = E - e sin E (Newton iteration, 1e-15 convergence).

## Tolerances (all in tests/test_kepler.py)

| Check | Tolerance | Basis |
|---|---|---|
| Initial energy and angular momentum | 1e-14 | float64 round-off |
| Exact solution closes after one period | 1e-12 | Newton solver convergence |
| RK4, 2000 steps/orbit, position error | < 1e-8 | ~10x observed 1.9e-9 |
| Euler max energy error, 10 orbits, 200 steps/orbit | > 0.1 | observed 0.5 to 0.9 |
| Verlet max energy error, 50 orbits | < 1e-2 and no growth (2nd half <= 1.2x 1st half) | observed 2.7e-3, bounded |
| Verlet angular momentum drift | < 1e-12 | exact for central force, observed 2e-14 |
| Convergence order (halving dt) | nominal +/- 0.3 | Euler 1, Verlet 2, RK4 4 |

## Results (e = 0.5, 200 steps/orbit, 50 orbits)

| Method | max abs(dE/E) | max abs(dL/L) | position error at t = 50 orbits |
|---|---|---|---|
| euler | 9.4e-1 | 4.9e-1 | 32 |
| verlet | 2.7e-3 | 2e-14 | 1.5 |
| rk4 | 4.7e-5 | 7e-6 | 2.0e-2 |

Note: Verlet keeps energy bounded (symplectic) but accumulates phase error;
RK4 is far more accurate over short runs but its energy error grows slowly over
very long runs.

## Limitations

- Fixed time step: no adaptive stepping, so very eccentric orbits (e > 0.9)
  need many steps near periapsis.
- Convergence is measured at t = 1 orbit; orders can degrade if steps are too coarse.
- 2D only; no units other than the normalized system above.
