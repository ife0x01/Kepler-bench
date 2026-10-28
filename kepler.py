"""Two-body (Kepler) problem in normalized units: GM = 1, semi-major axis a = 1.

Units: length = a, time = sqrt(a^3/GM) (so the orbital period is 2*pi),
specific energy = GM/a. Point masses, Newtonian gravity, no perturbations.
"""
import numpy as np

GM = 1.0
A = 1.0
PERIOD = 2.0 * np.pi * np.sqrt(A**3 / GM)  # Kepler's third law


def initial_state(e):
    """Start at periapsis, counter-clockwise. State = [x, y, vx, vy]."""
    if not 0.0 <= e < 1.0:
        raise ValueError("eccentricity must satisfy 0 <= e < 1 (bound orbit)")
    r_p = A * (1.0 - e)
    v_p = np.sqrt(GM * (1.0 + e) / r_p)  # vis-viva at periapsis
    return np.array([r_p, 0.0, 0.0, v_p])


def accel(pos):
    r = np.linalg.norm(pos)
    return -GM * pos / r**3


def energy(state):
    """Specific orbital energy; analytic value is -GM/(2a) = -0.5."""
    pos, vel = state[..., :2], state[..., 2:]
    r = np.linalg.norm(pos, axis=-1)
    return 0.5 * np.sum(vel**2, axis=-1) - GM / r


def ang_mom(state):
    """Specific angular momentum (z component), x*vy - y*vx."""
    return state[..., 0] * state[..., 3] - state[..., 1] * state[..., 2]


def _deriv(s):
    return np.concatenate([s[2:], accel(s[:2])])


def step_euler(s, dt):
    return s + dt * _deriv(s)


def step_rk4(s, dt):
    k1 = _deriv(s)
    k2 = _deriv(s + 0.5 * dt * k1)
    k3 = _deriv(s + 0.5 * dt * k2)
    k4 = _deriv(s + dt * k3)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def step_verlet(s, dt):
    """Velocity-Verlet (kick-drift-kick), symplectic, 2nd order."""
    pos, vel = s[:2], s[2:]
    vel = vel + 0.5 * dt * accel(pos)
    pos = pos + dt * vel
    vel = vel + 0.5 * dt * accel(pos)
    return np.concatenate([pos, vel])


METHODS = {"euler": step_euler, "verlet": step_verlet, "rk4": step_rk4}


def integrate(method, e, steps_per_orbit, orbits):
    """Return (t, states) with shape (N+1,), (N+1, 4)."""
    step = METHODS[method]
    n = int(round(steps_per_orbit * orbits))
    dt = PERIOD / steps_per_orbit
    states = np.empty((n + 1, 4))
    states[0] = initial_state(e)
    for i in range(n):
        states[i + 1] = step(states[i], dt)
    return np.arange(n + 1) * dt, states


def exact_position(t, e):
    """Analytic position from Kepler's equation M = E - e sin E (Newton solve)."""
    t = np.asarray(t, dtype=float)
    M = np.mod(2.0 * np.pi * t / PERIOD, 2.0 * np.pi)
    E = M.copy() if e < 0.8 else np.full_like(M, np.pi)
    for _ in range(50):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < 1e-15:
            break
    x = A * (np.cos(E) - e)
    y = A * np.sqrt(1.0 - e**2) * np.sin(E)
    return np.stack([x, y], axis=-1)
