import numpy as np

def init_buffers(particles):
    """Return numpy buffers initialized from given particle objects.

    Parameters
    ----------
    particles : list
        Iterable of particle objects that expose attributes similar to
        ``VerletObject``.

    Returns
    -------
    dict
        Dictionary containing structured numpy arrays used by the solver.
    """
    n = len(particles)
    buf = {
        "positions": np.empty((n, 2, 2), dtype=float),
        "velocities": np.empty((n, 2, 2), dtype=float),
        "accelerations": np.empty((n, 2), dtype=float),
        "radii": np.empty(n, dtype=float),
        "masses": np.empty(n, dtype=float),
        "fixated_mask": np.empty(n, dtype=bool),
        # work arrays reused across update steps
        "work_vec2": np.empty((n, 2), dtype=float),
        "work_vec2b": np.empty((n, 2), dtype=float),
        "work_scalar": np.empty(n, dtype=float),
        "work_scalar_b": np.empty(n, dtype=float),
        "accum": np.empty((n, 2), dtype=float),
    }
    for idx, p in enumerate(particles):
        buf["positions"][idx] = p.position
        buf["velocities"][idx] = p.velocity
        buf["accelerations"][idx] = p.acceleration
        buf["radii"][idx] = p.radius
        buf["masses"][idx] = p.mass
        buf["fixated_mask"][idx] = p.fixated
    return buf
