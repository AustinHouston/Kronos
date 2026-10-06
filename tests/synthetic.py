"""Small synthetic TimePix-like HDF5 files with known electrons."""

import h5py
import numpy as np

NEIGHBORS = np.array(
    [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
)


def make_events(n_electrons, duration, size=256, max_spread=5, seed=0):
    """Events of n_electrons; each is a center pixel plus a few 8-neighbors.

    Returns columns x, y, toa, tot and the true electron, sorted by toa.
    """
    rng = np.random.default_rng(seed)
    t0 = rng.integers(0, duration, n_electrons)
    cx = rng.integers(1, size - 1, n_electrons)
    cy = rng.integers(1, size - 1, n_electrons)
    n_extra = rng.poisson(3, n_electrons).clip(0, 8)
    electron = np.repeat(np.arange(n_electrons), 1 + n_extra)
    offsets = np.concatenate(
        [np.vstack([[0, 0], NEIGHBORS[rng.permutation(8)[:k]]]) for k in n_extra]
    )
    spread = rng.integers(0, max_spread + 1, len(electron))
    events = {
        "x": (cx[electron] + offsets[:, 0]).astype(np.uint16),
        "y": (cy[electron] + offsets[:, 1]).astype(np.uint16),
        "toa": (t0[electron] + spread).astype(np.uint64),
        "tot": rng.integers(1, 60, len(electron)).astype(np.uint16),
        "electron": electron,
    }
    order = np.argsort(events["toa"], kind="stable")
    return {name: values[order] for name, values in events.items()}


def write_event_files(directory, events, n_files=4, jitter=0, seed=1):
    """Split events across files; each file is sorted except for local jitter.

    jitter=0 gives sorted files; jitter=None gives fully shuffled files.
    """
    rng = np.random.default_rng(seed)
    owner = rng.integers(0, n_files, len(events["toa"]))
    paths = []
    for f in range(n_files):
        rows = np.flatnonzero(owner == f)
        if jitter is None:
            rows = rng.permutation(rows)
        else:
            key = events["toa"][rows] + rng.uniform(0, jitter + 1e-9, len(rows))
            rows = rows[np.argsort(key, kind="stable")]
        path = directory / f"synthetic_ch1_proc{f}.h5"
        with h5py.File(path, "w") as h5:
            for name in ("x", "y", "toa", "tot"):
                h5[name] = events[name][rows]
            h5["control"] = np.zeros(3, dtype=np.uint64)
        paths.append(path)
    return paths
