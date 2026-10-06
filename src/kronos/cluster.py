"""Group toa-sorted events into clusters (one cluster ~ one electron).

Two events belong together when they are within `radius` pixels in x and y
(a square neighborhood) and within `dt_max` toa ticks. Clusters are the
connected components of that relation.

    cluster_parts(merge_sort(files), directory, dt_max=2)
"""

import numba
import numpy as np
import pandas as pd

from kronos.events import concat, new_directory, take


def cluster_parts(parts, directory, dt_max, radius=1):
    """Cluster parts arriving in toa order; write one cluster table per part.

    Clusters that may continue into the next part (any event within dt_max of
    the part's last toa) are carried over and finished with the next part.
    """
    directory = new_directory(directory)
    carry, k = None, -1
    for k, part in enumerate(parts):
        events = part if carry is None else concat([carry, part])
        labels = label_events(events["x"], events["y"], events["toa"], dt_max, radius)
        late = events["toa"] >= int(events["toa"][-1]) - dt_max
        still_open = np.unique(labels[late])
        clusters = summarize_clusters(events, labels).drop(index=still_open)
        clusters.to_parquet(directory / f"clusters-{k:05d}.parquet", index=False)
        carry = take(events, np.isin(labels, still_open))
    if carry is not None and len(carry["toa"]):
        labels = label_events(carry["x"], carry["y"], carry["toa"], dt_max, radius)
        clusters = summarize_clusters(carry, labels)
        clusters.to_parquet(directory / f"clusters-{k + 1:05d}.parquet", index=False)
    return directory


def label_events(x, y, toa, dt_max, radius=1):
    """Cluster label per event (sorted by toa), numbered 0, 1, ... by first event."""
    x, y = np.asarray(x, np.int32), np.asarray(y, np.int32)
    toa = np.asarray(toa, np.int64)
    return _label_sorted(x, y, toa, dt_max, radius, x.max() + 1, y.max() + 1)


def summarize_clusters(events, labels):
    """One row per cluster (row index = label): first toa, duration,
    ToT-weighted centroid, total ToT and number of events.

    Events must be sorted by toa and labels numbered as label_events does.
    """
    toa = np.asarray(events["toa"], np.int64)
    tot = np.asarray(events["tot"], np.float64)
    first = np.flatnonzero(labels > np.maximum.accumulate(np.r_[-1, labels[:-1]]))
    last = np.full(len(first), np.iinfo(np.int64).min)
    np.maximum.at(last, labels, toa)
    tot_sum = np.bincount(labels, tot)
    weight = np.where(tot_sum > 0, tot_sum, 1.0)
    return pd.DataFrame(
        {
            "toa": toa[first],
            "duration": last - toa[first],
            "x": np.bincount(labels, tot * events["x"]) / weight,
            "y": np.bincount(labels, tot * events["y"]) / weight,
            "tot": tot_sum,
            "n_events": np.bincount(labels),
        }
    )


def neighbor_time_gaps(x, y, toa, radius=1):
    """For each event, toa ticks since the latest earlier event nearby (-1: none).

    A histogram shows a narrow same-electron peak on a flat floor of chance
    coincidences; choose dt_max where the peak meets the floor.
    """
    x, y = np.asarray(x, np.int32), np.asarray(y, np.int32)
    toa = np.asarray(toa, np.int64)
    return _neighbor_gaps(x, y, toa, radius, x.max() + 1, y.max() + 1)


@numba.njit(cache=True)
def _label_sorted(x, y, t, dt_max, radius, width, height):
    """Union each event with the latest earlier event on every nearby pixel.

    Keeping only the latest event per pixel is exact: an older event on that
    pixel within dt_max of the new one is also within dt_max of the latest,
    so it is already in the same cluster.
    """
    n = len(t)
    parent = np.arange(n)
    latest = np.full((height, width), -1, dtype=np.int64)
    for i in range(n):
        for py in range(max(0, y[i] - radius), min(height, y[i] + radius + 1)):
            for px in range(max(0, x[i] - radius), min(width, x[i] + radius + 1)):
                j = latest[py, px]
                if j >= 0 and t[i] - t[j] <= dt_max:
                    a, b = _find(parent, i), _find(parent, j)
                    parent[max(a, b)] = min(a, b)
        latest[y[i], x[i]] = i
    # Roots are the first event of each cluster, so numbering roots as they
    # appear gives labels 0, 1, 2, ... in order of first event.
    labels = np.empty(n, dtype=np.int64)
    n_clusters = 0
    for i in range(n):
        root = _find(parent, i)
        if root == i:
            labels[i] = n_clusters
            n_clusters += 1
        else:
            labels[i] = labels[root]
    return labels


@numba.njit(cache=True)
def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@numba.njit(cache=True)
def _neighbor_gaps(x, y, t, radius, width, height):
    gaps = np.full(len(t), -1, dtype=np.int64)
    latest = np.full((height, width), -1, dtype=np.int64)
    for i in range(len(t)):
        for py in range(max(0, y[i] - radius), min(height, y[i] + radius + 1)):
            for px in range(max(0, x[i] - radius), min(width, x[i] + radius + 1)):
                if latest[py, px] >= 0:
                    gap = t[i] - latest[py, px]
                    if gaps[i] < 0 or gap < gaps[i]:
                        gaps[i] = gap
        latest[y[i], x[i]] = t[i]
    return gaps
