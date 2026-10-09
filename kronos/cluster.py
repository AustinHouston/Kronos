"""Group toa-sorted events into clusters (one cluster ~ one electron).

Two events belong together when they are on the same or an edge-adjacent
pixel (|dx| + |dy| <= 1) and within dt_max toa ticks. Clusters are the
connected components of that relation. notebooks/02_Clustering_Choices.ipynb
shows why: charge is shared only with edge neighbors, and same-electron
events arrive within one tick of each other.

    cluster_parts(merge_sort(files), directory, dt_max=1)
    stats = pd.concat(cluster_tables(merge_sort(files), summarize=my_summary))
"""

import numba
import numpy as np
import pandas as pd

from kronos.events import concat, new_directory, take

# The pixel itself and its four edge neighbors, as (dx, dy).
neighbors = np.array([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int64)

# label_events cuts a part into time slices of at least this many events, one per task.
slice_rows = 100_000


def cluster_parts(parts, directory, dt_max=1):
    """Cluster parts arriving in toa order; write each table of cluster_tables as directory/clusters-#####.parquet."""
    directory = new_directory(directory)
    for k, clusters in enumerate(cluster_tables(parts, dt_max)):
        clusters.to_parquet(directory / f'clusters-{k:05d}.parquet', index=False)
    return directory


def label_events(x, y, toa, dt_max=1):
    """Cluster label per event (sorted by toa), numbered 0, 1, ... by first event."""
    x, y = np.asarray(x, np.int32), np.asarray(y, np.int32)
    toa = np.asarray(toa, np.int64)
    n_slices = max(1, min(4 * numba.get_num_threads(), len(toa) // slice_rows))
    return _label_sorted(x, y, toa, dt_max, x.max() + 1, y.max() + 1, n_slices)


def summarize_clusters(events, labels):
    """One row per cluster (row index = label): first toa, duration, ToT-weighted
    centroid (x, y), plain mean of the pixel positions (x_mean, y_mean), total ToT
    and number of events.

    The best impact estimate is a blend of the two centroids,
    0.5 * x_mean + 0.5 * x (notebooks/03_Cluster_Centroids.ipynb).
    Events must be sorted by toa and labels numbered as label_events does.
    """
    toa = np.asarray(events['toa'], np.int64)
    x = np.asarray(events['x'], np.float64)
    y = np.asarray(events['y'], np.float64)
    tot = np.asarray(events['tot'], np.float64)
    labels = np.asarray(labels, np.int64)
    columns = _summarize(labels, toa, x, y, tot, labels.max() + 1)
    names = ('toa', 'duration', 'x', 'y', 'x_mean', 'y_mean', 'tot', 'n_events')
    return pd.DataFrame(dict(zip(names, columns)), copy=False)


def cluster_tables(parts, dt_max=1, summarize=summarize_clusters):
    """Yield one cluster table per part, from parts arriving in toa order.

    summarize(events, labels) gives one row per cluster with the label as row index,
    as summarize_clusters does. Clusters that may continue into the next part (any event
    within dt_max of the part's last toa) are carried over and finished with the next
    part; one last table holds the clusters still open at the end.
    """
    carry = None
    for part in parts:
        events = part if carry is None else concat([carry, part])
        labels = label_events(events['x'], events['y'], events['toa'], dt_max)
        still_open = np.unique(labels[events['toa'] >= int(events['toa'][-1]) - dt_max])
        yield summarize(events, labels).drop(index=still_open)
        carry = take(events, np.isin(labels, still_open))
    yield summarize(carry, label_events(carry['x'], carry['y'], carry['toa'], dt_max))


def neighbor_time_gaps(x, y, toa):
    """For each event, toa ticks since the latest earlier event nearby (-1: none).

    A histogram shows a narrow same-electron peak on a flat floor of chance
    coincidences; choose dt_max where the peak meets the floor.
    """
    x, y = np.asarray(x, np.int32), np.asarray(y, np.int32)
    toa = np.asarray(toa, np.int64)
    return _neighbor_gaps(x, y, toa, x.max() + 1, y.max() + 1)


@numba.njit(parallel=True, cache=True)
def _label_sorted(x, y, t, dt_max, width, height, n_slices):
    """Label time slices in parallel, then link events across slice edges.

    Each slice is a contiguous run of events, so the threads touch disjoint
    parts of `parent`. A link across an edge joins events within dt_max of
    each other, so both lie within dt_max of the edge: relinking that window
    alone finds every one of them.
    """
    n = len(t)
    parent = np.arange(n)
    bounds = np.linspace(0, n, n_slices + 1).astype(np.int64)
    for s in numba.prange(n_slices):
        latest = np.full((height, width), -1, dtype=np.int64)
        _link(x, y, t, dt_max, parent, latest, bounds[s], bounds[s + 1])
    latest = np.full((height, width), -1, dtype=np.int64)
    for s in range(1, n_slices):
        lo = hi = bounds[s]
        while lo > 0 and t[lo - 1] >= t[bounds[s]] - dt_max:
            lo -= 1
        while hi < n and t[hi] <= t[bounds[s] - 1] + dt_max:
            hi += 1
        _link(x, y, t, dt_max, parent, latest, lo, hi)
    # Roots are the first event of each cluster, so numbering roots in event
    # order gives labels 0, 1, 2, ... in order of first event.
    roots = np.empty(n, dtype=np.int64)
    for i in numba.prange(n):
        root = np.int64(i)  # no path halving here: other threads read parent meanwhile
        while parent[root] != root:
            root = parent[root]
        roots[i] = root
    rank = np.cumsum(roots == np.arange(n)) - 1
    labels = np.empty(n, dtype=np.int64)
    for i in numba.prange(n):
        labels[i] = rank[roots[i]]
    return labels


@numba.njit(cache=True)
def _link(x, y, t, dt_max, parent, latest, lo, hi):
    """Union each event in lo..hi-1 with the latest earlier one on each neighbor pixel.

    Keeping only the latest event per pixel is exact: an older event on that
    pixel within dt_max of the new one is also within dt_max of the latest,
    so it is already in the same cluster.
    """
    height, width = latest.shape
    for i in range(lo, hi):
        for k in range(len(neighbors)):
            px, py = x[i] + neighbors[k, 0], y[i] + neighbors[k, 1]
            if 0 <= px < width and 0 <= py < height:
                j = latest[py, px]
                if lo <= j < i and t[i] - t[j] <= dt_max:
                    a, b = _find(parent, i), _find(parent, j)
                    parent[max(a, b)] = min(a, b)
        latest[y[i], x[i]] = i


@numba.njit(cache=True)
def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@numba.njit(cache=True)
def _summarize(labels, t, x, y, tot, n_clusters):
    first = np.empty(n_clusters, dtype=np.int64)
    duration = np.empty(n_clusters, dtype=np.int64)
    cx = np.zeros(n_clusters)
    cy = np.zeros(n_clusters)
    mx = np.zeros(n_clusters)
    my = np.zeros(n_clusters)
    tot_sum = np.zeros(n_clusters)
    count = np.zeros(n_clusters, dtype=np.int64)
    n_seen = 0
    for i in range(len(labels)):
        k = labels[i]
        if k == n_seen:  # labels appear in order, so this is the first event
            first[k] = t[i]
            n_seen += 1
        duration[k] = t[i] - first[k]  # events are sorted by toa
        cx[k] += tot[i] * x[i]
        cy[k] += tot[i] * y[i]
        mx[k] += x[i]
        my[k] += y[i]
        tot_sum[k] += tot[i]
        count[k] += 1
    for k in range(n_clusters):  # sums -> centroids
        mx[k] /= count[k]
        my[k] /= count[k]
        if tot_sum[k] > 0:
            cx[k] /= tot_sum[k]
            cy[k] /= tot_sum[k]
    return first, duration, cx, cy, mx, my, tot_sum, count


@numba.njit(cache=True)
def _neighbor_gaps(x, y, t, width, height):
    gaps = np.full(len(t), -1, dtype=np.int64)
    latest = np.full((height, width), -1, dtype=np.int64)
    for i in range(len(t)):
        for k in range(len(neighbors)):
            px, py = x[i] + neighbors[k, 0], y[i] + neighbors[k, 1]
            if 0 <= px < width and 0 <= py < height and latest[py, px] >= 0:
                gap = t[i] - latest[py, px]
                if gaps[i] < 0 or gap < gaps[i]:
                    gaps[i] = gap
        latest[y[i], x[i]] = t[i]
    return gaps
