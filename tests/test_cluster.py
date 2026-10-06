import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from synthetic import make_events, write_event_files

from kronos.cluster import (
    cluster_parts,
    label_events,
    neighbor_time_gaps,
    summarize_clusters,
)
from kronos.sort import merge_sort

DT_MAX = 8


def reference_labels(x, y, toa, dt_max, radius=1):
    """Slow, simple clustering: connected components of all neighbor pairs."""
    t = np.asarray(toa, np.int64)
    points = np.column_stack(
        [np.asarray(x) / radius, np.asarray(y) / radius, t / dt_max]
    )
    pairs = cKDTree(points).query_pairs(1 + 1e-9, p=np.inf, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), (len(t),) * 2)
    return connected_components(graph, directed=False)[1]


def same_grouping(a, b):
    """True when two labelings group the events identically."""
    pairs = pd.DataFrame({"a": a, "b": b}).drop_duplicates()
    return pairs.a.is_unique and pairs.b.is_unique


def labels_of(events):
    return label_events(events["x"], events["y"], events["toa"], DT_MAX)


def test_sparse_events_recover_electrons():
    events = make_events(n_electrons=2_000, duration=10_000_000)
    assert same_grouping(labels_of(events), events["electron"])


def test_matches_reference_on_dense_events():
    events = make_events(n_electrons=50_000, duration=200_000, size=64)
    labels = labels_of(events)
    reference = reference_labels(events["x"], events["y"], events["toa"], DT_MAX)
    assert same_grouping(labels, reference)
    assert labels.max() + 1 < 50_000  # dense enough that electrons overlap


def test_labels_are_numbered_by_first_event():
    labels = labels_of(make_events(n_electrons=1_000, duration=1_000_000))
    assert np.array_equal(pd.unique(labels), np.arange(labels.max() + 1))


def test_neighbor_gaps_peak_within_spread():
    events = make_events(n_electrons=2_000, duration=10_000_000, max_spread=5)
    gaps = neighbor_time_gaps(events["x"], events["y"], events["toa"])
    assert gaps[(gaps >= 0) & (gaps < 1_000)].max() <= 5


def test_summary_rows_follow_labels():
    events = {
        "x": [0, 9, 0, 9],
        "y": [0, 0, 0, 0],
        "toa": [1, 2, 3, 4],
        "tot": [1, 1, 2, 2],
    }
    summary = summarize_clusters(events, np.array([0, 1, 0, 1]))
    assert summary.toa.tolist() == [1, 2]
    assert summary.duration.tolist() == [2, 2]
    assert summary.x.tolist() == [0, 9]
    assert summary.n_events.tolist() == [2, 2]


def test_streamed_parts_match_one_big_part(tmp_path):
    events = make_events(n_electrons=30_000, duration=300_000, size=64)
    files = write_event_files(tmp_path, events, jitter=20)
    parts = merge_sort(files, max_disorder=100, chunk_rows=777, part_rows=2_000)
    directory = cluster_parts(parts, tmp_path / "clusters", DT_MAX)

    tables = [pd.read_parquet(p) for p in sorted(directory.glob("*.parquet"))]
    streamed = pd.concat(tables).sort_values(["toa", "x", "y"], ignore_index=True)
    whole = summarize_clusters(events, labels_of(events))
    whole = whole.sort_values(["toa", "x", "y"], ignore_index=True)
    pd.testing.assert_frame_equal(streamed, whole)
