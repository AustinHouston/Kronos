import numpy as np
import pandas as pd
import pytest
from synthetic import make_events, write_event_files

from kronos.events import concat, find_files
from kronos.sort import bucket_sort, merge_sort, time_disorder, time_order

SMALL = {"chunk_rows": 7_000, "part_rows": 10_000}


@pytest.fixture(params=[0, 50, None], ids=["sorted", "jittered", "shuffled"])
def event_files(tmp_path, request):
    events = make_events(n_electrons=20_000, duration=2_000_000)
    write_event_files(tmp_path, events, jitter=request.param)
    return find_files(tmp_path), events, request.param


def assert_same_events_sorted(parts, events):
    result = pd.DataFrame(concat(parts))
    assert np.all(np.diff(result.toa.to_numpy().astype(np.int64)) >= 0)
    expected = pd.DataFrame({k: events[k] for k in ("x", "y", "toa", "tot")})
    key = ["toa", "x", "y", "tot"]
    pd.testing.assert_frame_equal(
        result.sort_values(key, ignore_index=True)[key],
        expected.sort_values(key, ignore_index=True)[key],
    )


def test_merge_sort_keeps_every_event(event_files):
    files, events, jitter = event_files
    if jitter is None:
        pytest.skip("merge_sort needs near-sorted files")
    assert_same_events_sorted(
        list(merge_sort(files, max_disorder=100, **SMALL)), events
    )


def test_merge_sort_rejects_shuffled_files(tmp_path):
    events = make_events(n_electrons=20_000, duration=2_000_000)
    files = write_event_files(tmp_path, events, jitter=None)
    with pytest.raises(ValueError, match="jumps back"):
        list(merge_sort(files, max_disorder=100, **SMALL))


def test_bucket_sort_keeps_every_event(event_files, tmp_path):
    files, events, _ = event_files
    parts = list(bucket_sort(files, tmp_path / "tmp", **SMALL))
    assert_same_events_sorted(parts, events)
    assert max(len(p["toa"]) for p in parts) <= SMALL["part_rows"]
    assert not (tmp_path / "tmp").exists()


def test_time_order_matches_argsort():
    rng = np.random.default_rng(0)
    for toa in (rng.integers(0, 1_000, 5_000), rng.integers(0, 10**12, 5_000)):
        toa = toa.astype(np.uint64)
        assert np.array_equal(time_order(toa), np.argsort(toa, kind="stable"))


def test_time_disorder():
    assert time_disorder([1, 2, 3]) == 0
    assert time_disorder([5, 1, 6, 4]) == 4
