import numpy as np
import pytest
from synthetic import make_events, write_event_files

from kronos.events import find_files, load_parts, read_chunks, save_parts


def test_find_files_filters_and_sorts(tmp_path):
    for name in ("b_1.h5", "a_2.h5", "a_1.h5", "a.dat"):
        (tmp_path / name).touch()
    assert [p.name for p in find_files(tmp_path, "a_*.h5")] == ["a_1.h5", "a_2.h5"]


def test_read_chunks_covers_every_row(tmp_path):
    events = make_events(n_electrons=1_000, duration=100_000)
    (path,) = write_event_files(tmp_path, events, n_files=1)
    chunks = list(read_chunks(path, ("toa",), chunk_rows=333))
    assert sum(len(c["toa"]) for c in chunks) == len(events["toa"])
    assert max(len(c["toa"]) for c in chunks) == 333


def test_saved_parts_load_back_unchanged(tmp_path):
    parts = [{"x": np.arange(5, dtype=np.uint16), "toa": np.arange(5, dtype=np.uint64)}]
    directory = save_parts(parts, tmp_path / "parts")
    loaded = list(load_parts(directory, columns=("x", "toa")))
    assert np.array_equal(loaded[0]["toa"], parts[0]["toa"])
    with pytest.raises(FileExistsError):
        save_parts(parts, directory)
