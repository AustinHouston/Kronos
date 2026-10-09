import numpy as np
import pytest
from synthetic import make_events, write_event_files

from kronos.events import fine_steps, fine_toa, fine_tot, load_parts, read_chunks, save_parts


def test_read_chunks_covers_every_row(tmp_path):
    events = make_events(n_electrons=1_000, duration=100_000)
    (path,) = write_event_files(tmp_path, events, n_files=1)
    chunks = list(read_chunks(path, ('toa',), chunk_rows=333))
    assert sum(len(c['toa']) for c in chunks) == len(events['toa'])
    assert max(len(c['toa']) for c in chunks) == 333


def test_saved_parts_load_back_unchanged(tmp_path):
    parts = [{'x': np.arange(5, dtype=np.uint16), 'toa': np.arange(5, dtype=np.uint64)}]
    directory = save_parts(parts, tmp_path / 'parts')
    loaded = list(load_parts(directory, columns=('x', 'toa')))
    assert np.array_equal(loaded[0]['toa'], parts[0]['toa'])
    with pytest.raises(FileExistsError):
        save_parts(parts, directory)


def test_fine_tot_matches_manual_table_24_2():
    # Timepix4 manual Table 24.2: toa, uftoa_start, uftoa_stop, ftoa_rise, ftoa_fall,
    # tot and the manual's TOT_CALCULATED in ns.
    rows = np.array(
        [
            (35279, 0, 5, 9, 9, 1, 25.98),
            (42062, 0, 1, 4, 3, 2, 51.76),
            (48847, 0, 4, 25, 18, 0, 11.72),
            (55631, 0, 7, 30, 25, 0, 9.18),
            (33486, 0, 4, 0, 2, 1, 22.66),
            (11343, 0, 4, 30, 24, 0, 10.16),
            (40271, 0, 4, 13, 3, 1, 41.41),
        ]
    )
    names = ('toa', 'uftoa_start', 'uftoa_stop', 'ftoa_rise', 'ftoa_fall', 'tot')
    events = {name: rows[:, k].astype(np.int64) for k, name in enumerate(names)}
    tot_ns = fine_tot(events) * 25 / fine_steps
    assert np.allclose(tot_ns, rows[:, 6], atol=0.005)


def test_fine_toa_decode():
    events = {
        'toa': np.array([10, 10, 10, 10, 10]),
        'y': np.array([255, 256, 0, 511, 255]),
        'ftoa_rise': np.array([3, 3, 3, 3, 25]),
        'uftoa_start': np.array([0, 0, 0, 0, 0]),
        'uftoa_stop': np.array([5, 5, 5, 5, 5]),
    }
    # 128 * toa - 8 * ftoa + (start - stop) - 4 * (15 - SPGroup)
    expected = [1280 - 24 - 5, 1280 - 24 - 5, 1280 - 24 - 5 - 60, 1280 - 24 - 5 - 60]
    assert fine_toa(events)[:4].tolist() == expected
    assert fine_toa(events)[4] == 1280 - 8 * 9 - 5  # ftoa 25 is read as 9
