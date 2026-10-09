"""Event data: raw HDF5 files in, sorted *parts* out.

An event is one pixel record (x, y, toa, tot). A *part* is a dict of numpy
columns holding events sorted by toa; consecutive parts cover consecutive,
non-overlapping toa ranges. Sorters yield parts, clustering consumes them.

fine_toa and fine_tot decode the Timepix4 sub-tick timing; read the extra
columns with columns=event_columns + fine_columns.
"""

from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

event_columns = ('x', 'y', 'toa', 'tot')
fine_columns = ('ftoa_rise', 'ftoa_fall', 'uftoa_start', 'uftoa_stop')

fine_steps = 128  # fine time units per 25 ns toa tick: 195.3125 ps each
half_rows = 256  # rows per chip half; the top half's address map is flipped


def read_chunks(path, columns=event_columns, chunk_rows=200_000):
    """Yield consecutive row ranges of one HDF5 file as dicts of columns."""
    with h5py.File(path, 'r') as h5:
        for start in range(0, len(h5['toa']), chunk_rows):
            yield {name: h5[name][start : start + chunk_rows] for name in columns}


def take(part, index):
    """The rows of a part selected by a mask, slice or index array."""
    return {name: values[index] for name, values in part.items()}


def concat(parts):
    """Stack parts with the same columns into one."""
    return {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}


def save_parts(parts, directory):
    """Write each part as directory/part-#####.parquet."""
    directory = new_directory(directory)
    for k, part in enumerate(parts):
        pq.write_table(pa.table(part), directory / f'part-{k:05d}.parquet', row_group_size=1_000_000)
    return directory


def load_parts(directory, columns=event_columns, batch_rows=2_000_000):
    """Yield parts saved by save_parts, in toa order."""
    for path in sorted(Path(directory).glob('part-*.parquet')):
        batches = pq.ParquetFile(path).iter_batches(batch_rows, columns=list(columns))
        for batch in batches:
            yield {name: batch.column(name).to_numpy() for name in columns}


def new_directory(directory):
    """Create an output directory, refusing to overwrite an existing one."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f'Choose a new output directory: {directory}')
    directory.mkdir(parents=True)
    return directory


def fine_toa(events):
    """Time of arrival in fine units (1/128 tick, 195.3125 ps), as int64.

    Timepix4 manual, Listing 21.2 (decode_toa minus toa_clkdll_correction):

        toa - ftoa_rise/16 + (uftoa_start - uftoa_stop)/128 - (15 - SPGroup)/32

    in ticks, where SPGroup is the 16-row group counted from the chip's edge
    (y // 16 in the bottom half, 15 - (y - 256) // 16 in the top half): the
    column clock reaches rows near the edge last. uftoa values must already be
    thermometer-decoded to bins 0..7, as they are in the HDF5 files.
    """
    toa = np.asarray(events['toa'], np.int64)
    y = np.asarray(events['y'], np.int64)
    ftoa = _ftoa_count(events['ftoa_rise'])
    uftoa = np.asarray(events['uftoa_start'], np.int64) - events['uftoa_stop']
    spgroup = np.where(y < half_rows, y // 16, 15 - (y - half_rows) // 16)
    return fine_steps * toa - 8 * ftoa + uftoa - 4 * (15 - spgroup)


def fine_tot(events):
    """Time over threshold in fine units (1/128 tick, 195.3125 ps), as int64.

    Timepix4 manual, Listing 21.2 (decode_tot), in ticks:

        tot + (ftoa_rise - ftoa_fall)/16 - (uftoa_start - uftoa_stop)/128

    It reproduces the TOT_CALCULATED column of the manual's Table 24.2.
    Pixels with tot = 0 can come out slightly negative.
    """
    tot = np.asarray(events['tot'], np.int64)
    ftoa = _ftoa_count(events['ftoa_rise']) - _ftoa_count(events['ftoa_fall'])
    uftoa = np.asarray(events['uftoa_start'], np.int64) - events['uftoa_stop']
    return fine_steps * tot + 8 * ftoa - uftoa


def _ftoa_count(ftoa):
    """fToA as a count of 1.5625 ns cycles, 0..16.

    Values 17..31 occur almost only in tot = 0 packets (manual section 24.4);
    read as value - 16 they put those pixels in time with the rest of their
    cluster (notebooks/02_cluster_parameters.ipynb, section 6).
    """
    ftoa = np.asarray(ftoa, np.int64)
    return np.where(ftoa > 16, ftoa - 16, ftoa)
