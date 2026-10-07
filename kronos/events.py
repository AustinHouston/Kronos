"""Event data: raw HDF5 files in, sorted *parts* out.

An event is one pixel record (x, y, toa, tot). A *part* is a dict of numpy
columns holding events sorted by toa; consecutive parts cover consecutive,
non-overlapping toa ranges. Sorters yield parts, clustering consumes them.
"""

from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EVENT_COLUMNS = ("x", "y", "toa", "tot")


def read_chunks(path, columns=EVENT_COLUMNS, chunk_rows=200_000):
    """Yield consecutive row ranges of one HDF5 file as dicts of columns."""
    with h5py.File(path, "r") as h5:
        for start in range(0, len(h5["toa"]), chunk_rows):
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
        pq.write_table(
            pa.table(part),
            directory / f"part-{k:05d}.parquet",
            row_group_size=1_000_000,
        )
    return directory


def load_parts(directory, columns=EVENT_COLUMNS, batch_rows=2_000_000):
    """Yield parts saved by save_parts, in toa order."""
    for path in sorted(Path(directory).glob("part-*.parquet")):
        batches = pq.ParquetFile(path).iter_batches(batch_rows, columns=list(columns))
        for batch in batches:
            yield {name: batch.column(name).to_numpy() for name in columns}


def new_directory(directory):
    """Create an output directory, refusing to overwrite an existing one."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Choose a new output directory: {directory}")
    directory.mkdir(parents=True)
    return directory
