"""Read aligned HDF5 event columns as lazy DataFrame partitions."""
from pathlib import Path

import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
from dask import delayed


def read_batch(filename, start, stop, fields, file_id):
    # Open inside the task: no live HDF5 handles cross process boundaries.
    try:
        with h5py.File(filename, 'r') as h5:
            columns = {name: h5[name][start:stop] for name in fields}
    except BlockingIOError as exc:
        raise BlockingIOError(
            f'Cannot read {filename}: an HDF5 writer may hold a lock. '
            'Close the notebook NSIDReader with reader.close() and any '
            'other write-mode HDF5 handles before starting Dask.'
        ) from exc
    columns['file_id'] = np.full(stop - start, file_id, dtype=np.uint32)
    columns['event_id'] = np.arange(start, stop, dtype=np.uint64)
    return pd.DataFrame(columns)


def build_events(directory, batch_rows=100_000, fields=('x', 'y', 'toa', 'tot')):
    """Inspect shapes/dtypes only, then create deferred batch reads.

    fields=None includes all root-level 1D datasets aligned with toa in the
    first file. Empty timestamp/control arrays are separate streams, excluded.
    Every file must have the same selected schema. file_id maps to manifest.
    """
    if batch_rows <= 0:
        raise ValueError('batch_rows must be positive')
    paths = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in ('.h5', '.hdf5'))
    if not paths:
        raise ValueError('No HDF5 files found')
    parts, records, schema = [], [], None
    for file_id, filename in enumerate(paths):
        with h5py.File(filename, 'r') as h5:
            n = h5['toa'].shape[0]
            if fields is None:
                fields = tuple(name for name, obj in h5.items()
                               if isinstance(obj, h5py.Dataset)
                               and obj.ndim == 1 and obj.shape == (n,))
            current = {}
            for name in fields:
                obj = h5[name]
                if obj.ndim != 1 or obj.shape != (n,):
                    raise ValueError(f'{filename}: {name} is not aligned with toa')
                current[name] = obj.dtype
            if schema is None:
                schema = current
            elif schema != current:
                raise ValueError(f'{filename}: inconsistent column dtypes')
        records.append({'file_id': file_id, 'filename': str(filename), 'events': n})
        for start in range(0, n, batch_rows):
            parts.append(delayed(read_batch)(str(filename), start,
                         min(start + batch_rows, n), fields, file_id))
    meta = pd.DataFrame({name: pd.Series(dtype=dtype)
                         for name, dtype in schema.items()})
    meta['file_id'] = pd.Series(dtype='uint32')
    meta['event_id'] = pd.Series(dtype='uint64')
    if not parts:
        events = dd.from_pandas(meta, npartitions=1)
    else:
        events = dd.from_delayed(parts, meta=meta)
    return events, pd.DataFrame(records)


def sort_to_parquet(events, manifest, output, spill_directory,
                    worker_memory='2GiB', output_partitions=None):
    """Execute a time-range shuffle on one memory-managed worker process.

    Spill/pause/restart are safeguards, not a hard bound on an individual
    pandas sort. Repeated timestamps can produce oversized partitions.
    """
    import dask
    from distributed import Client, LocalCluster

    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    spill_directory = Path(spill_directory).resolve()
    spill_directory.mkdir(parents=True, exist_ok=True)
    if output_partitions is None:
        output_partitions = events.npartitions
    with dask.config.set({'distributed.worker.memory.target': 0.50,
                          'distributed.worker.memory.spill': 0.60,
                          'distributed.worker.memory.pause': 0.75,
                          'distributed.worker.memory.terminate': 0.90}):
        with LocalCluster(n_workers=1, threads_per_worker=1, processes=True,
                          memory_limit=worker_memory,
                          local_directory=str(spill_directory),
                          dashboard_address=None) as cluster:
            with Client(cluster):
                ordered = events.set_index('toa', shuffle_method='p2p',
                                           npartitions=output_partitions)
                ordered.to_parquet(str(output), engine='pyarrow',
                                   write_index=True, overwrite=False)
    manifest.to_json(output / '_source_manifest.json', orient='records', indent=2)
    return dd.read_parquet(str(output), engine='pyarrow')
