"""Lazy HDF5 event batches and a memory-managed time sort."""
from pathlib import Path

import dask
import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
from dask import delayed
from distributed import Client, LocalCluster


def read_batch(filename, start, stop, fields, file_id):
    """Read one batch, opening the file inside the worker process."""
    with h5py.File(filename, "r") as h5:
        columns = {name: h5[name][start:stop] for name in fields}
    columns["file_id"] = np.full(stop - start, file_id, dtype=np.uint32)
    columns["event_id"] = np.arange(start, stop, dtype=np.uint64)
    return pd.DataFrame(columns)


def build_events(directory, batch_rows=100_000, fields=("x", "y", "toa", "tot")):
    """Build lazy partitions; fields=None selects aligned event columns.

    Inputs must share a schema. Separate control/timestamp streams are excluded.
    """
    paths = sorted(p.resolve() for p in Path(directory).iterdir()
                   if p.suffix.lower() in (".h5", ".hdf5"))
    with h5py.File(paths[0], "r") as h5:
        if fields is None:
            fields = tuple(name for name, obj in h5.items()
                           if isinstance(obj, h5py.Dataset)
                           and obj.ndim == 1 and obj.shape == h5["toa"].shape)
        meta = pd.DataFrame({name: pd.Series(dtype=h5[name].dtype)
                             for name in fields})
    meta["file_id"] = pd.Series(dtype="uint32")
    meta["event_id"] = pd.Series(dtype="uint64")

    batches, records = [], []
    for file_id, filename in enumerate(paths):
        with h5py.File(filename, "r") as h5:
            count = len(h5["toa"])
        records.append({"file_id": file_id, "filename": str(filename), "events": count})
        batches.extend(delayed(read_batch)(str(filename), start,
                       min(start + batch_rows, count), fields, file_id)
                       for start in range(0, count, batch_rows))
    events = (dd.from_delayed(batches, meta=meta) if batches
              else dd.from_pandas(meta, npartitions=1))
    return events, pd.DataFrame(records)


def sort_to_parquet(events, manifest, output, spill_directory,
                    worker_memory="2GiB", output_partitions=None):
    """Sort by toa and write Parquet with one worker and disk spilling.

    Memory thresholds are safeguards, not a hard bound on a single task.
    Close write-mode HDF5 handles before calling this function.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    spill_directory = Path(spill_directory).resolve()
    spill_directory.mkdir(parents=True, exist_ok=True)
    memory_settings = {
        "distributed.worker.memory.target": 0.50,
        "distributed.worker.memory.spill": 0.60,
        "distributed.worker.memory.pause": 0.75,
        "distributed.worker.memory.terminate": 0.90,
    }
    with dask.config.set(memory_settings), LocalCluster(
        n_workers=1, threads_per_worker=1, processes=True,
        memory_limit=worker_memory, local_directory=str(spill_directory),
        dashboard_address=None,
    ) as cluster, Client(cluster):
        ordered = events.set_index(
            "toa", shuffle_method="p2p",
            npartitions=output_partitions or events.npartitions,
        )
        ordered.to_parquet(str(output), engine="pyarrow", write_index=True,
                           overwrite=False)
    manifest.to_json(output / "_source_manifest.json", orient="records", indent=2)
    return dd.read_parquet(str(output), engine="pyarrow")
