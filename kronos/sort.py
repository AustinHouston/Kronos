"""Sort events from many files by toa, in bounded memory.

Both sorters yield parts (see kronos.events) in toa order:

    merge_sort(files)                 # near-sorted files: one sequential read
    bucket_sort(files, tmp_directory) # any order: slower, uses temp files
"""

import numba
import numpy as np
import pandas as pd

from kronos.events import EVENT_COLUMNS, concat, new_directory, read_chunks, take

BIN_SHIFT = 16  # bucket_sort plans parts in coarse bins of 2**16 toa ticks


def merge_sort(
    files,
    max_disorder=1 << 16,
    chunk_rows=200_000,
    part_rows=2_000_000,
    columns=EVENT_COLUMNS,
):
    """Yield toa-sorted parts from files that are each sorted up to max_disorder.

    Files are read in step, always advancing the one furthest behind in time.
    No unread event can be older than safe = (slowest file's newest toa -
    max_disorder), so every buffered event older than that is sorted and
    yielded. Raises ValueError if an event is older than an already yielded
    part; then raise max_disorder or use bucket_sort. A file's disorder is its
    largest step back in toa: max(running max of toa - toa).

    Peak memory with clustering is ~200 bytes * (part_rows + len(files) *
    chunk_rows): ~1.4 GB for 32 files at the defaults.
    """
    chunks = {path: read_chunks(path, columns, chunk_rows) for path in files}
    newest = dict.fromkeys(files, -1)
    flush_rows = part_rows + len(files) * chunk_rows
    buffer, n_buffered, done_until = [], 0, 0
    while chunks:
        path = min(chunks, key=newest.get)
        chunk = next(chunks[path], None)
        if chunk is None:
            del chunks[path]
        elif len(chunk["toa"]):
            if int(chunk["toa"].min()) < done_until:
                raise ValueError(f"{path} jumps back more than {max_disorder} ticks")
            newest[path] = max(newest[path], int(chunk["toa"].max()))
            buffer.append(chunk)
            n_buffered += len(chunk["toa"])
        if buffer and (n_buffered >= flush_rows or not chunks):
            events = concat(buffer)
            if chunks:
                safe = max(done_until, min(newest[p] for p in chunks) - max_disorder)
            else:
                safe = int(events["toa"].max()) + 1
            older = events["toa"] < safe
            part = take(events, older)
            buffer = [take(events, ~older)]
            n_buffered = len(buffer[0]["toa"])
            del events, older  # free the merge buffer while the part is in use
            if len(part["toa"]):
                done_until = safe
                yield take(part, time_order(part["toa"]))


def bucket_sort(
    files,
    tmp_directory,
    part_rows=2_000_000,
    chunk_rows=10_000_000,
    columns=EVENT_COLUMNS,
):
    """Yield toa-sorted parts from files in any order, using temp files."""
    # 1. Count events per coarse time bin (toa only) and choose part edges so
    #    each part holds at most part_rows events (a bigger bin is its own part).
    counts = []
    for path in files:
        for chunk in read_chunks(path, ("toa",), chunk_rows):
            bins, n_events = np.unique(chunk["toa"] >> BIN_SHIFT, return_counts=True)
            counts.append(pd.Series(n_events, index=bins))
    counts = pd.concat(counts).groupby(level=0).sum()
    edges, in_part = [counts.index[0]], 0
    for coarse_bin, n_events in counts.items():
        if in_part and in_part + n_events > part_rows:
            edges.append(coarse_bin)
            in_part = 0
        in_part += n_events
    edges.append(counts.index[-1] + 1)
    edges = np.asarray(edges, dtype=np.uint64) << np.uint64(BIN_SHIFT)

    # 2. Append each chunk's events to the temp file of their part.
    tmp_directory = new_directory(tmp_directory)
    buckets = [tmp_directory / f"{k:05d}.bin" for k in range(len(edges) - 1)]
    for path in files:
        for chunk in read_chunks(path, columns, chunk_rows):
            records = np.rec.fromarrays(
                [chunk[name] for name in columns], names=columns
            )
            bucket_of = np.searchsorted(edges, records["toa"], "right") - 1
            order = np.argsort(bucket_of, kind="stable")
            bounds = np.searchsorted(bucket_of[order], np.arange(len(buckets) + 1))
            for k in np.flatnonzero(np.diff(bounds)):
                with open(buckets[k], "ab") as handle:
                    records[order[bounds[k] : bounds[k + 1]]].tofile(handle)

    # 3. Load, sort and yield one temp file at a time.
    for bucket in buckets:
        if bucket.exists():
            records = np.fromfile(bucket, dtype=records.dtype)
            bucket.unlink()
            part = {name: records[name] for name in columns}
            yield take(part, time_order(part["toa"]))
    tmp_directory.rmdir()


def time_order(toa):
    """Indices that sort toa (stable): counting sort when the toa span is small,
    as within a part, argsort otherwise."""
    t = np.asarray(toa).astype(np.int64)
    if len(t) == 0:
        return np.empty(0, dtype=np.int64)
    first, span = t.min(), t.max() - t.min()
    if span > 4 * len(t) + 1_000_000:
        return np.argsort(t, kind="stable")
    return _counting_order(t, first, span)


@numba.njit(cache=True)
def _counting_order(t, first, span):
    starts = np.zeros(span + 2, dtype=np.int64)
    for value in t:
        starts[value - first + 1] += 1
    for k in range(1, span + 2):
        starts[k] += starts[k - 1]
    order = np.empty(len(t), dtype=np.int64)
    for i in range(len(t)):
        k = t[i] - first
        order[starts[k]] = i
        starts[k] += 1
    return order
