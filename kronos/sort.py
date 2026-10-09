"""Sort events from many files by toa, in bounded memory.

merge_sort(files) yields parts (see kronos.events) in toa order, in one sequential read of
files that are each nearly sorted.
"""

import numba
import numpy as np

from kronos.events import concat, event_columns, read_chunks, take


def merge_sort(files, max_disorder=1 << 16, chunk_rows=200_000, part_rows=2_000_000, columns=event_columns):
    """Yield toa-sorted parts from files that are each sorted up to max_disorder.

    Files are read in step, always advancing the one furthest behind in time.
    No unread event can be older than safe = (slowest file's newest toa -
    max_disorder), so every buffered event older than that is sorted and
    yielded. Raises ValueError if an event is older than an already yielded
    part; then raise max_disorder, which only costs memory. A file's disorder
    is its largest step back in toa: max(running max of toa - toa).

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
        else:
            if int(chunk['toa'].min()) < done_until:
                raise ValueError(f'{path} jumps back more than {max_disorder} ticks: raise max_disorder')
            newest[path] = max(newest[path], int(chunk['toa'].max()))
            buffer.append(chunk)
            n_buffered += len(chunk['toa'])
        if buffer and (n_buffered >= flush_rows or not chunks):
            events = concat(buffer)
            if chunks:
                safe = max(done_until, min(newest[p] for p in chunks) - max_disorder)
            else:
                safe = int(events['toa'].max()) + 1
            older = events['toa'] < safe
            part = take(events, older)
            buffer = [take(events, ~older)]
            n_buffered = len(buffer[0]['toa'])
            del events, older  # free the merge buffer while the part is in use
            if len(part['toa']):
                done_until = safe
                yield take(part, time_order(part['toa']))


def time_order(toa):
    """Indices that sort toa (stable): counting sort when the toa span is small,
    as within a part, argsort otherwise."""
    t = np.asarray(toa).astype(np.int64)
    first, span = t.min(), t.max() - t.min()
    if span > 4 * len(t) + 1_000_000:
        return np.argsort(t, kind='stable')
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
