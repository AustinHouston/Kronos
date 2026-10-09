# Kronos

Sort and cluster TimePix electron-microscopy event data that is far too large for memory.

Each raw **event** is one pixel record: position `x, y`, time of arrival `toa` (25 ns ticks)
and time over threshold `tot`. One electron usually fires a few neighboring pixels, so events
are grouped into **clusters**, about one per electron.

## Quick start

```bash
uv sync          # installs dependencies and the kronos package
uv run pytest    # synthetic-data tests, a few seconds
```

```python
from pathlib import Path

from kronos.cluster import cluster_parts
from kronos.sort import merge_sort

files = sorted(Path('/Volumes/Extreme SSD/Timepix/MoS2_Kai_5b').glob('TestSeq1_1k_*.h5'))
cluster_parts(merge_sort(files), 'outputs/TestSeq1_1k/clusters', dt_max=1)
```

The walkthrough is `notebooks/01_sort_and_cluster.ipynb`. Why clusters join edge-neighboring
pixels within `dt_max = 1` tick, and how fast clustering is, is shown in
`notebooks/02_cluster_parameters.ipynb`.

For the detector-level background behind the event fields, read
`docs/TimePixForDummies.md` and run `notebooks/05_TimePixForDummies.ipynb`.

## How it works

Everything streams **parts**: dicts of numpy columns holding events sorted by `toa`,
covering consecutive, non-overlapping time ranges.

1. **Sort** (`kronos.sort`). `merge_sort` reads every file once, front to back, all files
   in step, and counting-sorts everything older than the slowest file's newest `toa` minus
   `max_disorder`. It raises if a file is more jumbled than that; `bucket_sort` handles any
   order, at the cost of temp files and extra passes.
2. **Cluster** (`kronos.cluster`). `cluster_parts` labels each part with a union-find over a
   pixel grid, one time slice per core (`label_events`), writes one summary row per cluster
   (`summarize_clusters`), and carries clusters still open at a part's end into the next part.
   `cluster_tables` does the same but yields the tables, with any per-cluster summary you pass.
3. **Save / load** (`kronos.events`). `save_parts` and `load_parts` keep sorted events on disk
   as Parquet when you want them for other analyses.

## Layout

```
kronos/
    events.py    read HDF5 chunks, part helpers, save/load parts
    sort.py      merge_sort, bucket_sort, time_order
    cluster.py   cluster_parts, cluster_tables, label_events, summarize_clusters, neighbor_time_gaps
tests/           synthetic HDF5 data with known electrons
notebooks/       numbered walkthroughs; plot_style.py is their shared plot style
outputs/         results, git-ignored
```
