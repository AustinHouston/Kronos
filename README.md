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

The notebooks, each a self-contained story:

1. `notebooks/01_Sorting_Events.ipynb`: how the event files are laid out, how the fine time is
   decoded, and how `merge_sort` turns shuffled files into one time-ordered stream, feeding
   clustering in the same pass.
2. `notebooks/02_Clustering_Choices.ipynb`: how clusters are found, and why they join
   edge-neighboring pixels within `dt_max = 1` tick (accuracy and speed of the alternatives).
3. `notebooks/03_Cluster_Centroids.ipynb`: where in its cluster the electron landed, from a
   simulation of electrons in silicon fitted to the real clusters.
4. `notebooks/04_connecting_timing.ipynb`: how the scan generator's line pulses, timestamped on the
   detector clock, turn clusters into STEM images: line, dwell and flyback measured from the data,
   and a darkfield image with virtual detectors chosen after a single pass.
5. `notebooks/05_Moire_Analysis.ipynb`: the same run as a 4D-STEM dataset: the time bin per diffraction
   pattern (4 × 4 probe positions), one pass into a 4D cube, zero-beam alignment on the disk edge, NMF of the
   zero beam and of the scattered electrons across the moiré, and probe deconvolution of the position-averaged
   pattern, which shows two crystals rotated by ~15°, as in the image FFT.

## How it works

Everything streams **parts**: dicts of numpy columns holding events sorted by `toa`,
covering consecutive, non-overlapping time ranges.

1. **Sort** (`kronos.sort`). `merge_sort` reads every file once, front to back, all files
   in step, and counting-sorts everything older than the slowest file's newest `toa` minus
   `max_disorder`. It raises if a file is more jumbled than that; a larger `max_disorder` only
   costs memory.
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
    sort.py      merge_sort, time_order
    cluster.py   cluster_parts, cluster_tables, label_events, summarize_clusters, neighbor_time_gaps
tests/           synthetic HDF5 data with known electrons
notebooks/       numbered walkthroughs; plot_style.py is their shared plot style
outputs/         results, git-ignored
```
