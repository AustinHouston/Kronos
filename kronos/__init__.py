"""Kronos: sort and cluster TimePix event data that does not fit in memory.

    from kronos.cluster import cluster_parts
    from kronos.sort import merge_sort

    files = sorted(Path(directory).glob("TestSeq1_1k_*.h5"))
    cluster_parts(merge_sort(files), "outputs/clusters", dt_max=2)

Modules: kronos.events (reading files, parts), kronos.sort, kronos.cluster.
"""
