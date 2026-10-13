"""Helpers shared by the clustering figures."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def align_to_reference(labels, reference):
    """Relabel a partition to best match the reference on their common bins (Hungarian).

    The overlap is scored over each partition's full label universe (counts taken
    on the common bins) so the assignment is a complete bijection — a cluster with
    no shared bins still receives a distinct reference colour rather than a spare id.
    """
    common = labels.index.intersection(reference.index)
    lab_c = labels.loc[common].to_numpy()
    ref_c = reference.loc[common].to_numpy()
    lab_ids = np.unique(labels.to_numpy())
    ref_ids = np.unique(reference.to_numpy())
    overlap = np.zeros((len(lab_ids), len(ref_ids)), dtype=int)
    for i, l in enumerate(lab_ids):
        for j, r in enumerate(ref_ids):
            overlap[i, j] = np.sum((lab_c == l) & (ref_c == r))

    rows, cols = linear_sum_assignment(-overlap)
    mapping = {lab_ids[i]: ref_ids[j] for i, j in zip(rows, cols)}

    next_id = int(ref_ids.max()) + 1
    for l in lab_ids:
        if l not in mapping:
            mapping[l] = next_id
            next_id += 1
    return labels.map(mapping)
