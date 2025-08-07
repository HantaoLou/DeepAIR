"""Utilities to load antibody-antigen complex data."""

import csv
import random

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}


def encode_sequence(seq):
    """Encode an amino acid sequence into a list of integers."""
    return [AA_TO_ID.get(aa, 0) for aa in seq]


def load_antibody_antigen_dataset(csv_path):
    """Load antibody-antigen complex dataset from a CSV file.

    The CSV file must contain three columns: ``antibody``, ``antigen`` and
    ``label``. Sequences are encoded into integer lists.

    Returns
    -------
    tuple of lists
        A tuple ``(antibodies, antigens, labels)`` where each element is a
        list with one entry per example.
    """
    antibodies, antigens, labels = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            antibodies.append(encode_sequence(row["antibody"]))
            antigens.append(encode_sequence(row["antigen"]))
            labels.append(float(row["label"]))
    return antibodies, antigens, labels


def batch_generator(antibodies, antigens, labels, batch_size=32, shuffle=True):
    """Yield mini-batches of the dataset.

    Parameters
    ----------
    antibodies, antigens, labels : list
        Lists returned from :func:`load_antibody_antigen_dataset`.
    batch_size : int, default ``32``
        Number of examples per batch.
    shuffle : bool, default ``True``
        Whether to shuffle the data before batching.

    Yields
    ------
    tuple
        ``(antibody_batch, antigen_batch, label_batch)`` where each element is
        a list of examples for the current batch.
    """
    idx = list(range(len(labels)))
    if shuffle:
        random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield [antibodies[i] for i in batch_idx], [antigens[i] for i in batch_idx], [labels[i] for i in batch_idx]
