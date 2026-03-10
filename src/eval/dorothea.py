from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_dorothea(path: str | Path, confidence_levels: Iterable[str] | None = None):
    df = pd.read_csv(path, sep="\t")
    columns = {col.lower(): col for col in df.columns}

    source_col = None
    target_col = None
    if "tf" in columns and "target" in columns:
        source_col = columns["tf"]
        target_col = columns["target"]
    elif "source" in columns and "target" in columns:
        source_col = columns["source"]
        target_col = columns["target"]

    if source_col is None or target_col is None:
        df = pd.read_csv(path, sep="\t", header=None)
        if df.shape[1] < 2:
            raise ValueError("DoRothEA file must include at least two columns")
        source_col = 0
        target_col = 1
        columns = {}

    if confidence_levels is not None and "confidence" in columns:
        df = df[df[columns["confidence"]].isin(confidence_levels)]

    edges = df[[source_col, target_col]].dropna().copy()
    edges.columns = ["source", "target"]
    edges = edges.drop_duplicates()
    return edges
