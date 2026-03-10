from __future__ import annotations

from pathlib import Path

from .preprocess import load_h5ad


def load_tabula_sapiens(path: str | Path):
    return load_h5ad(path)
