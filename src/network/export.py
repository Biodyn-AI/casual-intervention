from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


def export_edges_tsv(edges: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(path, sep="\t", index=False)


def export_graphml(edges: pd.DataFrame, path: str | Path):
    graph = nx.DiGraph()
    for _, row in edges.iterrows():
        graph.add_edge(row["source"], row["target"], weight=row["score"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)
