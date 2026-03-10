from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

import numpy as np
import pandas as pd


def edge_set(edges: pd.DataFrame) -> Set[Tuple[str, str]]:
    return {(row["source"], row["target"]) for _, row in edges.iterrows()}


def precision_recall_f1(pred_edges: pd.DataFrame, true_edges: pd.DataFrame) -> Dict[str, float]:
    pred = edge_set(pred_edges)
    truth = edge_set(true_edges)

    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    total_pos = labels.sum()
    if total_pos == 0:
        return 0.0

    for label in labels:
        if label:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_pos)

    return np.trapz(precisions, recalls)
