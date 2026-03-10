from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
import json


SPECIAL_TOKENS = ["<pad>", "[PAD]", "<PAD>", "PAD"]


@dataclass
class Vocab:
    gene_to_id: Dict[str, int]
    id_to_gene: Dict[int, str]
    pad_token: str | None = None
    pad_id: int | None = None


def load_vocab(path: str | Path) -> Vocab:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        gene_to_id = _parse_json_vocab(data)
    else:
        gene_to_id = _parse_text_vocab(path)

    id_to_gene = {idx: gene for gene, idx in gene_to_id.items()}
    pad_token, pad_id = _infer_pad_token(gene_to_id)
    return Vocab(gene_to_id=gene_to_id, id_to_gene=id_to_gene, pad_token=pad_token, pad_id=pad_id)


def _parse_json_vocab(data) -> Dict[str, int]:
    if isinstance(data, list):
        return {gene: idx for idx, gene in enumerate(data)}
    if isinstance(data, dict):
        if "stoi" in data and isinstance(data["stoi"], dict):
            return {gene: int(idx) for gene, idx in data["stoi"].items()}
        if "itos" in data and isinstance(data["itos"], list):
            return {gene: idx for idx, gene in enumerate(data["itos"])}
        if all(isinstance(value, int) for value in data.values()):
            return {gene: int(idx) for gene, idx in data.items()}
    raise ValueError("Unsupported vocab JSON format")


def _parse_text_vocab(path: Path) -> Dict[str, int]:
    genes = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            gene = line.strip()
            if gene:
                genes.append(gene)
    return {gene: idx for idx, gene in enumerate(genes)}


def _infer_pad_token(gene_to_id: Dict[str, int]):
    for token in SPECIAL_TOKENS:
        if token in gene_to_id:
            return token, gene_to_id[token]
    return None, None


def map_genes_to_vocab(genes: Iterable[str], vocab: Vocab):
    mapped = []
    missing = []
    for gene in genes:
        if gene in vocab.gene_to_id:
            mapped.append(vocab.gene_to_id[gene])
        else:
            missing.append(gene)
    return mapped, missing
