from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def normalize_symbol(symbol: str | None) -> str:
    if symbol is None:
        return ""
    sym = str(symbol).strip()
    if not sym:
        return ""
    sym = sym.replace(" ", "").replace("\t", "")
    sym = sym.strip().strip(".")
    sym = sym.upper()
    if sym.startswith("ENSG") and "." in sym:
        sym = sym.split(".", 1)[0]
    return sym


def _split_aliases(value: str) -> List[str]:
    if not value:
        return []
    parts = value.replace("|", ",").split(",")
    return [part.strip() for part in parts if part and part.strip()]


def load_hgnc_alias_map(path: str | Path | None) -> Dict[str, str]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}

    df = pd.read_csv(path, sep="\t", dtype=str)
    symbol_col = None
    for candidate in (
        "symbol",
        "app_sym",
        "hgnc_symbol",
        "approved_symbol",
        "Approved symbol",
    ):
        if candidate in df.columns:
            symbol_col = candidate
            break
    if symbol_col is None:
        return {}

    alias_cols = []
    for candidate in (
        "alias_symbol",
        "alias_symbols",
        "prev_symbol",
        "previous_symbol",
        "Alias symbols",
        "Previous symbols",
    ):
        if candidate in df.columns:
            alias_cols.append(candidate)

    alias_map: Dict[str, str] = {}

    def add_alias(alias: str, symbol: str) -> None:
        alias_norm = normalize_symbol(alias)
        symbol_norm = normalize_symbol(symbol)
        if alias_norm and symbol_norm and alias_norm not in alias_map:
            alias_map[alias_norm] = symbol_norm

    for _, row in df.iterrows():
        symbol = row.get(symbol_col)
        if not isinstance(symbol, str) or not symbol:
            continue
        symbol_norm = normalize_symbol(symbol)
        if symbol_norm and symbol_norm not in alias_map:
            alias_map[symbol_norm] = symbol_norm
        for col in alias_cols:
            aliases = row.get(col)
            if isinstance(aliases, str) and aliases:
                for alias in _split_aliases(aliases):
                    add_alias(alias, symbol)

    return alias_map


def build_hgnc_alias_index(path: str | Path | None) -> Dict[str, List[str]]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}

    df = pd.read_csv(path, sep="\t", dtype=str)
    symbol_col = None
    for candidate in (
        "symbol",
        "app_sym",
        "hgnc_symbol",
        "approved_symbol",
        "Approved symbol",
    ):
        if candidate in df.columns:
            symbol_col = candidate
            break
    if symbol_col is None:
        return {}

    alias_cols = []
    for candidate in (
        "alias_symbol",
        "alias_symbols",
        "prev_symbol",
        "previous_symbol",
        "Alias symbols",
        "Previous symbols",
    ):
        if candidate in df.columns:
            alias_cols.append(candidate)

    alias_index: Dict[str, set[str]] = {}

    def add_alias(alias: str, symbol: str) -> None:
        alias_norm = normalize_symbol(alias)
        symbol_norm = normalize_symbol(symbol)
        if alias_norm and symbol_norm:
            alias_index.setdefault(alias_norm, set()).add(symbol_norm)

    for _, row in df.iterrows():
        symbol = row.get(symbol_col)
        if not isinstance(symbol, str) or not symbol:
            continue
        add_alias(symbol, symbol)
        for col in alias_cols:
            aliases = row.get(col)
            if isinstance(aliases, str) and aliases:
                for alias in _split_aliases(aliases):
                    add_alias(alias, symbol)

    return {alias: sorted(list(symbols)) for alias, symbols in alias_index.items()}


def build_hgnc_alias_map(
    alias_index: Dict[str, List[str]],
    ambiguous_policy: str = "lexicographic",
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    alias_map: Dict[str, str] = {}
    ambiguous: Dict[str, List[str]] = {}

    for alias, symbols in alias_index.items():
        if not symbols:
            continue
        if len(symbols) == 1:
            alias_map[alias] = symbols[0]
            continue
        ambiguous[alias] = symbols
        if ambiguous_policy == "drop":
            continue
        if ambiguous_policy == "lexicographic":
            alias_map[alias] = sorted(symbols)[0]
            continue
        raise ValueError(f"Unknown ambiguous_policy: {ambiguous_policy}")

    return alias_map, ambiguous


def load_ensembl_map(path: str | Path | None) -> Dict[str, str]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["feature_id", "feature_name"], dtype=str)
    df = df.dropna(subset=["feature_id", "feature_name"])
    mapping = dict(zip(df["feature_id"].astype(str), df["feature_name"].astype(str)))
    return {normalize_symbol(key): normalize_symbol(value) for key, value in mapping.items()}


@dataclass
class SymbolMappingResult:
    symbol: str
    normalized: str
    mapped: str
    status: str
    ambiguous_options: str
    ensembl_mapped: bool


class SymbolMapper:
    def __init__(
        self,
        hgnc_alias_tsv: str | Path | None = None,
        gene_info_csv: str | Path | None = None,
        ambiguous_policy: str = "lexicographic",
        drop_unmapped: bool = False,
    ) -> None:
        alias_index = build_hgnc_alias_index(hgnc_alias_tsv)
        self.alias_map, self.ambiguous = build_hgnc_alias_map(alias_index, ambiguous_policy)
        self.ensembl_map = load_ensembl_map(gene_info_csv)
        self.ambiguous_policy = ambiguous_policy
        self.drop_unmapped = drop_unmapped

    def map_symbol(self, symbol: str | None) -> SymbolMappingResult:
        raw = "" if symbol is None else str(symbol)
        normalized = normalize_symbol(raw)
        if not normalized:
            return SymbolMappingResult(
                symbol=raw,
                normalized=normalized,
                mapped="",
                status="empty",
                ambiguous_options="",
                ensembl_mapped=False,
            )

        ensembl_mapped = False
        mapped = normalized
        if normalized in self.ensembl_map:
            mapped = normalize_symbol(self.ensembl_map[normalized])
            ensembl_mapped = True

        ambiguous_options = ""
        if mapped in self.ambiguous:
            ambiguous_options = "|".join(self.ambiguous[mapped])
            if self.ambiguous_policy == "drop":
                return SymbolMappingResult(
                    symbol=raw,
                    normalized=normalized,
                    mapped="",
                    status="ambiguous_dropped",
                    ambiguous_options=ambiguous_options,
                    ensembl_mapped=ensembl_mapped,
                )
            mapped = self.alias_map.get(mapped, mapped)
            return SymbolMappingResult(
                symbol=raw,
                normalized=normalized,
                mapped=mapped,
                status="ambiguous_resolved",
                ambiguous_options=ambiguous_options,
                ensembl_mapped=ensembl_mapped,
            )

        if mapped in self.alias_map:
            mapped_value = self.alias_map[mapped]
            status = "alias_resolved" if mapped_value != mapped else "unchanged"
            if ensembl_mapped and status == "unchanged":
                status = "ensembl_mapped"
            return SymbolMappingResult(
                symbol=raw,
                normalized=normalized,
                mapped=mapped_value,
                status=status,
                ambiguous_options=ambiguous_options,
                ensembl_mapped=ensembl_mapped,
            )

        if self.drop_unmapped:
            return SymbolMappingResult(
                symbol=raw,
                normalized=normalized,
                mapped="",
                status="unmapped",
                ambiguous_options=ambiguous_options,
                ensembl_mapped=ensembl_mapped,
            )

        status = "ensembl_mapped" if ensembl_mapped else "unmapped"
        return SymbolMappingResult(
            symbol=raw,
            normalized=normalized,
            mapped=mapped,
            status=status,
            ambiguous_options=ambiguous_options,
            ensembl_mapped=ensembl_mapped,
        )

    def map_symbols(self, symbols: Iterable[str | None]) -> Tuple[np.ndarray, pd.DataFrame]:
        results = [self.map_symbol(symbol) for symbol in symbols]
        mapped = np.array([result.mapped for result in results], dtype=object)
        report = pd.DataFrame(
            [
                {
                    "symbol": result.symbol,
                    "normalized": result.normalized,
                    "mapped": result.mapped,
                    "status": result.status,
                    "ambiguous_options": result.ambiguous_options,
                    "ensembl_mapped": int(result.ensembl_mapped),
                }
                for result in results
            ]
        )
        return mapped, report


def canonical_symbol(symbol: str | None, alias_map: Dict[str, str] | None = None) -> str:
    norm = normalize_symbol(symbol)
    if alias_map and norm in alias_map:
        return alias_map[norm]
    return norm


def normalize_gene_names(
    gene_names: np.ndarray | list[str],
    alias_map: Dict[str, str] | None = None,
) -> np.ndarray:
    return np.array([canonical_symbol(name, alias_map) for name in gene_names], dtype=object)


def normalize_edges(edges: pd.DataFrame, alias_map: Dict[str, str] | None = None) -> pd.DataFrame:
    df = edges.copy()
    sources = df["source"].astype(str).str.strip().str.upper()
    targets = df["target"].astype(str).str.strip().str.upper()
    if alias_map:
        sources = sources.map(lambda value: alias_map.get(value, value))
        targets = targets.map(lambda value: alias_map.get(value, value))
    df["source"] = sources
    df["target"] = targets
    df = df[(df["source"] != "") & (df["target"] != "")]
    return df.drop_duplicates()
