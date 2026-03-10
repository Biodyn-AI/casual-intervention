from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bias_protocol import (
    basic_normalize_symbols,
    bootstrap_auc_metrics,
    build_candidate_sets,
    dedupe_edges_with_score,
    evaluate_from_sets,
    map_edges_for_context,
    map_symbols_for_context,
    prepare_candidate_edges,
    read_gene_set,
    simulate_noise,
)
from src.eval.dorothea import load_dorothea
from src.eval.gene_symbols import SymbolMapper, normalize_symbol
from src.utils.config import load_config, resolve_path


def _resolve(path_value: str | Path, base_dir: Path) -> Path:
    return resolve_path(path_value, base_dir)


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**32 - 1)


def _load_predicted_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    columns = {col.lower(): col for col in df.columns}
    if "source" in columns and "target" in columns:
        source_col = columns["source"]
        target_col = columns["target"]
    else:
        if df.shape[1] < 2:
            raise ValueError(f"Predicted edges at {path} must have at least two columns")
        source_col = df.columns[0]
        target_col = df.columns[1]
    df = df.rename(columns={source_col: "source", target_col: "target"})
    return df


def _normalize_edges_basic(edges: pd.DataFrame) -> pd.DataFrame:
    df = edges.copy()
    df["source"] = df["source"].map(normalize_symbol)
    df["target"] = df["target"].map(normalize_symbol)
    df = df[(df["source"] != "") & (df["target"] != "")]
    return df.drop_duplicates().reset_index(drop=True)


def _coverage_stats(gene_set: List[str], edges: pd.DataFrame) -> Dict[str, float]:
    genes = set(symbol for symbol in gene_set if symbol)
    if not genes:
        return {
            "gene_set_size": 0,
            "gold_genes": 0,
            "gold_tfs": 0,
            "gold_targets": 0,
            "gold_edges": 0,
            "gene_overlap": 0,
            "tf_overlap": 0,
            "target_overlap": 0,
            "edge_overlap": 0,
            "gene_coverage": 0.0,
            "tf_coverage": 0.0,
            "target_coverage": 0.0,
            "edge_coverage": 0.0,
        }

    source_genes = set(edges["source"].unique())
    target_genes = set(edges["target"].unique())
    gold_genes = source_genes | target_genes
    edges_in_gene_set = edges[edges["source"].isin(genes) & edges["target"].isin(genes)]

    gene_overlap = len(genes & gold_genes)
    tf_overlap = len(source_genes & genes)
    target_overlap = len(target_genes & genes)

    return {
        "gene_set_size": len(genes),
        "gold_genes": len(gold_genes),
        "gold_tfs": len(source_genes),
        "gold_targets": len(target_genes),
        "gold_edges": len(edges),
        "gene_overlap": gene_overlap,
        "tf_overlap": tf_overlap,
        "target_overlap": target_overlap,
        "edge_overlap": len(edges_in_gene_set),
        "gene_coverage": gene_overlap / len(gold_genes) if gold_genes else 0.0,
        "tf_coverage": tf_overlap / len(source_genes) if source_genes else 0.0,
        "target_coverage": target_overlap / len(target_genes) if target_genes else 0.0,
        "edge_coverage": len(edges_in_gene_set) / len(edges) if len(edges) else 0.0,
    }


def _parse_gene_set_filter(value) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        names = {item.strip() for item in value.split(",")}
        return {name for name in names if name}
    if isinstance(value, (list, tuple, set)):
        names = {str(item).strip() for item in value if str(item).strip()}
        return names or None
    raise ValueError(f"Unsupported gene_sets value: {value!r}")


def _write_mapping_log(path: Path, config: dict, report: pd.DataFrame, errors: List[str]) -> None:
    lines = ["# Symbol Mapping Summary", ""]
    metadata = config.get("metadata", {})
    if metadata:
        lines.append("## Versions")
        for key, value in metadata.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    policies = config.get("symbol_mapping_policies")
    if policies:
        lines.append("## Policies")
        for policy in policies:
            name = policy.get("name", "default")
            lines.append(
                f"- {name}: ambiguous_policy={policy.get('ambiguous_policy', 'lexicographic')}, "
                f"drop_unmapped={bool(policy.get('drop_unmapped', False))}"
            )
        lines.append("")
    else:
        mapping_cfg = config.get("symbol_mapping", {})
        lines.append("## Policy")
        lines.append(f"- ambiguous_policy: {mapping_cfg.get('ambiguous_policy', 'lexicographic')}")
        lines.append(f"- drop_unmapped: {bool(mapping_cfg.get('drop_unmapped', False))}")
        lines.append("")

    lines.append("## Status Counts")
    if report.empty:
        lines.append("- No mapping report rows produced")
    else:
        if "mapping_policy" in report.columns:
            for policy in sorted(report["mapping_policy"].dropna().unique()):
                lines.append(f"### {policy}")
                counts = (
                    report[report["mapping_policy"] == policy]
                    .groupby(["context", "status"])
                    .size()
                    .reset_index(name="count")
                )
                for _, row in counts.iterrows():
                    lines.append(f"- {row['context']} | {row['status']}: {row['count']}")
                lines.append("")
        else:
            counts = report.groupby(["context", "status"]).size().reset_index(name="count")
            for _, row in counts.iterrows():
                lines.append(f"- {row['context']} | {row['status']}: {row['count']}")
    lines.append("")

    if errors:
        lines.append("## Skipped Inputs")
        for error in errors:
            lines.append(f"- {error}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_recommendations(
    path: Path,
    config: dict,
    coverage: pd.DataFrame,
    protocol: pd.DataFrame,
    noise: pd.DataFrame,
) -> None:
    lines: List[str] = ["# Evaluation Bias Recommendations", ""]
    lines.append("## Inputs")
    if not coverage.empty:
        gold_names = sorted(coverage["gold_standard"].unique())
        gene_set_names = sorted(coverage["gene_set"].unique())
        mapping_policies = sorted(coverage["mapping_policy"].unique()) if "mapping_policy" in coverage.columns else []
    elif not protocol.empty:
        gold_names = sorted(protocol["gold_standard"].unique())
        gene_set_names = sorted(protocol["gene_set"].unique())
        mapping_policies = sorted(protocol["mapping_policy"].unique()) if "mapping_policy" in protocol.columns else []
    else:
        gold_names = [spec.get("name") for spec in config.get("gold_standards", []) if spec.get("name")]
        gene_set_names = [spec.get("name") for spec in config.get("gene_sets", []) if spec.get("name")]
        mapping_policies = []
    lines.append(f"- Gold standards: {', '.join(gold_names) if gold_names else 'none'}")
    lines.append(f"- Gene sets: {', '.join(gene_set_names) if gene_set_names else 'none'}")
    if not protocol.empty and "prediction_method" in protocol.columns:
        method_names = sorted(protocol["prediction_method"].unique())
        lines.append(f"- Prediction methods: {', '.join(method_names)}")
    if mapping_policies:
        lines.append(f"- Mapping policies: {', '.join(mapping_policies)}")
    mapping_cfg = config.get("symbol_mapping", {})
    lines.append(f"- Ambiguity policy: {mapping_cfg.get('ambiguous_policy', 'lexicographic')}")
    lines.append(f"- Drop unmapped: {bool(mapping_cfg.get('drop_unmapped', False))}")
    lines.append("")

    if not coverage.empty:
        lines.append("## Coverage Highlights")
        best = coverage.sort_values("edge_coverage", ascending=False).head(5)
        for _, row in best.iterrows():
            lines.append(
                f"- {row['gene_set']} vs {row['gold_standard']} ({row['mapping_stage']}): "
                f"edge_coverage={row['edge_coverage']:.3f}, gene_coverage={row['gene_coverage']:.3f}"
            )
        lines.append("")

    if not protocol.empty:
        lines.append("## Candidate Set Sensitivity")
        pivot = protocol.groupby("candidate_set")["aupr"].median().sort_values(ascending=False)
        best_candidate = pivot.index[0] if not pivot.empty else None
        if best_candidate:
            lines.append(f"- Best median AUPR candidate set: {best_candidate}")
        aupr_range = protocol["aupr"].max() - protocol["aupr"].min()
        lines.append(f"- AUPR range across protocols: {aupr_range:.3f}")
        lines.append("")

    if not noise.empty and "aupr_std" in noise.columns:
        lines.append("## Noise Stability")
        stability = noise.sort_values("aupr_std").head(5)
        for _, row in stability.iterrows():
            lines.append(
                f"- {row['candidate_set']} ({row['noise_type']} {row['noise_rate']}): aupr_std={row['aupr_std']:.4f}"
            )
        lines.append("")

    lines.append("## Reporting Checklist")
    lines.append("- Include mapping stats (unmapped, ambiguous, alias-resolved counts)")
    lines.append("- Report candidate-set size and base rate for every metric")
    lines.append("- Version gold standards and HGNC mapping")
    lines.append("- Report sensitivity to candidate-set changes and noise")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation bias and symbol mapping analysis")
    parser.add_argument("--config", default="configs/eval_bias.yaml")
    parser.add_argument("--gene-sets", default=None, help="Comma-separated subset of gene set names")
    parser.add_argument("--methods", default=None, help="Comma-separated subset of prediction methods")
    parser.add_argument("--mapping-policies", default=None, help="Comma-separated subset of mapping policies")
    parser.add_argument("--skip-noise", action="store_true", help="Skip noise simulations")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap confidence intervals")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)
    base_dir = PROJECT_ROOT

    paths_cfg = cfg.get("paths", {})
    outputs_cfg = cfg.get("outputs", {})
    eval_cfg = cfg.get("evaluation", {})
    noise_cfg = cfg.get("noise", {})

    output_dir = _resolve(outputs_cfg.get("output_dir", "outputs/eval_bias"), base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted_sets_cfg = cfg.get("predicted_edge_sets")
    if not predicted_sets_cfg:
        predicted_sets_cfg = [
            {
                "name": "default",
                "path": paths_cfg.get("predicted_edges"),
                "score_column": eval_cfg.get("score_column", "score"),
            }
        ]

    selected_methods = None
    if args.methods:
        selected_methods = {name.strip() for name in args.methods.split(",") if name.strip()}

    mapping_policies = cfg.get("symbol_mapping_policies")
    if not mapping_policies:
        mapping_cfg = cfg.get("symbol_mapping", {})
        mapping_policies = [
            {
                "name": "default",
                "ambiguous_policy": mapping_cfg.get("ambiguous_policy", "lexicographic"),
                "drop_unmapped": bool(mapping_cfg.get("drop_unmapped", False)),
            }
        ]

    selected_policies = None
    if args.mapping_policies:
        selected_policies = {name.strip() for name in args.mapping_policies.split(",") if name.strip()}

    mapping_reports: List[pd.DataFrame] = []
    coverage_records: List[Dict[str, float]] = []
    protocol_records: List[Dict[str, float]] = []
    noise_records: List[Dict[str, float]] = []
    skipped_inputs: List[str] = []

    selected_gene_sets = None
    if args.gene_sets:
        selected_gene_sets = {name.strip() for name in args.gene_sets.split(",") if name.strip()}

    for policy in mapping_policies:
        policy_name = policy.get("name", "default")
        if selected_policies and policy_name not in selected_policies:
            continue

        mapper = SymbolMapper(
            hgnc_alias_tsv=_resolve(paths_cfg.get("hgnc_alias_tsv"), base_dir),
            gene_info_csv=_resolve(paths_cfg.get("gene_info_csv"), base_dir),
            ambiguous_policy=policy.get("ambiguous_policy", "lexicographic"),
            drop_unmapped=bool(policy.get("drop_unmapped", False)),
        )

        prediction_entries: List[Dict[str, object]] = []

        for pred_spec in predicted_sets_cfg:
            method_name = pred_spec.get("name") or "default"
            if selected_methods and method_name not in selected_methods:
                continue
            allowed_gene_sets = _parse_gene_set_filter(pred_spec.get("gene_sets"))
            if selected_gene_sets and allowed_gene_sets and allowed_gene_sets.isdisjoint(selected_gene_sets):
                continue
            pred_path_value = pred_spec.get("path") or paths_cfg.get("predicted_edges")
            if not pred_path_value:
                continue
            pred_path = _resolve(pred_path_value, base_dir)
            if not pred_path.exists():
                if pred_spec.get("optional", False):
                    skipped_inputs.append(f"predicted_edges:{method_name} -> missing")
                    continue
                raise FileNotFoundError(f"Predicted edges not found: {pred_path}")

            pred_edges_raw = _load_predicted_edges(pred_path)
            pred_edges, pred_report = map_edges_for_context(
                pred_edges_raw, mapper, f"predicted_edges:{method_name}"
            )
            pred_report["mapping_policy"] = policy_name
            pred_report["prediction_method"] = method_name
            mapping_reports.append(pred_report)

            score_col = pred_spec.get("score_column", eval_cfg.get("score_column", "score"))
            pred_edges = dedupe_edges_with_score(pred_edges, score_col)
            prediction_entries.append(
                {
                    "name": method_name,
                    "edges": pred_edges,
                    "score_col": score_col,
                    "gene_sets": allowed_gene_sets,
                }
            )

        if not prediction_entries:
            continue

        gold_standards: Dict[str, pd.DataFrame] = {}
        gold_standards_raw: Dict[str, pd.DataFrame] = {}

        for spec in cfg.get("gold_standards", []):
            name = spec.get("name")
            if not name:
                continue
            path_value = spec.get("path")
            if not path_value:
                continue
            path = _resolve(path_value, base_dir)
            if not path.exists():
                if spec.get("optional", False):
                    continue
                raise FileNotFoundError(f"Gold standard not found: {path}")

            fmt = spec.get("format", "edge_list")
            if fmt in {"dorothea", "trrust"}:
                edges = load_dorothea(path, confidence_levels=spec.get("confidence_levels"))
            else:
                edges = pd.read_csv(path, sep="\t")
                if "source" not in edges.columns or "target" not in edges.columns:
                    edges = edges.iloc[:, :2]
                    edges.columns = ["source", "target"]

            gold_standards_raw[name] = edges.copy()
            mapped_edges, report = map_edges_for_context(edges, mapper, f"gold_standard:{name}")
            report["mapping_policy"] = policy_name
            mapping_reports.append(report)
            gold_standards[name] = mapped_edges

        gene_sets: Dict[str, List[str]] = {}
        gene_sets_basic: Dict[str, List[str]] = {}

        for spec in cfg.get("gene_sets", []):
            name = spec.get("name")
            path_value = spec.get("h5ad")
            if not name or not path_value:
                continue
            if selected_gene_sets and name not in selected_gene_sets:
                continue
            path = _resolve(path_value, base_dir)
            try:
                raw_genes = read_gene_set(path)
            except Exception as exc:  # noqa: BLE001
                skipped_inputs.append(f"gene_set:{name} -> {exc}")
                continue
            basic_genes = basic_normalize_symbols(raw_genes)
            mapped_genes, report = map_symbols_for_context(raw_genes, mapper, f"gene_set:{name}")
            report["mapping_policy"] = policy_name
            mapping_reports.append(report)
            gene_sets[name] = [gene for gene in mapped_genes if gene]
            gene_sets_basic[name] = [gene for gene in basic_genes if gene]

        for gene_set_name, genes in gene_sets.items():
            basic_genes = gene_sets_basic.get(gene_set_name, [])
            for gold_name, edges in gold_standards.items():
                raw_edges = gold_standards_raw.get(gold_name, edges)
                basic_edges = _normalize_edges_basic(raw_edges)
                mapped_edges = edges.copy()

                for stage, gene_list, edge_df in (
                    ("normalized", basic_genes, basic_edges),
                    ("mapped", genes, mapped_edges),
                ):
                    stats = _coverage_stats(gene_list, edge_df)
                    stats.update({
                        "gene_set": gene_set_name,
                        "gold_standard": gold_name,
                        "mapping_stage": stage,
                        "mapping_policy": policy_name,
                    })
                    coverage_records.append(stats)

        allow_self_edges = bool(eval_cfg.get("allow_self_edges", False))
        top_k = eval_cfg.get("top_k", [50, 100, 200])
        rng = np.random.default_rng(noise_cfg.get("seed", 42))

        bootstrap_cfg = cfg.get("bootstrap", {})
        bootstrap_enabled = bool(bootstrap_cfg.get("enabled", False)) and not args.skip_bootstrap
        bootstrap_resamples = int(bootstrap_cfg.get("n_resamples", 200))
        bootstrap_seed = int(bootstrap_cfg.get("seed", 42))

        candidate_specs = cfg.get("candidate_sets", [])
        for gene_set_name, genes in gene_sets.items():
            predictions: Dict[str, pd.DataFrame] = {}
            score_cols: Dict[str, str] = {}
            for entry in prediction_entries:
                allowed_gene_sets = entry["gene_sets"]
                if allowed_gene_sets is not None and gene_set_name not in allowed_gene_sets:
                    continue
                method_name = entry["name"]
                if method_name in predictions:
                    raise ValueError(f"Duplicate prediction method for gene set {gene_set_name}: {method_name}")
                predictions[method_name] = entry["edges"]
                score_cols[method_name] = entry["score_col"]

            if not predictions:
                skipped_inputs.append(f"predicted_edges:{gene_set_name} -> no matching methods")
                continue
            candidates = build_candidate_sets(genes, gold_standards, candidate_specs, allow_self_edges)
            for gold_name, edges in gold_standards.items():
                for method_name, pred_edges in predictions.items():
                    score_col = score_cols[method_name]
                    for candidate in candidates:
                        pred_filtered, true_set = prepare_candidate_edges(pred_edges, edges, candidate)
                        metrics = evaluate_from_sets(pred_filtered, true_set, candidate, score_col, top_k)
                        metrics.update({
                            "gene_set": gene_set_name,
                            "gold_standard": gold_name,
                            "candidate_set": candidate.name,
                            "source_count": candidate.source_count,
                            "target_count": candidate.target_count,
                            "prediction_method": method_name,
                            "mapping_policy": policy_name,
                        })
                        if bootstrap_enabled:
                            if pred_filtered.empty:
                                scores = np.array([], dtype=float)
                                labels = np.array([], dtype=int)
                            else:
                                scores = pred_filtered[score_col].to_numpy(dtype=float)
                                labels = np.array(
                                    [
                                        1 if (src, tgt) in true_set else 0
                                        for src, tgt in zip(pred_filtered["source"], pred_filtered["target"])
                                    ],
                                    dtype=int,
                                )
                            total_pos = len(true_set)
                            total_neg = max(candidate.size - total_pos, 0)
                            seed = _stable_seed(
                                bootstrap_seed,
                                policy_name,
                                method_name,
                                gene_set_name,
                                gold_name,
                                candidate.name,
                            )
                            ci_metrics = bootstrap_auc_metrics(
                                scores,
                                labels,
                                total_pos,
                                total_neg,
                                bootstrap_resamples,
                                seed,
                            )
                            metrics.update(ci_metrics)
                        protocol_records.append(metrics)

                        if not args.skip_noise:
                            noise_metrics = simulate_noise(
                                pred_edges,
                                edges,
                                candidate,
                                score_col,
                                top_k,
                                rng,
                                noise_cfg.get("rates", [0.05, 0.1, 0.2, 0.3]),
                                int(noise_cfg.get("repeats", 5)),
                                noise_cfg.get("tf_dropout_rates", []),
                                noise_cfg.get("target_dropout_rates", []),
                                int(noise_cfg.get("structured_repeats", noise_cfg.get("repeats", 5))),
                            )
                            for record in noise_metrics:
                                record.update({
                                    "gene_set": gene_set_name,
                                    "gold_standard": gold_name,
                                    "candidate_set": candidate.name,
                                    "prediction_method": method_name,
                                    "mapping_policy": policy_name,
                                })
                                noise_records.append(record)

    mapping_report_df = pd.concat(mapping_reports, ignore_index=True) if mapping_reports else pd.DataFrame()
    mapping_report_path = _resolve(
        outputs_cfg.get("symbol_mapping_report", output_dir / "symbol_mapping_report.tsv"), base_dir
    )
    if not mapping_report_df.empty:
        mapping_report_df.to_csv(mapping_report_path, sep="\t", index=False)

    mapping_log_path = _resolve(outputs_cfg.get("symbol_mapping_log", output_dir / "symbol_mapping_log.md"), base_dir)
    _write_mapping_log(mapping_log_path, cfg, mapping_report_df, skipped_inputs)

    coverage_df = pd.DataFrame(coverage_records)
    coverage_path = _resolve(outputs_cfg.get("coverage_tables", output_dir / "coverage_tables.tsv"), base_dir)
    if not coverage_df.empty:
        coverage_df.to_csv(coverage_path, sep="\t", index=False)

    protocol_df = pd.DataFrame(protocol_records)
    protocol_path = _resolve(outputs_cfg.get("protocol_comparison", output_dir / "protocol_comparison.csv"), base_dir)
    if not protocol_df.empty:
        protocol_df.to_csv(protocol_path, index=False)

    noise_df = pd.DataFrame(noise_records)
    noise_summary = pd.DataFrame()
    noise_path = _resolve(outputs_cfg.get("noise_stability", output_dir / "noise_stability_summary.tsv"), base_dir)
    if not noise_df.empty:
        metric_cols = [
            col
            for col in noise_df.columns
            if col
            not in {
                "gene_set",
                "gold_standard",
                "candidate_set",
                "prediction_method",
                "mapping_policy",
                "noise_type",
                "noise_rate",
            }
        ]
        summary = (
            noise_df.groupby(
                [
                    "gene_set",
                    "gold_standard",
                    "candidate_set",
                    "prediction_method",
                    "mapping_policy",
                    "noise_type",
                    "noise_rate",
                ]
            )[metric_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
        summary.to_csv(noise_path, sep="\t", index=False)
        noise_summary = summary

    recommendations_path = _resolve(outputs_cfg.get("recommendations", output_dir / "recommendations.md"), base_dir)
    _write_recommendations(recommendations_path, cfg, coverage_df, protocol_df, noise_summary)


if __name__ == "__main__":
    main()
