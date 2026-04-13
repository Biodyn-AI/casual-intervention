# Causal Intervention Validation of Gene Regulatory Signals in scGPT

This repository contains the source code, experiment configurations, evaluation
artifacts, and manuscript for the paper:

> **Causal intervention validation of gene regulatory signals in scGPT**
> Ihor Kendiukhov, University of Tuebingen
> *Submitted to PLOS Computational Biology*

## Overview

We evaluate whether causal interventions on scGPT gene tokens recover known
transcription factor (TF)–target dependencies beyond attention-only heuristics.
Using Tabula Sapiens kidney, lung, and immune subsets plus an external lung
atlas, we ablate or swap TF token values and quantify changes in target-token
readouts. We compare intervention scores to curated gene regulatory references
(TRRUST, DoRothEA), include attention and coexpression baselines, trace
component-level circuits via activation patching, and validate on CRISPR
perturbation data from Adamson et al.

### Key findings

- **Lung tissue** shows strong enrichment (AUPR up to 0.78, permutation
  *p* = 0.001) that substantially exceeds attention and coexpression baselines
- **Kidney** enrichment is modest but reproducible across seeds (AUPR ≈ 0.60)
- **Immune** tissue remains at baseline, revealing tissue-dependent regulatory
  encoding
- **Mechanistic tracing** localizes effects to a small set of attention heads
  and MLP blocks with depth separation between self-targeting and cross-gene
  interactions
- **Quantitative comparison** against **eleven classical GRN inference
  methods** (Pearson, Spearman, distance correlation; partial correlation;
  mutual information, CLR, ARACNE; LASSO and elastic net; GRNBoost2 and
  GENIE3) on a matched substrate shows scGPT leads the strongest classical
  baseline by **+0.053 in kidney**, **+0.075 in lung**, and **+0.002 in
  immune** — consistent with the tissue-level pattern of the main runs

## Repository structure

```
casual-intervention/
├── src/                       # Core Python modules
│   ├── interpret/             # Causal intervention & activation patching
│   │   ├── causal_intervention.py   # Value ablation, token swap, readout extraction
│   │   ├── attention.py             # Attention score aggregation
│   │   └── attribution.py          # Attribution utilities
│   ├── eval/                  # Evaluation against reference databases
│   │   ├── dorothea.py              # DoRothEA TF-target loader
│   │   ├── metrics.py               # AUPR, AUROC, permutation tests
│   │   ├── gene_symbols.py          # HGNC symbol normalization
│   │   └── bias_protocol.py         # Bias evaluation protocol
│   ├── model/                 # scGPT model loading and wrapping
│   │   ├── scgpt_loader.py          # Model checkpoint loading
│   │   ├── wrapper.py               # ScGPTWrapper with hook support
│   │   ├── hooks.py                 # PyTorch forward/backward hooks
│   │   └── vocab.py                 # Gene vocabulary management
│   ├── data/                  # Dataset loading and preprocessing
│   │   ├── scgpt_dataset.py         # scGPT-compatible dataset class
│   │   ├── preprocess.py            # normalize_total, log1p, HVG selection
│   │   └── tabula_sapiens.py        # Tabula Sapiens tissue subsetting
│   └── network/               # Network inference and export
│       ├── infer.py                 # Causal edge inference
│       └── export.py                # TSV/JSON export utilities
├── scripts/                   # Executable analysis scripts
│   ├── run_causal_interventions.py          # Main pipeline entry point
│   ├── evaluate_causal_results.py           # Compute AUPR/AUROC metrics
│   ├── summarize_causal_metrics.py          # Aggregate across runs
│   ├── evaluate_perturbation_validation.py  # Adamson/Dixit/Shifrut validation
│   ├── ablate_head_contributions.py         # Head-level ablation analysis
│   ├── analyze_invariant_causal_edges.py    # Multi-seed edge stability
│   ├── analyze_invariant_blockers.py        # Non-causal edge identification
│   ├── run_eval_bias_protocol.py            # Symbol remapping bias test
│   ├── grn_baseline_comparison.py           # Eleven-method GRN benchmark
│   ├── plot_grn_baseline_comparison.py      # GRN benchmark figure
│   ├── plot_atlas_summary_panel.py          # Summary panel figure
│   ├── plot_head_baseline_heatmap.py        # Attention head heatmap
│   ├── plot_head_overlap_heatmap.py         # Head overlap visualization
│   └── plot_scaling_results.py              # Scaling experiment figures
├── configs/                   # YAML experiment configurations
│   ├── causal_intervention.yaml             # Base configuration
│   ├── causal_intervention_lung.yaml        # Lung tissue
│   ├── causal_intervention_kidney_rg.yaml   # Kidney tissue
│   ├── causal_intervention_immune.yaml      # Immune tissue
│   ├── causal_intervention_krasnow_lung.yaml # External lung dataset
│   ├── causal_intervention_adamson*.yaml    # Adamson perturbation configs
│   ├── causal_intervention_dorothea.yaml    # DoRothEA evaluation
│   ├── causal_intervention_*seed*.yaml      # Multi-seed runs
│   └── causal_metrics_manifest.csv          # Run manifest
├── figures/                   # Publication figures (PNG)
├── outputs/
│   └── grn_baseline_comparison/             # Per-tissue metric TSVs for the
│       ├── kidney_metrics.tsv               #   eleven-method benchmark
│       ├── lung_metrics.tsv
│       ├── immune_metrics.tsv
│       └── summary.tsv
├── manuscript/                # PLOS Computational Biology submission
│   ├── causal_intervention_ploscompbiol.tex # Main manuscript
│   ├── causal_intervention_ploscompbiol.pdf
│   ├── references.bib                       # Bibliography
│   ├── cover_letter.tex                     # Cover letter
│   ├── response_to_reviewers.tex            # Response to editor comments
│   └── response_to_reviewers.pdf
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 1.13+ (CPU sufficient for evaluation; GPU recommended for scaling)
- scGPT model checkpoint (whole-human, from the
  [scGPT model zoo](https://github.com/bowang-lab/scGPT))

### Setup

```bash
git clone https://github.com/Biodyn-AI/casual-intervention.git
cd casual-intervention
pip install -r requirements.txt
```

### Data

The following public datasets are used (downloaded automatically or via
provided scripts):

| Dataset | Source | Reference |
|---------|--------|-----------|
| Tabula Sapiens (kidney, lung, immune) | [CZ Biohub](https://tabula-sapiens-portal.ds.czbiohub.org) | Tabula Sapiens Consortium, *Science* 2022 |
| Krasnow lung atlas | [Human Lung Cell Atlas](https://hlca.ds.czbiohub.org) | Travaglini et al., *Nature* 2020 |
| Adamson CRISPR perturbation | [Harvard Dataverse](https://doi.org/10.7910/DVN/BIKGRL) | Adamson et al., *Cell* 2016 |
| TRRUST v2 | [GRNpedia](https://www.grnpedia.org/trrust) | Han et al., *Scientific Reports* 2015 |
| DoRothEA regulons | [decoupler-py](https://github.com/saezlab/decoupler-py) | Garcia-Alonso et al., *Genome Research* 2019 |

## Usage

### Running causal interventions

```bash
# Basic kidney run (4 seeds)
python scripts/run_causal_interventions.py --config configs/causal_intervention_kidney_rg.yaml

# Lung tissue run
python scripts/run_causal_interventions.py --config configs/causal_intervention_lung.yaml

# Adamson perturbation validation
python scripts/run_causal_interventions.py --config configs/causal_intervention_adamson_scaled.yaml
```

### Evaluating results

```bash
# Compute AUPR/AUROC against TRRUST and DoRothEA
python scripts/evaluate_causal_results.py --output-dir outputs/causal_lung/

# Perturbation validation against Adamson
python scripts/evaluate_perturbation_validation.py --output-dir outputs/causal_adamson/

# Aggregate metrics across all runs
python scripts/summarize_causal_metrics.py --manifest configs/causal_metrics_manifest.csv
```

### Generating figures

```bash
python scripts/plot_atlas_summary_panel.py --output-dir figures/
python scripts/plot_head_baseline_heatmap.py --output-dir figures/
```

### Benchmark against classical GRN inference methods

The script `scripts/grn_baseline_comparison.py` is a self-contained
CPU-only pipeline that evaluates eleven classical GRN baselines on a
matched reproduction of the main kidney, lung, and immune runs:

| Family | Methods |
|--------|---------|
| Coexpression | Pearson, Spearman, distance correlation (Székely et al. 2007) |
| Graphical model | Partial correlation on ridge-regularized precision matrix |
| Information-theoretic | Mutual information, CLR (Faith et al. 2007), ARACNE (Margolin et al. 2006) |
| Regularized regression | LASSO (Tibshirani 1996), Elastic net (Zou & Hastie 2005) |
| Tree-based regression | GRNBoost2 (Moerman et al. 2019), GENIE3 (Huynh-Thu et al. 2010) |

```bash
# Run the benchmark (requires the raw Tabula Sapiens h5ad files)
python scripts/grn_baseline_comparison.py --tissue all --permutations 1000

# Plot the grouped bar chart
python scripts/plot_grn_baseline_comparison.py
```

Per-tissue AUPR / AUROC / permutation p-values are written to
`outputs/grn_baseline_comparison/{kidney,lung,immune}_metrics.tsv` and a
combined `summary.tsv`. The benchmark reproduces the main runs'
preprocessing, cell sampling, deterministic TRRUST pair sampling, and
evidence filter, so the scGPT main-run numbers and the classical-method
numbers can be compared directly. All eleven baselines are implemented
with scikit-learn / scipy / dcor primitives and run on CPU in minutes per
tissue — no Dask or dedicated cluster backend is required.

## Configuration

Experiment configurations are YAML files in `configs/`. Key parameters:

```yaml
# Example: configs/causal_intervention_lung.yaml
dataset:
  name: tabula_sapiens
  tissue: lung
  max_cells: 120

intervention:
  type: ablation          # ablation | swap
  n_positive_pairs: 120
  n_negative_pairs: 120

evaluation:
  references: [trrust, dorothea]
  n_permutations: 1000
  seed: 42

model:
  checkpoint: whole_human  # scGPT checkpoint name
  deterministic: true      # disable fast transformers
```

## Results summary

### Main runs (TRRUST ablation)

| Tissue | Reference | AUPR | Permutation *p* |
|--------|-----------|------|-----------------|
| Lung | TRRUST | 0.727 | **0.001** |
| Lung | DoRothEA | 0.782 | **0.001** |
| Kidney (mean, 4 seeds) | TRRUST | 0.602 | 0.105 |
| Kidney (mean, 4 seeds) | DoRothEA | 0.537 | — |
| Immune | TRRUST | 0.596 | 0.359 |
| Lung (high-pair, 160+160) | TRRUST | 0.757 | **0.004** |

### Matched benchmark vs. eleven classical GRN methods

scGPT causal ablation (main-run AUPR) vs. the strongest classical
baseline on the matched substrate:

| Tissue | scGPT causal | Best classical | Δ (scGPT − best) |
|--------|--------------|----------------|------------------|
| Kidney | 0.602 | Spearman correlation 0.549 | +0.053 |
| Lung   | 0.727 | Partial correlation 0.652 (*p*=0.001) | +0.075 |
| Immune | 0.596 | Spearman correlation 0.594 (*p*=0.014) | +0.002 |

Full per-tissue per-method AUPR, AUROC, and permutation p-values are in
`outputs/grn_baseline_comparison/`.

## Citation

If you use this code or data, please cite:

```bibtex
@article{kendiukhov_causal_intervention_2026,
  title={Causal intervention validation of gene regulatory signals in {scGPT}},
  author={Kendiukhov, Ihor},
  journal={Submitted to PLOS Computational Biology},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.

## Contact

Ihor Kendiukhov — kenduhov.ig@gmail.com
Department of Computer Science, University of Tuebingen
