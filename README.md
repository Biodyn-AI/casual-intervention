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
├── manuscript/                # PLOS Computational Biology submission
│   ├── causal_intervention_ploscompbiol.tex # Main manuscript
│   ├── references.bib                       # Bibliography (31 references)
│   ├── cover_letter.tex                     # Cover letter
│   └── *.pdf                               # Compiled PDFs
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

| Tissue | Reference | AUPR | Permutation *p* |
|--------|-----------|------|-----------------|
| Lung | TRRUST | 0.727 | **0.001** |
| Lung | DoRothEA | 0.782 | **0.001** |
| Kidney (mean, 4 seeds) | TRRUST | 0.602 | 0.105 |
| Kidney (mean, 4 seeds) | DoRothEA | 0.537 | — |
| Immune | TRRUST | 0.596 | 0.359 |
| Lung (high-pair, 160+160) | TRRUST | 0.757 | **0.004** |

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
