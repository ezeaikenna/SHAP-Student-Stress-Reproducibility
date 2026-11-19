# SHAP Student-Stress Reproducibility Repository

This repository provides the analysis scripts, saved results, computational
environment information, and execution records supporting the study of
machine-learning robustness and SHAP explanation stability in student-stress
prediction.

## Analysis components

1. **Primary analysis:** evaluates six classifiers, predictive performance,
   robustness to synthetic input corruption, feature ablation, and global SHAP
   explanation stability for the Logistic Regression model selected using
   cross-validation on the training partition.
2. **Blood-pressure feature-exclusion sensitivity analysis:** evaluates the
   effect of excluding `blood_pressure` on predictive performance and SHAP
   summaries.
3. **Diagnostic-run reproduction:** preserves a separate archived run of the
   broad six-model workflow. One retained supplementary output is the Decision
   Tree case-level probability and log-loss table. This diagnostic run uses a
   different probability-clipping convention for that case-level calculation,
   so those log-loss values are reported separately.

## Repository contents

- `scripts/primary/`: primary analysis implementation.
- `scripts/sensitivity/`: blood-pressure feature-exclusion sensitivity analysis.
- `scripts/diagnostics/`: archived diagnostic-run reproduction, including the Decision Tree case-probability output.
- `scripts/archive/`: exact historical primary and diagnostic scripts retained for provenance checks.
- `results/primary_analysis/`: primary analysis tables, figures, confusion
  matrices, and workbook.
- `results/blood_pressure_sensitivity/`: sensitivity-analysis results and report.
- `results/decision_tree_diagnostic/`: the retained Decision Tree case-level probability CSV and its output description; this is a selected diagnostic output, not the complete output set of the broader diagnostic run.
- `reproduction_evidence/`: execution record, output verification, reference
  output comparison, and saved outputs.
- `environment/`: computational environment information.
- `requirements.txt` and `requirements-execution.txt`: package dependencies.

## Dataset

The study uses `StressLevelDataset.csv`, comprising 1,100 observations,
20 predictors, and the `stress_level` target. The dataset is not redistributed
in this repository.

Original dataset source:
https://www.kaggle.com/datasets/samyakb/student-stress-factors

Obtain the dataset from its source and review the source's current terms before
reuse. Place `StressLevelDataset.csv` in the repository root or provide its
location with `--csv`. The manifest records the SHA-256 checksum of the
dataset used in the recorded execution.

The original target-label generation procedure and substantive meanings of
the `blood_pressure` codes were not independently established. These
uncertainties limit interpretation of the observed predictive relationships.

## Reproducing the primary analysis

Run from the repository root:

```bash
python scripts/primary/primary_stress_robustness_analysis.py \
  --csv StressLevelDataset.csv \
  --outdir results/primary_analysis_reproduced
```

For the full analysis, do not use `--fast` or `--skip-shap`. The
`--shap-all-models` option changes the explanation scope and should not be
used to reproduce the manuscript's SHAP analysis, which focuses on the
training-CV-selected Logistic Regression model.

## Blood-pressure feature-exclusion sensitivity analysis

Run:

```bash
python scripts/sensitivity/blood_pressure_sensitivity_analysis.py \
  --csv StressLevelDataset.csv \
  --outdir results/blood_pressure_sensitivity_reproduced
```

## Diagnostic-run reproduction and retained Decision Tree output

Run:

```bash
python scripts/diagnostics/decision_tree_probability_diagnostic.py \
  --csv StressLevelDataset.csv \
  --outdir results/decision_tree_diagnostic_reproduced
```

The diagnostic script reproduces a broader six-model diagnostic run; it is
not a Decision-Tree-only pipeline. The repository currently retains one
selected supplementary output from that run:
`results/decision_tree_diagnostic/decision_tree_test_case_probabilities.csv`.
This CSV contains case-level Decision Tree predicted probabilities and
case-level log-loss values; it should not be mistaken for the complete output
set of the diagnostic run.

For this case-level calculation, diagnostic probabilities are clipped using
machine epsilon, whereas the primary analysis uses a 1e-15 lower bound.
Consequently, the two runs' case-level log-loss values are not directly
interchangeable. The exact historical primary and diagnostic scripts are
retained under `scripts/archive/`; their SHA-256 checksums are recorded in the
reproduction manifest.

| Aspect | Primary analysis | Diagnostic run |
|---|---|---|
| Scope | Six-model performance, robustness, feature ablation and SHAP analysis | Broad six-model diagnostic workflow |
| Retained script | `scripts/primary/primary_stress_robustness_analysis.py` | `scripts/diagnostics/decision_tree_probability_diagnostic.py` |
| Retained historical script | `scripts/archive/primary_analysis_historical_execution.py` | `scripts/archive/diagnostic_analysis_historical_execution.py` |
| Distinguishing detail | Primary case-level probability clipping uses a 1e-15 lower bound | Decision Tree case-level probability diagnostic uses machine-epsilon clipping |
| Available saved outputs | Primary analysis outputs under `results/primary_analysis/` | Selected case-probability CSV under `results/decision_tree_diagnostic/` |

## Computational environment

The recorded analysis environment used:

- Python 3.13.16
- NumPy 2.4.6
- pandas 2.2.3
- SciPy 1.16.3
- scikit-learn 1.6.1
- Matplotlib 3.10.0
- SHAP 0.54.1

Further platform and run metadata are recorded in
`reproduction_evidence/full_reproduction_2026-11-11/reproduction_manifest.json`.
Pinned package versions are listed in `requirements-execution.txt`.

## Execution record

The recorded full analysis used a stratified 80:20 train/test split, with
five-fold stratified cross-validation restricted to the training partition.
The held-out test set was not used for model selection. The archived output
comparison documents numerical agreement with reference outputs; this is not
external validation. The scripts write `shap_errors.csv` as a diagnostic record
of SHAP computation failures (including an empty file when no failures occur);
it is an execution-status log, not a model-performance result.


## Independent execution audit

On 19 November 2026, the primary analysis, diagnostic-run reproduction, and
blood-pressure sensitivity scripts each completed successfully using the
dataset whose LF-normalised SHA-256 checksum matches the recorded manifest.
The regenerated numerical tables matched the archived reference outputs;
the four primary figures were pixel-identical. The primary case-level
Decision Tree log-loss values retain the documented difference in probability
clipping convention. Full details, including the runtime-version qualification
and output-by-output comparisons, are provided in
`reproduction_evidence/independent_execution_audit_2026-11-19/`.

The independent execution used a runtime whose package versions differ from
the historical pinned environment. It therefore verifies successful execution
and output consistency in the tested environment, not recreation of the exact
historical software environment.

## Interpretation and limitations

- The dataset-specific results do not establish clinical validity.
- External validation was not performed.
- The target-label generation procedure remains unverified.
- The observed association involving `blood_pressure` does not by itself
  establish target leakage.
- Corruption levels are probabilities under the implemented synthetic
  perturbation procedure, not calibrated real-world measurement-error levels.
- SHAP results are global descriptive summaries for the selected Logistic
  Regression model. They are not causal explanations or participant-level
  stability estimates.

## Citation and licence

Citation metadata is provided in `CITATION.cff`. The code and accompanying
reproducibility documentation are released under the MIT Licence; see
`LICENSE`. The licence does not apply to the third-party dataset, which is
not redistributed here. Dataset access and reuse remain subject to the
provider's terms.

GitHub repository:
https://github.com/ezeaikenna/SHAP-Student-Stress-Reproducibility

Zenodo concept DOI (all versions):
https://doi.org/10.5281/zenodo.24250742

Zenodo version-specific DOI (Version 1.0.2):
https://doi.org/10.5281/zenodo.24267796
