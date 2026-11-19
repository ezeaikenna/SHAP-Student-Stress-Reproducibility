# SHAP Student-Stress Analysis: Reproducibility Materials

This repository contains the analysis code, saved numerical results, figures,
computational environment information and execution records for a study of
machine-learning performance and SHAP explanation stability in student-stress
prediction.

## Study analyses

1. **Primary analysis.** Six classifiers are evaluated for predictive
   performance, robustness to synthetic input corruption and feature
   ablation. Global SHAP explanation stability is evaluated for the
   Logistic Regression model selected using cross-validation on the training
   partition.
2. **Blood-pressure feature-exclusion sensitivity analysis.** Predictive
   performance and global SHAP summaries are compared with and without the
   `blood_pressure` predictor.
3. **Extended corruption and SHAP analysis.** The six classifiers are
   evaluated under 20 perturbation repetitions at each of 5%, 10% and 20%
   corruption, comparing all predictors with a condition excluding
   `blood_pressure`. Global Logistic Regression SHAP stability is calculated
   using a 100-row training background and all 220 held-out observations.
4. **Decision Tree case-level probabilities.** A supplementary output reports
   case-level predicted probabilities and log-loss values from a separate
   six-model analysis run. Its probability-clipping convention differs from
   that used in the primary analysis; the values are therefore reported
   separately.

## Repository structure

- `scripts/primary/`: primary analysis implementation.
- `scripts/sensitivity/`: blood-pressure feature-exclusion and extended
  corruption/SHAP analyses.
- `scripts/diagnostics/`: supplementary case-level probability analysis.
- `reproducibility_records/historical_source_snapshots/`: exact source snapshots associated with historical
  execution records; see `reproducibility_records/historical_source_snapshots/README.md`.
- `results/primary_analysis/`: primary analysis tables, figures, confusion
  matrices and workbook.
- `results/blood_pressure_sensitivity/`: feature-exclusion sensitivity
  results and report.
- `results/extended_bp_exclusion/`: extended corruption-performance and
  global SHAP stability outputs.
- `results/decision_tree_diagnostic/`: selected case-level probability
  output and its description.
- `reproducibility_records/`: execution metadata, generated-output inventories,
  reference comparisons and independent reproduction records.
- `environment/`: computational environment information.

## Dataset

The study uses `StressLevelDataset.csv`, comprising 1,100 observations,
20 predictors and the `stress_level` target. The dataset is not redistributed
in this repository.

Original dataset source:
https://www.kaggle.com/datasets/samyakb/student-stress-factors

Obtain the dataset from its source and review the provider's current terms
before reuse. Place `StressLevelDataset.csv` in the repository root or pass
its location with `--csv`. The SHA-256 checksum of the dataset used in the
recorded analyses is provided in the execution manifests.

The original target-label generation procedure and substantive meanings of
the `blood_pressure` codes were not established by the study materials.
These uncertainties limit interpretation of the observed predictive
relationships.

## Reproducing the primary analysis

Run from the repository root:

```bash
python scripts/primary/primary_stress_robustness_analysis.py \
  --csv StressLevelDataset.csv \
  --outdir results/primary_analysis_reproduced
```

For the full analysis, do not use `--fast` or `--skip-shap`. The
`--shap-all-models` option changes the explanation scope and should not be
used to reproduce the reported SHAP analysis, which focuses on the
training-CV-selected Logistic Regression model.

## Blood-pressure feature-exclusion sensitivity analysis

```bash
python scripts/sensitivity/blood_pressure_sensitivity_analysis.py \
  --csv StressLevelDataset.csv \
  --outdir results/blood_pressure_sensitivity_reproduced
```

## Extended corruption and global SHAP stability analysis

```bash
python scripts/sensitivity/extended_blood_pressure_corruption_shap_analysis.py \
  --csv StressLevelDataset.csv \
  --outdir results/extended_bp_exclusion_reproduced
```

The 100-row SHAP background is sampled without replacement using NumPy's
`default_rng` with seed 20261780. Feature-specific perturbation seeds are
derived deterministically from the base seed, feature name, corruption level
and repetition using SHA-256. The same sampled background-row positions and
the same perturbations for shared predictors are used across the two
predictor conditions. Saved analysis outputs are under
`results/extended_bp_exclusion/`.

The independent reproduction comparison dated 19 November 2026 covers the
primary, supplementary diagnostic, blood-pressure sensitivity and extended
corruption/global SHAP analyses. All five extended-analysis CSV outputs
matched their saved counterparts byte-for-byte. The comparison record and
experiment-specific reproduction details are documented in
`reproducibility_records/independent_reproduction_2026-11-19/` and
`reproducibility_records/extended_corruption_shap_analysis/`.

## Supplementary case-level probability analysis

```bash
python scripts/diagnostics/decision_tree_probability_diagnostic.py \
  --csv StressLevelDataset.csv \
  --outdir results/decision_tree_diagnostic_reproduced
```

This script reproduces a broader six-model analysis run; it is not a
Decision-Tree-only pipeline. The repository retains one selected supplementary
output at `results/decision_tree_diagnostic/decision_tree_test_case_probabilities.csv`.
For this case-level calculation, probabilities are clipped using machine
epsilon, whereas the primary analysis uses a 1e-15 lower bound. The resulting
case-level log-loss values are not directly interchangeable.

## Computational environment and execution records

The recorded primary analysis environment used Python 3.13.16, NumPy 2.4.6,
pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.6.1, Matplotlib 3.10.0 and
SHAP 0.54.1. Platform and run metadata are recorded in
`reproducibility_records/primary_analysis_2026-11-11/reproduction_manifest.json`.
Pinned package versions are listed in `requirements-execution.txt`.

The primary analysis used a stratified 80:20 train/test split and five-fold
stratified cross-validation restricted to the training partition. The held-out
test set was not used for model selection. The archived output comparisons
assess numerical consistency with saved reference outputs; they do not
constitute external validation.

An independent reproduction performed on 19 November 2026 confirmed that
the primary, supplementary diagnostic, blood-pressure sensitivity and
extended corruption/global SHAP scripts completed on a dataset matching the
recorded LF-normalised SHA-256 checksum. All five regenerated CSV outputs
from the extended analysis matched their saved references byte-for-byte.
The output comparisons are documented in
`reproducibility_records/independent_reproduction_2026-11-19/`. The runtime
package versions differed from the historically recorded environment, so
this is evidence of successful execution and output consistency in the tested
runtime, not recreation of the exact historical software environment.

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

The version-specific DOI for a release is provided by Zenodo when that release
is archived. Use the version-specific DOI associated with the deposited
release when citing a particular version.
