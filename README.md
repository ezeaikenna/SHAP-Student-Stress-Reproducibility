# SHAP Student-Stress Reproducibility Repository

This repository contains the primary Version 4 analysis implementation, a
separate blood-pressure feature-exclusion sensitivity analysis, supplementary
Decision Tree diagnostic material, saved results, and records from a successful
full reproduction run completed on 11 November 2026.

## Research scope and analysis lineage

The primary analysis is implemented by the Experiment VIII Version 4 script.
The blood-pressure feature-exclusion analysis (Experiment X) is a separate
sensitivity analysis. Experiment IX is included only as a supplementary
Decision Tree case-level diagnostic. These analyses are kept separate because
the Experiment IX log-loss clipping convention differs from that used in the
primary Version 4 implementation.

## Repository structure

- `scripts/primary/`: primary Version 4 analysis script.
- `scripts/sensitivity/`: original and path-portable blood-pressure
  feature-exclusion scripts.
- `scripts/diagnostics/`: supplementary Decision Tree diagnostic script.
- `results/primary_experiment_viii/`: primary analysis tables, figures,
  confusion matrices and workbook.
- `results/sensitivity_experiment_x/`: sensitivity-analysis results and report.
- `results/diagnostics_experiment_ix/`: supplementary case-level diagnostic.
- `reproduction_evidence/`: run manifest, console log, output records,
  reference-output comparison and generated outputs from the 11 November 2026 run.
- `data/`: dataset access and verification instructions.
- `environment/`: recorded software environment.
- `requirements.txt` and `requirements-execution.txt`: package dependencies.

## Dataset

The analysis uses `StressLevelDataset.csv`, described as containing 1,100
observations, 20 predictors and the `stress_level` target. The dataset is not
redistributed in this repository.

The dataset download page used for this study is:
https://www.kaggle.com/datasets/samyakb/student-stress-factors

Obtain the dataset from its source and review the source's current terms before
reuse. Place `StressLevelDataset.csv` in the repository root or supply its path
with `--csv`. Verify the downloaded file against the SHA-256 value in the
reproduction manifest before treating it as the same input used in the recorded run.

The original target-label generation procedure and substantive meanings of the `blood_pressure` codes were not independently established. These limitations
constrain interpretation of the observed predictive relationships.

## Primary analysis: Version 4

Reference implementation:

`scripts/primary/stress_robustness_experiment_v4_release.py`

Run from the repository root:

```bash
python scripts/primary/stress_robustness_experiment_v4_release.py \
  --csv StressLevelDataset.csv \
  --outdir results/reproduced_experiment_viii
```

For a full reproduction, do not use `--fast` or `--skip-shap`.
The `--shap-all-models` option changes the explanation scope and should not be
used to reproduce the manuscript's Logistic-Regression-only SHAP analysis.

## Blood-pressure feature-exclusion sensitivity analysis

The original script is preserved at:

`scripts/sensitivity/blood_pressure_sensitivity_analysis_original.py`

It uses fixed `/mnt/data/` paths from its original execution environment.
For a user-specified dataset and output directory, use the path-portable
derivative:

```bash
python scripts/sensitivity/blood_pressure_sensitivity_analysis_portable.py \
  --csv StressLevelDataset.csv \
  --outdir results/reproduced_experiment_x
```

The path-portable derivative changes path handling and output archive location.
It has been syntax-checked but was not rerun as part of the 11 November 2026
Version 4 reproduction. Its outputs should therefore be treated as the saved
sensitivity-analysis results, not as a newly reproduced execution of that
derivative.

## Supplementary Decision Tree diagnostic

Experiment IX provides a case-level Decision Tree probability/log-loss
diagnostic. Its clipping convention differs from the primary Version 4
implementation. It is retained separately and must not be combined with or
substituted for the primary Version 4 outputs without a documented
methodological reason.

The diagnostic script accepts an explicit dataset path through `--csv` and an
output directory through `--outdir`. From the repository root, run:

```bash
python scripts/diagnostics/experiment_ix_diagnostic_script.py \
  --csv StressLevelDataset.csv \
  --outdir results/reproduced_experiment_ix
```

If `--csv` is omitted, the script checks several default locations, including
paths from its original execution environment. Supplying `--csv` is therefore
recommended on Windows, macOS and Linux. The script writes its generated files
to the directory specified by `--outdir`. This diagnostic is supplementary;
it is not the primary manuscript analysis, and its log-loss clipping convention
must remain distinct from the primary Version 4 implementation.

## Recorded computational environment

The successful full Version 4 run recorded:

- Python 3.13.16
- NumPy 2.4.6
- pandas 2.2.3
- SciPy 1.16.3
- scikit-learn 1.6.1
- Matplotlib 3.10.0
- SHAP 0.54.1

The platform and complete run metadata are recorded in
`reproduction_evidence/version4_full_reproduction_2026-11-11/reproduction_manifest.json`.
The pinned package versions are listed in `requirements-execution.txt`.
The recorded environment does not imply bit-identical results on every platform.

## Output naming and legacy filenames

Some output filenames retain the names emitted by the checksum-verified
Version 4 implementation, including `table_1_data_audit.csv`. In this context,
the file reports dataset characteristics and data checks; the filename does
not refer to an internal manuscript or peer-review audit. The primary script
and its output filenames are preserved to maintain consistency with the
recorded reproduction evidence.

## Reproduction record

The full Version 4 run exited successfully, and the run record indicates that
the core outputs and SHAP stability outputs were generated. The reference-output
comparison contains 12 comparisons with zero absolute difference. This is
numerical agreement with the archived reference outputs, not external validation.

Model selection was based on stratified cross-validation using the training
partition. The held-out test set was not used to select the model. Logistic
Regression was the training-CV-selected model used for the SHAP analysis.

## Interpretation and limitations

- The dataset-specific results do not establish clinical validity.
- No external validation was performed.
- The target-label generation procedure remains unverified.
- The observed association involving `blood_pressure` does not by itself prove
  target leakage; feature exclusion also cannot establish whether the target
  was generated using that variable.
- Corruption levels are probabilities under the implemented perturbation
  protocol, not calibrated real-world measurement-error magnitudes.
- SHAP results are global descriptive summaries for the selected Logistic
  Regression model. They are not causal explanations or participant-level
  stability estimates.

## Citation, licence and data reuse

Citation metadata is provided in `CITATION.cff`. The code and accompanying
reproducibility documentation in this repository are released under the MIT
Licence; see `LICENSE`. This licence does not apply to third-party datasets,
which are not redistributed here. Obtain the dataset from its original provider,
check the provider's current terms, and comply with any separate restrictions
on dataset access, reuse, or redistribution. A repository URL and DOI will be
added to the citation metadata after the public GitHub/Zenodo deposit.
