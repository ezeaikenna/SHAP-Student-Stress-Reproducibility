# Independent reproduction record — 19 November 2026

## Execution summary

The primary analysis, supplementary diagnostic analysis, blood-pressure
sensitivity analysis and extended corruption/global SHAP analysis scripts
completed successfully using the supplied `StressLevelDataset.csv` in an
extracted copy of the repository:

- Primary analysis: exit code 0; approximately 30.1 seconds.
- Supplementary diagnostic analysis: exit code 0; approximately 28.2 seconds.
- Blood-pressure sensitivity analysis: exit code 0; approximately 29.4 seconds.
- Extended corruption and global SHAP analysis: exit code 0; approximately 55.0 seconds.

The primary and supplementary diagnostic scripts generated their core
outputs and SHAP stability results. The sensitivity analysis completed with
an empty SHAP computation-message file.

## Dataset identity

The supplied CSV contained 1,100 rows and 21 columns (20 predictors and the
`stress_level` target), with no schema discrepancy. The uploaded file used
CRLF line endings. Normalising line endings to LF, without changing field
values, produced SHA-256:

`14a45e92708b0c063ad4ab04563aa8fd4e3fc27157fd282e1c0658dc5161faed`

This matches the dataset checksum in the recorded execution manifest.

## Output comparison

The machine-readable comparisons are provided in `output_comparison.csv`.

- Primary analysis: compared numerical result tables matched the saved
  reference values, including cross-validation, held-out performance, noise
  robustness, feature ablation, SHAP stability and SHAP importance.
- Primary classification reports and confusion matrices matched the saved
  references.
- The four regenerated primary figures were pixel-identical to their saved
  counterparts.
- Supplementary diagnostic analysis: the retained Decision Tree
  case-probability CSV matched its saved reference.
- Blood-pressure sensitivity analysis: the six compared CSV outputs matched
  their saved references.
- Extended corruption and global SHAP analysis: all five generated CSV
  outputs matched their saved references byte-for-byte; all five SHA-256
  checksums were identical. The regenerated experiment metadata also matched
  the saved metadata exactly.
- The primary script's case-level Decision Tree log-loss values differed
  from the earlier saved values because the current primary implementation
  uses a 1e-15 clipping lower bound while the earlier run used machine-epsilon
  clipping. Predicted probabilities and labels were identical. The maximum
  absolute case-level log-loss difference was 1.5048769942.
- The dataset-characteristics output records the dataset filename rather
  than the temporary execution path; measured dataset characteristics were
  unchanged.

## Runtime environment qualification

This reproduction used Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SciPy
1.17.0, scikit-learn 1.8.0, Matplotlib 3.10.8 and SHAP 0.50.0. These differ
from the historically recorded environment (Python 3.13.16, NumPy 2.4.6,
pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.6.1, Matplotlib 3.10.0 and
SHAP 0.54.1).

Accordingly, this record documents successful execution and output
consistency in the available runtime. It does not claim recreation of the
exact historical software environment. The compared numerical outputs
matched as described above.

A spreadsheet-runtime warm-up warning appeared on the sensitivity script's
standard error stream. The script nevertheless completed, generated its
outputs, and reported no SHAP computation messages. The warning did not
prevent completion.

## Reproduction scope

The results support reproducibility of the reported computations in the
tested runtime, including the extended corruption and global SHAP analysis.
They do not establish external validation, clinical validity, or verification
of the dataset's original target-generation procedure.


## Extended analysis execution note

The extended analysis process returned exit code 0 and generated all five
expected CSV outputs. A host-level spreadsheet-runtime warm-up warning was
printed to standard error during Python startup; it did not originate from
the analysis script, prevent execution, or alter the generated outputs.
