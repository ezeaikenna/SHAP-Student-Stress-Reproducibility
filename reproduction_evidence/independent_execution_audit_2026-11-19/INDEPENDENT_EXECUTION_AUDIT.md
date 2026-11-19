# Independent execution audit — 19 November 2026

## Overall result

All three current analysis scripts completed successfully against the uploaded
`StressLevelDataset.csv` in an extracted copy of the repository:

- Primary analysis: exit code 0; completed in approximately 30.1 seconds.
- Diagnostic-run reproduction: exit code 0; completed in approximately 28.2 seconds.
- Blood-pressure sensitivity analysis: exit code 0; completed in approximately 29.4 seconds.

The primary and diagnostic scripts reported `Core outputs: PASS`, `SHAP
stability: PASS`, and no missing core outputs. The sensitivity script reported
`SHAP errors: []`.

## Dataset identity

The uploaded CSV contained 1,100 rows and 21 columns (20 predictors and the
`stress_level` target), with no schema discrepancy. The uploaded file used
CRLF line endings. Normalising line endings to LF, without changing field
values, produced SHA-256:

`14a45e92708b0c063ad4ab04563aa8fd4e3fc27157fd282e1c0658dc5161faed`

This exactly matches the dataset checksum in the historical reproduction
manifest. The uploaded dataset is therefore byte-equivalent to the manifest's
LF-normalised source file.

## Output comparison

The machine-readable comparison is in `output_comparison.csv`.

- Primary analysis: all compared numeric result tables matched the archived
  reference values exactly, including cross-validation, held-out performance,
  noise robustness, feature ablation, SHAP stability and SHAP importance.
- Primary classification reports and confusion matrices matched exactly.
- The four regenerated primary figures were pixel-identical to their archived
  counterparts.
- Diagnostic-run reproduction: the retained Decision Tree case-probability
  CSV matched the repository's saved diagnostic CSV exactly.
- Blood-pressure sensitivity analysis: all six compared CSV outputs matched
  the repository's saved sensitivity outputs exactly.
- The primary script's case-level Decision Tree CSV differed from the
  historical case-level log-loss values only because the current primary
  implementation uses a 1e-15 clipping lower bound while the historical
  run used machine-epsilon clipping. All other columns, including predicted
  probabilities and labels, were identical. The maximum absolute case-level
  log-loss difference was 1.5048769942. This is the documented clipping
  distinction, not a prediction discrepancy.
- The dataset-characteristics CSV records the actual temporary execution path
  and uses updated capitalization for two limitation labels; the measured
  dataset characteristics themselves are unchanged.

## Runtime environment qualification

This execution used the available Python 3.13.5 runtime and installed package
versions: NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0,
Matplotlib 3.10.8 and SHAP 0.50.0. These differ from the historical pinned
environment (Python 3.13.16, NumPy 2.4.6, pandas 2.2.3, SciPy 1.16.3,
scikit-learn 1.6.1, Matplotlib 3.10.0 and SHAP 0.54.1).

Accordingly, this is a successful independent execution and output-consistency
check in the available environment. It is not a bit-for-bit recreation of the
historical software environment. The observed tabular numerical outputs
matched as described above.

A spreadsheet-runtime warm-up warning appeared on the sensitivity script's
standard error stream, but the script exited successfully, generated its
outputs, and its SHAP error list was empty. The warning did not prevent the
analysis from completing.

## Reproduction conclusion

The current primary, diagnostic and blood-pressure sensitivity scripts all
executed successfully on the checksum-verified dataset. The key numerical
outputs matched their archived references exactly, subject to the documented
case-level probability-clipping difference. This supports a strong
reproducibility claim for the reported computations in the tested runtime.

It does not establish external validation, clinical validity, or verification
of the dataset's original target-generation procedure.
