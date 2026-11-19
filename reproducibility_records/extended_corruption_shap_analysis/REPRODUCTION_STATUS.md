# Extended corruption and global SHAP analysis: reproduction record

## Analysis materials

The implementation is `scripts/sensitivity/extended_blood_pressure_corruption_shap_analysis.py`.
The canonical saved outputs are in `results/extended_bp_exclusion/`.

The dataset used for reproduction contained 1,100 rows and 21 columns
(20 predictors and the `stress_level` target). Its LF-normalised SHA-256
digest is:

`14a45e92708b0c063ad4ab04563aa8fd4e3fc27157fd282e1c0658dc5161faed`

This matches the dataset identity recorded in the experiment metadata.

## Independent reproduction

The analysis was rerun on 19 November 2026 using the supplied dataset and
the repository script. The process completed with exit code 0 in
approximately 55 seconds.

Runtime:

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- SHAP 0.50.0

All five regenerated CSV outputs matched their saved counterparts
byte-for-byte, and their SHA-256 checksums matched:

- `BP_exclusion_corruption_performance_repeats.csv`
- `BP_exclusion_corruption_performance_summary.csv`
- `BP_exclusion_SHAP_stability_repeats.csv`
- `BP_exclusion_SHAP_stability_summary.csv`
- `BP_exclusion_SHAP_importance_by_condition.csv`

The regenerated `extended_experiment_metadata.json` also matched the
recorded metadata exactly. The machine-readable comparisons are included in
`reproducibility_records/independent_reproduction_2026-11-19/output_comparison.csv`.

A host-level spreadsheet-runtime warm-up warning appeared on standard error
during Python startup. The analysis process nevertheless returned exit code
0 and produced byte-identical outputs.

## Reproduction command

Run from the repository root:

```bash
python scripts/sensitivity/extended_blood_pressure_corruption_shap_analysis.py \
  --csv StressLevelDataset.csv \
  --outdir results/extended_bp_exclusion_reproduced
```

## Interpretation and limitations

This analysis uses synthetic corruption, one fixed train/test partition and
global SHAP summaries. It does not provide participant-level SHAP stability
or external validation. Excluding `blood_pressure` does not resolve target
provenance or establish the absence of target leakage.
