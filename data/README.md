# Dataset

The analysis uses `StressLevelDataset.csv` (1,100 observations, 20 predictors
and the `stress_level` target). The dataset is not redistributed here.

Dataset source page:
https://www.kaggle.com/datasets/samyakb/student-stress-factors

Obtain the file from the source and review its current terms before reuse.
Place it in the repository root or pass its location using `--csv`.

Verify the downloaded file against the SHA-256 checksum in the
`reproduction_manifest.json` before treating it as the same dataset used in
the recorded reproduction. The expected structure is 1,100 rows and 21 columns.

The original target-label generation procedure and substantive meanings of
the `blood_pressure` codes were not independently established by the study.
