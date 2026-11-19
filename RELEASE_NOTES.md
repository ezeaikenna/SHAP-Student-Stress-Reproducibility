# Release information

## Version 1.0.4 — 19 November 2026

This version expands the repository's reproducibility materials and clarifies
the scope and provenance of the analyses.

### Contents and changes

- Adds the extended blood-pressure-exclusion corruption and global SHAP
  stability analysis script and its generated result files.
- Documents the extended experiment's background sampling and deterministic
  feature-specific perturbation seed procedure.
- Provides execution metadata, saved outputs and independent reproduction
  records for the primary, supplementary diagnostic, blood-pressure
  sensitivity and extended corruption/global SHAP analyses. In the recorded
  validation, all five regenerated CSV outputs matched their saved reference
  outputs exactly when compared as dataframes; this statement does not imply
  byte-for-byte identity of the CSV files.
- Excludes `StressLevelDataset.csv` because it is a third-party dataset whose
  redistribution terms have not been established by this project. Users
  obtain the dataset from its original source and are responsible for
  complying with the provider's terms.
- Retains the SHA-256 checksum of the dataset used in the recorded analyses
  so that users can check the identity of their local copy.
- Includes a SHA-256 manifest for the files in this source distribution.

### Scientific scope and limitations

The release does not establish the original target-label generation procedure
or the substantive meanings of the `blood_pressure` codes. Excluding that
predictor does not establish the absence of target leakage. The extended
perturbations are synthetic, SHAP stability is global rather than
participant-level, and the analyses use one fixed held-out partition.

### Licence and dataset

The MIT licence applies to the code and accompanying reproducibility
documentation authored for this repository. It does not apply to the
third-party dataset, which is not included. Dataset access and reuse remain
subject to the provider's terms.

### Repository archive

The source archive contains the repository files required for analysis and
reproduction. Git history is maintained in the GitHub repository rather than
in this ZIP source distribution.

### DOI

Zenodo concept DOI (all versions):
https://doi.org/10.5281/zenodo.24250742

Version-specific DOI for Version 1.0.4:
https://doi.org/10.5281/zenodo.24381007

Cite the version-specific DOI when referring specifically to Version 1.0.4.
Use the concept DOI when referring to the repository across versions.
