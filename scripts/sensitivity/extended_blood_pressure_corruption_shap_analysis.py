#!/usr/bin/env python3
"""Extended paired corruption and global-SHAP analysis with/without blood_pressure.

Uses the Version 1.0.3 fixed stratified split and the repository's canonical
model definitions and corruption operator. Extends SHAP explanation coverage
to all held-out observations, a 100-row training background, and 20 repeats
per corruption level. Shared predictors receive the same feature-specific
perturbation seed across predictor conditions.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "primary"))
import primary_stress_robustness_analysis as primary

LEVELS = (0.05, 0.10, 0.20)
N_REPEATS = 20
N_BACKGROUND = 100

def perturb_stable(X, reference, level, repeat, seed_base=primary.SEED):
    """Apply the repository corruption operator with stable feature-specific seeds."""
    out = X.copy(deep=True)
    for feature in X.columns:
        seed_bytes = hashlib.sha256(
            f"{seed_base}|{feature}|{level:.2f}|{repeat}".encode()
        ).digest()[:8]
        seed = int.from_bytes(seed_bytes, "little") % (2**32 - 1)
        rng = np.random.default_rng(seed)
        values = reference[feature].dropna().to_numpy()
        unique = np.unique(values)
        mask = rng.random(len(out)) < level
        if len(unique) <= 12:
            for row in np.where(mask)[0]:
                alternatives = unique[unique != out.iloc[row][feature]]
                if len(alternatives):
                    out.iat[row, out.columns.get_loc(feature)] = rng.choice(alternatives)
        else:
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if sd > 0:
                out.loc[mask, feature] = (
                    out.loc[mask, feature] + rng.normal(0, level * sd, int(mask.sum()))
                )
                out[feature] = out[feature].clip(float(np.min(values)), float(np.max(values)))
    return out

def shap_global(model, background, explained, feature_names, n_classes):
    """Mean absolute SHAP averaged over rows and class outputs."""
    estimator = model.steps[-1][1]
    bg = model[:-1].transform(background)
    xx = model[:-1].transform(explained)
    explainer = shap.LinearExplainer(estimator, bg)
    values = explainer.shap_values(xx)
    if isinstance(values, list):
        arr = np.stack([np.asarray(v) for v in values], axis=-1)
    else:
        arr = np.asarray(values)
    n, p = len(explained), len(feature_names)
    if arr.ndim == 2 and arr.shape == (n, p):
        return np.mean(np.abs(arr), axis=0)
    if arr.ndim == 3:
        if arr.shape[0] == n and arr.shape[1] == p:       # rows, features, classes
            normalized = arr
        elif arr.shape[0] == n_classes and arr.shape[1] == n and arr.shape[2] == p:
            normalized = np.transpose(arr, (1, 2, 0))    # classes, rows, features
        elif arr.shape[0] == n and arr.shape[2] == p:
            normalized = np.transpose(arr, (0, 2, 1))    # rows, classes, features
        else:
            raise ValueError(f"Unrecognized SHAP array shape: {arr.shape}")
        return np.mean(np.abs(normalized), axis=(0, 2))
    raise ValueError(f"Unexpected SHAP array dimensions: {arr.shape}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(REPO / "StressLevelDataset.csv"))
    parser.add_argument("--outdir", default=str(REPO / "results" / "extended_bp_exclusion"))
    args = parser.parse_args()
    csv_path, out = Path(args.csv), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    Xall, y = df.drop(columns=["stress_level"]), df["stress_level"]
    Xtr0, Xte0, ytr, yte = train_test_split(
        Xall, y, test_size=0.20, stratify=y, random_state=primary.SEED
    )
    conditions = {
        "all_predictors": list(Xall.columns),
        "without_blood_pressure": [c for c in Xall.columns if c != "blood_pressure"],
    }
    rng = np.random.default_rng(primary.SEED + 771)
    bg_positions = np.sort(rng.choice(len(Xtr0), size=N_BACKGROUND, replace=False))
    Xbg0 = Xtr0.iloc[bg_positions].copy()
    performance, shap_rows, importance_rows = [], [], []
    fitted_lr = {}
    for condition, columns in conditions.items():
        Xtr, Xte = Xtr0[columns].copy(), Xte0[columns].copy()
        for model_name, model in primary.make_models().items():
            fitted = clone(model).fit(Xtr, ytr)
            if model_name == "LogisticRegression":
                fitted_lr[condition] = fitted
            clean = f1_score(yte, fitted.predict(Xte), average="macro", zero_division=0)
            performance.append({
                "condition": condition, "model": model_name, "noise_level": 0.0,
                "repeat": 0, "macro_f1": clean, "delta_macro_f1": 0.0
            })
            for level in LEVELS:
                for repeat in range(1, N_REPEATS + 1):
                    perturbed = perturb_stable(Xte, Xtr, level, repeat)
                    score = f1_score(
                        yte, fitted.predict(perturbed), average="macro", zero_division=0
                    )
                    performance.append({
                        "condition": condition, "model": model_name,
                        "noise_level": level, "repeat": repeat, "macro_f1": score,
                        "delta_macro_f1": score - clean
                    })
    perf = pd.DataFrame(performance)
    perf.to_csv(out / "BP_exclusion_corruption_performance_repeats.csv", index=False)
    summary = (perf[perf.noise_level > 0]
        .groupby(["condition", "model", "noise_level"])
        .agg(n_repeats=("repeat", "count"), macro_f1_mean=("macro_f1", "mean"),
             macro_f1_sd=("macro_f1", "std"), delta_macro_f1_mean=("delta_macro_f1", "mean"),
             delta_macro_f1_sd=("delta_macro_f1", "std")).reset_index())
    summary.to_csv(out / "BP_exclusion_corruption_performance_summary.csv", index=False)

    for condition, columns in conditions.items():
        Xtr, Xte = Xtr0[columns].copy(), Xte0[columns].copy()
        background = Xbg0[columns].copy()
        model = fitted_lr[condition]
        base = shap_global(model, background, Xte, columns, len(np.unique(ytr)))
        for feature, value in zip(columns, base):
            importance_rows.append({
                "condition": condition, "noise_level": 0.0, "repeat": 0,
                "feature": feature, "mean_abs_shap": float(value)
            })
        for level in LEVELS:
            for repeat in range(1, N_REPEATS + 1):
                perturbed = perturb_stable(Xte, Xtr, level, repeat)
                current = shap_global(model, background, perturbed, columns, len(np.unique(ytr)))
                rank = primary.rank_metrics(base, current)
                shap_rows.append({
                    "condition": condition, "noise_level": level, "repeat": repeat, **rank
                })
                for feature, value in zip(columns, current):
                    importance_rows.append({
                        "condition": condition, "noise_level": level, "repeat": repeat,
                        "feature": feature, "mean_abs_shap": float(value)
                    })
    shap_repeat_df = pd.DataFrame(shap_rows)
    importance_df = pd.DataFrame(importance_rows)
    shap_repeat_df.to_csv(out / "BP_exclusion_SHAP_stability_repeats.csv", index=False)
    importance_df.to_csv(out / "BP_exclusion_SHAP_importance_by_condition.csv", index=False)
    shap_summary = (shap_repeat_df.groupby(["condition", "noise_level"])
        .agg(n_repeats=("repeat", "count"),
             spearman_mean=("spearman_rank_correlation", "mean"),
             spearman_sd=("spearman_rank_correlation", "std"),
             kendall_mean=("kendall_rank_correlation", "mean"),
             kendall_sd=("kendall_rank_correlation", "std"),
             cosine_mean=("importance_cosine_similarity", "mean"),
             cosine_sd=("importance_cosine_similarity", "std"),
             top5_overlap_mean=("top_5_overlap", "mean"),
             top5_overlap_sd=("top_5_overlap", "std")).reset_index())
    shap_summary.to_csv(out / "BP_exclusion_SHAP_stability_summary.csv", index=False)
    metadata = {
        "experiment": "Extended paired corruption and global SHAP stability with/without blood_pressure",
        "dataset": csv_path.name,
        "dataset_sha256_lf": hashlib.sha256(csv_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "seed_split": primary.SEED, "split": "stratified 80:20; fixed Version 1.0.3 split",
        "train_n": len(Xtr0), "test_n": len(Xte0), "background_n": N_BACKGROUND,
        "shap_explanation_n": len(Xte0), "shap_repetitions_per_level": N_REPEATS,
        "corruption_levels": list(LEVELS), "models": list(primary.make_models().keys()),
        "shap_model": "LogisticRegression in both conditions; fixed model family",
        "perturbation": "per-feature Bernoulli selection; <=12 unique training values replaced by another training category; >12 unique values receive Gaussian noise with SD=level*training SD and clipping to training min/max",
        "paired_corruption_seeds": "stable feature-specific seed; shared predictors receive identical perturbations across conditions",
        "background_sampling": "100 training rows without replacement; same row positions in both conditions",
        "explanation_rows": "all held-out observations",
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "pandas": pd.__version__, "scikit_learn": __import__("sklearn").__version__,
                    "shap": shap.__version__},
        "limitations": ["Synthetic corruption only", "One fixed train/test partition",
            "Global, not participant-level, SHAP stability",
            "Feature exclusion does not resolve target provenance or prove absence of leakage"]
    }
    (out / "extended_experiment_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Extended experiment completed:", out)

if __name__ == "__main__":
    main()
