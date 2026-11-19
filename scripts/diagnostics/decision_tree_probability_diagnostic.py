#!/usr/bin/env python3
"""Reproduction of the archived diagnostic analysis run.

This script reproduces the diagnostic run's six-model evaluation, robustness,
feature-ablation and SHAP analyses. Its Decision Tree case-probability output
uses machine-epsilon clipping for case-level log loss; the primary analysis
uses a 1e-15 lower bound. Consequently, case-level log-loss values from the
two runs are not directly interchangeable. The diagnostic run is not a
Decision-Tree-only workflow.

Important: target-label provenance is unverified. Results support benchmarking
on this dataset only, not clinical validity or future prediction.
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, classification_report,
                             confusion_matrix, roc_auc_score, log_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

warnings.filterwarnings('ignore')
SEED = 20261009
NOISE_LEVELS = (0.05, 0.10, 0.20)
N_NOISE_REPEATS = 10
N_BOOTSTRAP = 1000
N_SHAP_REPEATS = 5
SHAP_N_EXPLAIN = 20
SHAP_BACKGROUND = 20


def make_models(seed=SEED):
    return {
        'LogisticRegression': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight=None, random_state=seed)),
        'SVM_RBF': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
            SVC(C=1.0, kernel='rbf', probability=True, random_state=seed)),
        'RandomForest': make_pipeline(SimpleImputer(strategy='median'),
            RandomForestClassifier(n_estimators=50, min_samples_leaf=1, n_jobs=-1, random_state=seed)),
        'ExtraTrees': make_pipeline(SimpleImputer(strategy='median'),
            ExtraTreesClassifier(n_estimators=50, min_samples_leaf=1, n_jobs=-1, random_state=seed)),
        'MLP': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, early_stopping=True,
                          random_state=seed)),
        'DecisionTree': make_pipeline(SimpleImputer(strategy='median'),
            DecisionTreeClassifier(random_state=seed, min_samples_leaf=2)),
    }


def metrics(y, pred, prob=None):
    out = {'accuracy': accuracy_score(y, pred),
           'balanced_accuracy': balanced_accuracy_score(y, pred),
           'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
           'macro_precision': precision_score(y, pred, average='macro', zero_division=0),
           'macro_recall': recall_score(y, pred, average='macro', zero_division=0)}
    if prob is not None:
        try: out['roc_auc_ovr_macro'] = roc_auc_score(y, prob, multi_class='ovr', average='macro')
        except Exception: out['roc_auc_ovr_macro'] = np.nan
        try: out['log_loss'] = log_loss(y, prob, labels=np.unique(y))
        except Exception: out['log_loss'] = np.nan
    return out


def perturb_frame(X, level, seed, reference):
    """Corrupt evaluation rows using statistics/categories learned from training only.
    `reference` must be Xtr; held-out feature values are never used to define
    replacement categories, clipping bounds, or noise scale.
    """
    rng = np.random.default_rng(seed)
    out = X.copy(deep=True)
    for c in X.columns:
        vals = reference[c].dropna().to_numpy()
        uniq = np.unique(vals)
        n = len(out)
        mask = rng.random(n) < level
        if len(uniq) <= 12:
            rows = np.where(mask)[0]
            if len(rows):
                for r in rows:
                    alternatives = uniq[uniq != out.iloc[r][c]]
                    if len(alternatives): out.iat[r, out.columns.get_loc(c)] = rng.choice(alternatives)
        else:
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if sd > 0:
                out.loc[mask, c] = out.loc[mask, c] + rng.normal(0, level * sd, mask.sum())
                out[c] = out[c].clip(float(np.min(vals)), float(np.max(vals)))
    return out


def bootstrap_ci(y, pred, metric_func, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pred = np.asarray(pred)
    vals=[]; n=len(y)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        if len(np.unique(y[ix])) < 2: continue
        vals.append(metric_func(y[ix], pred[ix]))
    if not vals: return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rank_metrics(reference, current, top_k=5):
    a=np.asarray(reference, dtype=float); b=np.asarray(current, dtype=float)
    if np.std(a)==0 or np.std(b)==0: rho=np.nan; tau=np.nan
    else:
        rho=float(spearmanr(a,b).statistic); tau=float(kendalltau(a,b).statistic)
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    cosine=float(np.dot(a,b)/denom) if denom else np.nan
    ka=set(np.argsort(a)[::-1][:top_k]); kb=set(np.argsort(b)[::-1][:top_k])
    overlap=len(ka & kb)/top_k
    return {'spearman_rank_correlation':rho, 'kendall_rank_correlation':tau,
            'importance_cosine_similarity':cosine, f'top_{top_k}_overlap':overlap}


def get_shap_importance(model, X_bg, X_exp, feature_names):
    import shap
    # Use exact/analytic explainers where available to keep the primary rerun practical.
    final_est = model.steps[-1][1] if hasattr(model, 'steps') else model
    if isinstance(final_est, LogisticRegression) and hasattr(model, 'steps'):
        bg = model[:-1].transform(X_bg)
        xx = model[:-1].transform(X_exp)
        explainer = shap.LinearExplainer(final_est, bg)
        sv = explainer.shap_values(xx)
        if isinstance(sv, list):
            values = np.stack([np.asarray(v) for v in sv], axis=-1)
        else:
            values = np.asarray(sv)
    elif final_est.__class__.__name__ in ('RandomForestClassifier','ExtraTreesClassifier','DecisionTreeClassifier') and hasattr(model, 'steps'):
        bg = model[:-1].transform(X_bg)
        xx = model[:-1].transform(X_exp)
        explainer = shap.TreeExplainer(final_est, data=bg, feature_perturbation='interventional')
        values = np.asarray(explainer.shap_values(xx))
        if isinstance(explainer.shap_values(xx), list):
            values = np.stack([np.asarray(v) for v in explainer.shap_values(xx)], axis=-1)
    else:
        # Model-agnostic permutation SHAP fallback for other model families.
        masker = shap.maskers.Independent(X_bg, max_samples=len(X_bg))
        explainer = shap.Explainer(model.predict_proba, masker, algorithm='permutation')
        exp = explainer(X_exp, max_evals=2*len(feature_names)+1, batch_size=100)
        values = np.asarray(exp.values)
    # Normalize multiclass SHAP output to (rows, features, classes).
    if values.ndim == 3 and values.shape[1] == len(feature_names):
        imp = np.mean(np.abs(values), axis=(0, 2))
    elif values.ndim == 3 and values.shape[0] == len(np.unique(final_est.classes_)):
        imp = np.mean(np.abs(values), axis=(0, 2))
    elif values.ndim == 2 and values.shape[1] == len(feature_names):
        imp = np.mean(np.abs(values), axis=0)
    else:
        raise ValueError(f'Unexpected SHAP array shape: {values.shape}')
    return np.asarray(imp, dtype=float)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', default=None, help='Path to StressLevelDataset.csv')
    ap.add_argument('--outdir', default='decision_tree_diagnostic_outputs')
    ap.add_argument('--fast', action='store_true', help='Run a reduced-computation analysis for exploratory checks; do not use these outputs as the full study results')
    ap.add_argument('--skip-shap', action='store_true', help='Skip SHAP if not installed or runtime is limited')
    ap.add_argument('--shap-all-models', action='store_true', help='Run the slower model-agnostic SHAP stability analysis for all six models; default is the CV-selected model only')
    # Jupyter/Google Colab injects its own command-line arguments (e.g. -f kernel.json).
    # Ignore unknown notebook arguments so the script can run inside a notebook cell.
    args, unknown_args = ap.parse_known_args()
    if unknown_args:
        print('Ignoring notebook/runtime arguments:', ' '.join(unknown_args), flush=True)
    out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    n_noise=3 if args.fast else N_NOISE_REPEATS
    n_boot=200 if args.fast else N_BOOTSTRAP
    n_shap=2 if args.fast else N_SHAP_REPEATS
    n_shap_exp=12 if args.fast else SHAP_N_EXPLAIN
    started=time.time()
    if args.csv:
        csv_path=Path(args.csv)
    else:
        candidates=[Path('StressLevelDataset.csv')]
        csv_path=next((p for p in candidates if p.exists()), None)
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError('CSV not found. Pass --csv /path/to/StressLevelDataset.csv')
    df=pd.read_csv(csv_path)
    target='stress_level'
    if target not in df.columns: raise ValueError(f'Missing required target column: {target}')
    if df[target].isna().any(): raise ValueError('Target contains missing values; resolve before running.')
    X=df.drop(columns=[target]).copy(); y=df[target].copy()
    # Dataset characteristics
    dataset_profile=pd.DataFrame([
        ['source_file', str(csv_path)], ['rows', len(df)], ['columns_including_target', df.shape[1]],
        ['feature_count', X.shape[1]], ['missing_cells', int(df.isna().sum().sum())],
        ['duplicate_rows', int(df.duplicated().sum())], ['target_column', target],
        ['target_provenance_verified', 'NO - requires independent documentation'],
        ['external_validation', 'NOT PERFORMED'],
    ], columns=['item','value'])
    dataset_profile.to_csv(out/'table_1_dataset_characteristics.csv', index=False)
    y.value_counts().sort_index().rename_axis('stress_level').reset_index(name='n').to_csv(out/'table_target_distribution.csv', index=False)
    # One fixed holdout. Everything else is fit on training partition only.
    Xtr, Xte, ytr, yte=train_test_split(X,y,test_size=0.20,stratify=y,random_state=SEED)
    models=make_models()
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    scoring={'accuracy':'accuracy','balanced_accuracy':'balanced_accuracy','macro_f1':'f1_macro'}
    cv_rows=[]; fitted={}; test_rows=[]; pred_store={}
    print(f'Data: {df.shape}; train={len(Xtr)}, test={len(Xte)}; features={X.shape[1]}')
    for name, model in models.items():
        print('CV and fit:', name, flush=True)
        cvres=cross_validate(clone(model), Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
        cv_rows.append({'model':name, **{f'cv_{k[5:]}_mean':float(np.mean(v)) for k,v in cvres.items() if k.startswith('test_')},
                        **{f'cv_{k[5:]}_sd':float(np.std(v,ddof=1)) for k,v in cvres.items() if k.startswith('test_')}})
        m=clone(model); m.fit(Xtr,ytr); fitted[name]=m
        pred=m.predict(Xte); prob=m.predict_proba(Xte); pred_store[name]=(pred,prob)
        if name == 'DecisionTree':
            case = pd.DataFrame({'test_row_position': np.arange(len(yte)), 'true_label': np.asarray(yte),
                                 'predicted_label': np.asarray(pred), 'true_class_probability': prob[np.arange(len(yte)), [list(m.classes_).index(v) for v in yte]]})
            for j, cls in enumerate(m.classes_): case[f'prob_class_{cls}'] = prob[:, j]
            eps = np.finfo(prob.dtype).eps
            case['case_log_loss'] = -np.log(np.clip(case['true_class_probability'], eps, 1.0 - eps))
            case.sort_values('case_log_loss', ascending=False).to_csv(out/'decision_tree_test_case_probabilities.csv', index=False)
        row={'model':name, **metrics(yte,pred,prob)}
        lo,hi=bootstrap_ci(yte,pred,lambda a,b:f1_score(a,b,average='macro',zero_division=0),n_boot=n_boot,seed=SEED)
        row['macro_f1_ci95_low']=lo; row['macro_f1_ci95_high']=hi
        test_rows.append(row)
        pd.DataFrame(classification_report(yte,pred,output_dict=True,zero_division=0)).T.to_csv(out/f'classification_report_{name}.csv')
        pd.DataFrame(confusion_matrix(yte,pred,labels=sorted(y.unique())),index=[f'true_{c}' for c in sorted(y.unique())],columns=[f'pred_{c}' for c in sorted(y.unique())]).to_csv(out/f'confusion_matrix_{name}.csv')
    cv_df=pd.DataFrame(cv_rows).sort_values('cv_macro_f1_mean',ascending=False)
    cv_df.to_csv(out/'table_2_cross_validation.csv',index=False)
    test_df=pd.DataFrame(test_rows).sort_values('macro_f1',ascending=False)
    test_df.to_csv(out/'table_3_test_performance.csv',index=False)
    # Repeated input perturbation on the fixed held-out data, model not refit.
    noise_rows=[]
    for name,m in fitted.items():
        base_pred,base_prob=pred_store[name]; base=metrics(yte,base_pred,base_prob)
        for level in NOISE_LEVELS:
            for rep in range(n_noise):
                Xpert=perturb_frame(Xte,level,SEED+1000+rep+int(level*10000),Xtr)
                p=m.predict(Xpert); pr=m.predict_proba(Xpert)
                noise_rows.append({'model':name,'noise_level':level,'repeat':rep+1,**metrics(yte,p,pr),
                    'delta_macro_f1_vs_clean':f1_score(yte,p,average='macro',zero_division=0)-base['macro_f1']})
    noise=pd.DataFrame(noise_rows); noise.to_csv(out/'table_4_noise_robustness_repeats.csv',index=False)
    noise.groupby(['model','noise_level']).agg({c:['mean','std','min','max'] for c in ['accuracy','balanced_accuracy','macro_f1','delta_macro_f1_vs_clean']}).to_csv(out/'table_4_noise_robustness_summary.csv')
    # Retrained feature ablation: remove one feature, refit each model from scratch.
    ab_rows=[]
    for name, model in models.items():
        base_pred,base_prob=pred_store[name]; base=metrics(yte,base_pred,base_prob)
        for feat in X.columns:
            m=clone(model); m.fit(Xtr.drop(columns=[feat]),ytr)
            pred=m.predict(Xte.drop(columns=[feat])); prob=m.predict_proba(Xte.drop(columns=[feat]))
            mm=metrics(yte,pred,prob)
            ab_rows.append({'model':name,'removed_feature':feat,**{f'ablated_{k}':v for k,v in mm.items()},
                'baseline_macro_f1':base['macro_f1'],'delta_macro_f1':mm['macro_f1']-base['macro_f1'],
                'macro_f1_drop':base['macro_f1']-mm['macro_f1']})
    ab=pd.DataFrame(ab_rows); ab.to_csv(out/'table_5_retrained_feature_ablation.csv',index=False)
    ab.groupby(['model']).apply(lambda g:g.sort_values('macro_f1_drop',ascending=False).head(5),include_groups=False).reset_index().to_csv(out/'table_5b_top_ablation_features.csv',index=False)
    # SHAP stability: common background/explanation sample; explain same fitted model
    # on clean inputs and independently perturbed versions. Repeat perturbations.
    shap_rows=[]; shap_importance_rows=[]; shap_errors=[]
    if not args.skip_shap:
        try:
            import shap  # noqa
            rng=np.random.default_rng(SEED)
            bg_idx=rng.choice(len(Xtr),size=min(SHAP_BACKGROUND,len(Xtr)),replace=False)
            exp_idx=rng.choice(len(Xte),size=min(n_shap_exp,len(Xte)),replace=False)
            Xbg=Xtr.iloc[bg_idx].copy(); Xexp=Xte.iloc[exp_idx].copy()
            shap_model_names = list(fitted.keys()) if args.shap_all_models else [str(cv_df.iloc[0]['model'])]
            print('SHAP model scope:', ', '.join(shap_model_names), '(selected using training-only CV)' if not args.shap_all_models else '(all models)', flush=True)
            for name in shap_model_names:
                m = fitted[name]
                print('SHAP stability:',name,flush=True)
                try:
                    base_imp=get_shap_importance(m,Xbg,Xexp,list(X.columns))
                    for feat,val in zip(X.columns,base_imp):
                        shap_importance_rows.append({'model':name,'condition':'clean','repeat':0,'feature':feat,'mean_abs_shap':val})
                    for level in (0.0,)+NOISE_LEVELS:
                        repeats=1 if level==0 else n_shap
                        for rep in range(repeats):
                            xx=Xexp.copy() if level==0 else perturb_frame(Xexp,level,SEED+90000+rep+int(level*10000),Xtr)
                            imp=base_imp if level==0 else get_shap_importance(m,Xbg,xx,list(X.columns))
                            for feat,val in zip(X.columns,imp):
                                shap_importance_rows.append({'model':name,'condition':'clean' if level==0 else f'noise_{int(level*100)}pct','repeat':rep+1,'feature':feat,'mean_abs_shap':val})
                            if level>0:
                                row={'model':name,'noise_level':level,'repeat':rep+1,**rank_metrics(base_imp,imp)}
                                shap_rows.append(row)
                except Exception as e:
                    shap_errors.append({'model':name,'error':repr(e)})
                    print(f'WARNING SHAP failed for {name}: {e}',flush=True)
            pd.DataFrame(shap_rows).to_csv(out/'table_6_shap_explanation_stability.csv',index=False)
            pd.DataFrame(shap_importance_rows).to_csv(out/'table_6b_shap_importance_by_condition.csv',index=False)
            if shap_rows:
                pd.DataFrame(shap_rows).groupby(['model','noise_level']).agg(['mean','std']).to_csv(out/'table_6c_shap_stability_summary.csv')
            pd.DataFrame(shap_errors,columns=['model','error']).to_csv(out/'shap_errors.csv',index=False)
        except Exception as e:
            print('WARNING: SHAP unavailable; remaining experiment will still finish:',repr(e))
            pd.DataFrame([{'error':repr(e)}]).to_csv(out/'shap_errors.csv',index=False)
    # Visual summaries
    for metric, filename, ylabel in [('macro_f1','figure_1_cv_macro_f1.png','5-fold CV macro-F1'),('macro_f1','figure_2_test_macro_f1.png','Held-out test macro-F1')]:
        if 'cv' in filename:
            d=cv_df.sort_values('cv_macro_f1_mean',ascending=True); vals=d['cv_macro_f1_mean']; errs=d['cv_macro_f1_sd']
            labels=d['model']
        else:
            d=test_df.sort_values('macro_f1',ascending=True); vals=d['macro_f1']; errs=None; labels=d['model']
        fig,ax=plt.subplots(figsize=(9,5)); ax.barh(labels,vals,xerr=errs,capsize=3 if errs is not None else 0)
        ax.set_xlabel(ylabel); ax.set_xlim(max(0,float(vals.min())-.08),min(1,float(vals.max())+.04)); ax.grid(axis='x',alpha=.25)
        fig.tight_layout(); fig.savefig(out/filename,dpi=200); plt.close(fig)
    piv=noise.groupby(['model','noise_level'])['macro_f1'].mean().unstack()
    fig,ax=plt.subplots(figsize=(9,5))
    for name,row in piv.iterrows(): ax.plot(row.index*100,row.values,marker='o',label=name)
    ax.set_xlabel('Perturbation level (%)'); ax.set_ylabel('Mean macro-F1'); ax.set_title('Repeated input-noise robustness'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'figure_3_noise_robustness.png',dpi=200); plt.close(fig)
    if not ab.empty:
        top=ab.groupby('removed_feature')['macro_f1_drop'].mean().sort_values(ascending=True)
        fig,ax=plt.subplots(figsize=(9,7)); ax.barh(top.index,top.values); ax.axvline(0,linewidth=1)
        ax.set_xlabel('Mean macro-F1 drop after removing feature (across models)'); ax.set_title('Retrained feature ablation')
        fig.tight_layout(); fig.savefig(out/'figure_4_feature_ablation.png',dpi=200); plt.close(fig)
    # Verify that required analysis outputs were generated.
    required_outputs = [
        'table_1_dataset_characteristics.csv', 'table_target_distribution.csv',
        'table_2_cross_validation.csv', 'table_3_test_performance.csv',
        'table_4_noise_robustness_repeats.csv', 'table_4_noise_robustness_summary.csv',
        'table_5_retrained_feature_ablation.csv', 'table_5b_top_ablation_features.csv',
        'figure_1_cv_macro_f1.png', 'figure_2_test_macro_f1.png',
        'figure_3_noise_robustness.png', 'figure_4_feature_ablation.png'
    ]
    output_checks = []
    for rel in required_outputs:
        output_checks.append({'output': rel, 'exists': (out / rel).exists(),
                              'status': 'PASS' if (out / rel).exists() else 'MISSING'})
    shap_expected = not args.skip_shap
    shap_table = out / 'table_6_shap_explanation_stability.csv'
    shap_rows_valid = bool(shap_rows) if not args.skip_shap else False
    output_checks.append({'output': 'table_6_shap_explanation_stability.csv',
                          'exists': shap_table.exists(),
                          'status': ('PASS' if shap_rows_valid else ('SKIPPED_BY_USER' if args.skip_shap else 'INCOMPLETE_OR_FAILED'))})
    pd.DataFrame(output_checks).to_csv(out / 'generated_output_inventory.csv', index=False)
    missing_core = [r['output'] for r in output_checks if r['output'] != 'table_6_shap_explanation_stability.csv' and not r['exists']]
    shap_status = 'PASS' if shap_rows_valid else ('SKIPPED_BY_USER' if args.skip_shap else 'INCOMPLETE_OR_FAILED')
    (out / 'OUTPUT_STATUS.txt').write_text(
        'ANALYSIS OUTPUT VERIFICATION\n'
        + f'Core outputs: {"PASS" if not missing_core else "FAIL"}\n'
        + f'Missing core outputs: {missing_core if missing_core else "None"}\n'
        + f'SHAP stability: {shap_status}\n'
        + 'A successful script exit alone does not imply SHAP succeeded. Check this file and generated_output_inventory.csv.\n'
        + ('Reduced-computation exploratory run: these outputs are not the full study results.\n' if args.fast else '')
        + 'Target label provenance remains unverified; external validation not performed.\n', encoding='utf-8')

    # Manifest / interpretation guardrails
    notes=f'''ANALYSIS EXECUTION DETAILS\nSource CSV: {csv_path}\nRows: {len(df)}; features: {X.shape[1]}; target: {target}\nSeed: {SEED}\nTrain/test: stratified 80/20; one fixed held-out test set\nCV: stratified 5-fold on training data only\nModels: {', '.join(models.keys())}\nNoise robustness: {n_noise} independent perturbation repeats per level, levels={NOISE_LEVELS}\nBootstrap: {n_boot} bootstrap resamples for test macro-F1 intervals\nFeature ablation: remove one feature, retrain each model, evaluate on the same holdout\nSHAP stability: common background and fixed explanation sample for {'all models' if args.shap_all_models else 'the training-CV-selected model'}; {n_shap} repeats per nonzero noise level; compare rank correlation, cosine similarity, and top-5 overlap\nTarget provenance verified: NO\nExternal validation: NOT PERFORMED\nClinical validity: NOT ESTABLISHED\nInterpretation: noise levels are corruption probabilities for low-cardinality integer-coded features, not calibrated measurement-error magnitudes.\nImportant: do not select the final model based on held-out test performance; use training-only CV for model selection and report the holdout once.\nRun mode: {'REDUCED-COMPUTATION EXPLORATORY RUN' if args.fast else 'FULL'}\nOutput verification: see generated_output_inventory.csv and OUTPUT_STATUS.txt\nElapsed seconds: {time.time()-started:.1f}\n'''
    (out/'RUN_NOTES.txt').write_text(notes,encoding='utf-8')
    # zip all outputs
    zip_path=out.with_suffix('.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file(): z.write(p,arcname=f'{out.name}/{p.relative_to(out)}')
    print('\n=== CV SUMMARY ===\n',cv_df[['model','cv_macro_f1_mean','cv_macro_f1_sd']].to_string(index=False))
    print('\n=== HELD-OUT TEST SUMMARY ===\n',test_df[['model','macro_f1','macro_f1_ci95_low','macro_f1_ci95_high']].to_string(index=False))
    print(f'\nDone. Results folder: {out.resolve()}\nZIP: {zip_path.resolve()}')

if __name__=='__main__': main()
