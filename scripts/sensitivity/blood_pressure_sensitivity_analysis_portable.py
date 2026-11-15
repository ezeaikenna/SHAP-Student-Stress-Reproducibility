#!/usr/bin/env python3
"""Paired blood_pressure feature-exclusion sensitivity experiment.
Same stratified holdout, CV folds, model definitions and seeds across conditions.
Target provenance remains unverified; this is sensitivity analysis, not proof of leakage.
"""
from pathlib import Path
import argparse, json, zipfile, warnings, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings('ignore')
SEED=20261009
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--csv', default='StressLevelDataset.csv', help='Path to StressLevelDataset.csv')
parser.add_argument('--outdir', default='blood_pressure_sensitivity_results', help='Directory for sensitivity outputs')
args = parser.parse_args()
OUT = Path(args.outdir)
OUT.mkdir(parents=True, exist_ok=True)
CSV = Path(args.csv)
def models():
 return {
 'LogisticRegression':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1500,random_state=SEED)),
 'SVM_RBF':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),SVC(C=1.0,kernel='rbf',probability=True,random_state=SEED)),
 'RandomForest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=50,min_samples_leaf=1,n_jobs=-1,random_state=SEED)),
 'ExtraTrees':make_pipeline(SimpleImputer(strategy='median'),ExtraTreesClassifier(n_estimators=50,min_samples_leaf=1,n_jobs=-1,random_state=SEED)),
 'MLP':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32),max_iter=150,early_stopping=True,random_state=SEED)),
 'DecisionTree':make_pipeline(SimpleImputer(strategy='median'),DecisionTreeClassifier(random_state=SEED,min_samples_leaf=2))}
def metric(y,p,proba):
 try: auc=roc_auc_score(y,proba,multi_class='ovr',average='macro')
 except Exception: auc=np.nan
 return {'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0),'macro_precision':precision_score(y,p,average='macro',zero_division=0),'macro_recall':recall_score(y,p,average='macro',zero_division=0),'roc_auc_ovr_macro':auc,'log_loss':log_loss(y,proba,labels=[0,1,2])}
def boot_diff(y,pa,pb,n=2000,seed=SEED):
 # Difference A minus B, paired bootstrap over same held-out cases.
 rng=np.random.default_rng(seed); y=np.asarray(y); pa=np.asarray(pa); pb=np.asarray(pb); vals=[]
 for _ in range(n):
  ix=rng.integers(0,len(y),len(y))
  if len(np.unique(y[ix]))<3: continue
  vals.append(f1_score(y[ix],pa[ix],average='macro',zero_division=0)-f1_score(y[ix],pb[ix],average='macro',zero_division=0))
 return np.quantile(vals,[.025,.975]).tolist() if vals else [np.nan,np.nan]
def perturb(X, ref, level, seed):
 rng=np.random.default_rng(seed); out=X.copy()
 for c in X.columns:
  vals=ref[c].dropna().to_numpy(); uniq=np.unique(vals); mask=rng.random(len(out))<level
  if len(uniq)<=12:
   for i in np.where(mask)[0]:
    alt=uniq[uniq!=out.iloc[i][c]]
    if len(alt): out.iat[i,out.columns.get_loc(c)]=rng.choice(alt)
  else:
   sd=np.std(vals,ddof=1) if len(vals)>1 else 0
   if sd>0:
    out.loc[mask,c]=out.loc[mask,c]+rng.normal(0,level*sd,mask.sum())
    out[c]=out[c].clip(vals.min(),vals.max())
 return out
def rank_stats(a,b):
 denom=np.linalg.norm(a)*np.linalg.norm(b)
 return {'spearman':float(spearmanr(a,b).statistic),'kendall':float(kendalltau(a,b).statistic),'cosine':float(np.dot(a,b)/denom) if denom else np.nan,'top5_overlap':len(set(np.argsort(a)[-5:])&set(np.argsort(b)[-5:]))/5}

def main():
 t=time.time(); df=pd.read_csv(CSV); y=df.stress_level
 Xall=df.drop(columns='stress_level'); conditions={'all_predictors':Xall,'without_blood_pressure':Xall.drop(columns='blood_pressure')}
 Xtr0,Xte0,ytr,yte=train_test_split(Xall,y,test_size=.20,stratify=y,random_state=SEED)
 cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
 cvrows=[]; testrows=[]; predictions={}; fitted={}
 for cond,X in conditions.items():
  Xtr=X.loc[Xtr0.index]; Xte=X.loc[Xte0.index]
  for name,model in models().items():
   cvres=cross_validate(clone(model),Xtr,ytr,cv=cv,scoring={'macro_f1':'f1_macro','accuracy':'accuracy','balanced_accuracy':'balanced_accuracy'},n_jobs=1)
   cvrows.append({'condition':cond,'model':name,**{f'{k}_mean':float(np.mean(v)) for k,v in cvres.items() if k.startswith('test_')},**{f'{k}_sd':float(np.std(v,ddof=1)) for k,v in cvres.items() if k.startswith('test_')}})
   fitted[(cond,name)]=clone(model).fit(Xtr,ytr); m=fitted[(cond,name)]
   p=m.predict(Xte); pro=m.predict_proba(Xte); predictions[(cond,name)]=(p,pro)
   testrows.append({'condition':cond,'model':name,**metric(yte,p,pro)})
 cvdf=pd.DataFrame(cvrows); testdf=pd.DataFrame(testrows)
 cvdf.to_csv(OUT/'cv_comparison.csv',index=False); testdf.to_csv(OUT/'heldout_comparison.csv',index=False)
 # Paired per-model holdout differences and bootstrap CIs (all minus no BP).
 diffs=[]
 for name in models():
  ma=next(r for r in testrows if r['condition']=='all_predictors' and r['model']==name)
  mb=next(r for r in testrows if r['condition']=='without_blood_pressure' and r['model']==name)
  pa=predictions[('all_predictors',name)][0]; pb=predictions[('without_blood_pressure',name)][0]
  lo,hi=boot_diff(yte,pa,pb)
  diffs.append({'model':name,'macro_f1_all':ma['macro_f1'],'macro_f1_without_bp':mb['macro_f1'],'delta_macro_f1_all_minus_without_bp':ma['macro_f1']-mb['macro_f1'],'paired_bootstrap_95ci_low':lo,'paired_bootstrap_95ci_high':hi,'delta_balanced_accuracy':ma['balanced_accuracy']-mb['balanced_accuracy'],'delta_log_loss_all_minus_without_bp':ma['log_loss']-mb['log_loss'],'delta_roc_auc':ma['roc_auc_ovr_macro']-mb['roc_auc_ovr_macro']})
 pd.DataFrame(diffs).to_csv(OUT/'paired_holdout_differences.csv',index=False)
 # SHAP global importance and perturbation stability for LR in both predictor conditions.
 shaprows=[]; stabrows=[]; shaperr=[]
 try:
  import shap
  rng=np.random.default_rng(SEED); bgpos=rng.choice(len(Xtr0),min(40,len(Xtr0)),replace=False); exppos=rng.choice(len(Xte0),min(60,len(Xte0)),replace=False)
  for cond,X in conditions.items():
   Xtr=X.loc[Xtr0.index]; Xte=X.loc[Xte0.index]; m=fitted[(cond,'LogisticRegression')]
   bg=Xtr.iloc[bgpos]; exp=Xte.iloc[exppos]
   # Pipeline transform is fitted on training data; explain same held-out rows in both conditions.
   bgz=m[:-1].transform(bg); expz=m[:-1].transform(exp); expl=shap.LinearExplainer(m[-1],bgz)
   def imp_for(x):
    vals=expl.shap_values(m[:-1].transform(x))
    vals=np.asarray(vals)
    if vals.ndim==3: return np.mean(np.abs(vals),axis=(0,2))
    return np.mean(np.abs(vals),axis=0)
   base=imp_for(exp)
   for f,v in zip(X.columns,base): shaprows.append({'condition':cond,'feature':f,'mean_abs_shap':float(v),'rank_descending':int(np.where(np.argsort(base)[::-1]==list(X.columns).index(f))[0][0]+1)})
   for level in [0.05,0.10,0.20]:
    for rep in range(1,6):
     xp=perturb(exp,Xtr,level,SEED+30000+int(level*1000)+rep)
     cur=imp_for(xp); stabrows.append({'condition':cond,'noise_level':level,'repeat':rep,**rank_stats(base,cur)})
 except Exception as e:
  shaperr.append({'error':repr(e)})
 pd.DataFrame(shaprows).to_csv(OUT/'shap_importance_lr.csv',index=False)
 pd.DataFrame(stabrows).to_csv(OUT/'shap_stability_lr.csv',index=False)
 pd.DataFrame(shaperr).to_csv(OUT/'shap_errors.csv',index=False)
 if len(stabrows): pd.DataFrame(stabrows).groupby(['condition','noise_level']).agg(['mean','std']).to_csv(OUT/'shap_stability_summary.csv')
 # Write interpretation and methods note.
 report=['BLOOD_PRESSURE FEATURE-EXCLUSION SENSITIVITY ANALYSIS','',f'Dataset: {CSV}',f'Rows: {len(df)}; predictors: {Xall.shape[1]}; classes: {dict(y.value_counts().sort_index())}',f'Seed: {SEED}; stratified holdout: 80/20; 5-fold stratified CV on training partition.', 'Conditions use the same train/test indices, CV folds, six model definitions, and preprocessing pipelines.','Holdout bootstrap CI uses paired resampling of the same test cases (2,000 resamples).','SHAP: Logistic Regression in each condition; same held-out row positions and background row positions; 5 perturbation repeats at each 5%, 10%, and 20% corruption level.','Target provenance remains unverified; feature exclusion cannot establish whether the target was generated using blood pressure.','Noise procedure is synthetic perturbation, not a calibrated measurement-error model.','']
 report+=['CROSS-VALIDATION (sorted within condition):',cvdf.sort_values(['condition','test_macro_f1_mean'],ascending=[True,False]).to_string(index=False),'','HELD-OUT PERFORMANCE:',testdf.to_string(index=False),'','PAIRED HOLDOUT DIFFERENCES (all predictors minus without blood_pressure):',pd.DataFrame(diffs).to_string(index=False),'','SHAP IMPORTANCE:',pd.DataFrame(shaprows).sort_values(['condition','mean_abs_shap'],ascending=[True,False]).to_string(index=False) if shaprows else 'SHAP unavailable/failed.','',f'Elapsed seconds: {time.time()-t:.1f}']
 (OUT/'REPORT.txt').write_text('\n'.join(report),encoding='utf-8')
 zip_path = OUT.with_suffix('.zip')
 with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
  for p in OUT.iterdir():
   if p.is_file(): z.write(p, arcname=f'{OUT.name}/{p.name}')
 print('OUTPUT', OUT); print('ZIP', zip_path)
 print('\nCV summary:'); print(cvdf[['condition','model','test_macro_f1_mean','test_macro_f1_sd']].to_string(index=False))
 print('\nHoldout differences:'); print(pd.DataFrame(diffs).to_string(index=False))
 print('\nSHAP errors:',shaperr)
if __name__=='__main__': main()
