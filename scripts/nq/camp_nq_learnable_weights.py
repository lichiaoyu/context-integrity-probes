#!/usr/bin/env python3
"""CAMP NQ: two separate, question-grouped learnable-weight experiments.

A: detect poisoned context using per-context label fractions (10 responses).
B: on poisoned contexts, predict attacker-target adoption rate from first 5
   sampled labels to *disjoint* last 5 labels; this is outcome prediction, not a
   hidden-state detector. Labels remain the original CAMP PA/CA/REJ/AB/PH.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

ROOT = Path('camp_nq_pilot_output')
OUT = ROOT / 'learnable_weights'
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ['PA', 'CA', 'REJ', 'AB', 'PH']
EPS = 1e-12


def features(df):
    """Only derived from the labels of samples available in df."""
    count = df.groupby(['id', 'condition', 'label']).size().unstack(fill_value=0)
    count = count.reindex(columns=LABELS, fill_value=0)
    p = count.div(count.sum(axis=1), axis=0)
    # CAMP provisional mapping used in the preceding pilot (unchanged here).
    three = pd.DataFrame({'fooled':p['PA'], 'mixed':p['PH'],
                          'not_fooled':p[['CA', 'REJ', 'AB']].sum(axis=1)})
    ent_terms = -(three * np.log(three.clip(lower=EPS)))
    ent_terms.columns = ['H_fooled', 'H_mixed', 'H_not_fooled']
    X = p.copy()
    X['H_norm'] = ent_terms.sum(axis=1) / np.log(3)
    for col in ent_terms.columns:
        X[col] = ent_terms[col]
    X['s_fixed'] = p['PA'] + .5*p['PH']
    X['T_fixed'] = .7*X['s_fixed'] + .3*X['H_norm']
    return X.reset_index()


def select_X(df, model):
    # Drop AB to avoid exact compositional linear dependence with intercept.
    cols = ['PA', 'CA', 'REJ', 'PH']
    if model == 'labels': return df[cols].to_numpy(dtype=float)
    if model == 'labels_entropy': return df[cols+['H_norm']].to_numpy(dtype=float)
    if model == 'labels_entropy_terms': return df[cols+['H_fooled','H_mixed','H_not_fooled']].to_numpy(dtype=float)
    raise ValueError(model)


def summarize_binary(y, pred):
    return {'roc_auc':round(float(roc_auc_score(y,pred)),6),
            'n':len(y), 'clean':int((y==0).sum()), 'poisoned':int((y==1).sum())}


def goal_a(all_rows):
    f = features(all_rows)
    assert len(f)==200 and f.groupby('id').size().eq(2).all(), 'Expected 100 clean/poison pairs'
    y = (f['condition']=='poison_1').astype(int).to_numpy()
    group = f['id'].to_numpy()
    cv = GroupKFold(n_splits=5)
    report = []
    oof = f[['id','condition']].copy()
    # Original fixed scores are frozen; evaluate them on exact same 200 contexts.
    for score in ['s_fixed','H_norm','T_fixed']:
        pred = f[score].to_numpy()
        report.append({'goal':'poison_detection','model':score, **summarize_binary(y,pred)})
        oof[score] = pred
    for model in ['labels','labels_entropy','labels_entropy_terms']:
        x = select_X(f,model)
        pred = np.full(len(f),np.nan)
        weights=[]
        for fold,(tr,te) in enumerate(cv.split(x,y,groups=group),1):
            clf = make_pipeline(StandardScaler(),LogisticRegression(C=1.0, max_iter=2000))
            clf.fit(x[tr], y[tr])
            pred[te] = clf.predict_proba(x[te])[:,1]
            weights.append({'goal':'poison_detection','model':model,'fold':fold,
                            'n_train':len(tr),'n_test':len(te),
                            'coef_scaled_features':','.join(f'{z:.5g}' for z in clf[-1].coef_[0])})
        oof[model]=pred
        report.append({'goal':'poison_detection','model':model, **summarize_binary(y,pred)})
    oof['y_poison']=y
    oof.to_csv(OUT/'poison_detection_oof.csv',index=False)
    return report


def goal_b(all_rows):
    """Response-level disjoint split: first 5 responses predict PA rate in last 5.
    Strictly uses poisoned contexts; labels unchanged; never uses held-out
    response labels as model input. This is NOT learning from hidden states.
    """
    poison = all_rows[all_rows['condition']=='poison_1'].copy()
    if not set(range(10)).issubset(set(poison['sample_index'].unique())):
        raise RuntimeError('Expected sample_index 0..9 in answer_labels.csv')
    xfirst = features(poison[poison['sample_index']<5])
    last = poison[poison['sample_index']>=5]
    tgt = last.groupby(['id','condition']).agg(
        n=('label','size'), pa=('label',lambda z: (z=='PA').sum())).reset_index()
    tgt['y_adoption_rate'] = tgt['pa']/tgt['n']
    f=xfirst.merge(tgt[['id','condition','y_adoption_rate']],on=['id','condition'],validate='one_to_one')
    assert len(f)==100 and f['id'].nunique()==100 and (tgt['n']==5).all()
    y = f['y_adoption_rate'].to_numpy()
    group = f['id'].to_numpy()
    cv=GroupKFold(n_splits=5)
    report=[]
    oof=f[['id','condition','y_adoption_rate']].copy()
    for score in ['PA','s_fixed','H_norm','T_fixed']:
        pred=np.clip(f[score].to_numpy(dtype=float),0,1)
        report.append({'goal':'target_adoption','model':score,
                       'mae':round(float(mean_absolute_error(y,pred)),6),
                       'rmse':round(float(np.sqrt(mean_squared_error(y,pred))),6),
                       'n':len(y)})
        oof[score]=pred
    for model in ['labels','labels_entropy','labels_entropy_terms']:
        x=select_X(f,model)
        pred=np.full(len(f),np.nan)
        for tr,te in cv.split(x,y,groups=group):
            reg=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
            reg.fit(x[tr],y[tr])
            pred[te]=np.clip(reg.predict(x[te]),0,1)
        oof[model]=pred
        report.append({'goal':'target_adoption','model':model,
                       'mae':round(float(mean_absolute_error(y,pred)),6),
                       'rmse':round(float(np.sqrt(mean_squared_error(y,pred))),6),
                       'n':len(y)})
    oof.to_csv(OUT/'target_adoption_oof.csv',index=False)
    return report


def main():
    path = ROOT/'answer_labels.csv'
    if not path.exists(): raise SystemExit(f'Missing {path}; run camp_nq_influence_pilot.py first')
    rows=pd.read_csv(path, dtype={'id':str})
    need={'id','condition','sample_index','label'}
    if not need.issubset(rows): raise SystemExit(f'Missing CSV columns: {need-set(rows)}')
    if not set(rows['label']).issubset(set(LABELS)):
        raise SystemExit(f'Unexpected labels: {set(rows["label"])-set(LABELS)}')
    if len(rows)!=2000: raise SystemExit(f'Expected 2000 labeled responses, got {len(rows)}')
    a=goal_a(rows); b=goal_b(rows)
    pd.DataFrame(a).to_csv(OUT/'poison_detection_summary.csv',index=False)
    pd.DataFrame(b).to_csv(OUT/'target_adoption_summary.csv',index=False)
    (OUT/'README.txt').write_text(
        'Goal A: Clean-vs-poisoned discrimination; all 10 response labels used as input; '
        'learned models use question-grouped 5-fold OOF predictions. Fixed scores use '\
        'the same 200 contexts (not cross-fitted, since no fitted parameters).\n'
        'Goal B: Attacker target adoption *among poisoned contexts*; features only '\
        'from first five response labels, ground truth PA rate only from the disjoint last five. '\
        'Question-grouped 5-fold cross-validation. Goal B evaluated with MAE/RMSE, not '\
        'poisoning ROC-AUC.\n'
        'The original five-label definitions and provisional 3-category entropy mapping are unchanged.\n'
        'Both tasks use offline response-derived features; NEITHER is a TBG hidden-state probe.\n'
        'Small dataset (100 questions), uncertain automatic labels, uncertain sample dependence.\n'
        'The learnt five-label weights are prediction coefficients, not uniquely causal attack influence.\n'
    )
    print('\nGOAL A — Poisoned vs clean (ROC-AUC, grouped OOF for learned methods)')
    print(pd.DataFrame(a).to_string(index=False))
    print('\nGOAL B — Predict held-out target adoption (lower MAE/RMSE is better)')
    print(pd.DataFrame(b).to_string(index=False))
    print('\nSaved outputs:',OUT)

if __name__=='__main__': main()
