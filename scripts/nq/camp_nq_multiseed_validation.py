#!/usr/bin/env python3
"""CAMP NQ 100-question reproducibility check: 5 seeds, grouped 5-fold OOF.
Same Probe architecture and objectives as camp_train_nq_probes.py.
Includes per-fold leakage audit, grouped bootstrap AUC CIs, and paired contrasts.
Not a substitute for independent held-out validation or matched-pair attack controls.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn

from camp_train_nq_probes import Probe

MODELS = ('direct', 'success', 'fixed_T', 'learned_T')


def load_data(root):
    raw = np.load(root / 'camp_nq_hidden/tbg_activations.npy', mmap_mode='r')
    meta = [json.loads(line) for line in (root/'camp_nq_hidden/metadata.jsonl').read_text().splitlines() if line.strip()]
    scores = pd.read_csv(root/'camp_nq_pilot_output/context_scores.csv', dtype={'id':str})
    if len(raw)!=200 or len(meta)!=200 or len(scores)!=200:
        raise ValueError('Expected exactly 200 activations, metadata records and scores')
    if raw.ndim!=3 or not np.isfinite(raw).all():
        raise ValueError('Invalid activation array')
    if scores.duplicated(['id','condition']).any():
        raise ValueError('Duplicate target row')
    lookup = scores.set_index(['id','condition'])
    aligned=[]
    for m in meta:
        key=(str(m['id']),m['condition'])
        if key not in lookup.index:raise ValueError(f'Missing target {key}')
        aligned.append(lookup.loc[key])
    s=pd.DataFrame(aligned).reset_index(drop=True)
    group=np.array([str(m['id']) for m in meta])
    y=np.array([int(m['condition']=='poison_1') for m in meta], dtype=int)
    unique=np.unique(group)
    if len(unique)!=100 or any(np.sum(group==g)!=2 for g in unique):
        raise ValueError('Not 100 matched question pairs')
    for g in unique:
        if set(y[group==g])!={0,1}:raise ValueError(f'Missing class in question {g}')
    L=raw.shape[1]
    X=np.concatenate([np.asarray(raw[:,L//2,:],dtype=np.float32),
                      np.asarray(raw[:,L-1,:],dtype=np.float32),
                      np.asarray(raw.mean(axis=1),dtype=np.float32)],axis=1)
    props=np.stack([s[k].to_numpy(dtype=float)/s['n'].to_numpy(dtype=float) for k in ['PA','CA','REJ','PH']],axis=1)
    targets={'direct':y.astype(float),
             'success':s['success_s'].to_numpy(dtype=float),
             'fixed_T':s['T_alpha_0p7'].to_numpy(dtype=float)}
    if not np.isfinite(X).all() or not np.isfinite(props).all() or not all(np.isfinite(t).all() for t in targets.values()):
        raise ValueError('Non-finite features or targets')
    if any(np.min(t)<-1e-6 or np.max(t)>1+1e-6 for t in targets.values()):
        raise ValueError('Expected training targets in [0,1]')
    return X,group,y,unique,props,targets


def grouped_bootstrap(preds, y, group, n_reps, rng):
    """Resample question IDs, carrying both clean and poison rows together."""
    ids=np.unique(group)
    index={g:np.flatnonzero(group==g) for g in ids}
    values={k:[] for k in MODELS}
    diff=[]
    for _ in range(n_reps):
        chosen=rng.choice(ids,size=len(ids),replace=True)
        ix=np.concatenate([index[g] for g in chosen])
        yy=y[ix]
        current={k:roc_auc_score(yy,preds[k][ix]) for k in MODELS}
        for k,v in current.items():values[k].append(v)
        diff.append(current['learned_T']-current['fixed_T'])
    return values,diff


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--outdir',default='camp_nq_multiseed_output')
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55])
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--bootstrap',type=int,default=1000)
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds):raise ValueError('Seed values must be distinct')
    root=Path(a.root)
    X,group,y,unique,props,targets=load_data(root)
    out=root/a.outdir;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device={device}, X={X.shape}, questions={len(unique)}, seeds={a.seeds}',flush=True)
    all_summary=[]; all_folds=[]; all_pred=[]; all_ci=[]; all_audits=[]
    for seed in a.seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        cv=KFold(n_splits=5,shuffle=True,random_state=seed)
        result={k:np.full(len(y),np.nan) for k in MODELS}
        for fold,(tr_q,te_q) in enumerate(cv.split(unique),1):
            train_ids=set(unique[tr_q]); test_ids=set(unique[te_q])
            if not train_ids.isdisjoint(test_ids):raise RuntimeError('QUESTION LEAKAGE')
            tr=np.flatnonzero(np.isin(group,list(train_ids)))
            te=np.flatnonzero(np.isin(group,list(test_ids)))
            if len(tr)!=160 or len(te)!=40:raise RuntimeError('Unexpected grouped split size')
            scaler=StandardScaler()
            xtr=np.clip(scaler.fit_transform(X[tr]),-8,8).astype(np.float32)
            xte=np.clip(scaler.transform(X[te]),-8,8).astype(np.float32)
            teacher=LogisticRegression(C=1.0,max_iter=2000)
            teacher.fit(props[tr],y[tr])  # train-only teacher, NEVER use test targets
            t_targets={**{k:values[tr] for k,values in targets.items()},
                       'learned_T':teacher.predict_proba(props[tr])[:,1]}
            xt=torch.from_numpy(xtr).to(device)
            xv=torch.from_numpy(xte).to(device)
            for j,name in enumerate(MODELS):
                torch.manual_seed(seed+fold*10+j)
                if torch.cuda.is_available():torch.cuda.manual_seed_all(seed+fold*10+j)
                model=Probe(X.shape[1]).to(device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
                yy=torch.as_tensor(t_targets[name],dtype=torch.float32,device=device)
                for epoch in range(a.epochs):
                    model.train()
                    perm=torch.randperm(len(tr),device=device)
                    for ind in perm.split(32):
                        z=model(xt[ind])
                        if name=='direct':loss=nn.functional.binary_cross_entropy_with_logits(z,yy[ind])
                        else:loss=nn.functional.mse_loss(torch.sigmoid(z),yy[ind])
                        optimizer.zero_grad();loss.backward();optimizer.step()
                model.eval()
                with torch.no_grad(): pred=torch.sigmoid(model(xv)).cpu().numpy()
                result[name][te]=pred
                auc=roc_auc_score(y[te],pred)
                all_folds.append({'seed':seed,'fold':fold,'model':name,'auc':float(auc),'n_test':len(te)})
                print(f'Seed {seed} fold {fold} {name}: AUC={auc:.4f}',flush=True)
            all_audits.append({'seed':seed,'fold':fold,'n_train_questions':len(train_ids),
                               'n_test_questions':len(test_ids),'overlap':len(train_ids&test_ids),
                               'n_train_contexts':len(tr),'n_test_contexts':len(te),
                               'teacher_train_only':True,'scaler_train_only':True})
        for name in MODELS:
            score=result[name]
            if not np.isfinite(score).all():raise RuntimeError(f'Missing OOF score: {seed} {name}')
            all_summary.append({'seed':seed,'model':name,'oof_roc_auc':roc_auc_score(y,score),
                                'oof_auprc':average_precision_score(y,score)})
        for i in range(len(group)):
            all_pred.append({'seed':seed,'id':group[i],'condition':'poison_1' if y[i] else 'clean',
                             'y_poison':int(y[i]),**{name:float(result[name][i]) for name in MODELS}})
        boot,diff=grouped_bootstrap(result,y,group,a.bootstrap,np.random.default_rng(seed+2026))
        for name in MODELS:
            lo,hi=np.percentile(boot[name],[2.5,97.5])
            all_ci.append({'seed':seed,'comparison':name,'point':roc_auc_score(y,result[name]),
                           'bootstrap_ci_low':lo,'bootstrap_ci_high':hi})
        lo,hi=np.percentile(diff,[2.5,97.5])
        all_ci.append({'seed':seed,'comparison':'learned_T-minus-fixed_T',
                       'point':roc_auc_score(y,result['learned_T'])-roc_auc_score(y,result['fixed_T']),
                       'bootstrap_ci_low':lo,'bootstrap_ci_high':hi})
        print('Completed seed',seed,flush=True)
    summary=pd.DataFrame(all_summary)
    summary.to_csv(out/'per_seed_summary.csv',index=False)
    pd.DataFrame(all_folds).to_csv(out/'per_fold_summary.csv',index=False)
    pd.DataFrame(all_pred).to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(all_ci).to_csv(out/'question_bootstrap_ci.csv',index=False)
    pd.DataFrame(all_audits).to_csv(out/'leakage_audit.csv',index=False)
    aggregate=summary.groupby('model')['oof_roc_auc'].agg(['mean','std','min','max']).reindex(MODELS)
    aggregate.to_csv(out/'aggregate_auc.csv')
    print('\n=== FIVE-SEED SUMMARY (GROUPED OOF ROC-AUC) ===',flush=True)
    print(aggregate.to_string(float_format=lambda v:f'{v:.4f}'),flush=True)
    diffs=[]
    for seed in a.seeds:
        z=summary[summary.seed==seed].set_index('model')['oof_roc_auc']
        diffs.append(z['learned_T']-z['fixed_T'])
    print('Learned_T minus fixed_T by seed:',[round(float(x),4) for x in diffs],flush=True)
    print('Mean delta:',round(float(np.mean(diffs)),4),flush=True)
    print('NOTE: grouped bootstrap CI conditional on each seed; 5 seeds are sensitivity checks, not independent datasets.',flush=True)
    print('NOTE: This is a 100-question controlled-replacement pilot. 1% FPR is too coarse with 20 clean examples per fold.',flush=True)
    print('Saved:',out,flush=True)

if __name__=='__main__':main()
