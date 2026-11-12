#!/usr/bin/env python3
"""CAMP NQ dual-task pilot. First five generations are supervision;
last five are disjoint response-level evaluation. Group questions across folds.
Not independent documents / new prompts; 5 samples yield noisy PA estimates.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from scipy.stats import spearmanr
from camp_train_nq_probes import Probe

NAMES=['direct','success','fixed_T','learned_T']

def make_data(root):
    raw=np.load(root/'camp_nq_hidden/tbg_activations.npy',mmap_mode='r')
    meta=[json.loads(line) for line in (root/'camp_nq_hidden/metadata.jsonl').read_text().splitlines() if line.strip()]
    labs=pd.read_csv(root/'camp_nq_pilot_output/answer_labels.csv',dtype={'id':str})
    required={'id','condition','sample_index','label'}
    if not required.issubset(labs):raise ValueError(f'Missing answer labels: {required-set(labs.columns)}')
    if len(meta)!=len(raw)!=200:raise ValueError('Unexpected activation / metadata count')
    if len(meta)!=200 or raw.ndim!=3:raise ValueError('Expected 200 TBG contexts with 3 dimensions')
    if labs.duplicated(['id','condition','sample_index']).any():raise ValueError('Duplicate answer labels')
    if set(labs.label)-{'PA','CA','PH','REJ','AB'}:raise ValueError('Unexpected response labels')
    labmap={(str(q),c):z.sort_values('sample_index') for (q,c),z in labs.groupby(['id','condition'])}
    keys=[(str(z['id']),z['condition']) for z in meta]
    if len(set(keys))!=200:raise ValueError('Duplicate metadata keys')
    group=np.asarray([k[0] for k in keys]); y=np.array([int(k[1]=='poison_1') for k in keys])
    uq=np.unique(group)
    if len(uq)!=100 or any(sorted(y[group==q].tolist())!=[0,1] for q in uq):raise ValueError('Missing matched pairs')
    train_proportions=[]; evaluation=[]
    for key in keys:
        if key not in labmap:raise ValueError(f'Missing answer labels: {key}')
        z=labmap[key]
        if len(z)!=10 or z.sample_index.tolist()!=list(range(10)):
            raise ValueError(f'Expected ordered sample indices 0..9 for {key}')
        p=z.iloc[:5].label.value_counts()
        train_proportions.append([p.get(k,0)/5 for k in ['PA','CA','REJ','PH','AB']])
        evaluation.append((z.iloc[5:].label=='PA').mean())
    prop=np.array(train_proportions,dtype=float)
    heldout=np.asarray(evaluation,dtype=float)
    fooled=prop[:,0];mixed=prop[:,3];notfooled=1-fooled-mixed
    probs=np.stack([fooled,mixed,notfooled],axis=1)
    h=-np.sum(np.where(probs>0,probs*np.log(np.maximum(probs,1e-12)),0.),axis=1)/np.log(3)
    success=fooled+0.5*mixed
    fixed=.7*success+.3*h
    L=raw.shape[1]
    X=np.concatenate([np.asarray(raw[:,L//2,:],dtype=np.float32),
                     np.asarray(raw[:,L-1,:],dtype=np.float32),
                     np.asarray(raw.mean(axis=1),dtype=np.float32)],axis=1)
    if not np.isfinite(X).all():raise ValueError('Nonfinite hidden states')
    return X,group,y,uq,prop,heldout,{'direct':y.astype(float),'success':success,'fixed_T':fixed}

def scoring(pred, y, heldout):
    out=[]
    pos=y==1
    success_binary=(heldout[pos]>=.6).astype(int)  # at least 3 of last 5 responses attacker answer
    for name in NAMES:
        z=pred[name]
        rho=spearmanr(z[pos],heldout[pos]).statistic
        auc=(roc_auc_score(success_binary,z[pos]) if len(np.unique(success_binary))==2 else np.nan)
        out.append({'model':name,'poison_auc':roc_auc_score(y,z),
                    'poison_auprc':average_precision_score(y,z),
                    'attack_success_spearman_poison_only':rho,
                    'attack_majority_auc_poison_only':auc,
                    'attack_majority_prevalence':success_binary.mean(),
                    'mae_vs_heldout_PA_rate_uncalibrated':mean_absolute_error(heldout[pos],z[pos]),
                    'n_poison':int(pos.sum())})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--outdir',default='camp_nq_dual_task_output')
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    ap.add_argument('--epochs',type=int,default=100)
    args=ap.parse_args()
    root=Path(args.root)
    X,groups,y,uq,prop,heldout,targets=make_data(root)
    out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:',device,'X:',X.shape,'questions:',len(uq),'heldout PA rate mean poison:',heldout[y==1].mean(),flush=True)
    scores=[]; predrows=[]; audit=[]
    for seed in args.seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        predictions={name:np.full(len(y),np.nan) for name in NAMES}
        cv=KFold(n_splits=5,shuffle=True,random_state=seed)
        for fold,(trq,teq) in enumerate(cv.split(uq),1):
            train_ids=set(uq[trq]);test_ids=set(uq[teq])
            if train_ids & test_ids:raise RuntimeError('Question leakage')
            tr=np.flatnonzero(np.isin(groups,list(train_ids)))
            te=np.flatnonzero(np.isin(groups,list(test_ids)))
            scaler=StandardScaler()
            xtr=np.clip(scaler.fit_transform(X[tr]),-8,8).astype(np.float32)
            xte=np.clip(scaler.transform(X[te]),-8,8).astype(np.float32)
            teacher=LogisticRegression(max_iter=2000,C=1.)
            teacher.fit(prop[tr,:4],y[tr])
            tr_targets={**{k:arr[tr] for k,arr in targets.items()},
                        'learned_T':teacher.predict_proba(prop[tr,:4])[:,1]}
            xt=torch.from_numpy(xtr).to(device);xv=torch.from_numpy(xte).to(device)
            for j,name in enumerate(NAMES):
                torch.manual_seed(seed+fold*10+j)
                if torch.cuda.is_available():torch.cuda.manual_seed_all(seed+fold*10+j)
                model=Probe(X.shape[1]).to(device)
                opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
                target=torch.tensor(tr_targets[name],dtype=torch.float32,device=device)
                for epoch in range(args.epochs):
                    model.train()
                    perm=torch.randperm(len(tr),device=device)
                    for batch in perm.split(32):
                        z=model(xt[batch])
                        loss=(nn.functional.binary_cross_entropy_with_logits(z,target[batch]) if name=='direct'
                              else nn.functional.mse_loss(torch.sigmoid(z),target[batch]))
                        opt.zero_grad();loss.backward();opt.step()
                model.eval()
                with torch.no_grad():predictions[name][te]=torch.sigmoid(model(xv)).cpu().numpy()
            audit.append({'seed':seed,'fold':fold,'overlapping_question_ids':len(train_ids&test_ids),
                          'n_train_questions':len(train_ids),'n_test_questions':len(test_ids),
                          'n_train_contexts':len(tr),'n_test_contexts':len(te),
                          'teacher_fit_on_train_only':True,'heldout_samples_used_in_training':False})
        if not all(np.isfinite(pred).all() for pred in predictions.values()):raise RuntimeError('Missing OOF values')
        for score in scoring(predictions,y,heldout):scores.append({'seed':seed,**score})
        for i in range(len(y)):
            predrows.append({'seed':seed,'id':groups[i],'condition':'poison_1' if y[i] else 'clean',
                             'y_poison':y[i],'heldout_PA_rate_samples_5to9':heldout[i],
                             **{k:predictions[k][i] for k in NAMES}})
        print('Completed seed',seed,flush=True)
    detail=pd.DataFrame(scores)
    detail.to_csv(out/'per_seed_dual_task.csv',index=False)
    pd.DataFrame(predrows).to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'leakage_audit.csv',index=False)
    summary=detail.groupby('model').agg({'poison_auc':['mean','std'],
        'attack_success_spearman_poison_only':['mean','std'],
        'attack_majority_auc_poison_only':['mean','std'],
        'mae_vs_heldout_PA_rate_uncalibrated':['mean','std']})
    summary.to_csv(out/'dual_task_summary.csv')
    print('\n=== DUAL TASK SUMMARY ===',flush=True)
    print(summary.to_string(float_format=lambda x:f'{x:.4f}'),flush=True)
    print('NOTE: supervised with first 5 responses; evaluated on last 5 responses from SAME CONTEXTS (not independently collected contexts).',flush=True)
    print('NOTE: PA is automatic exact-phrase labeling; review semantic aliases, especially PH.',flush=True)
    print('NOTE: attack-majority AUC has noisy binary labels from only 5 responses; Spearman handles discrete/tied outcomes.',flush=True)
    print('NOTE: MAE uses UNCALIBRATED probe scores; compare with care because direct is trained for poisoning, not PA rate.',flush=True)
    print('NOTE: five seeds share the same questions and do not form independent replication.',flush=True)
    print('Saved:',out,flush=True)

if __name__=='__main__':main()
