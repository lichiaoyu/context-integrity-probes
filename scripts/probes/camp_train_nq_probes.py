#!/usr/bin/env python3
"""Question-grouped 5-fold CAMP feasibility probes on cached NQ Qwen TBG states.
Four-layer MLP, four targets. Learned-label target is fit on training questions ONLY.
This compares targets; 100 question pairs are NOT adequate for 1% FPR claims.
"""
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from torch import nn

class Probe(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(.2),
           nn.Linear(256,128),nn.ReLU(),nn.Dropout(.2),
           nn.Linear(128,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1))
    def forward(self,x):return self.layers(x).squeeze(-1)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',default='.')
    a.add_argument('--epochs',type=int,default=100)
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--outdir',default='camp_nq_probe_output')
    args=a.parse_args();root=Path(args.root)
    torch.set_num_threads(4)
    raw=np.load(root/'camp_nq_hidden/tbg_activations.npy',mmap_mode='r')
    meta=[json.loads(t) for t in (root/'camp_nq_hidden/metadata.jsonl').read_text().splitlines() if t.strip()]
    scores=pd.read_csv(root/'camp_nq_pilot_output/context_scores.csv')
    assert len(meta)==len(raw)==len(scores)==200, 'Expected exact 200 contexts; do not train on smoke-test cache'
    assert len({x['id'] for x in meta})==100
    scores=scores.set_index(['id','condition'])
    scores.index=pd.MultiIndex.from_tuples([(str(i),c) for i,c in scores.index])
    rows=[]
    for m in meta:
        key=(str(m['id']),m['condition'])
        if key not in scores.index:raise RuntimeError(f'Missing target: {key}')
        if (scores.index==key).sum()!=1:raise RuntimeError(f'Duplicate target: {key}')
        rows.append(scores.loc[key])
    s=pd.DataFrame(rows).reset_index(drop=True)
    # Preserve layer information, but avoid a huge 40k-dimensional model on 200 rows.
    # Mid-layer, last-layer, mean-layer embeddings; all standardized on train folds only.
    L=raw.shape[1]; idx=[L//2,L-1]
    X=np.concatenate([np.asarray(raw[:,idx[0],:],dtype=np.float32),
                      np.asarray(raw[:,idx[1],:],dtype=np.float32),
                      np.asarray(raw.mean(axis=1),dtype=np.float32)],axis=1)
    group=np.array([str(m['id']) for m in meta]);y=np.array([int(m['condition']=='poison_1') for m in meta])
    assert all((group==qid).sum()==2 for qid in set(group))
    label_p=np.stack([s[k].to_numpy(dtype=float)/s['n'].to_numpy(dtype=float) for k in ['PA','CA','REJ','PH']],axis=1)
    target={
        'direct':y.astype(float),
        'success':s['success_s'].to_numpy(dtype=float),
        'fixed_T':s['T_alpha_0p7'].to_numpy(dtype=float),
    }
    np.random.seed(args.seed)
    results={name:np.full(len(meta),np.nan) for name in [*target,'learned_T']}
    cv=GroupKFold(n_splits=5)
    per_fold=[]
    for fold,(tr,te) in enumerate(cv.split(X,y,groups=group),1):
        scaler=StandardScaler()
        xtr=np.clip(scaler.fit_transform(X[tr]),-8,8).astype(np.float32)
        xte=np.clip(scaler.transform(X[te]),-8,8).astype(np.float32)
        # Teacher uses only training questions. It predicts binary poisoning from outcome proportions.
        teacher=LogisticRegression(C=1.0,max_iter=2000)
        teacher.fit(label_p[tr],y[tr])
        fold_targets={**{name:values[tr] for name,values in target.items()},
                      'learned_T':teacher.predict_proba(label_p[tr])[:,1]}
        device='cuda' if torch.cuda.is_available() else 'cpu'
        xt=torch.from_numpy(xtr).to(device)
        xv=torch.from_numpy(xte).to(device)
        for j,(name,target_train) in enumerate(fold_targets.items()):
            torch.manual_seed(args.seed+fold*10+j)
            model=Probe(X.shape[1]).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
            yt=torch.tensor(target_train,dtype=torch.float32,device=device)
            for epoch in range(args.epochs):
                model.train()
                perm=torch.randperm(len(tr),device=device)
                for index in perm.split(32):
                    z=model(xt[index]); loss=nn.functional.binary_cross_entropy_with_logits(z,yt[index]) if name=='direct' else nn.functional.mse_loss(torch.sigmoid(z),yt[index])
                    optimizer.zero_grad();loss.backward();optimizer.step()
            model.eval()
            with torch.no_grad():pred=torch.sigmoid(model(xv)).cpu().numpy()
            results[name][te]=pred
            per_fold.append({'fold':fold,'model':name,'n_test':len(te),
                'auc':float(roc_auc_score(y[te],pred)), 'n_train':len(tr)})
            print(f'Fold {fold} {name}: AUC={per_fold[-1]["auc"]:.4f}',flush=True)
    out=root/args.outdir;out.mkdir(parents=True,exist_ok=True)
    report=[]
    for name,pred in results.items():
        assert np.isfinite(pred).all()
        report.append({'model':name,'oof_roc_auc':round(float(roc_auc_score(y,pred)),5),
                       'oof_auprc':round(float(average_precision_score(y,pred)),5),
                       'n':len(y),'n_questions':len(set(group))})
    pd.DataFrame(report).to_csv(out/'probe_oof_summary.csv',index=False)
    pd.DataFrame(per_fold).to_csv(out/'probe_per_fold.csv',index=False)
    pd.DataFrame({'id':group,'condition':[m['condition'] for m in meta],'y_poison':y,**results}).to_csv(out/'probe_oof_predictions.csv',index=False)
    print('\n=== CAMP HIDDEN-STATE PROBE OOF RESULTS ===',flush=True)
    print(pd.DataFrame(report).to_string(index=False),flush=True)
    print('\nNOTE: This is a tiny, selected, controlled-replacement pilot. Do not compare AUC directly with HotpotQA RevPRAG 0.978; 1% FPR too granular per fold.',flush=True)

if __name__=='__main__':main()
