#!/usr/bin/env python3
"""Same-fold question-grouped attention-only and early-feature-fusion MLP evaluation.
NO stacking of cross-fold OOF predictions: each model sees training fold only.
Targets are exactly those from ceg_fourprobe_eval.make_data (first 5 samples).
Attack evaluation uses held-out samples 5..9; NEVER used as training labels.
"""
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from camp_train_nq_probes import Probe
from camp_attention import attention_features
from ceg_fourprobe_eval import make_data

def metrics(y,heldout,p):
    pos=y==1; binary=(heldout[pos]>=.6).astype(int)
    return {'presence_auc':float(roc_auc_score(y,p)),
            'attack_majority_auc':float(roc_auc_score(binary,p[pos])) if len(set(binary))==2 else float('nan'),
            'attack_spearman':float(spearmanr(heldout[pos],p[pos]).statistic)}

def paired_ci(y,pnew,pbase,groups,nboot,seed):
    rng=np.random.default_rng(seed); qids=np.unique(groups)
    idx={q:np.flatnonzero(groups==q) for q in qids};dif=[]
    for _ in range(nboot):
        pick=np.concatenate([idx[q] for q in rng.choice(qids,len(qids),replace=True)])
        if len(np.unique(y[pick]))==2:
            dif.append(roc_auc_score(y[pick],pnew[pick])-roc_auc_score(y[pick],pbase[pick]))
    return [float(x) for x in np.percentile(dif,[2.5,97.5])] if dif else [float('nan')]*2

def train_probe(xtr,xte,target,name,seed,epochs,device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model=Probe(xtr.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
    xt=torch.tensor(xtr,device=device,dtype=torch.float32)
    xv=torch.tensor(xte,device=device,dtype=torch.float32)
    yt=torch.tensor(target,device=device,dtype=torch.float32)
    for _ in range(epochs):
        model.train(); perm=torch.randperm(len(xtr),device=device)
        for batch in perm.split(32):
            z=model(xt[batch]);loss=(nn.functional.binary_cross_entropy_with_logits(z,yt[batch]) if name=='direct'
                else nn.functional.mse_loss(torch.sigmoid(z),yt[batch]))
            opt.zero_grad();loss.backward();opt.step()
    model.eval()
    with torch.no_grad():return torch.sigmoid(model(xv)).cpu().numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='camp_ceg_matched_run')
    ap.add_argument('--attention-dir',default='camp_ceg_matched_attention')
    ap.add_argument('--pairs',default='camp_ceg_surface_matched_100.jsonl')
    ap.add_argument('--baseline-csv',default='camp_ceg_matched_run/camp_ceg_matched_fourprobe_output/oof_predictions.csv')
    ap.add_argument('--outdir',default='camp_ceg_attention_comparison')
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--bootstrap',type=int,default=2000)
    args=ap.parse_args(); torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    X,groups,y,uq,prop,heldout,targets=make_data(Path(args.root))
    md=[json.loads(v) for v in (Path(args.root)/'camp_ceg_100_hidden/metadata.jsonl').read_text().splitlines() if v.strip()]
    ameta=[json.loads(v) for v in (Path(args.attention_dir)/'metadata.jsonl').read_text().splitlines() if v.strip()]
    m=np.load(Path(args.attention_dir)/'attention_mass.npy',mmap_mode='r')
    lens=np.load(Path(args.attention_dir)/'passage_token_lengths.npy',mmap_mode='r')
    amap={(str(v['id']),v['condition']):i for i,v in enumerate(ameta)}
    assert len(ameta)==len(m)==len(lens)==200 and len(amap)==200
    order=[amap[(str(v['id']),v['condition'])] for v in md]
    A=[]; TOP=[]
    for i in order:
        a,t=attention_features(np.asarray(m[i]),np.asarray(lens[i]))
        A.append(a);TOP.append(t)
    A=np.asarray(A,dtype=np.float32);TOP=np.asarray(TOP,dtype=int)
    assert A.shape==(200,3*m.shape[1]) and np.isfinite(A).all()
    orig=pd.read_csv(args.baseline_csv,dtype={'id':str}).set_index(['seed','id','condition'])
    # Poison position: identify the one changed passage per question.
    rows=[json.loads(z) for z in open(args.pairs) if z.strip()]
    by={}
    for r in rows: by.setdefault(str(r['id']),{})[r['condition']]=r
    pslot={}
    for q,d in by.items():
        diff=[i for i,(c,p) in enumerate(zip(d['clean']['passages'],d['poison_1']['passages'])) if c[:650]!=p[:650]]
        assert len(diff)==1,(q,diff)
        pslot[q]=diff[0]
    loc_hits=np.asarray([TOP[i]==pslot[groups[i]] for i in range(len(y)) if y[i]==1])
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    loc=loc_hits.mean(axis=0)
    pd.DataFrame({'layer_1based':np.arange(1,len(loc)+1),'hit_rate':loc,'chance':.2}).to_csv(out/'localization_per_layer.csv',index=False)
    print('Localization pre-specified layer 15:',loc[14], 'all-layer mean:',loc.mean(),'chance:',.2,flush=True)
    results=[]; predr=[]; comps=[]; audit=[]
    for seed in args.seeds:
        cv=KFold(n_splits=5,shuffle=True,random_state=seed)
        p={(kind,target):np.full(len(y),np.nan) for kind in ('attention','fusion') for target in ('direct','success','learned_T')}
        for fold,(trq,teq) in enumerate(cv.split(uq),1):
            train_q=set(uq[trq]); test_q=set(uq[teq]);assert not(train_q & test_q)
            tr=np.flatnonzero(np.isin(groups,list(train_q)))
            te=np.flatnonzero(np.isin(groups,list(test_q)))
            sx=StandardScaler(); xt=np.clip(sx.fit_transform(X[tr]),-8,8).astype(np.float32); xv=np.clip(sx.transform(X[te]),-8,8).astype(np.float32)
            sa=StandardScaler(); at=np.clip(sa.fit_transform(A[tr]),-8,8).astype(np.float32); av=np.clip(sa.transform(A[te]),-8,8).astype(np.float32)
            fused_tr=np.concatenate([xt,at],axis=1);fused_te=np.concatenate([xv,av],axis=1)
            teacher=LogisticRegression(max_iter=2000,C=1.).fit(prop[tr,:4],y[tr])
            tg={'direct':y[tr].astype(float),'success':targets['success'][tr],
                'learned_T':teacher.predict_proba(prop[tr,:4])[:,1]}
            for ti,name in enumerate(('direct','success','learned_T')):
                for ki,kind in enumerate(('attention','fusion')):
                    xxtr,xxte=(at,av) if kind=='attention' else (fused_tr,fused_te)
                    p[kind,name][te]=train_probe(xxtr,xxte,tg[name],name,seed+fold*10+{'direct':0,'success':1,'learned_T':3}[name]+100*(ki+1),args.epochs,device)
            audit.append({'seed':seed,'fold':fold,'train_questions':len(train_q),'test_questions':len(test_q),
                'overlap_questions':len(train_q & test_q),'teacher_train_only':True,'heldout_used_in_training':False})
        for (kind,name),scores in p.items():
            assert np.isfinite(scores).all()
            met=metrics(y,heldout,scores)
            results.append({'seed':seed,'model':f'{kind}_{name}',**met})
            for i,v in enumerate(md):predr.append({'seed':seed,'id':str(v['id']),'condition':v['condition'],
                'model':f'{kind}_{name}','score':float(scores[i])})
            base=np.asarray([orig.loc[(seed,str(v['id']),v['condition']),name] for v in md])
            pos=y==1
            d_presence=met['presence_auc']-roc_auc_score(y,base)
            ci_p=paired_ci(y,scores,base,groups,args.bootstrap,seed)
            att_label=(heldout[pos]>=.6).astype(int)
            d_attack=(roc_auc_score(att_label,scores[pos])-roc_auc_score(att_label,base[pos]) if len(set(att_label))==2 else float('nan'))
            ci_a=(paired_ci(att_label,scores[pos],base[pos],groups[pos],args.bootstrap,seed+123) if len(set(att_label))==2 else [float('nan')]*2)
            comps.append({'seed':seed,'model':f'{kind}_{name}','baseline':name,'delta_presence_auc':d_presence,
                'presence_ci_low':ci_p[0],'presence_ci_high':ci_p[1],
                'delta_attack_auc':d_attack,'attack_ci_low':ci_a[0],'attack_ci_high':ci_a[1]})
        print('completed seed',seed,flush=True)
    pd.DataFrame(results).to_csv(out/'attention_fusion_per_seed.csv',index=False)
    pd.DataFrame(predr).to_csv(out/'attention_fusion_oof.csv',index=False)
    pd.DataFrame(comps).to_csv(out/'paired_delta_vs_hidden_per_seed.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'fold_audit.csv',index=False)
    avg=pd.DataFrame(results).groupby('model').agg(['mean','std'])
    avg.to_csv(out/'attention_fusion_summary.csv');print(avg.to_string(),flush=True)
    print('Paired deltas (each seed bootstrap over QUESTIONS; seeds are not independent replications):',flush=True)
    print(pd.DataFrame(comps).to_string(index=False),flush=True)
    print('Note: Fusion is early FEATURE concatenation, not leakage-prone OOF-score stacking. No re-use of heldout answer labels for training.',flush=True)
if __name__=='__main__':main()
