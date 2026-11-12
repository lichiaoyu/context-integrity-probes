#!/usr/bin/env python3
"""Extract Qwen TBG hidden states + pre-generation per-passage attention from SAME forward pass.
Matches ceg_extract_tbg.py prompt, tokenization, row ordering, and 650-char cap.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from camp_attention import forward_with_attention

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path('camp_ceg_surface_matched_100.jsonl'))
    p.add_argument('--outdir',type=Path,default=Path('camp_ceg_matched_attention'))
    p.add_argument('--model',default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--passage-chars',type=int,default=650)
    p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--limit-pairs',type=int,default=100)
    a=p.parse_args()
    by=defaultdict(dict)
    for ln in a.input.open():
        if not ln.strip(): continue
        r=json.loads(ln); k=str(r['id']); c=r['condition']
        assert c in ('clean','poison_1') and c not in by[k] and len(r['passages'])==5
        by[k][c]=r
    assert by and all(set(v)=={'clean','poison_1'} for v in by.values())
    selected=list(by.items())[:a.limit_pairs]
    tok=AutoTokenizer.from_pretrained(a.model, use_fast=True)
    assert tok.is_fast, 'Fast tokenizer required for offset_mapping'
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
    hs_all=[]; m_all=[]; l_all=[]; meta=[]
    for ki,(qid,d) in enumerate(selected,1):
        assert d['clean']['question']==d['poison_1']['question']
        for c in ('clean','poison_1'):
            r=d[c]; passages=[v[:a.passage_chars] for v in r['passages']]
            contents='\n\n'.join(f'Passage {j}: {v}' for j,v in enumerate(passages,1))
            prompt=f'Use the following passages to answer the question.\n\n{contents}\n\nQuestion: {r["question"]}\nAnswer briefly:'
            formatted=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
            ntok=len(tok(formatted,add_special_tokens=False)['input_ids'])
            assert ntok<=a.max_tokens,(qid,c,ntok)
            # Eager same-pass attention, as provided by teammate.  Rows/format exactly match old extraction.
            hs,mass,lens=forward_with_attention(model,tok,formatted,passages,add_special_tokens=False)
            assert hs.ndim==2 and mass.shape==(hs.shape[0]-1,5) and lens.shape==(5,)
            if not (np.isfinite(hs).all() and np.isfinite(mass).all() and (lens>0).all()):
                print("INVALID OUTPUT:", qid, c, flush=True)
                for name, arr in [("hidden", hs), ("attention", mass), ("lengths", lens)]:
                    print(name, "shape=", arr.shape,
                          "nan=", int(np.isnan(arr).sum()),
                          "inf=", int(np.isinf(arr).sum()),
                          "min=", float(np.nanmin(arr)),
                          "max=", float(np.nanmax(arr)),
                          flush=True)
                raise RuntimeError("Non-finite attention extraction output")
            hs_all.append(hs[1:].astype(np.float16)) # exclude embeddings: original extractor used hidden_states[1:]
            m_all.append(mass.astype(np.float32));l_all.append(lens.astype(np.int32))
            meta.append({'id':qid,'condition':c,'question':r['question'],'model':a.model,'prompt_tokens':ntok})
        if ki%10==0 or ki==len(selected):print(f'Extracted pairs {ki}/{len(selected)}',flush=True)
    a.outdir.mkdir(parents=True,exist_ok=True)
    np.save(a.outdir/'tbg_activations.npy',np.stack(hs_all))
    np.save(a.outdir/'attention_mass.npy',np.stack(m_all))
    np.save(a.outdir/'passage_token_lengths.npy',np.stack(l_all))
    with (a.outdir/'metadata.jsonl').open('w') as f:
        for v in meta:f.write(json.dumps(v,ensure_ascii=False)+'\n')
    print('SAVED',a.outdir,'hidden',np.stack(hs_all).shape,'attention',np.stack(m_all).shape,'lengths',np.stack(l_all).shape,flush=True)
if __name__=='__main__':main()
