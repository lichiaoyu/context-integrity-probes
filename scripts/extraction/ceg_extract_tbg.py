#!/usr/bin/env python3
"""Extract Qwen TBG activations from *verified* clean/poison_1 matched contexts.
Input JSONL example:
{"id":"q1","condition":"clean","question":"...","passages":["...","...","...","...","..."]}
{"id":"q1","condition":"poison_1","question":"...","passages":["...","...","...","...","..."]}
Do NOT infer labels from merely belonging to poisoned corpus: inspect retrieved top-5.
"""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--outdir',type=Path,default=Path('camp_ceg_hidden'))
p.add_argument('--model',default='Qwen/Qwen2.5-1.5B-Instruct')
p.add_argument('--max-tokens',type=int,default=2048)
p.add_argument('--passage-chars',type=int,default=650)
p.add_argument('--limit-pairs',type=int,default=2)
a=p.parse_args()
rows=[json.loads(z) for z in a.input.read_text().splitlines() if z.strip()]
by=defaultdict(dict)
for r in rows:
    assert all(k in r for k in ('id','condition','question','passages'))
    assert r['condition'] in ('clean','poison_1')
    assert isinstance(r['passages'],list) and len(r['passages'])==5
    assert all(isinstance(z,str) and z.strip() for z in r['passages'])
    key=str(r['id']);condition=r['condition']
    assert condition not in by[key],f'duplicate {key}/{condition}'
    by[key][condition]=r
assert by and all(set(d)=={'clean','poison_1'} for d in by.values()),'Need exact clean + poisoned matched pairs'
for k,d in by.items():
    assert d['clean']['question']==d['poison_1']['question'],f'Question mismatch: {k}'
selected=list(by.items())[:a.limit_pairs] if a.limit_pairs>0 else list(by.items())
print('Validated pairs',len(by),'processing pairs',len(selected),flush=True)
tok=AutoTokenizer.from_pretrained(a.model)
model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16).to('cuda').eval()
acts=[];meta=[]
for k,d in selected:
    for condition in ('clean','poison_1'):
        r=d[condition]
        # This prompt is independently constructed. Align with original se_rag_large_scale.py
        # build_prompt + chat_template BEFORE comparing AUC to original CAMP.
        contents='\n\n'.join(f'Passage {j}: {z[:a.passage_chars]}' for j,z in enumerate(r['passages'],1))
        prompt=f'Use the following passages to answer the question.\n\n{contents}\n\nQuestion: {r["question"]}\nAnswer briefly:'
        formatted=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        enc=tok(formatted,return_tensors='pt',add_special_tokens=False)
        if enc['input_ids'].shape[1]>a.max_tokens:raise ValueError(f'Over max tokens {k}/{condition}: {enc["input_ids"].shape[1]}')
        enc={k:v.cuda() for k,v in enc.items()}
        with torch.inference_mode():z=model(**enc,output_hidden_states=True,use_cache=False,return_dict=True)
        h=np.stack([t[0,-1].float().cpu().numpy() for t in z.hidden_states[1:]]).astype(np.float16)
        acts.append(h);meta.append({'id':k,'condition':condition,'question':r['question'],'model':a.model,'prompt_tokens':enc['input_ids'].shape[1]})
    print('extracted',k,flush=True)
a.outdir.mkdir(parents=True,exist_ok=True)
np.save(a.outdir/'tbg_activations.npy',np.stack(acts))
with (a.outdir/'metadata.jsonl').open('w') as f:
    for m in meta:f.write(json.dumps(m,ensure_ascii=False)+'\n')
print('Saved',np.stack(acts).shape,'to',a.outdir,flush=True)
