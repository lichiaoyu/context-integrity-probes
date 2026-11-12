#!/usr/bin/env python3
"""Reconstruct NQ pilot prompts from se_rag_large_scale.py and cache Qwen TBG states.
Only run on Explorer GPU compute nodes. Does NOT sample new responses.
"""
import argparse, json, importlib.util
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_source(path):
    spec = importlib.util.spec_from_file_location('se_rag_large_scale', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--raw',default='se_large_output/raw_nq_n100_p1_qwen-qwen2-5-1-5b-instruct.jsonl')
    p.add_argument('--outdir',default='camp_nq_hidden')
    p.add_argument('--max-input-tokens',type=int,default=2048)
    p.add_argument('--passage-chars',type=int,default=650)
    p.add_argument('--limit',type=int,default=0,help='0 means all 100 questions; use 2 for smoke test')
    args=p.parse_args()
    repo=args.repo.resolve()
    raw_path=repo/args.raw
    rows=[json.loads(s) for s in raw_path.read_text().splitlines() if s.strip()]
    if args.limit: rows=rows[:args.limit]
    assert rows and len({str(r['id']) for r in rows})==len(rows)
    assert all(set(r['conditions'])=={'clean','poison_1'} for r in rows)
    model_name=rows[0]['model']; assert all(r['model']==model_name for r in rows)
    assert all(r['prompt_style']=='short' for r in rows)
    print('Questions:',len(rows),'Model:',model_name,flush=True)
    src=load_source(repo/'se_rag_large_scale.py')
    items=src.read_attack_items(repo/'results/adv_targeted_results/nq.json')
    attacks={str(r['id']):r for r in items}
    retrieval=json.loads((repo/'results/beir_results/nq-contriever.json').read_text())
    clean_ids={}; needed=set()
    for r in rows:
        qid=str(r['id']); assert qid in attacks and qid in retrieval
        ids=list(retrieval[qid].keys())[:5]
        assert len(ids)==5
        # Refuse silently changed retrieval identifiers/order.
        if 'clean_doc_ids' in r and r['clean_doc_ids'] != ids:
            raise RuntimeError(f'Clean retrieval mismatch for {qid}: {ids} != {r["clean_doc_ids"]}')
        clean_ids[qid]=ids; needed.update(ids)
    docs=src.read_needed_corpus(repo/'datasets/nq/corpus.jsonl',needed)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).to('cuda').eval()
    outputs=[]; metadata=[]
    for i,r in enumerate(rows):
        qid=str(r['id']); clean=[docs[z] for z in clean_ids[qid]]
        adv=attacks[qid].get('adv_texts') or []
        if isinstance(adv,str): adv=[adv]
        if not adv: raise RuntimeError(f'Missing poisoned passage: {qid}')
        if str(attacks[qid]['question'])!=str(r['question']):
            raise RuntimeError(f'Question mismatch at {qid}')
        for condition,passages in [('clean',clean),('poison_1',[adv[0]]+clean[:4])]:
            clipped=[s[:args.passage_chars] for s in passages]
            prompt=src.build_prompt(str(r['question']),clipped,'short')
            if tokenizer.chat_template:
                formatted=tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=False)
                enc=tokenizer(formatted,return_tensors='pt',add_special_tokens=False)
            else:
                enc=tokenizer(prompt,return_tensors='pt')
            length=enc['input_ids'].shape[1]
            # Original generation script may truncate on the left; refuse mismatched oversized prompts
            # rather than making an undocumented approximation.
            if length>args.max_input_tokens:
                raise RuntimeError(f'{qid}/{condition}: input length {length} > {args.max_input_tokens}. Confirm exact original truncation code (lines 305-320) before extraction.')
            enc={k:v.to('cuda') for k,v in enc.items()}
            with torch.inference_mode():
                output=model(**enc,output_hidden_states=True,use_cache=False,return_dict=True)
            # Transformer block outputs only; skip embedding-state index 0
            hidden=torch.stack([state[0,-1,:].detach().float().cpu() for state in output.hidden_states[1:]])
            outputs.append(hidden.numpy().astype(np.float16))
            metadata.append({'id':qid,'condition':condition,'prompt_tokens':length,'question':r['question'],'model':model_name})
        if (i+1)%10==0 or i==len(rows)-1: print(f'Extracted {i+1}/{len(rows)} questions',flush=True)
    outdir=repo/args.outdir;outdir.mkdir(parents=True,exist_ok=True)
    arr=np.stack(outputs)
    np.save(outdir/'tbg_activations.npy',arr)
    with (outdir/'metadata.jsonl').open('w') as f:
        for row in metadata:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print('Saved',arr.shape,arr.dtype,outdir,flush=True)

if __name__=='__main__':main()
