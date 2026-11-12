#!/usr/bin/env python3
"""CAMP CEG-RAG controlled-replacement reconstruction, with surface matching.

Not real mixed-corpus retrieval. Uses existing CEG-RAG MS-MARCO evaluation
records and clean/poison chunk corpora; avoids a fixed poison position and
matches the donor clean passage to the poison on the TEXT ACTUALLY PROMPTED
(by default, first 650 characters). No GPU/dependencies beyond Python stdlib.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

def tok(s):
    return set(re.findall(r'[a-z0-9]+', str(s).lower()))

def overlap(q, text):
    qset = tok(q)
    return len(qset & tok(text))/max(len(qset),1)

def stable_int(s):
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest()[:16],16)

def avg(seq):
    return round(sum(seq)/len(seq),3) if seq else None

def auc(pos, neg):
    if not pos or not neg: return None
    return round(sum((p>n)+0.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg)),4)

def read(path):
    with open(path,encoding='utf-8') as f: return json.load(f)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--base',type=Path,default=Path('external/CEG-RAG/Data'))
    a.add_argument('--limit-pairs',type=int,default=100)
    a.add_argument('--out',type=Path,default=Path('camp_ceg_surface_matched_100.jsonl'))
    a.add_argument('--passage-chars',type=int,default=650,help='MUST match ceg_extract_tbg.py --passage-chars')
    a.add_argument('--max-len-gap',type=int,default=100)
    a.add_argument('--max-overlap-gap',type=float,default=0.15)
    a.add_argument('--min-query-overlap',type=float,default=0.50)
    a.add_argument('--seed',type=int,default=33)
    a.add_argument('--strict',action='store_true',help='Exit nonzero if fewer than requested pairs found')
    args=a.parse_args()
    if args.passage_chars < 50: a.error('passage chars must be at least 50')
    cases=read(args.base/'Final_test_Msmarco_poisoned_rag.json')
    clean=[r for r in read(args.base/'msmarco_contriever_chunks.json') if str(r.get('text','')).strip()]
    poisons={str(r.get('chunk_id')):r for r in read(args.base/'msmarco_contriever_chunks_poisoned_gpt (1).json')}
    # All scoring features reflect the exact substring used by the extraction script.
    clean_feat=[]
    for r in clean:
        m=r.get('metadata') or {}
        clean_feat.append((r, str(m.get('query_id','')), r['text'][:args.passage_chars],tok(r['text'][:args.passage_chars])))
    rng=random.Random(args.seed)
    cases=list(cases)
    rng.shuffle(cases)  # avoid choosing only the first 100 original evaluation examples
    rows=[]; audit=[]; skips=Counter(); lens_c=[];lens_p=[]; ovs_c=[];ovs_p=[];poscount=Counter()
    seen=set()
    for case in cases:
        if len(audit)>=args.limit_pairs:break
        qid=str(case.get('query_id','')).strip()
        q=str(case.get('query','')).strip()
        correct=str(case.get('clean_answer','')).strip()
        target=str(case.get('desired_answer','')).strip()
        if not all((qid,q,correct,target)) or correct.casefold()==target.casefold():
            skips['missing_or_identical_answers']+=1;continue
        if qid in seen:
            skips['duplicate_question_id']+=1;continue
        qwords=tok(q)
        if not qwords:skips['empty_query']+=1;continue
        # Pick a reusable top-5 clean context by lexical retrieval, ensuring donor and
        # remaining passages are not the same chunk. This is NOT Contriever retrieval.
        scoring=[]
        for r, ownid, text, ts in clean_feat:
            ov=len(qwords&ts)/len(qwords)
            if not text.strip():continue
            scoring.append((ov, int(ownid==qid), r, text))
        scoring.sort(key=lambda e:(-e[0],-e[1],str(e[2].get('chunk_id'))))
        if len(scoring)<5:skips['few_clean']+=1;continue
        # Match the replaced clean slot to an available poison candidate. Both
        # query overlap and length are matched on the *truncated* passages.
        candidates=[]
        for rank,p in enumerate(case.get('reranker_top') or []):
            cid=str(p.get('chunk_id',''))
            pr=poisons.get(cid)
            if not pr or str((pr.get('metadata') or {}).get('query_id',''))!=qid:continue
            ptext=str(pr.get('text',''))
            if not ptext.strip():continue
            pshort=ptext[:args.passage_chars]
            pov=overlap(q,pshort)
            for c_ov, cown, cr, cshort in scoring[:min(len(scoring),200)]:
                ld=abs(len(cshort)-len(pshort))
                od=abs(c_ov-pov)
                if (ld<=args.max_len_gap and od<=args.max_overlap_gap and
                    c_ov>=args.min_query_overlap and pov>=args.min_query_overlap):
                    cost=(ld/max(1,args.max_len_gap) + od/max(.0001,args.max_overlap_gap))
                    candidates.append((cost,-c_ov,rank,-cown,str(cr.get('chunk_id')),cid,cr,pr,ld,od,c_ov,pov))
        if not candidates:
            skips['no_surface_matched_donor']+=1;continue
        candidates.sort(key=lambda z:z[:6])
        _, _,prank,_,donor_id,poison_id,donor,pr,ld,od,cov,pov=candidates[0]
        # Remaining four clean chunks selected by the same approximate retrieval.
        other=[];used={donor_id}
        for _,_,r,_ in scoring:
            rid=str(r.get('chunk_id'))
            if rid not in used:
                other.append(r);used.add(rid)
            if len(other)==4:break
        if len(other)<4:skips['insufficient_distinct_clean']+=1;continue
        # Position deterministic per question, balanced over accepted cases.
        position=len(audit)%5
        donor_at_slot=[None]*5
        donor_at_slot[position]=donor
        slots=iter(other)
        for i in range(5):
            if donor_at_slot[i] is None:donor_at_slot[i]=next(slots)
        cp=[r['text'] for r in donor_at_slot]
        pp=cp.copy();pp[position]=pr['text']
        id_=f'ceg_msmarco_{qid}'
        shared=dict(id=id_,question=q,clean_answer=correct,desired_answer=target,
                    source_dataset='CEG-RAG/msmarco',
                    protocol='controlled_surface_matched_replacement_NOT_REAL_RETRIEVAL',
                    poison_chunk_id=poison_id,poison_position=position+1)
        rows.extend([{**shared,'condition':'clean','passages':cp,'label':0},
                     {**shared,'condition':'poison_1','passages':pp,'label':1}])
        audit.append(dict(id=id_,qid=qid,poison_chunk_id=poison_id,
                          original_saved_reranker_rank=prank+1,donor_clean_chunk_id=donor_id,
                          remaining_clean_ids=[str(r.get('chunk_id')) for r in other],
                          poison_position=position+1,effective_clean_length=len(donor['text'][:args.passage_chars]),
                          effective_poison_length=len(pr['text'][:args.passage_chars]),
                          length_gap=ld,query_overlap_gap=od,
                          clean_query_overlap=cov,poison_query_overlap=pov))
        lens_c.append(len(donor['text'][:args.passage_chars]));lens_p.append(len(pr['text'][:args.passage_chars]));
        ovs_c.append(cov);ovs_p.append(pov);poscount[position+1]+=1;seen.add(qid)
    if not audit:raise SystemExit(f'No matched pairs found; skips: {dict(skips)}. Relax matching thresholds or inspect data.')
    report=dict(pair_count=len(audit),requested=args.limit_pairs,passage_chars=args.passage_chars,
                thresholds=dict(max_len_gap=args.max_len_gap,max_overlap_gap=args.max_overlap_gap,min_query_overlap=args.min_query_overlap),
                paired_positions=dict(poscount),skips=dict(skips),
                audit_metrics=dict(effective_length_clean_mean=avg(lens_c),effective_length_poison_mean=avg(lens_p),
                                   length_single_feature_auc=auc(lens_p,lens_c),
                                   length_best_direction_auc=max(auc(lens_p,lens_c),1-auc(lens_p,lens_c)),
                                   query_overlap_clean_mean=avg(ovs_c),query_overlap_poison_mean=avg(ovs_p),
                                   overlap_single_feature_auc=auc(ovs_p,ovs_c),
                                   overlap_best_direction_auc=max(auc(ovs_p,ovs_c),1-auc(ovs_p,ovs_c))),
                notes=['Controlled replacement, NOT real retrieval.',
                       'Clean donor chosen to match poison on features seen by the model (650-char cap by default).',
                       'Matching on measured surface features is not proof of eliminating all shortcuts.',
                       'New contexts require NEW activation extraction AND response sampling.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    args.out.with_suffix('.audit.json').write_text(json.dumps({'summary':report,'pairs':audit},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Saved',args.out,'and',args.out.with_suffix('.audit.json'))
    if len(audit)<args.limit_pairs:
        print('WARNING: Only',len(audit),'of',args.limit_pairs,'pairs met the thresholds.')
        if args.strict:raise SystemExit(2)
    if report['audit_metrics']['length_best_direction_auc']>0.75 or report['audit_metrics']['overlap_best_direction_auc']>0.75:
        print('WARNING: high residual length/overlap discrimination; adjust matching thresholds before GPU use.')

if __name__=='__main__':main()
