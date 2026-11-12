#!/usr/bin/env python3
"""CAMP NQ offline-label target separability: paired ROC AUC and alpha ablation.

This is NOT hidden-state probe performance. Entropy mapping and response labels
are provisional until validated. Requires only pandas/numpy/scikit-learn.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def pick(df, choices, purpose):
    canon = {str(c).casefold().replace('-', '_').replace(' ', '_'): c for c in df.columns}
    for name in choices:
        key = name.casefold().replace('-', '_').replace(' ', '_')
        if key in canon:
            return canon[key]
    raise ValueError(f"Can't find {purpose}. CSV columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='camp_nq_pilot_output/context_scores.csv')
    ap.add_argument('--output-dir', default='camp_nq_pilot_output/auc_ablation')
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    print('Input columns:', list(df.columns))
    condition = pick(df, ['condition','context_condition','variant'], 'condition')
    score = pick(df, ['s','s_x','attack_score','influence_component','mean_s','attack_success_score','success_score','s_score','score_s','s_value'], 'attack score s(x)')
    entropy = pick(df, ['H_norm','h_normalized','normalized_entropy','entropy_normalized','entropy_norm','h_norm_3','h_norm3','H_normalized','mean_H_norm','normalized_H','h_over_log3'], 'normalized outcome entropy')
    qid = pick(df, ['id','question_id','qid','query_id','sample_id'], 'question ID')

    work = df[[qid,condition,score,entropy]].copy()
    work.columns = ['qid','condition','s','h_norm']
    work['condition'] = work['condition'].astype(str).str.lower().str.strip()
    work = work.loc[work['condition'].isin(['clean','poison_1'])].copy()
    work['s'] = pd.to_numeric(work['s'], errors='raise')
    work['h_norm'] = pd.to_numeric(work['h_norm'], errors='raise')
    if work.empty:
        raise ValueError('No clean or poison_1 rows found')
    if work.duplicated(['qid','condition']).any():
        raise ValueError('More than one row per question/condition; inspect the input.')
    work['y_poison'] = (work['condition']=='poison_1').astype(int)
    if work.y_poison.nunique() != 2:
        raise ValueError('Both clean and poison_1 conditions required')

    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    metrics = []
    col_map = {'s(x)': work.s, 'H(x)/log(3)': work.h_norm}
    for alpha in alphas:
        col_map[f'T_alpha_{alpha:.1f}'] = alpha * work.s + (1-alpha)*work.h_norm
    for name, vals in col_map.items():
        auc = roc_auc_score(work.y_poison, vals)
        clean_mean = float(vals.loc[work.y_poison==0].mean())
        poisoned_mean = float(vals.loc[work.y_poison==1].mean())
        metrics.append({'metric':name, 'roc_auc':float(auc), 'clean_mean':clean_mean,
                        'poisoned_mean':poisoned_mean, 'mean_difference':poisoned_mean-clean_mean})
    summary = pd.DataFrame(metrics)
    print('\nOFFLINE TARGET SEPARABILITY (not a detector/probe AUC):')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))

    wide = work.pivot(index='qid', columns='condition', values=['s','h_norm']).dropna()
    if len(wide) == 0:
        raise ValueError('No paired questions with both conditions')
    paired = pd.DataFrame({'qid':wide.index,
        'delta_s':(wide[('s','poison_1')]-wide[('s','clean')]).values,
        'delta_h_norm':(wide[('h_norm','poison_1')]-wide[('h_norm','clean')]).values})
    for alpha in alphas:
        paired[f'delta_T_{alpha:.1f}'] = alpha*paired.delta_s+(1-alpha)*paired.delta_h_norm
    print(f'\nMatched question pairs: {len(paired)}')
    for column in ['delta_s','delta_h_norm','delta_T_0.7']:
        values = paired[column].to_numpy()
        print(f'{column}: mean={values.mean():.4f}; poison>clean={(values>0).sum()}/{len(values)}; ties={(values==0).sum()}')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out/'auc_alpha_summary.csv',index=False)
    paired.to_csv(out/'paired_differences.csv',index=False)
    (out/'evaluation_notes.txt').write_text(
        'IMPORTANT: This experiment measures OFFLINE SUPERVISION TARGET separability.\n'
        'It uses gold correct/attacker answers in response labeling; do NOT call these\n'
        'figures pre-generation hidden-state probe performance.\n'
        'PA/CA/REJ/AB/PH automated labels and the entropy mapping are provisional.\n'
        'Paired differences compare clean and poison_1 within the same question.\n'
        'Alpha is evaluated retrospectively; do not select alpha by test AUC and then\n'
        'report that same AUC as unbiased generalization.\n')
    print('\nSaved:', out/'auc_alpha_summary.csv', out/'paired_differences.csv')


if __name__=='__main__':
    main()
