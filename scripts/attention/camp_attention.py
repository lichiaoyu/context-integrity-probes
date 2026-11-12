"""
camp_attention.py: attention features for CAMP, read in the SAME forward pass as the TBG hidden states.

Why: hidden states and attention then come from identical inputs, so probes built on either (or both)
can be compared on identical question-grouped folds with a paired bootstrap.

------------------------------------------------------------------------------------------------
1) Load the model with eager attention (other kernels do not return attention weights):
       model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, attn_implementation="eager")

2) In the extraction loop, replace the plain forward pass with:
       hs, mass, lens = forward_with_attention(model, tok, prompt_text, passages)
   - prompt_text: the exact string you already feed the model (after apply_chat_template)
   - passages:    the 5 passage strings, exactly as they appear inside prompt_text
   - hs:   (L+1, D) last-prompt-token hidden states, embeddings + every layer (same as before)
   - mass: (L, K)   attention from the last prompt token to each passage, averaged over heads
   - lens: (K,)     passage lengths in tokens
   Save mass and lens next to your hidden states (np.savez), e.g. arrays of shape (N, L, K) and (N, K).
   IMPORTANT: use the same add_special_tokens setting your pipeline already uses, or token positions shift.

3) Analysis (CPU only, no model needed):
       F, top = attention_features(mass, lens)        # F: 3L features per context; top: (L,) top passage per layer
       oof    = oof_scores(X, y, groups, seed)        # out-of-fold scores, folds grouped by question
       fused  = fused_scores(probe_oof, F_all, y, groups, seed)   # your probe's OOF score + attention
       d, ci  = paired_auc_diff(y, fused, probe_oof, groups)      # does attention add anything? CI excludes 0?
       rate   = localization_hit_rate(top_all, poisoned_idx)      # (L,) how often top passage is poisoned

Suggested tests on the reconstructed (shortcut-controlled) pairs:
  A. Presence:        Direct probe vs Direct + attention            (paired bootstrap)
  B. Susceptibility:  Success / Learned T(x) vs + attention          (poisoned contexts, held-out labels)
  C. Localization:    hit rate vs chance, now that poison rank is balanced across 1-5
------------------------------------------------------------------------------------------------
"""
import numpy as np


# ------------------------------------------------------------------ extraction (needs torch + model)
def passage_token_spans(tokenizer, prompt_text, passages, add_special_tokens=False):
    """Token span [start, end) of each passage inside prompt_text, found in order of appearance."""
    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    offs = enc["offset_mapping"]
    spans, cursor = [], 0
    for k, p in enumerate(passages):
        a = prompt_text.find(p, cursor)
        if a < 0:
            raise ValueError(f"passage {k} not found verbatim in the prompt (check whitespace or truncation)")
        b = a + len(p)
        cursor = b
        idx = [t for t, (s, e) in enumerate(offs) if s < b and e > a]
        if not idx:
            raise ValueError(f"passage {k} maps to no tokens")
        spans.append((idx[0], idx[-1] + 1))
    return enc, spans


def forward_with_attention(model, tokenizer, prompt_text, passages, add_special_tokens=False):
    """One forward pass -> (hidden states of last prompt token, attention mass per layer and passage, lengths)."""
    import torch
    enc, spans = passage_token_spans(tokenizer, prompt_text, passages, add_special_tokens)
    dev = next(model.parameters()).device
    inputs = {k: torch.tensor([enc[k]], device=dev) for k in ("input_ids", "attention_mask") if k in enc}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError('no attention returned: load the model with attn_implementation="eager"')
    hs = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()          # (L+1, D)
    mass = np.array([[float(a[0, :, -1, s:e].float().mean(0).sum()) for (s, e) in spans]    # (L, K)
                     for a in out.attentions], dtype=np.float32)
    lens = np.array([e - s for s, e in spans], dtype=np.int32)
    return hs, mass, lens


# ------------------------------------------------------------------ features (numpy only)
def attention_features(mass, lens):
    """mass (L, K), lens (K,) -> (features of length 3L: entropy, max share, variance per layer; top passage per layer)."""
    dens = mass / np.maximum(lens, 1)[None, :]
    w = dens / np.maximum(dens.sum(1, keepdims=True), 1e-12)
    ent = -(w * np.log(np.clip(w, 1e-12, None))).sum(1)
    return np.concatenate([ent, w.max(1), w.var(1)]).astype(np.float32), w.argmax(1)


def localization_hit_rate(top_all, poisoned_idx):
    """top_all (N, L) top passage per layer; poisoned_idx: list of int or list of ints per context -> (L,) hit rate."""
    hits = np.array([np.isin(top_all[i], np.atleast_1d(poisoned_idx[i])) for i in range(len(top_all))])
    return hits.mean(0)


# ------------------------------------------------------------------ evaluation (sklearn)
def oof_scores(X, y, groups, seed=0, C=0.1, n_splits=5):
    """Out-of-fold probabilities from a logistic probe, folds grouped by question and shuffled by seed."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    folds = _group_folds(groups, n_splits, seed)
    s = np.zeros(len(y), dtype=float)
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C))
        s[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return s


def fused_scores(probe_oof, attn_X, y, groups, seed=0, C=0.1, n_splits=5):
    """Stack an existing probe's out-of-fold score with attention features (same grouped folds, no leakage)."""
    X = np.column_stack([probe_oof, attn_X])
    return oof_scores(X, y, groups, seed, C, n_splits)


def paired_auc_diff(y, s_new, s_base, groups, n_boot=2000, seed=0):
    """AUC(s_new) - AUC(s_base) with a 95% CI from resampling questions (both scores resampled together)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    qs = np.unique(groups)
    idx = {q: np.where(groups == q)[0] for q in qs}
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[q] for q in rng.choice(qs, len(qs), replace=True)])
        if len(np.unique(y[pick])) == 2:
            diffs.append(roc_auc_score(y[pick], s_new[pick]) - roc_auc_score(y[pick], s_base[pick]))
    d = roc_auc_score(y, s_new) - roc_auc_score(y, s_base)
    return float(d), [float(v) for v in np.percentile(diffs, [2.5, 97.5])]


def _group_folds(groups, n_splits, seed):
    """Question-grouped folds whose assignment changes with the seed (GroupKFold alone ignores seeds)."""
    rng = np.random.default_rng(seed)
    qs = np.unique(groups)
    rng.shuffle(qs)
    fold_of = {q: i % n_splits for i, q in enumerate(qs)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)]
