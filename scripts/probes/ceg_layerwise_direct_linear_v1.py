#!/usr/bin/env python3
"""CAMP: 28-layer direct linear probes, question-grouped CV."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ACT = Path("camp_ceg_successful_attention")
LABELS = Path("camp_ceg_successful_samples/answer_labels_v2.csv")
OUT = Path("camp_ceg_layerwise_direct_linear_v1")
SEEDS = [11, 22, 33, 44, 55]

# Never overwrite existing experiment results.
OUT.mkdir(parents=True, exist_ok=False)

raw = np.load(ACT / "tbg_activations.npy", mmap_mode="r")
meta = [
    json.loads(line)
    for line in (ACT / "metadata.jsonl").read_text().splitlines()
    if line.strip()
]
labs = pd.read_csv(LABELS, dtype={"id": str})

assert raw.shape == (898, 28, 1536)
assert len(meta) == len(raw)
assert not labs.duplicated(
    ["id", "condition", "sample_index"]
).any()

groups = np.array([str(m["id"]) for m in meta])
y = np.array([int(m["condition"] == "poison_1") for m in meta])
uq = np.unique(groups)

assert len(uq) == 449
assert all(sorted(y[groups == q].tolist()) == [0, 1] for q in uq)

# Held-out response labels: samples 5 through 9.
labelmap = {
    (str(q), c): z.sort_values("sample_index")
    for (q, c), z in labs.groupby(["id", "condition"])
}

heldout = []
for m in meta:
    key = (str(m["id"]), m["condition"])
    z = labelmap[key]
    assert len(z) == 10
    assert z.sample_index.tolist() == list(range(10))
    heldout.append((z.iloc[5:].label == "PA").mean())

heldout = np.array(heldout)
poison_mask = y == 1
success_binary = (heldout[poison_mask] >= 0.6).astype(int)

assert len(np.unique(success_binary)) == 2, \
    "Attack success labels have only one class"

print("Samples:", len(y))
print("Questions:", len(uq))
print("Layers:", raw.shape[1])
print("Success distribution:",
      np.bincount(success_binary).tolist(), flush=True)

# Reuse the same question-level splits across all layers.
splits = {}
for seed in SEEDS:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits[seed] = []
    for trq, teq in cv.split(uq):
        tr = np.flatnonzero(np.isin(groups, uq[trq]))
        te = np.flatnonzero(np.isin(groups, uq[teq]))
        assert not set(groups[tr]) & set(groups[te])
        splits[seed].append((tr, te))

results = []

for layer in range(raw.shape[1]):
    X = np.asarray(raw[:, layer, :], dtype=np.float32)
    assert np.isfinite(X).all()

    for seed in SEEDS:
        pred = np.full(len(y), np.nan)

        for tr, te in splits[seed]:
            scaler = StandardScaler()
            xtr = np.clip(scaler.fit_transform(X[tr]), -8, 8)
            xte = np.clip(scaler.transform(X[te]), -8, 8)

            model = LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver="lbfgs"
            )
            model.fit(xtr, y[tr])
            pred[te] = model.predict_proba(xte)[:, 1]

        assert np.isfinite(pred).all()

        presence_auc = roc_auc_score(y, pred)
        success_auc = roc_auc_score(
            success_binary, pred[poison_mask]
        )

        results.append({
            "layer": layer + 1,
            "seed": seed,
            "probe": "direct_linear",
            "presence_auc": presence_auc,
            "attack_success_auc": success_auc
        })

    print(f"Completed layer {layer + 1}/28", flush=True)

detail = pd.DataFrame(results)

summary = detail.groupby("layer").agg(
    presence_auc_mean=("presence_auc", "mean"),
    presence_auc_std=("presence_auc", "std"),
    success_auc_mean=("attack_success_auc", "mean"),
    success_auc_std=("attack_success_auc", "std")
).reset_index()

with (OUT / "per_seed.csv").open("x") as f:
    detail.to_csv(f, index=False)

with (OUT / "layer_summary.csv").open("x") as f:
    summary.to_csv(f, index=False)

print("\n=== TOP PRESENCE LAYERS ===")
print(summary.nlargest(5, "presence_auc_mean").to_string(index=False))

print("\n=== TOP ATTACK SUCCESS LAYERS ===")
print(summary.nlargest(5, "success_auc_mean").to_string(index=False))

print("\nSaved to:", OUT)
