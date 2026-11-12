import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ACT = Path("camp_ceg_casecontrolled_attention")
LABELS = Path("camp_ceg_successful_samples/answer_labels_v2.csv")
OUT = Path("camp_ceg_layerwise_success_linear")
SEEDS = [11, 22, 33, 44, 55]

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
poison_mask = np.array(
    [m["condition"] == "poison_1" for m in meta]
)

labelmap = {
    (str(q), c): z.sort_values("sample_index")
    for (q, c), z in labs.groupby(["id", "condition"])
}

heldout = []
for m in meta:
    z = labelmap[(str(m["id"]), m["condition"])]
    assert len(z) == 10
    assert z.sample_index.tolist() == list(range(10))
    heldout.append((z.iloc[5:].label == "PA").mean())

heldout = np.asarray(heldout)
Xraw = raw[poison_mask]
qids = groups[poison_mask]
y = (heldout[poison_mask] >= 0.6).astype(int)

assert len(y) == 449
assert len(np.unique(qids)) == 449
assert len(np.unique(y)) == 2

print("Poisoned contexts:", len(y))
print("Success distribution:", np.bincount(y).tolist(), flush=True)

results = []

for layer in range(28):
    X = np.asarray(Xraw[:, layer, :], dtype=np.float32)
    assert np.isfinite(X).all()

    for seed in SEEDS:
        pred = np.full(len(y), np.nan)
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)

        for tr, te in cv.split(qids):
            assert not set(qids[tr]) & set(qids[te])

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

        auc = roc_auc_score(y, pred)
        results.append({
            "layer": layer + 1,
            "seed": seed,
            "success_auc": auc
        })

    print(f"Completed layer {layer + 1}/28", flush=True)

detail = pd.DataFrame(results)
summary = detail.groupby("layer").agg(
    success_auc_mean=("success_auc", "mean"),
    success_auc_std=("success_auc", "std")
).reset_index()

detail.to_csv(OUT / "per_seed.csv", index=False)
summary.to_csv(OUT / "layer_summary.csv", index=False)

print("\n=== TOP SUCCESS LAYERS ===")
print(summary.nlargest(10, "success_auc_mean").to_string(index=False))
print("\nSaved to:", OUT)
