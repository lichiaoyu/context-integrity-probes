import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ACT = Path("camp_ceg_casecontrolled_attention")
X = np.load(ACT / "tbg_activations.npy", mmap_mode="r")[:, 17, :].astype(np.float32)

meta = [json.loads(s) for s in (ACT / "metadata.jsonl").read_text().splitlines() if s.strip()]
groups = np.array([str(m["id"]) for m in meta])
y = np.array([int(m["condition"] == "poison_1") for m in meta])
uq = np.unique(groups)

assert X.shape == (898, 1536)
assert len(uq) == 449
assert all(sorted(y[groups == q]) == [0, 1] for q in uq)

for permute in [False, True]:
    pred = np.full(len(y), np.nan)
    cv = KFold(n_splits=5, shuffle=True, random_state=11)

    for fold, (trq, teq) in enumerate(cv.split(uq)):
        tr = np.flatnonzero(np.isin(groups, uq[trq]))
        te = np.flatnonzero(np.isin(groups, uq[teq]))

        scaler = StandardScaler()
        xtr = np.clip(scaler.fit_transform(X[tr]), -8, 8)
        xte = np.clip(scaler.transform(X[te]), -8, 8)

        ytr = y[tr].copy()

        if permute:
            rng = np.random.default_rng(100 + fold)
            rng.shuffle(ytr)

        model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        model.fit(xtr, ytr)
        pred[te] = model.predict_proba(xte)[:, 1]

    print(
        "PERMUTED" if permute else "REAL",
        "Layer 18 AUC:",
        round(roc_auc_score(y, pred), 4),
        flush=True
    )
