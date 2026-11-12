import json
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

path = "camp_ceg_successful_matched_800_casecontrolled.jsonl"
rows = [json.loads(s) for s in open(path) if s.strip()]

def features(r):
    passages = r["passages"]
    text = " ".join(passages)
    words = re.findall(r"\b\w+\b", text.lower())
    qwords = set(re.findall(r"\b\w+\b", r["question"].lower()))
    pwords = set(words)

    lengths = [len(p) for p in passages]

    return [
        len(text),
        len(words),
        np.mean([len(w) for w in words]) if words else 0,
        len(qwords & pwords) / max(len(qwords), 1),
        sum(c.isdigit() for c in text) / max(len(text), 1),
        sum(c in ".,;:!?" for c in text) / max(len(text), 1),
        np.mean(lengths),
        np.std(lengths),
        max(lengths),
        min(lengths),
    ]

X = np.asarray([features(r) for r in rows])
y = np.asarray([int(r["condition"] == "poison_1") for r in rows])
groups = np.asarray([str(r["id"]) for r in rows])

assert len(rows) == 898
assert Counter(y) == {0:449, 1:449}

seeds = [11,22,33,44,55]
scores = []

for seed in seeds:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.full(len(y), np.nan)

    for train, test in cv.split(X, y, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000)
        )
        model.fit(X[train], y[train])
        pred[test] = model.predict_proba(X[test])[:,1]

    auc = roc_auc_score(y, pred)
    scores.append(auc)
    print(f"Seed {seed}: AUC={auc:.4f}")

print(f"\nSurface-only Presence AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
