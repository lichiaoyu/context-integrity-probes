import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path("camp_ceg_paraphrase_pilot")
SEEDS = [11, 22, 33, 44, 55]
LAYERS = [17, 18, 19, 23]  # 1-based
VERSIONS = ["Original", "Qwen"]
CONDITIONS = ["clean", "poison_1"]

TEXT_FILES = {
    "Original": BASE / "pilot_100_original.jsonl",
    "Qwen": BASE / "qwen_100/paraphrased.jsonl",
}
HIDDEN_DIRS = {
    "Original": BASE / "hidden_original_100",
    "Qwen": BASE / "hidden_qwen_100",
}


def load_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def surface_features(r, truncate=True):
    passages = [
        p[:650] if truncate else p
        for p in r["passages"]
    ]
    text = " ".join(passages)
    words = re.findall(r"\b\w+\b", text.lower())
    qwords = set(re.findall(r"\b\w+\b", r["question"].lower()))
    pwords = set(words)
    lengths = [len(p) for p in passages]

    return np.array([
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
    ], dtype=np.float32)


def keyed_rows(rows):
    result = {}
    for r in rows:
        key = (str(r["id"]), r["condition"])
        assert key not in result, f"Duplicate key: {key}"
        result[key] = r
    return result


text_rows = {
    name: keyed_rows(load_jsonl(path))
    for name, path in TEXT_FILES.items()
}

assert text_rows["Original"].keys() == text_rows["Qwen"].keys()
ids = sorted({k[0] for k in text_rows["Original"]})
assert len(ids) == 100

keys = [
    (qid, condition)
    for qid in ids
    for condition in CONDITIONS
]
assert set(keys) == set(text_rows["Original"])

y = np.array([int(c == "poison_1") for _, c in keys])

surface = {
    name: np.stack([
        surface_features(rows[k])
        for k in keys
    ])
    for name, rows in text_rows.items()
}

hidden = {}
for name, folder in HIDDEN_DIRS.items():
    activations = np.load(
        folder / "tbg_activations.npy"
    ).astype(np.float32)
    metadata = load_jsonl(folder / "metadata.jsonl")

    assert activations.shape == (200, 28, 1536)
    assert len(metadata) == 200

    activation_rows = {}
    for i, r in enumerate(metadata):
        key = (str(r["id"]), r["condition"])
        assert key not in activation_rows
        activation_rows[key] = activations[i]

    assert set(activation_rows) == set(keys)
    hidden[name] = np.stack([
        activation_rows[k] for k in keys
    ])

assert all(np.isfinite(x).all() for x in surface.values())
assert all(np.isfinite(x).all() for x in hidden.values())

print("VALIDATION PASSED")
print("Questions:", len(ids))
print("Surface shape:", surface["Original"].shape)
print("Hidden shape:", hidden["Original"].shape)
print("Seeds:", SEEDS)


def get_features(kind, version, layer=None):
    if kind == "Surface":
        return surface[version]
    return hidden[version][:, layer - 1, :]


def evaluate(kind, train_version, test_version, seed, layer=None):
    X_train = get_features(kind, train_version, layer)
    X_test = get_features(kind, test_version, layer)

    scores = np.full(200, np.nan)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    for train_questions, test_questions in kf.split(ids):
        # Each question contributes exactly two adjacent rows:
        # clean followed by poison_1.
        train_idx = np.array([
            2 * i + j
            for i in train_questions
            for j in (0, 1)
        ])
        test_idx = np.array([
            2 * i + j
            for i in test_questions
            for j in (0, 1)
        ])

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000)
        )
        clf.fit(X_train[train_idx], y[train_idx])
        scores[test_idx] = clf.predict_proba(
            X_test[test_idx]
        )[:, 1]

    assert np.isfinite(scores).all()
    auc = roc_auc_score(y, scores)
    pairwise = np.mean(scores[1::2] > scores[0::2])
    return auc, pairwise, scores


# Paired question-level bootstrap, preserving clean/poison pairs.
# Applied to pooled out-of-fold scores.
def bootstrap_auc_ci(scores, seed=2026, n_boot=2000):
    rng = np.random.default_rng(seed)
    aucs = []

    for _ in range(n_boot):
        sampled_questions = rng.integers(0, 100, size=100)
        idx = np.array([
            2 * i + j
            for i in sampled_questions
            for j in (0, 1)
        ])
        aucs.append(roc_auc_score(y[idx], scores[idx]))

    return np.percentile(aucs, [2.5, 97.5])


experiments = [("Surface", None)] + [
    ("Hidden", layer) for layer in LAYERS
]

results = {}

for kind, layer in experiments:
    label = kind if layer is None else f"Hidden-L{layer}"

    for train in VERSIONS:
        for test in VERSIONS:
            metrics = []
            first_scores = None

            for seed in SEEDS:
                auc, pairwise, scores = evaluate(
                    kind, train, test, seed, layer
                )
                metrics.append((auc, pairwise))
                if seed == SEEDS[0]:
                    first_scores = scores

            metrics = np.array(metrics)
            ci = bootstrap_auc_ci(first_scores)
            results[(label, train, test)] = metrics

            print(
                f"{label:11s} {train:8s}->{test:8s} "
                f"AUC={metrics[:,0].mean():.4f}"
                f" +/- {metrics[:,0].std(ddof=1):.4f} "
                f"Pairwise={metrics[:,1].mean():.3f} "
                f"Bootstrap95CI(seed11)="
                f"[{ci[0]:.4f},{ci[1]:.4f}]"
            )

print("\n=== CROSS-VERSION AUC DROP ===")
for kind, layer in experiments:
    label = kind if layer is None else f"Hidden-L{layer}"
    orig = results[(label, "Original", "Original")][:, 0]
    cross = results[(label, "Original", "Qwen")][:, 0]
    drop = orig - cross

    print(
        f"{label:11s} "
        f"Original->Original={orig.mean():.4f} "
        f"Original->Qwen={cross.mean():.4f} "
        f"Drop={drop.mean():.4f} "
        f"+/- {drop.std(ddof=1):.4f}"
    )
