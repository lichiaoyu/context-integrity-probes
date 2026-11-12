#!/usr/bin/env python3
"""Layer-wise Direct vs Learned T(x), with question-grouped CV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def fit_soft_logistic(X, soft_targets):
    """Fit logistic regression to probabilistic labels via weighted examples."""
    n = len(soft_targets)
    X2 = np.concatenate([X, X], axis=0)
    y2 = np.concatenate([
        np.ones(n, dtype=int),
        np.zeros(n, dtype=int)
    ])
    weights = np.concatenate([soft_targets, 1 - soft_targets])

    model = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs"
    )
    model.fit(X2, y2, sample_weight=weights)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--activations-dir",
        default="camp_ceg_successful_attention"
    )
    ap.add_argument(
        "--labels",
        default="camp_ceg_successful_samples/answer_labels_v2.csv"
    )
    ap.add_argument(
        "--outdir",
        default="camp_ceg_layerwise_learned_teacher"
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int,
        default=[11, 22, 33, 44, 55]
    )
    args = ap.parse_args()

    act = Path(args.activations_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    raw = np.load(act / "tbg_activations.npy", mmap_mode="r")
    meta = [
        json.loads(line)
        for line in (act / "metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    labs = pd.read_csv(args.labels, dtype={"id": str})

    assert raw.shape == (898, 28, 1536), raw.shape
    assert len(meta) == len(raw)
    assert not labs.duplicated(
        ["id", "condition", "sample_index"]
    ).any()

    groups = np.array([str(m["id"]) for m in meta])
    y = np.array([
        int(m["condition"] == "poison_1") for m in meta
    ])

    uq = np.unique(groups)
    assert len(uq) == 449
    assert all(
        sorted(y[groups == q].tolist()) == [0, 1]
        for q in uq
    )

    labelmap = {
        (str(q), c): z.sort_values("sample_index")
        for (q, c), z in labs.groupby(["id", "condition"])
    }

    prop, heldout = [], []

    for m in meta:
        key = (str(m["id"]), m["condition"])
        z = labelmap[key]

        assert len(z) == 10
        assert z.sample_index.tolist() == list(range(10))

        counts = z.iloc[:5].label.value_counts()

        prop.append([
            counts.get(k, 0) / 5
            for k in ["PA", "CA", "REJ", "PH", "AB"]
        ])

        heldout.append(
            (z.iloc[5:].label == "PA").mean()
        )

    prop = np.asarray(prop, dtype=float)
    heldout = np.asarray(heldout, dtype=float)

    poison = y == 1
    success_binary = (heldout[poison] >= 0.6).astype(int)

    assert len(np.unique(success_binary)) == 2

    print("Pairs:", len(uq), flush=True)
    print("Contexts:", len(y), flush=True)
    print("Layers:", raw.shape[1], flush=True)
    print(
        "Attack-majority distribution:",
        np.bincount(success_binary).tolist(),
        flush=True
    )

    results = []
    prediction_rows = []

    for seed in args.seeds:
        cv = KFold(
            n_splits=5, shuffle=True, random_state=seed
        )

        splits = []

        for trq, teq in cv.split(uq):
            tr_ids = set(uq[trq])
            te_ids = set(uq[teq])

            assert not tr_ids & te_ids

            tr = np.flatnonzero(
                np.isin(groups, list(tr_ids))
            )
            te = np.flatnonzero(
                np.isin(groups, list(te_ids))
            )

            teacher = LogisticRegression(
                C=1.0, max_iter=2000
            )

            # Same teacher protocol as ceg_fourprobe_eval.py.
            teacher.fit(prop[tr, :4], y[tr])
            soft = teacher.predict_proba(prop[tr, :4])[:, 1]

            splits.append((tr, te, soft))

        for layer in range(raw.shape[1]):
            X = np.asarray(
                raw[:, layer, :], dtype=np.float32
            )

            assert np.isfinite(X).all()

            predictions = {
                "direct": np.full(len(y), np.nan),
                "learned_T": np.full(len(y), np.nan)
            }

            for tr, te, soft in splits:
                scaler = StandardScaler()

                xtr = np.clip(
                    scaler.fit_transform(X[tr]), -8, 8
                )
                xte = np.clip(
                    scaler.transform(X[te]), -8, 8
                )

                direct = LogisticRegression(
                    C=1.0, max_iter=1000,
                    solver="lbfgs"
                )
                direct.fit(xtr, y[tr])

                learned = fit_soft_logistic(xtr, soft)

                predictions["direct"][te] = (
                    direct.predict_proba(xte)[:, 1]
                )
                predictions["learned_T"][te] = (
                    learned.predict_proba(xte)[:, 1]
                )

            for name, pred in predictions.items():
                assert np.isfinite(pred).all()

                rho = spearmanr(
                    pred[poison], heldout[poison]
                ).statistic

                results.append({
                    "seed": seed,
                    "layer": layer + 1,
                    "model": name,
                    "poison_auc": roc_auc_score(y, pred),
                    "attack_majority_auc": roc_auc_score(
                        success_binary, pred[poison]
                    ),
                    "attack_success_spearman": rho
                })

                for i in range(len(y)):
                    prediction_rows.append({
                        "seed": seed,
                        "layer": layer + 1,
                        "model": name,
                        "id": groups[i],
                        "condition": (
                            "poison_1" if y[i] else "clean"
                        ),
                        "prediction": float(pred[i]),
                        "heldout_PA_rate": heldout[i]
                    })

            print(
                f"Seed {seed}: layer {layer + 1}/28",
                flush=True
            )

    detail = pd.DataFrame(results)

    summary = detail.groupby(
        ["layer", "model"], as_index=False
    ).agg(
        poison_auc_mean=("poison_auc", "mean"),
        poison_auc_std=("poison_auc", "std"),
        attack_auc_mean=("attack_majority_auc", "mean"),
        attack_auc_std=("attack_majority_auc", "std"),
        spearman_mean=("attack_success_spearman", "mean"),
        spearman_std=("attack_success_spearman", "std")
    )

    detail.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "layer_summary.csv", index=False)

    pd.DataFrame(prediction_rows).to_csv(
        out / "oof_predictions.csv", index=False
    )

    print("\n=== TOP POISON DETECTION LAYERS ===")
    print(
        summary.nlargest(
            10, "poison_auc_mean"
        ).to_string(index=False)
    )

    print("\n=== TOP ATTACK SUCCESS LAYERS ===")
    print(
        summary.nlargest(
            10, "attack_auc_mean"
        ).to_string(index=False)
    )

    print("\nSaved to:", out, flush=True)


if __name__ == "__main__":
    main()
