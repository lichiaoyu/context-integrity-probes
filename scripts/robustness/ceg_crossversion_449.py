#!/usr/bin/env python3
"""Original-to-Qwen paraphrase robustness, grouped CV, aligned by ID."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def load_dataset(path):
    path = Path(path)
    meta = [
        json.loads(line)
        for line in (path / "metadata.jsonl").open()
        if line.strip()
    ]
    keys = [(str(m["id"]), m["condition"]) for m in meta]
    assert len(keys) == len(set(keys)), "Duplicate keys"

    hidden = np.load(path / "tbg_activations.npy")
    attention = np.load(path / "attention_mass.npy")
    lengths = np.load(path / "passage_token_lengths.npy")

    assert len(keys) == len(hidden) == len(attention) == len(lengths)
    assert hidden.shape[1:] == (28, 1536)
    assert attention.shape[1:] == (28, 5)
    assert lengths.shape[1:] == (5,)
    assert np.isfinite(hidden).all()
    assert np.isfinite(attention).all()
    assert (lengths > 0).all()

    return keys, hidden, attention, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--original",
        default="camp_ceg_successful_attention"
    )
    ap.add_argument(
        "--paraphrased",
        default="camp_ceg_paraphrase_449/qwen_449_activations"
    )
    ap.add_argument(
        "--outdir",
        default="camp_ceg_crossversion_449"
    )
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    ok, oh, oa, ol = load_dataset(args.original)
    qk, qh, qa, ql = load_dataset(args.paraphrased)

    assert set(ok) == set(qk), "Context IDs differ"

    # Reorder paraphrased arrays to match original metadata.
    qindex = {key: i for i, key in enumerate(qk)}
    order = np.array([qindex[key] for key in ok])

    qh = qh[order]
    qa = qa[order]
    ql = ql[order]

    assert [qk[i] for i in order] == ok

    groups = np.array([key[0] for key in ok])
    y = np.array(
        [int(key[1] == "poison_1") for key in ok]
    )
    uq = np.unique(groups)

    assert len(uq) == 449
    assert len(y) == 898
    assert all(
        sorted(y[groups == q].tolist()) == [0, 1]
        for q in uq
    )

    print("ALIGNED: 449 pairs, 898 contexts", flush=True)

    # Each attention feature includes the 28x5 attention mass,
    # plus 5 passage token lengths.
    # The scaler is always fitted on original training data.
    features = []

    for layer in range(28):
        features.append((
            f"hidden_L{layer + 1:02d}",
            np.asarray(oh[:, layer, :], dtype=np.float32),
            np.asarray(qh[:, layer, :], dtype=np.float32)
        ))

    features.append((
        "attention_all",
        np.concatenate([
            oa.reshape(len(y), -1),
            ol.astype(np.float32)
        ], axis=1),
        np.concatenate([
            qa.reshape(len(y), -1),
            ql.astype(np.float32)
        ], axis=1)
    ))

    seeds = [11, 22, 33, 44, 55]
    results = []
    predictions = []

    for seed in seeds:
        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=seed
        )

        splits = []

        for trq, teq in cv.split(uq):
            train_ids = uq[trq]
            test_ids = uq[teq]

            assert not set(train_ids) & set(test_ids)

            tr = np.flatnonzero(np.isin(groups, train_ids))
            te = np.flatnonzero(np.isin(groups, test_ids))
            splits.append((tr, te))

        for name, xo, xq in features:
            oo = np.full(len(y), np.nan)
            oq = np.full(len(y), np.nan)

            for tr, te in splits:
                scaler = StandardScaler()

                xtr = np.clip(
                    scaler.fit_transform(xo[tr]), -8, 8
                )
                xoo = np.clip(
                    scaler.transform(xo[te]), -8, 8
                )
                xoq = np.clip(
                    scaler.transform(xq[te]), -8, 8
                )

                model = LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    solver="lbfgs"
                )
                model.fit(xtr, y[tr])

                oo[te] = model.predict_proba(xoo)[:, 1]
                oq[te] = model.predict_proba(xoq)[:, 1]

            assert np.isfinite(oo).all()
            assert np.isfinite(oq).all()

            auc_oo = roc_auc_score(y, oo)
            auc_oq = roc_auc_score(y, oq)

            results.append({
                "seed": seed,
                "feature": name,
                "auc_OO": auc_oo,
                "auc_OQ": auc_oq,
                "auc_drop": auc_oo - auc_oq
            })

            for i, key in enumerate(ok):
                predictions.append({
                    "seed": seed,
                    "feature": name,
                    "id": key[0],
                    "condition": key[1],
                    "y": int(y[i]),
                    "pred_OO": float(oo[i]),
                    "pred_OQ": float(oq[i])
                })

            print(
                f"Seed {seed} {name}: "
                f"O->O={auc_oo:.4f}, "
                f"O->Q={auc_oq:.4f}, "
                f"drop={auc_oo - auc_oq:+.4f}",
                flush=True
            )

    detail = pd.DataFrame(results)

    summary = detail.groupby(
        "feature", as_index=False
    ).agg(
        auc_OO_mean=("auc_OO", "mean"),
        auc_OO_std=("auc_OO", "std"),
        auc_OQ_mean=("auc_OQ", "mean"),
        auc_OQ_std=("auc_OQ", "std"),
        drop_mean=("auc_drop", "mean"),
        drop_std=("auc_drop", "std")
    )

    detail.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        out / "oof_predictions.csv", index=False
    )

    print("\n=== TOP O->Q FEATURES ===")
    print(
        summary.nlargest(
            10, "auc_OQ_mean"
        ).to_string(index=False),
        flush=True
    )
    print("\nSaved:", out, flush=True)


if __name__ == "__main__":
    main()
