#!/usr/bin/env python3
"""Four-way original/paraphrased probe evaluation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ceg_crossversion_449 import load_dataset


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
        default="camp_ceg_crossversion_4way"
    )
    ap.add_argument(
        "--layers",
        default="all",
        help="all or comma-separated 1-based layers, e.g. 17,18"
    )
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    ok, oh, oa, ol = load_dataset(args.original)
    qk, qh, qa, ql = load_dataset(args.paraphrased)

    assert set(ok) == set(qk)

    qindex = {key: i for i, key in enumerate(qk)}
    order = np.array([qindex[key] for key in ok])

    qh, qa, ql = qh[order], qa[order], ql[order]

    assert [qk[i] for i in order] == ok

    groups = np.array([key[0] for key in ok])
    y = np.array([int(key[1] == "poison_1") for key in ok])
    uq = np.unique(groups)

    assert len(uq) == 449
    assert len(y) == 898
    assert all(
        sorted(y[groups == q].tolist()) == [0, 1]
        for q in uq
    )

    print("ALIGNED: 449 pairs, 898 contexts", flush=True)

    if args.layers == "all":
        layers = list(range(28))
    else:
        layers = [
            int(x.strip()) - 1
            for x in args.layers.split(",")
        ]
        assert all(0 <= layer < 28 for layer in layers)

    features = []

    for layer in layers:
        features.append((
            f"hidden_L{layer + 1:02d}",
            np.asarray(oh[:, layer, :], dtype=np.float32),
            np.asarray(qh[:, layer, :], dtype=np.float32)
        ))

        # Five passage-level attention mass features at this layer.
        features.append((
            f"attention_L{layer + 1:02d}",
            np.asarray(oa[:, layer, :], dtype=np.float32),
            np.asarray(qa[:, layer, :], dtype=np.float32)
        ))

    # Preserve the original all-layer attention baseline.
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
    records = []

    for seed in seeds:
        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=seed
        )

        splits = []
        for trq, teq in cv.split(uq):
            tr = np.flatnonzero(
                np.isin(groups, uq[trq])
            )
            te = np.flatnonzero(
                np.isin(groups, uq[teq])
            )
            assert not set(groups[tr]) & set(groups[te])
            splits.append((tr, te))

        for name, xo, xq in features:
            predictions = {
                setting: np.full(len(y), np.nan)
                for setting in ["OO", "OQ", "QQ", "QO"]
            }

            for tr, te in splits:
                for train_version, xtrain in [
                    ("O", xo),
                    ("Q", xq)
                ]:
                    scaler = StandardScaler()

                    xtr = np.clip(
                        scaler.fit_transform(xtrain[tr]),
                        -8, 8
                    )

                    model = LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        solver="lbfgs"
                    )
                    model.fit(xtr, y[tr])

                    for test_version, xtest in [
                        ("O", xo),
                        ("Q", xq)
                    ]:
                        setting = train_version + test_version

                        xte = np.clip(
                            scaler.transform(xtest[te]),
                            -8, 8
                        )

                        predictions[setting][te] = (
                            model.predict_proba(xte)[:, 1]
                        )

            aucs = {}
            for setting, pred in predictions.items():
                assert np.isfinite(pred).all()
                aucs[setting] = roc_auc_score(y, pred)

            records.append({
                "seed": seed,
                "feature": name,
                **{
                    f"auc_{setting}": auc
                    for setting, auc in aucs.items()
                }
            })

            print(
                f"Seed {seed} {name}: "
                + " ".join(
                    f"{k}={v:.4f}"
                    for k, v in aucs.items()
                ),
                flush=True
            )

    detail = pd.DataFrame(records)

    summary = detail.groupby(
        "feature", as_index=False
    ).agg(
        **{
            f"auc_{setting}_mean": (
                f"auc_{setting}", "mean"
            )
            for setting in ["OO", "OQ", "QQ", "QO"]
        },
        **{
            f"auc_{setting}_std": (
                f"auc_{setting}", "std"
            )
            for setting in ["OO", "OQ", "QQ", "QO"]
        }
    )

    detail.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    print("\n=== TOP Q->Q FEATURES ===")
    print(
        summary.nlargest(
            10, "auc_QQ_mean"
        ).to_string(index=False),
        flush=True
    )

    print("\nSaved:", out, flush=True)


if __name__ == "__main__":
    main()
