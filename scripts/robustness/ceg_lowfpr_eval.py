#!/usr/bin/env python3
"""Low-FPR evaluation with training-only threshold calibration."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ceg_crossversion_449 import load_dataset

SEEDS = [11, 22, 33, 44, 55]
TARGET_FPRS = [0.01, 0.05]


def fit_probe(x, y, train_idx):
    scaler = StandardScaler()
    x_train = np.clip(
        scaler.fit_transform(x[train_idx]), -8, 8
    )
    model = LogisticRegression(
        C=1.0, max_iter=2000, solver="lbfgs"
    )
    model.fit(x_train, y[train_idx])
    return scaler, model


def predict_probe(fitted, x, indices):
    scaler, model = fitted
    features = np.clip(scaler.transform(x[indices]), -8, 8)
    return model.predict_proba(features)[:, 1]


def calibrate_threshold(scores, labels, target_fpr):
    """Conservative threshold: at most floor(alpha*N) calibration FPs."""
    negatives = np.sort(scores[labels == 0])[::-1]
    allowed_fp = int(np.floor(target_fpr * len(negatives)))
    if allowed_fp >= len(negatives):
        return -np.inf
    return float(np.nextafter(negatives[allowed_fp], np.inf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="camp_ceg_successful_attention")
    ap.add_argument(
        "--paraphrased",
        default="camp_ceg_paraphrase_449/qwen_449_activations"
    )
    ap.add_argument("--outdir", default="camp_ceg_lowfpr")
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

    features = []

    for layer in [14, 17, 18]:
        features.append((
            f"hidden_L{layer:02d}",
            oh[:, layer - 1, :].astype(np.float32),
            qh[:, layer - 1, :].astype(np.float32)
        ))

    for layer in [20, 22]:
        features.append((
            f"attention_L{layer:02d}",
            oa[:, layer - 1, :].astype(np.float32),
            qa[:, layer - 1, :].astype(np.float32)
        ))

    features.append((
        "attention_all",
        np.concatenate(
            [oa.reshape(len(y), -1), ol.astype(np.float32)],
            axis=1
        ),
        np.concatenate(
            [qa.reshape(len(y), -1), ql.astype(np.float32)],
            axis=1
        )
    ))

    records = []
    predictions = []

    for seed in SEEDS:
        outer = KFold(
            n_splits=5, shuffle=True, random_state=seed
        )

        for feature_name, xo, xq in features:
            for fold, (trq, teq) in enumerate(outer.split(uq), 1):
                tr = np.flatnonzero(np.isin(groups, uq[trq]))
                te = np.flatnonzero(np.isin(groups, uq[teq]))

                assert not set(groups[tr]) & set(groups[te])

                for train_version, xtrain in [("O", xo), ("Q", xq)]:
                    # Inner grouped CV: calibration scores for training
                    # questions only, without in-sample predictions.
                    inner = StratifiedGroupKFold(
                        n_splits=4,
                        shuffle=True,
                        random_state=seed + fold
                    )

                    calibration = np.full(len(tr), np.nan)

                    for inner_tr, inner_val in inner.split(
                        xtrain[tr], y[tr], groups[tr]
                    ):
                        fitted_inner = fit_probe(
                            xtrain, y, tr[inner_tr]
                        )
                        calibration[inner_val] = predict_probe(
                            fitted_inner, xtrain, tr[inner_val]
                        )

                    assert np.isfinite(calibration).all()

                    fitted_outer = fit_probe(xtrain, y, tr)

                    for test_version, xtest in [("O", xo), ("Q", xq)]:
                        setting = train_version + test_version
                        test_scores = predict_probe(
                            fitted_outer, xtest, te
                        )

                        auc = roc_auc_score(y[te], test_scores)

                        for fpr_target in TARGET_FPRS:
                            threshold = calibrate_threshold(
                                calibration, y[tr], fpr_target
                            )
                            predicted = test_scores > threshold

                            tpr = predicted[y[te] == 1].mean()
                            fpr = predicted[y[te] == 0].mean()

                            records.append({
                                "seed": seed,
                                "fold": fold,
                                "feature": feature_name,
                                "setting": setting,
                                "target_fpr": fpr_target,
                                "auc": auc,
                                "tpr": tpr,
                                "actual_fpr": fpr,
                                "threshold": threshold
                            })

                        for idx, score in zip(te, test_scores):
                            predictions.append({
                                "seed": seed,
                                "fold": fold,
                                "feature": feature_name,
                                "setting": setting,
                                "id": ok[idx][0],
                                "condition": ok[idx][1],
                                "y": int(y[idx]),
                                "score": float(score)
                            })

                print(
                    f"Seed {seed} {feature_name} fold {fold} done",
                    flush=True
                )

    detail = pd.DataFrame(records)
    detail.to_csv(out / "per_fold.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        out / "oof_predictions.csv", index=False
    )

    summary = detail.groupby(
        ["feature", "setting", "target_fpr"],
        as_index=False
    ).agg(
        tpr_mean=("tpr", "mean"),
        tpr_std=("tpr", "std"),
        actual_fpr_mean=("actual_fpr", "mean"),
        auc_mean=("auc", "mean")
    )

    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print("Saved:", out, flush=True)


if __name__ == "__main__":
    main()
