#!/usr/bin/env python3
"""Repeated stratified CV for a fixed GSE132112 baseline.

Requires numpy and pandas. Feature ranking, scaling, and logistic fitting
are performed separately inside every training fold.
"""

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(x, y, penalty=0.1):
    design = np.column_stack((np.ones(len(x)), x))
    coef = np.zeros(design.shape[1])
    reg = np.diag([0.0] + [penalty] * x.shape[1])
    for _ in range(50):
        p = sigmoid(design @ coef)
        grad = design.T @ (p - y) / len(y) + reg @ coef
        hessian = (design.T * (p * (1.0 - p) / len(y))) @ design + reg
        hessian[0, 0] += 1e-9
        step = np.linalg.solve(hessian, grad)
        coef -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return coef


def stratified_folds(y, n_splits, rng):
    parts = []
    for cls in (0, 1):
        members = rng.permutation(np.where(y == cls)[0])
        parts.append(np.array_split(members, n_splits))
    for fold in range(n_splits):
        test = np.sort(np.concatenate((parts[0][fold], parts[1][fold])))
        train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        yield train, test


def select_features(x, y, count):
    a, b = x[y == 0], x[y == 1]
    numerator = b.mean(axis=0) - a.mean(axis=0)
    denominator = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
    score = np.abs(numerator / np.maximum(denominator, 1e-10))
    return np.sort(np.argpartition(score, -count)[-count:])


def pca_logistic_scores(x_train_all, y_train, x_test_all):
    """Unsupervised train-only variance filter and PCA, then logistic model."""
    variance = x_train_all.var(axis=0)
    chosen = np.sort(np.argpartition(variance, -500)[-500:])
    mean = x_train_all[:, chosen].mean(axis=0)
    scale = x_train_all[:, chosen].std(axis=0)
    scale[scale == 0] = 1.0
    train = (x_train_all[:, chosen] - mean) / scale
    test = (x_test_all[:, chosen] - mean) / scale
    u, s, vt = np.linalg.svd(train, full_matrices=False)
    train_pc = u[:, :5] * s[:5]
    test_pc = test @ vt[:5].T
    pc_scale = train_pc.std(axis=0)
    pc_scale[pc_scale == 0] = 1.0
    train_pc /= pc_scale
    test_pc /= pc_scale
    coef = fit_logistic(train_pc, y_train.astype(float))
    train_score = sigmoid(np.column_stack((np.ones(len(train_pc)), train_pc)) @ coef)
    test_score = sigmoid(np.column_stack((np.ones(len(test_pc)), test_pc)) @ coef)
    return train_score, test_score


def auc(y, score):
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.arange(1, len(score) + 1, dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2
        start = end
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[np.argsort(order)][pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, score):
    order = np.argsort(-score, kind="mergesort")
    s, truth = score[order], y[order]
    n_pos = int(truth.sum())
    tp = 0
    ap = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and s[end] == s[start]:
            end += 1
        gained = int(truth[start:end].sum())
        tp += gained
        ap += gained / n_pos * (tp / end)
        start = end
    return float(ap)


def metrics(y, prob):
    predicted = prob >= 0.5
    response_recall = float(np.mean(predicted[y == 1]))
    nonresponse_recall = float(np.mean(~predicted[y == 0]))
    return {
        "auroc_response": auc(y, prob),
        "auprc_nonresponse": average_precision(1 - y, 1 - prob),
        "balanced_accuracy": (response_recall + nonresponse_recall) / 2,
        "response_recall": response_recall,
        "nonresponse_recall": nonresponse_recall,
        "brier_score": float(np.mean((prob - y) ** 2)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--exclude-samples", nargs="*", default=[])
    args = parser.parse_args()

    frame = pd.read_csv(args.matrix, compression="gzip")
    genes = frame["gene_symbol"].to_numpy()
    samples = list(frame.columns[2:])
    x = frame.iloc[:, 2:].to_numpy(dtype=np.float64).T
    labels = pd.read_csv(args.labels).set_index("matrix_sample_id").loc[samples]
    y = labels["response_binary"].to_numpy(dtype=int)
    if x.shape != (95, 23934) or Counter(y) != Counter({1: 71, 0: 24}):
        raise ValueError("Unexpected matrix dimensions or label counts")
    unknown = set(args.exclude_samples) - set(samples)
    if unknown:
        raise ValueError(f"Unknown excluded samples: {sorted(unknown)}")
    if args.exclude_samples:
        keep = np.array([sample not in args.exclude_samples for sample in samples])
        x, y = x[keep], y[keep]
        samples = [sample for sample, included in zip(samples, keep) if included]
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite input")

    rng = np.random.default_rng(args.seed)
    all_metrics = []
    pca_metrics = []
    baseline_metrics = []
    selected = Counter()
    train_aucs = []
    pca_train_aucs = []
    predictions = []
    for repeat in range(args.repeats):
        prob = np.full(len(y), np.nan)
        pca_prob = np.full(len(y), np.nan)
        null_prob = np.full(len(y), np.nan)
        fold_index = np.full(len(y), -1)
        for fold, (train, test) in enumerate(stratified_folds(y, 5, rng), 1):
            if len(np.intersect1d(train, test)):
                raise AssertionError("Train/test overlap")
            chosen = select_features(x[train], y[train], 10)
            selected.update(genes[chosen])
            mean = x[train][:, chosen].mean(axis=0)
            scale = x[train][:, chosen].std(axis=0)
            scale[scale == 0] = 1.0
            x_train = (x[train][:, chosen] - mean) / scale
            x_test = (x[test][:, chosen] - mean) / scale
            coef = fit_logistic(x_train, y[train].astype(float))
            train_score = sigmoid(np.column_stack((np.ones(len(train)), x_train)) @ coef)
            train_aucs.append(auc(y[train], train_score))
            prob[test] = sigmoid(np.column_stack((np.ones(len(test)), x_test)) @ coef)
            pca_train_score, pca_test_score = pca_logistic_scores(x[train], y[train], x[test])
            pca_train_aucs.append(auc(y[train], pca_train_score))
            pca_prob[test] = pca_test_score
            null_prob[test] = y[train].mean()
            fold_index[test] = fold
        if np.isnan(prob).any() or np.isnan(pca_prob).any() or np.isnan(null_prob).any() or (fold_index < 0).any():
            raise AssertionError("Incomplete out-of-fold predictions")
        all_metrics.append(metrics(y, prob))
        pca_metrics.append(metrics(y, pca_prob))
        baseline_metrics.append(metrics(y, null_prob))
        for i, sample in enumerate(samples):
            predictions.append([repeat + 1, int(fold_index[i]), sample, int(y[i]), float(prob[i]), float(pca_prob[i]), float(null_prob[i])])

    Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat", "fold", "matrix_sample_id", "response_binary", "ten_gene_response_score", "pca_response_score", "prevalence_baseline_score"])
        writer.writerows(predictions)

    result = {
        "config": {"repeats": args.repeats, "folds": 5, "seed": args.seed, "selected_genes_per_fold": 10, "pca_variance_genes": 500, "pca_components": 5, "logistic_l2_penalty": 0.1, "excluded_samples": args.exclude_samples, "sample_count": len(samples), "response_count": int(y.sum()), "nonresponse_count": int((1-y).sum())},
        "ten_gene_metrics_by_repeat": all_metrics,
        "pca_metrics_by_repeat": pca_metrics,
        "prevalence_baseline_metrics_by_repeat": baseline_metrics,
        "ten_gene_training_auroc_by_fold": train_aucs,
        "pca_training_auroc_by_fold": pca_train_aucs,
        "selected_gene_frequency_top_15": selected.most_common(15),
    }
    with open(args.summary, "w") as handle:
        json.dump(result, handle, indent=2)
    for title, rows in (("ten gene logistic", all_metrics), ("PCA logistic", pca_metrics), ("prevalence baseline", baseline_metrics)):
        print(title)
        for key in rows[0]:
            values = [row[key] for row in rows]
            print(f"  {key}: mean={np.mean(values):.3f}, min={min(values):.3f}, max={max(values):.3f}")
    print("top selected genes:", selected.most_common(10))
    print(f"ten gene training AUROC by fold: mean={np.mean(train_aucs):.3f}, min={min(train_aucs):.3f}, max={max(train_aucs):.3f}")
    print(f"PCA training AUROC by fold: mean={np.mean(pca_train_aucs):.3f}, min={min(pca_train_aucs):.3f}, max={max(pca_train_aucs):.3f}")


if __name__ == "__main__":
    main()
