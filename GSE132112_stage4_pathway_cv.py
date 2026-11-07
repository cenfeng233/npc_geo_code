#!/usr/bin/env python3
"""Fixed MSigDB Hallmark pathway models with train-only transformations.

Requires numpy and pandas. Run from the project root after building the Phase 2
gene matrix. All three gene sets are fixed before reading response labels.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from GSE132112_baseline_cv import auc, fit_logistic, metrics, sigmoid, stratified_folds


SET_NAMES = (
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_HYPOXIA",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
)
MODEL_NAMES = (
    "dna_repair_only",
    "hypoxia_only",
    "emt_only",
    "three_pathway_combined",
    "prevalence_baseline",
)


def read_gene_sets(path):
    sets = {}
    with open(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or fields[0] in sets:
                raise ValueError("Invalid GMT file")
            sets[fields[0]] = {"source": fields[1], "genes": fields[2:]}
    if tuple(sets) != SET_NAMES:
        raise ValueError("Unexpected gene sets or ordering")
    if [len(sets[name]["genes"]) for name in SET_NAMES] != [150, 200, 200]:
        raise ValueError("Gene set sizes changed")
    for name in SET_NAMES:
        if len(set(sets[name]["genes"])) != len(sets[name]["genes"]):
            raise ValueError(f"Duplicate symbol in {name}")
    return sets


def pathway_scores(x_train, x_test, index_lists):
    """Average gene z scores, estimating all moments on training samples."""
    train_scores, test_scores = [], []
    for indices in index_lists:
        a, b = x_train[:, indices], x_test[:, indices]
        mean, scale = a.mean(axis=0), a.std(axis=0)
        scale[scale == 0] = 1.0
        train_scores.append(((a - mean) / scale).mean(axis=1))
        test_scores.append(((b - mean) / scale).mean(axis=1))
    train = np.column_stack(train_scores)
    test = np.column_stack(test_scores)
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale == 0] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--sets", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--permutations", type=int, default=0)
    args = parser.parse_args()

    gene_sets = read_gene_sets(args.sets)
    frame = pd.read_csv(args.matrix, compression="gzip")
    genes = list(frame["gene_symbol"])
    samples = list(frame.columns[2:])
    x = frame.iloc[:, 2:].to_numpy(dtype=np.float64).T
    labels = pd.read_csv(args.labels).set_index("matrix_sample_id").loc[samples]
    y = labels["response_binary"].to_numpy(dtype=int)
    if x.shape != (95, 23934) or Counter(y) != Counter({1: 71, 0: 24}):
        raise ValueError("Unexpected matrix dimensions or labels")
    if not np.isfinite(x).all() or len(set(genes)) != len(genes):
        raise ValueError("Invalid gene matrix")

    gene_index = {gene: i for i, gene in enumerate(genes)}
    index_lists = []
    coverage = {}
    for name in SET_NAMES:
        members = gene_sets[name]["genes"]
        matched = [gene for gene in members if gene in gene_index]
        index_lists.append(np.array([gene_index[gene] for gene in matched]))
        coverage[name] = {
            "source_gene_count": len(members),
            "matched_gene_count": len(matched),
            "missing_genes": [gene for gene in members if gene not in gene_index],
            "source_url": gene_sets[name]["source"],
        }
        if len(matched) < 100:
            raise ValueError(f"Too few mapped genes for {name}")

    rng = np.random.default_rng(args.seed)
    results = {model: [] for model in MODEL_NAMES}
    predictions = []
    for repeat in range(args.repeats):
        scores = {model: np.full(len(y), np.nan) for model in MODEL_NAMES}
        fold_index = np.full(len(y), -1)
        for fold, (train, test) in enumerate(stratified_folds(y, 5, rng), 1):
            z_train, z_test = pathway_scores(x[train], x[test], index_lists)
            for model, cols in (
                ("dna_repair_only", [0]),
                ("hypoxia_only", [1]),
                ("emt_only", [2]),
                ("three_pathway_combined", [0, 1, 2]),
            ):
                coef = fit_logistic(z_train[:, cols], y[train].astype(float), penalty=0.1)
                design = np.column_stack((np.ones(len(test)), z_test[:, cols]))
                scores[model][test] = sigmoid(design @ coef)
            scores["prevalence_baseline"][test] = y[train].mean()
            fold_index[test] = fold
        if (fold_index < 0).any() or any(np.isnan(s).any() for s in scores.values()):
            raise AssertionError("Incomplete fold-out predictions")
        for model in MODEL_NAMES:
            results[model].append(metrics(y, scores[model]))
        for i, sample in enumerate(samples):
            predictions.append([repeat + 1, int(fold_index[i]), sample, int(y[i])] + [float(scores[model][i]) for model in MODEL_NAMES])

    Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat", "fold", "matrix_sample_id", "response_binary"] + [name + "_response_score" for name in MODEL_NAMES])
        writer.writerows(predictions)
    summary = {
        "config": {"repeats": args.repeats, "folds": 5, "seed": args.seed, "logistic_l2_penalty": 0.1, "threshold_response_probability": 0.5, "sample_count": len(y), "response_count": int(y.sum()), "nonresponse_count": int((1-y).sum())},
        "gene_set_coverage": coverage,
        "metrics_by_repeat": results,
    }
    if args.permutations:
        if args.permutations < 1:
            raise ValueError("--permutations must be positive")
        generator = np.random.default_rng(args.seed + 100000)
        observed = {model: float(np.mean([row["auroc_response"] for row in results[model]])) for model in MODEL_NAMES[:-1]}
        observed_max = max(observed.values())
        null_max, null_emt = [], []
        for permutation in range(args.permutations):
            shuffled = generator.permutation(y)
            split_rng = np.random.default_rng(args.seed)
            permutation_aucs = {model: [] for model in MODEL_NAMES[:-1]}
            for repeat in range(args.repeats):
                permutation_scores = {model: np.full(len(y), np.nan) for model in MODEL_NAMES[:-1]}
                for train, test in stratified_folds(shuffled, 5, split_rng):
                    z_train, z_test = pathway_scores(x[train], x[test], index_lists)
                    for model, cols in (
                        ("dna_repair_only", [0]),
                        ("hypoxia_only", [1]),
                        ("emt_only", [2]),
                        ("three_pathway_combined", [0, 1, 2]),
                    ):
                        coef = fit_logistic(z_train[:, cols], shuffled[train].astype(float), penalty=0.1)
                        design = np.column_stack((np.ones(len(test)), z_test[:, cols]))
                        permutation_scores[model][test] = sigmoid(design @ coef)
                for model in MODEL_NAMES[:-1]:
                    if np.isnan(permutation_scores[model]).any():
                        raise AssertionError("Incomplete permutation predictions")
                    permutation_aucs[model].append(auc(shuffled, permutation_scores[model]))
            means = {model: float(np.mean(values)) for model, values in permutation_aucs.items()}
            null_max.append(max(means.values()))
            null_emt.append(means["emt_only"])
            if (permutation + 1) % 50 == 0:
                print(f"Completed {permutation + 1}/{args.permutations} permutations", flush=True)
        summary["permutation_test"] = {
            "method": "shuffle patient labels, rerun all four models with identical repeated stratified CV; use maximum mean AUROC across four models to account for within-stage model comparison",
            "permutation_count": args.permutations,
            "observed_mean_auroc_by_model": observed,
            "observed_max_mean_auroc": observed_max,
            "max_model_p_value": (1 + sum(value >= observed_max for value in null_max)) / (args.permutations + 1),
            "emt_unadjusted_p_value": (1 + sum(value >= observed["emt_only"] for value in null_emt)) / (args.permutations + 1),
            "null_max_mean_auroc": null_max,
            "null_emt_mean_auroc": null_emt,
        }
    with open(args.summary, "w") as handle:
        json.dump(summary, handle, indent=2)
    for model, rows in results.items():
        print(model)
        for key in rows[0]:
            values = [row[key] for row in rows]
            print(f"  {key}: mean={np.mean(values):.3f}, min={min(values):.3f}, max={max(values):.3f}")


if __name__ == "__main__":
    main()
