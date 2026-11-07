#!/usr/bin/env python3
"""Build a label-independent gene expression matrix for GSE132112.

Input: GEO processed probe-level expression matrix and GPL17585 annotation.
Rule: keep PSR exon-region probes with exactly one gene symbol, then take
the per-sample median across probes for each gene. No response labels are read.
"""

import argparse
import csv
import gzip
import json
import math
import statistics
from array import array
from collections import Counter, defaultdict
from pathlib import Path


def read_probe_map(path):
    mapping = {}
    counts = Counter()
    with open(path, newline="") as handle:
        rows = csv.reader((line for line in handle if not line.startswith("#")), delimiter="\t")
        header = next(rows)
        symbol_col = header.index("gene_symbols")
        for row in rows:
            if not row:
                continue
            counts["platform_rows"] += 1
            probe = row[0]
            if not probe.startswith("PSR"):
                counts["not_psr"] += 1
                continue
            symbol = row[symbol_col].strip()
            if not symbol or symbol == "---":
                counts["no_symbol"] += 1
            elif "|" in symbol:
                counts["multiple_symbols"] += 1
            else:
                mapping[probe] = symbol
                counts["eligible_psr"] += 1
    return mapping, counts


def build_matrix(expression, annotation, output):
    probe_map, counts = read_probe_map(annotation)
    gene_values = defaultdict(lambda: array("f"))
    gene_probe_count = Counter()
    with gzip.open(expression, "rt", newline="") as handle:
        rows = csv.reader(handle, delimiter="\t")
        header = next(rows)
        samples = header[1:]
        if len(samples) != 95 or len(set(samples)) != 95:
            raise ValueError("Expected 95 unique sample columns")
        if samples != [f"LA-NPC_{i}" for i in range(1, 96)]:
            raise ValueError("Unexpected sample order")
        for row in rows:
            counts["expression_rows"] += 1
            if len(row) != len(header):
                raise ValueError(f"Wrong column count at expression row {counts['expression_rows']}")
            gene = probe_map.get(row[0])
            if gene is None:
                continue
            values = array("f", (float(value) for value in row[1:]))
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"Nonfinite value at probe {row[0]}")
            gene_values[gene].extend(values)
            gene_probe_count[gene] += 1
            counts["used_probes"] += 1

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene_symbol", "n_probes", *samples])
        for gene in sorted(gene_values):
            values = gene_values[gene]
            n = gene_probe_count[gene]
            if len(values) != n * len(samples):
                raise ValueError(f"Incomplete values for {gene}")
            medians = [statistics.median(values[j::len(samples)]) for j in range(len(samples))]
            writer.writerow([gene, n, *(f"{value:.5f}" for value in medians)])

    counts["genes"] = len(gene_values)
    counts["samples"] = len(samples)
    counts["min_probes_per_gene"] = min(gene_probe_count.values())
    counts["max_probes_per_gene"] = max(gene_probe_count.values())
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_matrix(args.expression, args.annotation, args.output), indent=2))


if __name__ == "__main__":
    main()
