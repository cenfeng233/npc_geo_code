#!/usr/bin/env python3
"""Extract public GSE132112 sample IDs and response labels from GEO metadata."""

import argparse
import csv
import gzip
import re
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-matrix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fields = {}
    with gzip.open(args.series_matrix, "rt") as handle:
        for line in handle:
            if not line.startswith("!Sample_"):
                continue
            row = next(csv.reader([line], delimiter="\t"))
            fields.setdefault(row[0], []).append(row[1:])

    titles = fields["!Sample_title"][0]
    accessions = fields["!Sample_geo_accession"][0]
    sources = fields["!Sample_source_name_ch1"][0]
    responses = [row for row in fields["!Sample_characteristics_ch1"]
                 if all(value.startswith("response: ") for value in row)]
    if len(responses) != 1 or not all(len(row) == 95 for row in (titles, accessions, sources, responses[0])):
        raise ValueError("Expected one response row and 95 aligned samples")

    category = {"response: complete": "CR",
                "response: partial response": "PR",
                "response: stable disease": "SD"}
    records = []
    for index, (title, accession, source, response) in enumerate(
            zip(titles, accessions, sources, responses[0]), 1):
        match = re.fullmatch(r"(LA-NPC_\d+)-(Complete Response|Partial Response|Stable Disease)", source)
        if not match or match.group(1) != f"LA-NPC_{index}":
            raise ValueError(f"Unexpected sample source at position {index}: {source}")
        label = category[response]
        if (label == "SD") != title.startswith("LA-NPC_NR"):
            raise ValueError(f"Title and response disagree for {accession}")
        records.append((match.group(1), accession, title, label, int(label != "SD")))

    if Counter(row[3] for row in records) != Counter({"CR": 32, "PR": 39, "SD": 24}):
        raise ValueError("Unexpected response counts")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("matrix_sample_id", "geo_accession", "geo_title", "response_category", "response_binary"))
        writer.writerows(records)


if __name__ == "__main__":
    main()
