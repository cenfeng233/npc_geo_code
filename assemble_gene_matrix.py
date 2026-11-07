#!/usr/bin/env python3
"""Reassemble and verify the split GSE132112 gene-expression matrix."""

import argparse
import hashlib
from pathlib import Path


EXPECTED_SHA256 = "8f45417fbbd407fdbb93b3f6c64a596a4d78235d10e3d194c52b0c6ef89d3f70"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="GSE132112_gene_expression.csv.gz")
    args = parser.parse_args()

    directory = Path(__file__).resolve().parent
    parts = [directory / f"GSE132112_gene_expression.csv.gz.part{i:02d}" for i in range(8)]
    missing = [part.name for part in parts if not part.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing matrix parts: {', '.join(missing)}")

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with destination.open("wb") as output:
        for part in parts:
            with part.open("rb") as input_file:
                while chunk := input_file.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)

    if digest.hexdigest() != EXPECTED_SHA256:
        destination.unlink()
        raise ValueError("Reassembled matrix failed SHA-256 verification")
    print(f"Verified {destination} (SHA-256: {digest.hexdigest()})")


if __name__ == "__main__":
    main()
