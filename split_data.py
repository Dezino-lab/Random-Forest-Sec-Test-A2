"""Create 80/20 train/test copies of BEED CSV and ARFF with the same split."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "BEED_Data.csv"
ARFF_PATH = ROOT / "BEED_Data_preprocessed.arff"
SEED = 42
TRAIN_FRAC = 0.80


def stratified_indices(labels: list[str], train_frac: float, seed: int) -> tuple[list[int], list[int]]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)

    rng = random.Random(seed)
    train: list[int] = []
    test: list[int] = []
    for label in sorted(by_label):
        idxs = by_label[label]
        rng.shuffle(idxs)
        n_train = int(round(len(idxs) * train_frac))
        # Keep at least one example in each split when a class is large enough.
        if len(idxs) >= 2:
            n_train = min(max(n_train, 1), len(idxs) - 1)
        train.extend(idxs[:n_train])
        test.extend(idxs[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_arff(path: Path, header_lines: list[str], rows: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(header_lines)
        if header_lines and not header_lines[-1].endswith("\n"):
            f.write("\n")
        for row in rows:
            f.write(row if row.endswith("\n") else row + "\n")


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        csv_header = next(reader)
        csv_rows = [row for row in reader if row]

    with ARFF_PATH.open(encoding="utf-8") as f:
        arff_lines = f.readlines()

    data_idx = next(i for i, line in enumerate(arff_lines) if line.strip().lower() == "@data")
    arff_header = arff_lines[: data_idx + 1]
    arff_rows = [line for line in arff_lines[data_idx + 1 :] if line.strip()]

    if len(csv_rows) != len(arff_rows):
        raise SystemExit(
            f"Row count mismatch: CSV has {len(csv_rows)} data rows, ARFF has {len(arff_rows)}"
        )

    labels = [row[-1] for row in csv_rows]
    train_idx, test_idx = stratified_indices(labels, TRAIN_FRAC, SEED)

    csv_train = [csv_rows[i] for i in train_idx]
    csv_test = [csv_rows[i] for i in test_idx]
    arff_train = [arff_rows[i] for i in train_idx]
    arff_test = [arff_rows[i] for i in test_idx]

    write_csv(ROOT / "BEED_Data_train.csv", csv_header, csv_train)
    write_csv(ROOT / "BEED_Data_test.csv", csv_header, csv_test)
    write_arff(ROOT / "BEED_Data_preprocessed_train.arff", arff_header, arff_train)
    write_arff(ROOT / "BEED_Data_preprocessed_test.arff", arff_header, arff_test)

    def counts(rows: list[list[str]]) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for row in rows:
            out[row[-1]] += 1
        return dict(sorted(out.items()))

    print(f"Total: {len(csv_rows)}")
    print(f"Train: {len(csv_train)} ({len(csv_train) / len(csv_rows):.1%})")
    print(f"Test:  {len(csv_test)} ({len(csv_test) / len(csv_rows):.1%})")
    print(f"Train class counts: {counts(csv_train)}")
    print(f"Test class counts:  {counts(csv_test)}")
    print("Wrote:")
    print("  BEED_Data_train.csv")
    print("  BEED_Data_test.csv")
    print("  BEED_Data_preprocessed_train.arff")
    print("  BEED_Data_preprocessed_test.arff")


if __name__ == "__main__":
    main()
