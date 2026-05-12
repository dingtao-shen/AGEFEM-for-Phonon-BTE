#!/usr/bin/env python3
"""Compare iteration residual histories stored as CSV."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResidualMetrics:
    column: str
    count: int
    max_abs: float
    l2: float
    rms: float
    rel_l2: float
    max_reference_abs: float
    tolerance: float
    passed: bool


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def compare_column(
    reference: list[dict[str, str]],
    candidate: list[dict[str, str]],
    column: str,
    atol: float,
    rtol: float,
) -> ResidualMetrics:
    if len(reference) != len(candidate):
        raise ValueError(f"Residual history length differs: {len(reference)} vs {len(candidate)}")

    diffs: list[float] = []
    ref_values: list[float] = []
    for index, (ref_row, cand_row) in enumerate(zip(reference, candidate), start=1):
        ref_step = int(ref_row["step"])
        cand_step = int(cand_row["step"])
        if ref_step != cand_step:
            raise ValueError(f"Step mismatch at row {index}: {ref_step} vs {cand_step}")
        if column not in ref_row:
            raise ValueError(f"Reference CSV does not contain column '{column}'")
        if column not in cand_row:
            raise ValueError(f"Candidate CSV does not contain column '{column}'")

        ref_value = float(ref_row[column])
        cand_value = float(cand_row[column])
        ref_values.append(ref_value)
        diffs.append(cand_value - ref_value)

    max_abs = max(abs(value) for value in diffs)
    l2 = math.sqrt(sum(value * value for value in diffs))
    rms = l2 / math.sqrt(len(diffs))
    ref_l2 = math.sqrt(sum(value * value for value in ref_values))
    rel_l2 = l2 / ref_l2 if ref_l2 > 0.0 else (0.0 if l2 == 0.0 else math.inf)
    max_ref = max(abs(value) for value in ref_values)
    tolerance = atol + rtol * max_ref
    passed = max_abs <= tolerance
    return ResidualMetrics(column, len(diffs), max_abs, l2, rms, rel_l2, max_ref, tolerance, passed)


def print_metrics(metrics_list: list[ResidualMetrics], csv_output: bool) -> None:
    if csv_output:
        print("column,count,max_abs,l2,rms,rel_l2,max_reference_abs,tolerance,pass")
        for metrics in metrics_list:
            print(
                f"{metrics.column},{metrics.count},{metrics.max_abs:.16e},{metrics.l2:.16e},"
                f"{metrics.rms:.16e},{metrics.rel_l2:.16e},{metrics.max_reference_abs:.16e},"
                f"{metrics.tolerance:.16e},{int(metrics.passed)}"
            )
        return

    print(
        f"{'column':>12} {'max_abs':>16} {'l2':>16} {'rms':>16} "
        f"{'rel_l2':>16} {'tol':>16} {'pass':>6}"
    )
    for metrics in metrics_list:
        print(
            f"{metrics.column:>12} {metrics.max_abs:16.8e} {metrics.l2:16.8e} "
            f"{metrics.rms:16.8e} {metrics.rel_l2:16.8e} {metrics.tolerance:16.8e} "
            f"{'yes' if metrics.passed else 'no':>6}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference residual CSV.")
    parser.add_argument("candidate", type=Path, help="Candidate residual CSV.")
    parser.add_argument("--column", help="Single column to compare.")
    parser.add_argument("--columns", default="residual", help="Comma-separated columns to compare.")
    parser.add_argument("--atol", type=float, default=1.0e-12, help="Absolute max-norm tolerance.")
    parser.add_argument("--rtol", type=float, default=1.0e-10, help="Relative max-norm tolerance.")
    parser.add_argument("--csv", action="store_true", help="Print machine-readable CSV metrics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("Tolerances must be nonnegative.")
    reference = read_csv(args.reference)
    candidate = read_csv(args.candidate)
    selected = [args.column] if args.column else [item.strip() for item in args.columns.split(",") if item.strip()]
    if not selected:
        raise ValueError("At least one residual-history column must be selected.")
    metrics = [compare_column(reference, candidate, column, args.atol, args.rtol) for column in selected]
    print_metrics(metrics, args.csv)
    return 0 if all(item.passed for item in metrics) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"compare_residual.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
