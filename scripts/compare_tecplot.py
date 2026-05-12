#!/usr/bin/env python3
"""Compare two structured Tecplot ASCII files written by the Callaway solvers."""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TecplotData:
    variables: list[str]
    zone: dict[str, int]
    rows: list[list[float]]


@dataclass(frozen=True)
class ColumnMetrics:
    name: str
    count: int
    max_abs: float
    l2: float
    rms: float
    rel_l2: float
    max_reference_abs: float
    tolerance: float
    passed: bool


def _parse_variables(line: str) -> list[str]:
    quoted = re.findall(r'"([^"]+)"', line)
    if quoted:
        return quoted

    _, _, tail = line.partition("=")
    variables = [item.strip() for item in tail.split(",") if item.strip()]
    if not variables:
        raise ValueError(f"Unable to parse Tecplot VARIABLES line: {line.rstrip()}")
    return variables


def _parse_zone(line: str) -> dict[str, int]:
    zone: dict[str, int] = {}
    for key, value in re.findall(r"\b([IJK])\s*=\s*([0-9]+)", line, flags=re.IGNORECASE):
        zone[key.upper()] = int(value)
    if not zone:
        raise ValueError(f"Unable to parse Tecplot ZONE line: {line.rstrip()}")
    return zone


def read_tecplot(path: Path) -> TecplotData:
    variables: list[str] | None = None
    zone: dict[str, int] | None = None
    rows: list[list[float]] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("VARIABLES"):
                variables = _parse_variables(line)
                continue
            if upper.startswith("ZONE"):
                zone = _parse_zone(line)
                continue
            if variables is None or zone is None:
                continue
            try:
                values = [float(item) for item in line.replace(",", " ").split()]
            except ValueError as exc:
                raise ValueError(f"Failed to parse numeric row in {path}: {line}") from exc
            if len(values) != len(variables):
                raise ValueError(
                    f"Row in {path} has {len(values)} columns, expected {len(variables)}: {line}"
                )
            rows.append(values)

    if variables is None:
        raise ValueError(f"Missing Tecplot VARIABLES line in {path}")
    if zone is None:
        raise ValueError(f"Missing Tecplot ZONE line in {path}")

    expected_rows = 1
    for dim in zone.values():
        expected_rows *= dim
    if expected_rows != len(rows):
        raise ValueError(f"{path} has {len(rows)} data rows, expected {expected_rows} from ZONE")

    return TecplotData(variables=variables, zone=zone, rows=rows)


def _selected_indices(variables: list[str], names: Iterable[str] | None) -> list[int]:
    if names is None:
        return list(range(len(variables)))

    by_name = {name: index for index, name in enumerate(variables)}
    indices: list[int] = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"Requested column '{name}' is not present. Available: {variables}")
        indices.append(by_name[name])
    return indices


def compare(
    reference: TecplotData,
    candidate: TecplotData,
    column_names: list[str] | None,
    atol: float,
    rtol: float,
) -> list[ColumnMetrics]:
    if reference.variables != candidate.variables:
        raise ValueError(
            f"Variable lists differ:\n  reference={reference.variables}\n  candidate={candidate.variables}"
        )
    if reference.zone != candidate.zone:
        raise ValueError(f"ZONE dimensions differ: reference={reference.zone}, candidate={candidate.zone}")
    if len(reference.rows) != len(candidate.rows):
        raise ValueError("Data row counts differ.")

    indices = _selected_indices(reference.variables, column_names)
    metrics: list[ColumnMetrics] = []
    for index in indices:
        diffs: list[float] = []
        ref_values: list[float] = []
        for ref_row, cand_row in zip(reference.rows, candidate.rows):
            diffs.append(cand_row[index] - ref_row[index])
            ref_values.append(ref_row[index])

        max_abs = max((abs(value) for value in diffs), default=0.0)
        l2 = math.sqrt(sum(value * value for value in diffs))
        rms = l2 / math.sqrt(len(diffs)) if diffs else 0.0
        ref_l2 = math.sqrt(sum(value * value for value in ref_values))
        rel_l2 = l2 / ref_l2 if ref_l2 > 0.0 else (0.0 if l2 == 0.0 else math.inf)
        max_ref = max((abs(value) for value in ref_values), default=0.0)
        tolerance = atol + rtol * max_ref
        passed = max_abs <= tolerance
        metrics.append(
            ColumnMetrics(
                name=reference.variables[index],
                count=len(diffs),
                max_abs=max_abs,
                l2=l2,
                rms=rms,
                rel_l2=rel_l2,
                max_reference_abs=max_ref,
                tolerance=tolerance,
                passed=passed,
            )
        )

    return metrics


def print_table(metrics: list[ColumnMetrics], csv: bool) -> None:
    if csv:
        print("column,count,max_abs,l2,rms,rel_l2,max_reference_abs,tolerance,pass")
        for item in metrics:
            print(
                f"{item.name},{item.count},{item.max_abs:.16e},{item.l2:.16e},"
                f"{item.rms:.16e},{item.rel_l2:.16e},{item.max_reference_abs:.16e},"
                f"{item.tolerance:.16e},{int(item.passed)}"
            )
        return

    print(
        f"{'column':>12} {'max_abs':>16} {'l2':>16} {'rms':>16} "
        f"{'rel_l2':>16} {'tol':>16} {'pass':>6}"
    )
    for item in metrics:
        print(
            f"{item.name:>12} {item.max_abs:16.8e} {item.l2:16.8e} "
            f"{item.rms:16.8e} {item.rel_l2:16.8e} {item.tolerance:16.8e} "
            f"{'yes' if item.passed else 'no':>6}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference Tecplot ASCII file.")
    parser.add_argument("candidate", type=Path, help="Candidate Tecplot ASCII file.")
    parser.add_argument(
        "--columns",
        help="Comma-separated variable names to compare. Defaults to every Tecplot variable.",
    )
    parser.add_argument("--atol", type=float, default=1.0e-12, help="Absolute max-norm tolerance.")
    parser.add_argument("--rtol", type=float, default=1.0e-10, help="Relative max-norm tolerance.")
    parser.add_argument("--csv", action="store_true", help="Print machine-readable CSV metrics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = None
    if args.columns:
        selected = [item.strip() for item in args.columns.split(",") if item.strip()]
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("Tolerances must be nonnegative.")

    reference = read_tecplot(args.reference)
    candidate = read_tecplot(args.candidate)
    metrics = compare(reference, candidate, selected, args.atol, args.rtol)
    print_table(metrics, args.csv)
    return 0 if all(item.passed for item in metrics) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"compare_tecplot.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
