#!/usr/bin/env python3
"""Extract Fortran iteration residual lines from DGACC stdout into CSV."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
ITERATION_RE = re.compile(rf"^\s*(\d+)\s+({FLOAT_RE})\s+({FLOAT_RE})\s*$")


def parse_stdout(path: Path) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ITERATION_RE.match(line)
        if not match:
            continue
        step = int(match.group(1))
        residual = float(match.group(2).replace("D", "E").replace("d", "e"))
        mass = float(match.group(3).replace("D", "E").replace("d", "e"))
        rows.append((step, residual, mass))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stdout", type=Path, help="Fortran stdout log.")
    parser.add_argument("-o", "--output", type=Path, help="CSV output path. Defaults to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = parse_stdout(args.stdout)
    if not rows:
        raise ValueError(f"No iteration residual rows found in {args.stdout}")

    lines = ["step,residual,mass"]
    lines.extend(f"{step},{residual:.16e},{mass:.16e}" for step, residual, mass in rows)
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"parse_fortran_residual.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
