#!/usr/bin/env python3
"""Plot temperature contours from matching Fortran and C++ Tecplot files."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-callaway")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_tecplot(path: Path) -> tuple[list[str], dict[str, int], np.ndarray]:
    variables: list[str] | None = None
    zone: dict[str, int] | None = None
    rows: list[list[float]] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("VARIABLES"):
                variables = re.findall(r'"([^"]+)"', line)
                if not variables:
                    raise ValueError(f"Failed to parse VARIABLES line in {path}")
            elif line.upper().startswith("ZONE"):
                zone = {
                    key.upper(): int(value)
                    for key, value in re.findall(r"\b([IJK])\s*=\s*([0-9]+)", line, flags=re.IGNORECASE)
                }
                if not zone:
                    raise ValueError(f"Failed to parse ZONE line in {path}")
            elif variables is not None and zone is not None:
                values = [float(item) for item in line.replace(",", " ").split()]
                if len(values) != len(variables):
                    raise ValueError(f"Unexpected row width in {path}: {line}")
                rows.append(values)

    if variables is None or zone is None:
        raise ValueError(f"Missing Tecplot header in {path}")
    data = np.asarray(rows, dtype=float)
    expected = np.prod(list(zone.values()))
    if data.shape[0] != expected:
        raise ValueError(f"{path} has {data.shape[0]} rows, expected {expected}")
    return variables, zone, data


def grid(values: np.ndarray, zone: dict[str, int], variables: list[str], column: str) -> np.ndarray:
    if "I" not in zone or "J" not in zone:
        raise ValueError("Only structured 2D ZONE I/J Tecplot files are supported.")
    index = variables.index(column)
    return values[:, index].reshape((zone["J"], zone["I"]))


def plot_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    path: Path,
    title: str,
    cmap: str,
    levels: np.ndarray | int,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    contour = ax.contourf(x, y, z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.contour(x, y, z, levels=levels, colors="black", linewidths=0.18, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(contour, ax=ax, label="Temperature")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fortran", required=True, type=Path, help="Fortran Tecplot field file.")
    parser.add_argument("--cpp", required=True, type=Path, help="C++ Tecplot field file.")
    parser.add_argument("--scheme", required=True, help="Scheme label, e.g. CIS or GSIS.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for PNG files.")
    parser.add_argument("--levels", type=int, default=40, help="Contour level count.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    f_vars, f_zone, f_data = read_tecplot(args.fortran)
    c_vars, c_zone, c_data = read_tecplot(args.cpp)
    if f_vars != c_vars:
        raise ValueError("Fortran and C++ variable lists differ.")
    if f_zone != c_zone:
        raise ValueError("Fortran and C++ zone dimensions differ.")

    x = grid(f_data, f_zone, f_vars, "x")
    y = grid(f_data, f_zone, f_vars, "y")
    tf = grid(f_data, f_zone, f_vars, "T")
    tc = grid(c_data, c_zone, c_vars, "T")
    diff = tc - tf

    t_min = float(min(tf.min(), tc.min()))
    t_max = float(max(tf.max(), tc.max()))
    t_levels = np.linspace(t_min, t_max, args.levels)
    scheme = args.scheme.upper()
    stem = args.scheme.lower()

    plot_field(
        x,
        y,
        tf,
        args.output_dir / f"{stem}_fortran_temperature.png",
        f"{scheme} Fortran Temperature",
        "viridis",
        t_levels,
        t_min,
        t_max,
    )
    plot_field(
        x,
        y,
        tc,
        args.output_dir / f"{stem}_cpp_temperature.png",
        f"{scheme} C++ Temperature",
        "viridis",
        t_levels,
        t_min,
        t_max,
    )

    max_abs = float(np.max(np.abs(diff)))
    if max_abs == 0.0:
        diff_levels: np.ndarray | int = args.levels
        vmin = vmax = None
    else:
        diff_levels = np.linspace(-max_abs, max_abs, args.levels)
        vmin = -max_abs
        vmax = max_abs
    plot_field(
        x,
        y,
        diff,
        args.output_dir / f"{stem}_temperature_difference_cpp_minus_fortran.png",
        f"{scheme} Temperature Difference (C++ - Fortran)",
        "coolwarm",
        diff_levels,
        vmin,
        vmax,
    )

    print(f"Wrote {scheme} contour plots to {args.output_dir}")
    print(f"Temperature max abs difference: {max_abs:.16e}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"plot_temperature_contours.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
