#!/usr/bin/env python3
"""Run and summarize the square-domain validation cases from Liu et al. JCP 2022."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = REPO_ROOT / "build" / "callaway_mfem"
DEFAULT_VALIDATION_DIR = REPO_ROOT / "validation"
DEFAULT_MESH = "../../reference/FortranCodes/A1_Nx11_Ny11.msh"


@dataclass(frozen=True)
class PaperCase:
    case_id: str
    knr: float
    knn: float
    ntheta: int
    nphi: int
    expected_gsis_steps: int
    expected_cis_steps: int | None
    notes: str = ""

    @property
    def tau_threshold(self) -> float:
        return 100.0 if math.isclose(self.knr, 10.0) else 1.0


CASES: tuple[PaperCase, ...] = (
    PaperCase("knr1e-3_knn1e5", 1.0e-3, 1.0e5, 20, 40, 43, None, "Fourier-limit case; paper reports CIS > 1e6 steps."),
    PaperCase("knr1e-2_knn1e5", 1.0e-2, 1.0e5, 40, 80, 24, 13234),
    PaperCase("knr1e-1_knn1e5", 1.0e-1, 1.0e5, 40, 80, 26, 269),
    PaperCase("knr1_knn1", 1.0, 1.0, 80, 160, 28, 29),
    PaperCase("knr10_knn1e-2", 10.0, 1.0e-2, 40, 80, 47, 1883),
)

SCHEMES = ("gsis", "cis")
SOLVE_RE = re.compile(
    r"(?P<scheme>GSIS|CIS) solve: steps=(?P<steps>\d+), "
    r"converged=(?P<converged>yes|no), final_residual=(?P<residual>[-+0-9.eE]+), "
    r"mass=(?P<mass>[-+0-9.eE]+)"
)


def format_float(value: float) -> str:
    return f"{value:.16g}"


def config_text(case: PaperCase, scheme: str, output_prefix: Path, sample_count: int, max_steps: int) -> str:
    enabled = "true" if scheme == "gsis" else "false"
    trace_preconditioner = "direct" if scheme == "gsis" else "none"
    boundary_heat_flux = "true" if scheme == "gsis" and math.isclose(case.knr, 10.0) and math.isclose(case.knn, 1.0e-2) else "false"
    return f"""iteration:
  tolerance: 1.0e-7
  max_steps: {max_steps}

gsis:
  enabled: {enabled}
  trace_relative_tolerance: 1.0e-10
  trace_absolute_tolerance: 1.0e-14
  trace_max_iterations: 1000
  trace_print_level: -1
  trace_preconditioner: {trace_preconditioner}
  boundary_heat_flux_from_vdf: {boundary_heat_flux}

velocity_mesh:
  polar_angles: {case.ntheta}
  azimuthal_angles: {case.nphi}

dg:
  order: 3

flow:
  specific_heat: 1.0
  group_velocity: 1.0
  tau_r: {format_float(case.knr)}
  tau_n: {format_float(case.knn)}
  tau_threshold: {format_float(case.tau_threshold)}

files:
  mesh: {DEFAULT_MESH}
  output_prefix: {output_prefix.as_posix()}
  output_samples: {sample_count}

boundary_conditions:
  - name: SWall
    physical_id: 11
    type: thermalizing
    temperature: 0.0
    x_offset: 0.0
    y_offset: 0.0
  - name: NWall
    physical_id: 12
    type: thermalizing
    temperature: 1.0
    x_offset: 0.0
    y_offset: 0.0
  - name: EWall
    physical_id: 13
    type: thermalizing
    temperature: 0.0
    x_offset: 0.0
    y_offset: 0.0
  - name: WWall
    physical_id: 14
    type: thermalizing
    temperature: 0.0
    x_offset: 0.0
    y_offset: 0.0
"""


def write_case_configs(validation_dir: Path, sample_count: int, max_steps: int) -> dict[tuple[str, str], Path]:
    config_dir = validation_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    result: dict[tuple[str, str], Path] = {}
    for case in CASES:
        for scheme in SCHEMES:
            output_prefix = Path("validation") / "results" / case.case_id / scheme / scheme
            path = config_dir / f"{case.case_id}_{scheme}.yaml"
            path.write_text(config_text(case, scheme, output_prefix, sample_count, max_steps))
            result[(case.case_id, scheme)] = path
    return result


def expected_steps(case: PaperCase, scheme: str) -> int | None:
    return case.expected_gsis_steps if scheme == "gsis" else case.expected_cis_steps


def output_prefix(validation_dir: Path, case_id: str, scheme: str) -> Path:
    return validation_dir / "results" / case_id / scheme / scheme


def run_case(
    executable: Path,
    config_path: Path,
    validation_dir: Path,
    case: PaperCase,
    scheme: str,
    timeout_seconds: int | None,
    resume: bool,
) -> dict[str, object]:
    prefix = output_prefix(validation_dir, case.case_id, scheme)
    result_dir = prefix.parent
    result_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = result_dir / "stdout.log"
    metadata_path = result_dir / "run_metadata.json"
    field_path = prefix.with_name(prefix.name + "_field.dat")
    residual_path = prefix.with_name(prefix.name + "_residual.csv")
    reference_path = prefix.with_name(prefix.name + "_reference.dat")
    paraview_path = prefix.with_name(prefix.name + "_paraview")

    if resume and metadata_path.exists() and field_path.exists() and residual_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata["resumed"] = True
        return metadata

    command = [
        str(executable),
        "--config",
        str(config_path),
        "--solve",
        "--write-output",
    ]
    if case.ntheta * case.nphi >= 20_000:
        command.append("--no-cache-local-lu")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/callaway_validation_mplconfig")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        env=env,
        check=False,
    )
    elapsed = time.monotonic() - started
    stdout_path.write_text(completed.stdout)

    solve_match = SOLVE_RE.search(completed.stdout)
    metadata: dict[str, object] = {
        "case_id": case.case_id,
        "scheme": scheme,
        "knr": case.knr,
        "knn": case.knn,
        "ntheta": case.ntheta,
        "nphi": case.nphi,
        "tau_threshold": case.tau_threshold,
        "expected_steps_paper": expected_steps(case, scheme),
        "returncode": completed.returncode,
        "wall_time_s": elapsed,
        "stdout": str(stdout_path.relative_to(REPO_ROOT)),
        "field": str(field_path.relative_to(REPO_ROOT)),
        "residual": str(residual_path.relative_to(REPO_ROOT)),
        "reference": str(reference_path.relative_to(REPO_ROOT)),
        "paraview": str(paraview_path.relative_to(REPO_ROOT)),
        "notes": case.notes,
        "cache_local_lu": case.ntheta * case.nphi < 20_000,
    }
    if solve_match:
        metadata.update(
            {
                "steps": int(solve_match.group("steps")),
                "converged": solve_match.group("converged") == "yes",
                "final_residual": float(solve_match.group("residual")),
                "mass": float(solve_match.group("mass")),
            }
        )
    else:
        metadata.update({"steps": None, "converged": False, "final_residual": None, "mass": None})

    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def read_residual(path: Path) -> tuple[int | None, float | None]:
    if not path.exists():
        return None, None
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None, None
    last = rows[-1]
    return int(last["step"]), float(last["residual"])


def parse_tecplot(path: Path) -> dict[str, object]:
    import numpy as np

    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Tecplot file is too short: {path}")
    variables = re.findall(r'"([^"]+)"', lines[0])
    zone_match = re.search(r"I\s*=\s*(\d+)\s+J\s*=\s*(\d+)", lines[1], re.IGNORECASE)
    if not zone_match:
        raise ValueError(f"Could not parse Tecplot zone in {path}")
    nx = int(zone_match.group(1))
    ny = int(zone_match.group(2))
    data = np.loadtxt(path, skiprows=2)
    if data.shape[0] != nx * ny:
        raise ValueError(f"Tecplot row count mismatch in {path}")
    columns = {name: data[:, i].reshape((ny, nx)) for i, name in enumerate(variables)}
    return {"nx": nx, "ny": ny, "variables": variables, "columns": columns}


def field_temperature_metrics(a_path: Path, b_path: Path) -> dict[str, float]:
    import numpy as np

    a = parse_tecplot(a_path)["columns"]["T"]
    b = parse_tecplot(b_path)["columns"]["T"]
    diff = a - b
    return {
        "max_abs_temperature": float(np.max(np.abs(diff))),
        "rms_temperature": float(np.sqrt(np.mean(diff * diff))),
        "relative_l2_temperature": float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(b.ravel()), 1.0e-300)),
    }


def write_summary(validation_dir: Path, metadata_rows: list[dict[str, object]]) -> None:
    case_order = {case.case_id: index for index, case in enumerate(CASES)}
    scheme_order = {scheme: index for index, scheme in enumerate(SCHEMES)}
    metadata_rows = sorted(
        metadata_rows,
        key=lambda row: (
            case_order.get(str(row.get("case_id")), 10_000),
            scheme_order.get(str(row.get("scheme")), 10_000),
        ),
    )
    summary_path = validation_dir / "summary.csv"
    fields = [
        "case_id",
        "scheme",
        "knr",
        "knn",
        "ntheta",
        "nphi",
        "tau_threshold",
        "expected_steps_paper",
        "steps",
        "converged",
        "final_residual",
        "wall_time_s",
        "returncode",
        "max_abs_T_vs_fourier",
        "rms_T_vs_fourier",
        "rel_l2_T_vs_fourier",
        "field",
        "residual",
        "reference",
        "paraview",
        "stdout",
        "notes",
    ]
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    comparison_path = validation_dir / "scheme_comparison.csv"
    comparison_fields = [
        "case_id",
        "knr",
        "knn",
        "gsis_steps",
        "cis_steps",
        "paper_gsis_steps",
        "paper_cis_steps",
        "step_ratio_cis_over_gsis",
        "max_abs_T_gsis_minus_cis",
        "rms_T_gsis_minus_cis",
        "rel_l2_T_gsis_minus_cis",
    ]
    by_key = {(row["case_id"], row["scheme"]): row for row in metadata_rows}
    with comparison_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields)
        writer.writeheader()
        for case in CASES:
            gsis = by_key.get((case.case_id, "gsis"), {})
            cis = by_key.get((case.case_id, "cis"), {})
            row: dict[str, object] = {
                "case_id": case.case_id,
                "knr": case.knr,
                "knn": case.knn,
                "gsis_steps": gsis.get("steps", ""),
                "cis_steps": cis.get("steps", ""),
                "paper_gsis_steps": case.expected_gsis_steps,
                "paper_cis_steps": case.expected_cis_steps or ">1e6",
                "step_ratio_cis_over_gsis": "",
                "max_abs_T_gsis_minus_cis": "",
                "rms_T_gsis_minus_cis": "",
                "rel_l2_T_gsis_minus_cis": "",
            }
            if isinstance(gsis.get("steps"), int) and isinstance(cis.get("steps"), int) and gsis["steps"]:
                row["step_ratio_cis_over_gsis"] = float(cis["steps"]) / float(gsis["steps"])
            gsis_field_value = gsis.get("field")
            cis_field_value = cis.get("field")
            gsis_field = REPO_ROOT / str(gsis_field_value) if gsis_field_value else None
            cis_field = REPO_ROOT / str(cis_field_value) if cis_field_value else None
            if gsis_field and cis_field and gsis_field.exists() and cis_field.exists():
                metrics = field_temperature_metrics(gsis_field, cis_field)
                row["max_abs_T_gsis_minus_cis"] = metrics["max_abs_temperature"]
                row["rms_T_gsis_minus_cis"] = metrics["rms_temperature"]
                row["rel_l2_T_gsis_minus_cis"] = metrics["relative_l2_temperature"]
            writer.writerow(row)

    write_markdown_summary(validation_dir, metadata_rows)


def write_markdown_summary(validation_dir: Path, metadata_rows: list[dict[str, object]]) -> None:
    by_key = {(row["case_id"], row["scheme"]): row for row in metadata_rows}

    def step_text(row: dict[str, object]) -> object:
        if not row:
            return ""
        if row.get("returncode") == "timeout":
            return "timeout"
        return row.get("steps", "")

    lines = [
        "# Square-Domain Paper Validation",
        "",
        "These results reproduce Section 4.3 / Table 3 of Liu et al., JCP 467 (2022) 111436.",
        "",
        "| Case | KnR | KnN | Ntheta | Nphi | GSIS steps | CIS steps | Paper GSIS | Paper CIS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        gsis = by_key.get((case.case_id, "gsis"), {})
        cis = by_key.get((case.case_id, "cis"), {})
        lines.append(
            "| {case} | {knr:g} | {knn:g} | {ntheta} | {nphi} | {gsis_steps} | {cis_steps} | {pgsis} | {pcis} |".format(
                case=case.case_id,
                knr=case.knr,
                knn=case.knn,
                ntheta=case.ntheta,
                nphi=case.nphi,
                gsis_steps=step_text(gsis),
                cis_steps=step_text(cis),
                pgsis=case.expected_gsis_steps,
                pcis=case.expected_cis_steps if case.expected_cis_steps is not None else ">1e6",
            )
        )
    lines.extend(
        [
            "",
            "Generated files:",
            "",
            "- `summary.csv`: per-run iteration, residual, timing, and Fourier-limit metrics.",
            "- `scheme_comparison.csv`: GSIS/CIS pairwise temperature-field comparisons.",
            "- `figures/*_temperature_contours.png`: side-by-side temperature contours.",
            "- `results/<case>/<scheme>/`: residual CSV, Tecplot field, ParaView collection, Fourier reference, stdout, and run metadata.",
            "",
        ]
    )
    (validation_dir / "README.md").write_text("\n".join(lines))


def enrich_with_fourier_metrics(validation_dir: Path, metadata_rows: list[dict[str, object]]) -> None:
    for row in metadata_rows:
        if row.get("case_id") != "knr1e-3_knn1e5":
            continue
        field_value = row.get("field")
        reference_value = row.get("reference")
        if not field_value or not reference_value:
            continue
        field = REPO_ROOT / str(field_value)
        reference = REPO_ROOT / str(reference_value)
        if field.exists() and reference.exists():
            metrics = field_temperature_metrics(field, reference)
            row["max_abs_T_vs_fourier"] = metrics["max_abs_temperature"]
            row["rms_T_vs_fourier"] = metrics["rms_temperature"]
            row["rel_l2_T_vs_fourier"] = metrics["relative_l2_temperature"]


def plot_contours(validation_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = validation_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        panels: list[tuple[str, Path]] = []
        for scheme in SCHEMES:
            path = output_prefix(validation_dir, case.case_id, scheme)
            field = path.with_name(path.name + "_field.dat")
            if field.exists():
                panels.append((scheme.upper(), field))
        ref = output_prefix(validation_dir, case.case_id, "gsis").with_name("gsis_reference.dat")
        if case.case_id == "knr1e-3_knn1e5" and ref.exists():
            panels.append(("Fourier", ref))
        if not panels:
            continue

        parsed = [(label, parse_tecplot(path)) for label, path in panels]
        temperatures = [item[1]["columns"]["T"] for item in parsed]
        vmin = min(float(np.min(t)) for t in temperatures)
        vmax = max(float(np.max(t)) for t in temperatures)
        levels = np.linspace(vmin, vmax, 31)

        fig, axes = plt.subplots(1, len(parsed), figsize=(5.0 * len(parsed), 4.5), constrained_layout=True)
        if len(parsed) == 1:
            axes = [axes]
        contour = None
        for axis, (label, data) in zip(axes, parsed):
            columns = data["columns"]
            contour = axis.contourf(columns["x"], columns["y"], columns["T"], levels=levels, cmap="inferno")
            axis.contour(columns["x"], columns["y"], columns["T"], levels=levels[::5], colors="white", linewidths=0.35, alpha=0.6)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("x1")
            axis.set_ylabel("x2")
            axis.set_title(f"{label}: KnR={case.knr:g}, KnN={case.knn:g}")
        if contour is not None:
            fig.colorbar(contour, ax=axes, label="Temperature")
        fig.savefig(figures_dir / f"{case.case_id}_temperature_contours.png", dpi=180)
        plt.close(fig)


def selected_cases(case_ids: list[str]) -> list[PaperCase]:
    if not case_ids or case_ids == ["all"]:
        return list(CASES)
    by_id = {case.case_id: case for case in CASES}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise SystemExit(f"Unknown case id(s): {', '.join(missing)}")
    return [by_id[case_id] for case_id in case_ids]


def selected_schemes(schemes: list[str]) -> list[str]:
    if not schemes or schemes == ["all"]:
        return list(SCHEMES)
    unknown = [scheme for scheme in schemes if scheme not in SCHEMES]
    if unknown:
        raise SystemExit(f"Unknown scheme(s): {', '.join(unknown)}")
    return schemes


def load_existing_metadata(validation_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted((validation_dir / "results").glob("*/*/run_metadata.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--sample-count", type=int, default=109)
    parser.add_argument("--max-steps", type=int, default=8_000_000)
    parser.add_argument("--case", action="append", default=None, help="Case id to run; repeatable. Default: all.")
    parser.add_argument("--scheme", action="append", default=None, help="gsis, cis, or all. Repeatable. Default: all.")
    parser.add_argument("--resume", action="store_true", help="Skip runs that already have metadata, residual, and field output.")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate summaries and contour plots from existing outputs.")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    args = parser.parse_args()

    validation_dir = args.validation_dir.resolve()
    validation_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/callaway_validation_mplconfig")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    config_paths = write_case_configs(validation_dir, args.sample_count, args.max_steps)

    if args.plot_only:
        rows = load_existing_metadata(validation_dir)
        enrich_with_fourier_metrics(validation_dir, rows)
        write_summary(validation_dir, rows)
        plot_contours(validation_dir)
        return 0

    executable = args.executable.resolve()
    if not executable.exists():
        raise SystemExit(f"Executable not found: {executable}")

    rows = load_existing_metadata(validation_dir)
    existing_keys = {(row["case_id"], row["scheme"]) for row in rows}

    for case in selected_cases(args.case):
        for scheme in selected_schemes(args.scheme):
            if args.resume and (case.case_id, scheme) in existing_keys:
                continue
            print(f"running {case.case_id} {scheme}", flush=True)
            try:
                row = run_case(
                    executable,
                    config_paths[(case.case_id, scheme)],
                    validation_dir,
                    case,
                    scheme,
                    args.timeout_seconds,
                    args.resume,
                )
            except subprocess.TimeoutExpired as exc:
                row = {
                    "case_id": case.case_id,
                    "scheme": scheme,
                    "knr": case.knr,
                    "knn": case.knn,
                    "ntheta": case.ntheta,
                    "nphi": case.nphi,
                    "tau_threshold": case.tau_threshold,
                    "expected_steps_paper": expected_steps(case, scheme),
                    "returncode": "timeout",
                    "wall_time_s": args.timeout_seconds,
                    "steps": None,
                    "converged": False,
                    "final_residual": None,
                    "notes": f"Timed out after {args.timeout_seconds}s. {case.notes}",
                }
                timeout_dir = output_prefix(validation_dir, case.case_id, scheme).parent
                timeout_dir.mkdir(parents=True, exist_ok=True)
                (timeout_dir / "timeout.log").write_text(str(exc))
                (timeout_dir / "run_metadata.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows = [old for old in rows if not (old.get("case_id") == case.case_id and old.get("scheme") == scheme)]
            rows.append(row)
            enrich_with_fourier_metrics(validation_dir, rows)
            write_summary(validation_dir, rows)
            plot_contours(validation_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
