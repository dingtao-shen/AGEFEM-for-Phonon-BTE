#!/usr/bin/env python3
"""Convert the legacy Fortran control.in namelist file to the C++ YAML format."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any


BOUNDARY_TYPES = {
    1: "thermalizing",
    2: "non_thermalizing",
    3: "periodic",
    4: "symmetry",
}


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "!" and not in_single and not in_double:
            return line[:index]
    return line


def _split_values(text: str) -> list[str]:
    values: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    for char in text:
        if char == "'" and not in_double:
            in_single = not in_single
            current.append(char)
        elif char == '"' and not in_single:
            in_double = not in_double
            current.append(char)
        elif char == "," and not in_single and not in_double:
            value = "".join(current).strip()
            if value:
                values.append(value)
            current = []
        else:
            current.append(char)
    value = "".join(current).strip()
    if value:
        values.append(value)
    return values


def _convert_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]

    normalized = re.sub(r"([0-9.])([dD])([+-]?[0-9]+)", r"\1e\3", value)
    try:
        if not any(marker in normalized.lower() for marker in (".", "e")):
            return int(normalized)
        return float(normalized)
    except ValueError:
        return value


def parse_namelists(path: Path) -> dict[str, dict[str, list[Any]]]:
    sections: dict[str, dict[str, list[Any]]] = {}
    current_section: str | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = _strip_comment(raw).strip()
            if not line:
                continue
            if line.startswith("&"):
                current_section = line[1:].strip().upper()
                sections[current_section] = {}
                continue
            if line == "/":
                current_section = None
                continue
            if current_section is None:
                continue
            if "=" not in line:
                raise ValueError(f"Expected key/value line in section {current_section}: {raw.rstrip()}")

            key, _, value_text = line.partition("=")
            key = key.strip().upper()
            values = [_convert_scalar(value) for value in _split_values(value_text)]
            sections[current_section][key] = values

    return sections


def _single(sections: dict[str, dict[str, list[Any]]], section: str, key: str) -> Any:
    try:
        values = sections[section][key]
    except KeyError as exc:
        raise ValueError(f"Missing namelist value {section}/{key}") from exc
    if len(values) != 1:
        raise ValueError(f"Expected one value for {section}/{key}, got {values}")
    return values[0]


def _array(sections: dict[str, dict[str, list[Any]]], section: str, key: str) -> list[Any]:
    try:
        return sections[section][key]
    except KeyError as exc:
        raise ValueError(f"Missing namelist value {section}/{key}") from exc


def to_config(sections: dict[str, dict[str, list[Any]]], control_path: Path) -> dict[str, Any]:
    nbc = int(_single(sections, "N_BC", "NBC"))
    names = _array(sections, "BC", "BC_NAME")
    physical_ids = _array(sections, "BC", "BC_PHYID")
    boundary_types = _array(sections, "BC", "BC_TYP")
    temperatures = _array(sections, "BC", "BC_TEMP")
    x_offsets = _array(sections, "BC", "BC_XOFF")
    y_offsets = _array(sections, "BC", "BC_YOFF")

    arrays = [names, physical_ids, boundary_types, temperatures, x_offsets, y_offsets]
    if any(len(values) != nbc for values in arrays):
        raise ValueError("Boundary-condition arrays do not match NBC.")

    mesh_value = Path(str(_single(sections, "FILENAME", "FNAME_MSH")))
    if not mesh_value.is_absolute():
        mesh_value = (control_path.parent / mesh_value).resolve()

    boundary_conditions: list[dict[str, Any]] = []
    for index in range(nbc):
        type_id = int(boundary_types[index])
        if type_id not in BOUNDARY_TYPES:
            raise ValueError(f"Unsupported Fortran boundary type id: {type_id}")
        boundary_conditions.append(
            {
                "name": str(names[index]),
                "physical_id": int(physical_ids[index]),
                "type": BOUNDARY_TYPES[type_id],
                "temperature": float(temperatures[index]),
                "x_offset": float(x_offsets[index]),
                "y_offset": float(y_offsets[index]),
            }
        )

    return {
        "iteration": {
            "tolerance": float(_single(sections, "ITERATION", "TOL")),
            "max_steps": int(_single(sections, "ITERATION", "TMAX")),
        },
        "gsis": {
            "enabled": int(_single(sections, "GSIS", "ACCFLAG")) != 0,
            "trace_relative_tolerance": 1.0e-10,
            "trace_absolute_tolerance": 1.0e-14,
            "trace_max_iterations": 500,
            "trace_print_level": -1,
            "trace_preconditioner": "none",
        },
        "velocity_mesh": {
            "polar_angles": int(_single(sections, "VELMSH", "NPOLE")),
            "azimuthal_angles": int(_single(sections, "VELMSH", "NAZIM")),
        },
        "dg": {"order": int(_single(sections, "DG", "DEG"))},
        "flow": {
            "specific_heat": float(_single(sections, "FLOW", "CV")),
            "group_velocity": float(_single(sections, "FLOW", "VG")),
            "tau_r": float(_single(sections, "FLOW", "TAU_R")),
            "tau_n": float(_single(sections, "FLOW", "TAU_N")),
            "tau_threshold": float(_single(sections, "FLOW", "TAU_THR")),
        },
        "files": {
            "mesh": str(mesh_value),
            "output_prefix": "output",
            "output_samples": 109,
        },
        "boundary_conditions": boundary_conditions,
    }


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.16g}"
    return str(value)


def render_yaml(config: dict[str, Any]) -> str:
    lines: list[str] = []
    for section_name in (
        "iteration",
        "gsis",
        "velocity_mesh",
        "dg",
        "flow",
        "files",
    ):
        lines.append(f"{section_name}:")
        for key, value in config[section_name].items():
            lines.append(f"  {key}: {_format_scalar(value)}")
        lines.append("")

    lines.append("boundary_conditions:")
    for item in config["boundary_conditions"]:
        lines.append(f"  - name: {item['name']}")
        lines.append(f"    physical_id: {item['physical_id']}")
        lines.append(f"    type: {item['type']}")
        lines.append(f"    temperature: {_format_scalar(item['temperature'])}")
        lines.append(f"    x_offset: {_format_scalar(item['x_offset'])}")
        lines.append(f"    y_offset: {_format_scalar(item['y_offset'])}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path, help="Legacy Fortran control.in file.")
    parser.add_argument("-o", "--output", type=Path, help="YAML output path. Defaults to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sections = parse_namelists(args.control)
    config = to_config(sections, args.control)
    yaml_text = render_yaml(config)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml_text, encoding="utf-8")
    else:
        print(yaml_text, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"convert_control_in.py: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
