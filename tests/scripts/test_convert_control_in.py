#!/usr/bin/env python3
"""Unit test for the legacy control.in converter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_converter(script_path: Path):
    spec = importlib.util.spec_from_file_location("convert_control_in", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: test_convert_control_in.py SCRIPT CONTROL", file=sys.stderr)
        return 2

    converter = load_converter(Path(sys.argv[1]))
    control_path = Path(sys.argv[2]).resolve()
    sections = converter.parse_namelists(control_path)
    config = converter.to_config(sections, control_path)

    assert config["iteration"]["tolerance"] == 1.0e-8
    assert config["iteration"]["max_steps"] == 8000000
    assert config["gsis"]["enabled"] is False
    assert config["velocity_mesh"]["polar_angles"] == 20
    assert config["velocity_mesh"]["azimuthal_angles"] == 40
    assert config["dg"]["order"] == 3
    assert config["flow"]["specific_heat"] == 1.0
    assert config["flow"]["tau_r"] == 1.0e-3
    assert config["flow"]["tau_n"] == 1.0e5
    assert config["files"]["mesh"] == str((control_path.parent / "A1_Nx11_Ny11.msh").resolve())
    assert len(config["boundary_conditions"]) == 4
    assert config["boundary_conditions"][1]["name"] == "NWall"
    assert config["boundary_conditions"][1]["physical_id"] == 12
    assert config["boundary_conditions"][1]["type"] == "thermalizing"
    assert config["boundary_conditions"][1]["temperature"] == 1.0

    yaml_text = converter.render_yaml(config)
    assert "trace_preconditioner: none" in yaml_text
    assert "boundary_conditions:" in yaml_text
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
