# AGEFEM examples

End-to-end demonstrations that exercise the solver on the same meshes
the original AGEFEM prototype used and contrast a fine straight-sided
discretisation (SMBDG) against a sparse curved-geometry discretisation
(AGEDG).

Reference:
> Shen & Su, *Accurate-Geometry-Embodied Finite Element Method for
> Phonon Boltzmann Transport Equation*, Comput. Phys. Commun. 313 (2025)
> 109623.

## Prerequisites

- Build the solver:
  ```
  cmake -S . -B build-openmp -DCALLAWAY_USE_OPENMP=ON
  cmake --build build-openmp -j
  ```
  The examples expect the executable at `build-openmp/callaway_mfem`.
- Python 3 with `numpy` and `matplotlib`.

## `porous/` — paper §5.2 (nano-porous medium)

Reproduces the test set up by the original prototype as case
`Porous_4_1_1`, with **two different mesh resolutions** chosen to
highlight the AGEFEM headline claim:

- **SMBDG** runs on `porous_6.msh` — the prototype's mesh index 6,
  ~1.5k triangles. Boundaries are interpreted as the polygonal chord
  approximation of the pores (no AGE sidecar).
- **AGEDG** runs on `porous_2.msh` — the prototype's mesh index 2,
  ~0.7k triangles. The same five circular pore boundaries are described
  analytically via the geometry sidecar, so boundary-adjacent elements
  use exact-arc quadrature even at this coarse resolution.

Both runs use the prototype's **2D circular angular quadrature**
(`polar_angles=1, azimuthal_angles=40`) — `AngularQuadrature` treats
`polar_angles == 1` as 2D mode, collapsing to the equatorial circle
with `cx = vg·cos(phi)`, `cy = vg·sin(phi)`, weight summing to 2π and
the equilibrium normalisation `e_eq = Cv·T/(2π)`. The DG order is 3
(prototype uses 4; 3 gives the same contour shape at lower cost). With
the matching angular setup the |T| amplitude lands near the
prototype's ±0.6 rather than the under-amplitude ±0.35 we got with the
default 3D spherical quadrature.

The two runs share Knudsen regime, DG order, and angular quadrature, so
the comparison isolates the effect of "dense polygonal mesh" vs
"sparse mesh + exact geometry."

Both meshes are copied verbatim from the prototype's collection in
`reference/AGEFEM-for-Phonon-BTE-Original/Mesh/Porous/` and retain
their `$Periodic` section — the solver's `MeshAdapter` now parses
that section directly and strips it on-the-fly before handing the
mesh to MFEM, so the original prototype meshes are usable without any
preprocessing.

- **Geometry**: five circular boundaries declared in
  `output/porous.age.yaml` and bound to physical tags 1, 5, 9, 10, 11.
  Tags 1 and 5 are the half-circles biting into the bottom and top of
  the `[-1, 1]²` unit cell; tags 9, 10, 11 are three interior pores of
  radii 0.25, 0.1, 0.1 (matching the prototype's `HoleSet`).
- **Knudsen regime**: τ_R = τ_N = 0.01 — Knudsen pair k = 1 in the
  prototype's sweep (near-diffusion / transitional regime).
- **Boundary conditions** (now match the prototype's BDINF setup on
  every side):

  | Tag | BC                          | Notes                                |
  |-----|-----------------------------|--------------------------------------|
  | 1, 5, 9, 10, 11 (pores)     | diffuse (non_thermalizing) | curved BC via CurvedFaceInflow |
  | 3 (left wall)               | specular (symmetry)        | vertical-wall cx-flip          |
  | 7 (right wall)              | specular (symmetry)        | vertical-wall cx-flip          |
  | 2 (bottom-left segment)     | periodic, ΔT_bc = −1      | paired with tag 4 (top-left)   |
  | 8 (bottom-right segment)    | periodic, ΔT_bc = −1      | paired with tag 6 (top-right)  |
  | 4 (top-left segment)        | periodic, ΔT_bc = +1      | paired with tag 2              |
  | 6 (top-right segment)       | periodic, ΔT_bc = +1      | paired with tag 8              |

  The solver interprets the BoundaryCondition's `temperature` field as
  ΔT_bc = T_self − T_partner. With the pairing above, the energy density
  shift at the periodic interface is `Cv·ΔT_bc/(4π) · inflow_speed ·
  ∫_face φ_row dl` per inflow angle, imposing a unit temperature
  difference across the periodic axis.

Run:
```
examples/porous/run_and_plot.py
```
Output: `examples/porous/porous_contours.png` — a 2-panel comparison:

1. **SMBDG**, 8885 elements on the dense straight-sided mesh
   (`porous_10.msh`).
2. **AGEDG**, 734 elements on the sparse AGE mesh (`porous_2.msh`).

Both panels share the same colour scale; the pore boundaries are
overlaid in black.

### What the example exercises

- **End-to-end pipeline on production geometry**: AGE preprocessor →
  AGE basis → AGE-aware integration cache → curved-face flux + diffuse
  curved-face BC refresh + specular straight-wall BC refresh +
  **periodic straight-wall BC refresh** → CIS sweep → cell-averaged
  output.
- **Gmsh `$Periodic` parsing in `MeshAdapter`**. The prototype's mesh
  family ships with `$Periodic` blocks; MFEM's Gmsh reader does not
  consume that section, so the adapter parses it directly, strips the
  block out for MFEM, and builds a face-pair table keyed by
  vertex correspondence + affine translation.
- **Periodic BC with ΔT** on the four split top/bottom segments
  (tags 2, 4, 6, 8). Periodic inflow is computed each iteration as
  `inflow_speed · [ NeighborFaceMass × dist_partner + Cv·ΔT_bc/(4π) ·
  ElementFaceIntegral ]`, where the partner-side basis is evaluated at
  the periodically-shifted physical point during integration-cache
  construction.

## Output files (per run)

- `<prefix>_cells.csv` — header `n_vertices n_elements`, then vertex
  coordinates, then `v0 v1 v2 T_avg qx_avg qy_avg` per element. Loaded
  by the plot script via line-wise parsing.
- `<prefix>_residual.csv` — CIS residual history.
- `<prefix>_paraview` — MFEM `ParaViewDataCollection` (cell averages,
  AGE-aware).
- `<prefix>_field.dat` / `_reference.dat` — Tecplot grid samples; for
  AGE meshes the gridded sampling is not yet AGE-aware (a Phase-4-
  deferred item), so out-of-mesh sample points (inside the pores) are
  emitted as zero rather than the curved-element-evaluated field. Use
  the ParaView output or the cells CSV for AGE visualization.

## How the SMBDG ↔ AGEDG switch works

The two runs differ in (a) the mesh file passed in and (b) whether the
control YAML supplies `files.geometry`. No sidecar →
`AgePreprocessor::BuildStraight` produces an AgeMesh with every
element `Straight`. With sidecar → `AgePreprocessor::Build` identifies
AGE elements, binds them to the declared curves, and the integration
cache computes curved-geometry volume tensors via the Υ-transformation,
plus per-curved-face quadrature records used by the upwind flux and the
diffuse-BC wall-inflow precompute. Periodic and specular BC handling
is independent of the AGE/straight choice and applies to whichever
straight boundary faces happen to carry the matching boundary type.
