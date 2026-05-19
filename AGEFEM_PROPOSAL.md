# AGEFEM Integration Proposal

Stage-1 development plan for adding Accurate-Geometry-Embodied Finite Element
Method (AGEFEM) support to the MFEM-based Callaway phonon-BTE DG solver.

Reference: D. Shen & W. Su, "Accurate-Geometry-Embodied Finite Element Method
for Phonon Boltzmann Transport Equation", Computer Physics Communications 313
(2025) 109623.

## 1. Objective and scope

The current `callaway` solver fully and verifiably solves the linearized
Callaway phonon BTE on **straight-sided** triangular DG meshes. AGEFEM adds
exact-geometry treatment of elements adjacent to curved or irregular
boundaries ("AGE elements"), eliminating the geometric-approximation error
that otherwise caps the accuracy of a high-order scheme and forces heavy
near-boundary refinement.

Scope of this stage:

- **2D only.** The current solver is 2D; 3D is out of scope for now.
- **Additive.** The validated straight-sided code path is preserved exactly.
  AGE support is a strict superset; an all-straight mesh executes no AGE code.
- **Fresh implementation.** The mathematics is taken from the paper. The
  original prototype (`reference/AGEFEM-for-Phonon-BTE-Original/`) is set aside
  as a contamination source and is **not** referenced during implementation.
- **Frontend/backend separation.** The defining goal: a concise, well
  encapsulated pre-processing pipeline that accepts curved-geometry input in
  several forms and emits **one unified representation**, so the solver
  maintains a **single interface** and never sees how geometry was specified.

## 2. What AGEFEM requires (anchor)

Three ingredients distinguish an AGE element from a standard straight-sided
triangle:

1. **Geometry.** An AGE element has two straight interior edges sharing the
   interior vertex `x0`, and a third edge that is the *exact* boundary — either
   a portion `C([λ1,λ2])` of a parametric curve, or a polyline `P_n`
   reconstructed from sampling nodes.
2. **Basis.** The nodal Lagrangian basis is defined **directly in physical
   coordinates** as a monomial expansion `ψ_l(x,y) = Σ c_{l,r,s} x^{r-s} y^s`
   on the element itself — there is no shared reference triangle. Interpolation
   nodes are placed on the curved element by the map `Ψ_n`; coefficients are
   solved from `ψ_l(x_{n,i}) = δ_{li}`.
3. **Integration.** Volume and edge integrals are evaluated by curved-geometry
   quadrature: for a parametric edge, the transformation
   `Υ(λ,θ) = (1-θ)C(λ) + θ·x0` over `[λ1,λ2]×[0,1]` with a tensor product of 1D
   Gauss–Legendre rules; for a sampling-node edge, sub-triangulation (one
   straight sub-triangle per polyline segment) with symmetric triangle
   quadrature.

AGEFEM is purely a **spatial** discretization technique — orthogonal to the
angular/velocity discretization and to the CIS/GSIS iteration.

## 3. Guiding principles

1. **Frontend / IR / backend split.** A geometry-agnostic pipeline produces an
   intermediate representation; solver kernels read only that.
2. **One geometry abstraction.** Every curved input — circle formula, NURBS,
   reconstructed polyline — is normalized to a single `BoundaryCurve` interface
   before any numerics touch it.
3. **Additive, not invasive.** The straight-sided path stays as validated.
4. **Encapsulation.** Input-format details never leak past the pipeline.
5. **Fresh, idiomatic implementation.** `callaway` namespace conventions, RAII,
   YAML configuration, one unit-test target per module.

## 4. Architecture overview

```
   Inputs                         Pre-processing pipeline                 Backend (≈ unchanged)
 ┌─────────────────┐      ┌──────────────────────────────────────┐   ┌──────────────────────┐
 │ straight .msh   │      │ A ingest → B bind geometry →         │   │ KineticSweepSolver   │
 │ + geometry      │ ───► │ C identify/enrich AGE elems →        │──►│ MomentCalculator     │
 │   sidecar       │      │ D validate  ⇒  AgeMesh               │   │ SyntheticAccelSolver │
 │ (formula/NURBS/ │      │ E AGE basis → F IntegrationCache      │   │ OutputManager        │
 │  sampling-node) │      └──────────────────────────────────────┘   └──────────────────────┘
 └─────────────────┘                      │                                   ▲
                                          └────────  unified IR  ──────────────┘
```

The contract between pipeline and solver rests on two structures: `AgeMesh`
(topology + geometry binding) and `IntegrationCache` (numerics). Solver kernels
touch `IntegrationCache` for all numerical tensors and `AgeMesh` for
connectivity; nothing else.

## 5. Unified data structures

### 5.1 `BoundaryCurve` — the uniform geometry interface

The single abstraction through which all curved geometry is accessed,
regardless of input source.

| Method | Returns | Use |
|---|---|---|
| `Point(λ)` | `(x,y)` | `C(λ)` — node placement, quadrature points |
| `Tangent(λ)` | `C'(λ)` | edge Jacobian `|C'|`, `Υ` Jacobian |
| `Normal(λ)` | outward unit normal | curved-face upwind flux, BC normals |
| `ParameterOf(x,y)` | `λ` | bind mesh vertices to curve sub-intervals |
| `Bounds()` | `[λa, λb]`, closed flag | orientation, wrap handling |
| `kind()` | enum | select quadrature strategy (`Υ` vs sub-triangulation) |

Implementations:

- `CircularArc` — formula-based: center, radius, orientation. Closed-form
  `ParameterOf`.
- `NurbsCurve` — control points, weights, knot vector; de Boor evaluation.
  `ParameterOf` by point projection (Newton iteration on `(C(λ)-x)·C'(λ)=0`).
- `PolylineCurve` — the paper's `C_P(λ)`: a sequence of sampling nodes with a
  piecewise-linear arc-length parametrization. `ParameterOf` by segment walk.

The interface is extensible (`EllipticArc`, generic analytic curve) without
touching the pipeline or solver. The paper's two AGE flavors map directly:
parametric → arc/NURBS/analytic; sampling-node → polyline.

### 5.2 `AgeMesh` — the unified mesh + geometry IR

The single, input-agnostic structure produced by the pipeline. It is the
mesh-side interface the solver sees; it composes the existing `MeshAdapter`
(topology, connectivity, straight-face data — unchanged) and adds:

- `ElementKind[]` — `Straight` or `Age`, per element.
- `AgeElementGeometry` (AGE elements only): curved local-face id, a
  `const BoundaryCurve*`, the **oriented** parameter sub-interval `[λ1,λ2]`,
  the interior vertex `x0`, and the curve endpoints `x1 = C(λ1)`, `x2 = C(λ2)`.
- `CurvedFace[]` — for curved boundary faces: curve handle, parameter
  interval, boundary attribute.

For an all-straight mesh, `AgeMesh` is a thin pass-through over `MeshAdapter`,
so the straight-sided path is unaffected.

### 5.3 `IntegrationCache` — the single numerical interface (extended in place)

Same accessor philosophy as today (`Mass(e,r,c)`, `GradX`, `GradY`,
`BasisIntegral`, `ElementFaceMass`, `NeighborFaceMass`, …); values are computed
AGE-aware under the hood:

- **Straight elements** — existing analytic reference-map path, unchanged.
- **AGE elements** — volume tensors (`Mass`, `GradX/Y`, `BasisIntegral`) built
  by curved-geometry quadrature against the per-element physical-coordinate
  basis; straight faces of AGE elements integrated along their straight
  segments as usual.
- **Curved faces** — the direction-dependent upwind contribution is exposed
  through a uniform accessor; the underlying storage strategy (precomputed vs
  on-the-fly, see §9.2) is an internal detail invisible to solver kernels.

The accessor surface is the "one interface" the solver maintains. Adding AGE
support changes how values are *computed*, not how kernels *consume* them.

## 6. Geometry input: the AGE geometry sidecar

**Proposed format (resolves open question 1).** Rather than inventing a custom
mesh-file format or depending on Gmsh 4.x parametric features, geometry is
declared in a **YAML sidecar** that accompanies a standard straight-sided Gmsh
`.msh`. This is maximally compatible (any mesh generator works), consistent
with the project's existing YAML configuration, and — crucially — makes the
three input forms a **single ingestion mechanism**: they are simply different
`BoundaryCurve` types declared in the same file.

The sidecar is referenced from the control YAML via `files.geometry`, or auto
discovered as `<mesh-stem>.age.yaml` next to the mesh. If absent, the run is
pure straight-sided. It is parsed by a dedicated loader (`age_geometry`), not
by the control-YAML parser: the project's control parser is intentionally a
flat line-based reader, so the sidecar grammar is kept within the same
capability — a list of curve entries with scalar fields only. Analytic curves
(e.g. `circular_arc`) carry their few parameters inline; `nurbs` and
`polyline` curves carry their bulk numeric data (control points, weights,
knots, or sampling nodes) in a referenced `data_file`, since nested numeric
arrays are deliberately out of grammar scope.

```yaml
# <mesh-stem>.age.yaml — binds boundary physical-ids to exact geometry
version: 1
curves:
  - boundary_id: 9            # Gmsh physical-id of the boundary
    type: circular_arc
    center: [-0.5, -0.5]      # inline 2-vector
    radius: 0.25
    orientation: ccw          # ccw | cw — fixes the outward-normal sense

  - boundary_id: 10
    type: nurbs
    degree: 2
    closed: true
    data_file: hole10_nurbs.txt   # control points, weights, knots

  - boundary_id: 20
    type: polyline
    closed: false
    orientation: ccw
    data_file: rough_top_nodes.txt   # sampling nodes
```

Each entry binds one boundary physical-id to one curve; the loader
instantiates the matching `BoundaryCurve` (through the `MakeBoundaryCurve`
factory) and the pipeline derives per-element parameter sub-intervals in
Stage C. A `data_file` is a whitespace/newline-delimited numeric file with
`#` comments:

- **nurbs** — `n`, then `n` rows of `x y weight`, then `m`, then `m` knot
  values.
- **polyline** — `n`, then `n` rows of `x y`.

The "NURBS-in-file" requirement is satisfied by `type: nurbs` plus its
`data_file`; the formula and sampling-node paths use the same sidecar and the
same flow. `config/geometry.example.age.yaml` is the worked schema reference.

**Input assumption (resolves open question 6).** The straight-sided mesh is
assumed to be generated such that the endpoints of every curved-boundary edge
already lie on the true curve. Stage C derives their parameters with
`ParameterOf` and projects them exactly onto the curve to remove round-off;
Stage D rejects, with a clear diagnostic, any endpoint farther than a tolerance
from its bound curve.

## 7. The pre-processing pipeline

| Stage | Responsibility |
|---|---|
| **A. Ingest** | Read the topological mesh through `MeshAdapter` (standard Gmsh via MFEM, as today). |
| **B. Bind geometry** | Parse the geometry sidecar; instantiate one `BoundaryCurve` per declared boundary id; attach to the corresponding boundary faces. |
| **C. Identify & enrich** | Mark elements adjacent to a bound curve as AGE; locate the curved local face; compute the oriented `[λ1,λ2]` from face-endpoint vertices via `ParameterOf`; project endpoints onto the curve; handle closed-curve parameter wrap with a consistent convention. |
| **D. Validate** | Check: exactly one curved face per AGE element, the two interior edges straight, monotone parametrization, positive curved-element area, endpoints within tolerance of the curve. Emit `AgeMesh`. |
| **E. AGE basis** | Straight elements: the existing shared `NodalBasis`. AGE elements: a per-element physical-coordinate monomial basis, nodes placed by `Ψ_n`, coefficients from a well-conditioned interpolation solve (§9.1). |
| **F. Integration cache** | Straight elements: existing analytic path. AGE elements: volume and curved-face tensors by `Υ`-transformation / sub-triangulation quadrature. |

Stages A–D are orchestrated by an `AgePreprocessor`; E and F are the existing
basis/cache construction made AGE-aware. After F the solver runs as today.

## 8. Solver-side changes (minimal, enumerated)

| Module | Change |
|---|---|
| `IntegrationCache` | Construction branches on `ElementKind`; curved-face contributions added behind a uniform accessor. |
| `KineticSweepSolver` | Curved-face inflow/outflow uses the curved-face accessor (pointwise `s·n` upwind split) instead of `speed × ElementFaceMass`. |
| Boundary conditions | Curved-face application: thermalizing first, then diffuse and specular reflection. The `boundary.hpp` data model is already adequate. |
| `OutputManager` | AGE-aware point-in-element location (test against the curved edge) and field evaluation (per-element physical basis). |
| `MomentCalculator` | None — consumes `IntegrationCache::BasisIntegral` abstractly. |
| `SweepOrdering` | **None** — curved faces are boundary faces, absent from the inter-element dependency graph; the two straight interior faces participate normally. |
| `SyntheticAccelerationSolver` | Follows once `IntegrationCache` is AGE-aware; curved-face trace handling is deferred to Phase 5. |

## 9. Technical improvements over the prototype's approach

### 9.1 Basis conditioning

The per-element interpolation system is built on a **shifted-and-scaled local
monomial basis** — monomials in `((x - xc)/h, (y - yc)/h)` with `xc` the
element centroid and `h` a characteristic element size — instead of raw
physical-coordinate monomials. This spans the same polynomial space but yields
a far better-conditioned Vandermonde at orders 3–4, directly addressing the
mass-matrix conditioning the paper itself flags. Evaluation applies the same
shift/scale.

### 9.2 Curved-face tensor strategy (resolves open question 4)

Two modes, selected by configuration, encapsulated inside `IntegrationCache`
so solver kernels are mode-agnostic:

- **`precomputed` (default).** Direction-dependent curved-face tensors (the
  outflow contribution to the local matrix and the inflow coupling) are built
  once at construction. Memory is bounded — only AGE elements, of which there
  are far fewer than interior elements.
- **`on_the_fly` (optional).** Only the geometry-side curved-face quadrature
  record (points, weights, basis values, pointwise normals) is stored; the
  `s·n` split is evaluated per direction at solve time. Trades memory for
  compute; useful for very fine boundary discretizations or many directions.

### 9.3 Config-driven geometry

All curved geometry is declared in the sidecar (§6). Nothing geometric is
hard-coded — replacing the prototype's baked-in circular-hole set.

### 9.4 General boundary-condition dispatch

Curved faces carry boundary conditions through the same `BoundaryCondition`
model and dispatch as straight faces, with no fixed circular-hole assumption.

## 10. Proposed module layout

New:

```
include/callaway/geometry/boundary_curve.hpp
include/callaway/geometry/circular_arc.hpp
include/callaway/geometry/nurbs_curve.hpp
include/callaway/geometry/polyline_curve.hpp
include/callaway/age_geometry.hpp
include/callaway/age_mesh.hpp
include/callaway/age_basis.hpp
include/callaway/age_preprocessor.hpp
src/geometry/boundary_curve.cpp   (+ circular_arc.cpp, nurbs_curve.cpp, polyline_curve.cpp)
src/age_geometry.cpp
src/age_mesh.cpp
src/age_basis.cpp
src/age_preprocessor.cpp
```

Extended in place: `integration_cache.{hpp,cpp}`, `config.{hpp,cpp}`,
`kinetic_sweep_solver.cpp`, `output_manager.cpp`, `CMakeLists.txt`.

## 11. Configuration schema additions

Control YAML:

```yaml
files:
  geometry: path/to/mesh.age.yaml   # optional; absent ⇒ pure straight-sided run

age:
  curved_face_tensors: precomputed  # precomputed | on_the_fly
  edge_quadrature_points: 15        # 1D GL points along a curved edge
  area_quadrature_points: 15        # 1D GL points per Υ direction
```

The geometry sidecar schema is defined in §6.

## 12. Phasing and validation targets

| Phase | Scope | Validation target |
|---|---|---|
| **0** | Freeze the `BoundaryCurve` / `AgeMesh` / `IntegrationCache` interfaces and the config + sidecar schema. | design review |
| **1** | Geometry layer: `BoundaryCurve` + `CircularArc`, `PolylineCurve`, `NurbsCurve`. | unit tests vs analytic point/tangent/normal/inverse-parameter/arc-length |
| **2** | Pipeline stages A–D ⇒ `AgeMesh`. Formula-based (`circular_arc`) is the first-class input path. | small mesh with one circular boundary |
| **3** | AGE basis (§9.1) + AGE-aware `IntegrationCache` (both tensor modes). | partition of unity on AGE elements; exact integration of polynomials over a circular-segment element with analytic values; a straight-line "curve" reproduces the straight-sided path bit-for-bit |
| **4** | Sweep curved-face flux + curved-face thermalizing BC + AGE-aware output. | **homogeneous concentric ring (paper §5.1)** — analytic temperature in ballistic and diffusion limits |
| **5** | GSIS on AGE meshes + diffuse/specular curved-face BCs + polyline path. | **nano-porous media (§5.2)**, **square with rough boundary (§5.3)** |
| **6** | Performance, threading, AGEFEM paper validation script. | CPU-time / accuracy tables vs the paper |

Input-path priority across phases: formula-based curves first (covers ring and
porous), polyline second (rough boundary), NURBS-from-file third.

## 13. Testing strategy

- **Geometry** — each `BoundaryCurve` against analytic point/tangent/normal,
  `ParameterOf` round-trips, arc length.
- **Pipeline** — AGE-element identification, parameter-interval orientation and
  wrap, validation diagnostics on malformed input.
- **Basis** — partition of unity and Kronecker-delta nodal property on AGE
  elements; conditioning at orders 1–4.
- **Integration** — exact integration of monomials over an AGE element with
  analytically known integrals; **degeneracy check**: a `BoundaryCurve` that is
  actually a straight segment must reproduce the straight-sided tensors and
  final fields bit-for-bit.
- **Regression** — the existing straight-sided suite must remain green
  throughout (additive guarantee); paper cases per the phase table.

## 14. Resolved decisions

1. **NURBS-in-file format** — a YAML geometry sidecar paired with a standard
   Gmsh mesh (§6); unifies formula, NURBS, and sampling-node input through one
   file and one flow.
2. **First-class input path** — formula-based curves first, polyline second,
   NURBS-from-file third.
3. **First validation target** — the homogeneous concentric ring (paper §5.1).
4. **Curved-face tensor strategy** — optional; default `precomputed`,
   `on_the_fly` available (§9.2).
5. **Dimensionality** — 2D only for this stage.
6. **Mesh provenance** — the straight-sided mesh is assumed to have
   curved-boundary edge endpoints on the true curve; the pipeline validates and
   projects (§6).

## 15. Open items / future work

- 3D AGE elements (the paper's stated outlook).
- GSIS trace handling on curved faces (Phase 5 brings it into scope).
- Non-gray / frequency-dependent BTE — AGEFEM is independent of the transport
  model, so this is purely a solver-side extension.
- Performance tuning of the `precomputed` curved-face tensors for large
  angular resolutions.
