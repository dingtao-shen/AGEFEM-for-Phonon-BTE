# MFEM C++ Refactor Plan for the 2D-2V Linearized Callaway DG Solver

## 1. Scope

This plan covers a C++ refactor of the current Fortran implementation of the steady, gray, linearized Callaway phonon Boltzmann equation with dual relaxation times. The target implementation should reproduce the current Fortran CIS and GSIS results first, then use MFEM abstractions where they improve mesh handling, finite element data management, sparse linear algebra, visualization, and future parallelization.

The refactor should preserve the algorithmic model:

- Discrete angular/velocity quadrature on the unit sphere.
- Discontinuous Galerkin discretization on 2D triangular spatial meshes.
- Element-local upwind kinetic sweeps for the conventional iterative scheme.
- Macroscopic synthetic equations and trace/global solve for GSIS acceleration.
- Thermalizing wall boundary conditions as the verified baseline.
- Non-thermalizing and periodic boundary support after the thermalizing-wall path is validated.

## 2. Sources Reviewed

- Local Fortran source files in the project root: `Callaway_2D_2V_DG.f90`, `Global_Parameter.f90`, `Boundary_Conditions.f90`, `Spatial_Mesh.f90`, `Velocity_Mesh.f90`, `Basis_Function.f90`, `Integration.f90`, `Velocity_Distribution.f90`, `Matrix.f90`, `Solvers.f90`, `Synthetic_Acceleration.f90`, `Synthetic_Acceleration1.f90`, `Out_Put_Result.f90`, `USD_Math.f90`, `Makefile`, `control.in`, and `A1_Nx11_Ny11.msh`.
- Local reference PDF: `reference/A fast converging scheme for the phonon Boltzmann equation with dual relaxation times.pdf`.
- JCP article page and DOI: https://doi.org/10.1016/j.jcp.2022.111436.
- arXiv preprint for accessible equation and algorithm text: https://arxiv.org/abs/2107.06688 and https://arxiv.org/pdf/2107.06688.
- MFEM official documentation:
  - Features: https://mfem.org/features/
  - Mesh formats: https://mfem.org/mesh-formats/
  - Periodic boundaries: https://mfem.org/howto/periodic-boundaries/
  - Bilinear form integrators: https://mfem.org/bilininteg/
  - MFEM 4.9 Doxygen index: https://docs.mfem.org/4.9/
  - DG example 9: https://docs.mfem.org/4.9/ex9_8cpp_source.html
  - `L2_FECollection`: https://docs.mfem.org/4.8/classmfem_1_1L2__FECollection.html
  - `DGTraceIntegrator`: https://docs.mfem.org/4.7/classmfem_1_1DGTraceIntegrator.html

Local MFEM installations were detected at:

- Serial MFEM 4.8.0: `/usr/local/lib/cmake/mfem/MFEMConfig.cmake`
- MPI MFEM 4.8.0: `/usr/local/mfem-mpi/lib/cmake/mfem/MFEMConfig.cmake`

The serial MFEM build has no MPI, OpenMP, LAPACK, SuiteSparse, SuperLU, or PARDISO enabled. The MPI build has MPI and METIS enabled, but also no direct sparse direct solver package enabled.

## 3. Feasibility Assessment

The refactor is feasible, but it should be treated as a custom kinetic-DG solver built on top of MFEM, not as a direct rewrite using only standard MFEM integrators.

MFEM is suitable for:

- Reading the current Gmsh 2.2 mesh. MFEM supports Gmsh 2.2 ASCII/binary meshes, and Gmsh physical tags become MFEM attributes.
- Managing triangular meshes, boundary attributes, element transformations, face transformations, and DG/L2 finite element spaces.
- Providing `Vector`, `DenseMatrix`, `SparseMatrix`, iterative solvers, optional external solver integrations, and GLVis/ParaView-friendly output.
- Providing a future path to `ParMesh`, `ParFiniteElementSpace`, and `HypreParMatrix`.

MFEM should not replace the whole algorithm blindly because:

- The kinetic solve is a direction-wise source iteration with topological upwind sweeps, not a single global DG advection solve.
- The GSIS acceleration system is a hybrid trace/global system derived from the Callaway moment equations, not a stock diffusion, advection, or elasticity operator.
- The Fortran basis is an equispaced nodal polynomial basis on triangles and segments. Standard MFEM DG/L2 spaces may use a different basis and local dof ordering, so direct replacement can change local matrices and roundoff-level results.

Recommended strategy:

1. Use MFEM for mesh, geometry, finite element metadata, sparse matrices, solvers, and output.
2. Preserve the Fortran mathematical kernels initially as custom C++ assemblers with explicit validation against Fortran tensors and iteration results.
3. After regression tests pass, replace selected hard-coded kernels with MFEM shape/derivative evaluation where the basis and orientation differences are controlled by tests.

## 4. Current Fortran Implementation Summary

### 4.1 Main Control Flow

`Callaway_2D_2V_DG.f90` performs:

1. Read `control.in` namelists.
2. Read boundary conditions.
3. Build spatial mesh from Gmsh.
4. Build angular/velocity quadrature.
5. Build per-direction element sweep order.
6. Build nodal basis functions.
7. Precompute volume and face integration tensors.
8. Initialize the distribution function.
9. If `ACCFLAG=1`, initialize GSIS matrices and PARDISO.
10. Iterate:
    - Solve kinetic DG equations for all directions.
    - For CIS, compute macroscopic moments directly from the distribution.
    - For GSIS, build high-order moment source, solve global trace problem, solve local macroscopic problem, and correct the distribution.
    - Compute residual from temperature change.
11. Write Tecplot-style output and runtime information.

### 4.2 Governing Discrete Model

The current code solves the steady linearized gray Callaway model in energy-density form. For each angular direction `c = (CX, CY)`, the local DG equation corresponds to:

```text
c . grad(e) = (e_R_eq - e) / tau_R + (e_N_eq - e) / tau_N
tau_C = 1 / (1/tau_R + 1/tau_N)
e_R_eq = Cv * T / (4*pi)
e_N_eq = Cv * T / (4*pi) + 3 * (q . c) / (4*pi*Vg^2)
```

The moments are:

```text
T  = integral_over_solid_angle(e) / Cv
qx = integral_over_solid_angle(CX * e)
qy = integral_over_solid_angle(CY * e)
```

The velocity mesh uses Gauss-Legendre quadrature in polar angle `theta` and split Gauss-Legendre quadrature in azimuth `phi`, with weights `sin(theta) * w_theta * w_phi`.

### 4.3 Fortran Module Map

| Fortran file | Responsibility | C++ target |
| --- | --- | --- |
| `Global_Parameter.f90` | Runtime parameters and constants | `Config`, `PhysicalParams`, `IterationParams` |
| `Boundary_Conditions.f90` | Boundary attributes and values | `BoundaryDatabase` |
| `Spatial_Mesh.f90` | Manual Gmsh parser, faces, normals, periodic pairing | `MeshAdapter`, mostly backed by `mfem::Mesh` |
| `Velocity_Mesh.f90` | Angular quadrature and direction vectors | `AngularQuadrature` |
| `Basis_Function.f90` | Equispaced nodal triangle/segment basis | `NodalBasis` or MFEM FE wrapper |
| `Integration.f90` | Volume/face tensors and quadrature cache | `ElementIntegrationCache`, `FaceIntegrationCache` |
| `Matrix.f90` | Direction-wise topological sweep ordering | `SweepOrdering` |
| `Velocity_Distribution.f90` | VDF storage and macroscopic moments | `Distribution`, `MomentCalculator` |
| `Solvers.f90` | CIS local DG sweep | `KineticSweepSolver` |
| `Synthetic_Acceleration.f90` | GSIS local/global trace systems and correction | `SyntheticAccelerationSolver` |
| `Out_Put_Result.f90` | Tecplot output and analytical Fourier solution | `OutputManager`, `ReferenceSolutions` |
| `USD_Math.f90` | Quadrature and small utilities | `QuadratureRules`, `MathUtils` |

`Synthetic_Acceleration1.f90` defines another `ACCELERATION` module but is not used by the Makefile. It contains materially different scalings and boundary formulas. The authoritative GSIS implementation for this refactor is `Synthetic_Acceleration.f90`; `Synthetic_Acceleration1.f90` should be treated only as an inactive experimental/reference variant.

## 5. Important Issues Found in the Fortran Baseline

These issues should be documented in the C++ implementation and covered by regression tests. Fixes are acceptable where they correct clear bugs without changing the intended algorithm.

1. `Velocity_Distribution.f90` resets `Qx` twice and does not reset `Qy` in `Calculate_Macro_Properties`. This can accumulate stale `Qy` values in CIS runs.
2. In `Solvers.f90`, non-thermalizing wall and periodic kinetic boundary branches are commented out. The active kinetic boundary path treats every boundary as thermalizing.
3. `Calculate_FLUX_WALL()` exists for non-thermalizing walls but is not called in the main loop.
4. `Synthetic_Acceleration.f90` and `Synthetic_Acceleration1.f90` disagree on several scalings by `tau_R`, boundary handling, limiter definition, and flux projections. Only `Synthetic_Acceleration.f90` is built.
5. The GSIS sparse matrix is assembled by scanning a dense row block for every face and then compressing manually for PARDISO. This should be replaced with triplet/CSR assembly.
6. The current local kinetic matrix is refactorized for every element, direction, and iteration, although it is iteration-invariant for fixed mesh, degree, parameters, and direction. The C++ version should precompute local LU factors or cache them by direction and element.
7. OpenMP privatization of module-level allocatable work arrays is fragile. The C++ version should use explicit thread-local scratch objects.
8. Periodic face pairing in Fortran uses exact floating-point coordinate equality. MFEM periodic topology or tolerance-based pairing should be used.
9. The Gmsh reader assumes version 2.2 and a narrow element-line layout. MFEM can read the current mesh directly, but the project should still reject unsupported mesh versions with a clear error.
10. The output routine hard-codes a `109 x 109` sampling grid and square-domain Fourier reference. This should become configurable.

## 6. MFEM Design Choices

### 6.1 Mesh and Boundary Handling

Use `mfem::Mesh mesh(mesh_file, 1, 1)` as the primary mesh loader. The current mesh is Gmsh 2.2 and compatible with MFEM. Gmsh physical tags should map to:

- Element attributes for material/region tags.
- Boundary attributes for `BC_PHYID` tags.

The C++ mesh adapter should expose:

- Element count, vertex coordinates, physical attributes.
- Boundary face attributes.
- Face adjacency: plus/minus elements and local face ids.
- Outward normals from `FaceElementTransformations` or a validated custom normal computation.
- Face length and element area.

Initial validation target for `A1_Nx11_Ny11.msh`:

- 121 vertices.
- 200 triangular elements.
- 40 boundary edges.
- Boundary attributes 11, 12, 13, 14.
- Element attribute 15.

### 6.2 Finite Element Space

Use one scalar discontinuous finite element space for the spatial DG basis:

```cpp
mfem::L2_FECollection fec(order, mesh.Dimension(), mfem::BasisType::GaussLegendre);
mfem::FiniteElementSpace fes(&mesh, &fec);
```

However, exact Fortran reproduction may require one of these paths:

- Path A: keep Fortran-equivalent equispaced nodal basis and local dof ordering in custom C++ kernels while still using MFEM for mesh topology and output.
- Path B, preferred after baseline validation: use MFEM basis functions and accept a controlled basis change, comparing physical fields and paper-level observables rather than requiring dof-by-dof tensor equality.
- Path C, advanced: implement a small custom MFEM `FiniteElementCollection` or element wrapper matching the Fortran equispaced nodal basis.

Exact dof-by-dof Fortran reproduction is not a long-term requirement. The first milestone should still use enough Fortran-equivalent kernels to establish a reliable baseline, then move toward MFEM-native finite element evaluation where regression tests show acceptable physical-field agreement.

### 6.3 Distribution Layout

Use a flat, cache-conscious layout:

```text
e[angle_id][element_id][local_dof]
angle_id = jpole * nazim + jazim
```

This makes each direction sweep contiguous in memory. Provide typed accessors:

```cpp
double& Distribution::operator()(int angle, int elem, int dof);
```

Keep `T_s`, `Qx_s`, and `Qy_s` as element-dof arrays, and keep cell averages as element arrays.

### 6.4 Kinetic CIS Sweep

The kinetic update remains a source iteration with fixed upwind transport matrix:

1. For each direction, traverse elements using `SweepOrdering`.
2. For each element, form or reuse the local matrix:
   - Collision mass: `M / tau_C`.
   - Volume advection: `-cx * Dx - cy * Dy`.
   - Outflow upwind face contribution: `max(c.n, 0) * F`.
3. Build the RHS from previous macroscopic moments and inflow boundary/upwind neighbor values.
4. Solve the local dense system for the element distribution coefficients.

Optimization:

- Precompute and factor local matrices per `(angle, element)`.
- Store `mfem::DenseMatrix` LU factors plus pivot arrays, or use a small custom dense LU wrapper.
- For large meshes, offer a memory-saving mode that caches by direction only or refactors on demand.

### 6.5 Boundary Conditions

Implement boundary conditions in this order:

1. Thermalizing wall, reproducing the active Fortran path:
   ```text
   e_in = Cv * T_wall / (4*pi)
   ```
2. Define interfaces for non-thermalizing wall:
   - Restore and test `Calculate_FLUX_WALL()` behavior.
   - Ensure incoming/outgoing conventions match the DG face normal.
3. Define interfaces for periodic boundaries:
   - Prefer MFEM periodic meshes when possible.
   - For non-periodic input with periodic BC metadata, build a tolerance-based face-pair table.

Only thermalizing-wall behavior is required in the first production milestone. Non-thermalizing and periodic boundary implementations can be added later, but the boundary-condition dispatch and data model should reserve clean extension points for them.

### 6.6 GSIS Acceleration

The GSIS solver should be implemented as a dedicated class, not as a standard MFEM bilinear form.

State variables per element dof:

```text
UQ = [T, qx, qy, Lxx, Lxy, Lyx, Lyy]
```

Trace variables per face dof:

```text
U_TRACE = [T_trace, qx_trace, qy_trace]
```

Implementation plan:

1. Build local macroscopic block matrices matching `Synthetic_Acceleration.f90`.
2. Replace explicit inverse storage with LU factors and multiple-RHS solves wherever possible.
3. Build trace-coupling blocks.
4. Assemble the global trace matrix directly into `mfem::SparseMatrix`.
5. Solve the unsymmetric trace system.
6. Reconstruct local `UQ`.
7. Correct the VDF using the same limiter and correction formula as Fortran.

Solver options:

- Portable iterative path: `mfem::GMRESSolver` with optional Jacobi
  preconditioning remains available for the unsymmetric trace system.
- Validated serial benchmark path: Eigen/SparseLU can be selected with
  `trace_preconditioner: direct`, reusing the cached trace matrix factorization
  across GSIS iterations.
- Optional future high-performance path: link to MKL PARDISO or rebuild MFEM
  with SuiteSparse/SuperLU if direct-solver parity or larger sparse solves are
  needed.
- MPI path: use `ParMesh`, `ParFiniteElementSpace`, and `HypreParMatrix` only
  after serial validation.

### 6.7 Output

Provide:

- Tecplot-compatible `.dat` output matching the current file layout.
- ParaView output through MFEM `ParaViewDataCollection`.
- Optional GLVis output for quick inspection.
- Configurable sampling grid instead of hard-coded `109 x 109`.

## 7. Proposed Project Structure

Create all C++ refactor files under `mfem_cpp_refactor/`.

```text
mfem_cpp_refactor/
  REFACTOR_PLAN.md
  CMakeLists.txt
  README.md
  config/
    control.example.yaml
    control.gsis_smoke.yaml
  include/callaway/
    config.hpp
    boundary.hpp
    angular_quadrature.hpp
    mesh_adapter.hpp
    nodal_basis.hpp
    integration_cache.hpp
    distribution.hpp
    sweep_ordering.hpp
    kinetic_sweep_solver.hpp
    synthetic_acceleration_solver.hpp
    moment_calculator.hpp
    output_manager.hpp
  src/
    main.cpp
    config.cpp
    boundary.cpp
    angular_quadrature.cpp
    mesh_adapter.cpp
    nodal_basis.cpp
    integration_cache.cpp
    distribution.cpp
    sweep_ordering.cpp
    kinetic_sweep_solver.cpp
    synthetic_acceleration_solver.cpp
    moment_calculator.cpp
    output_manager.cpp
  tests/
    unit/
```

The first implementation commit should add only build/config scaffolding and unit tests. Solver code should be added module by module with regression checkpoints.

## 7.1 Current Implementation Status

The current C++ tree has completed the first serial implementation milestones for
the thermalizing-wall path:

- YAML configuration, MFEM mesh loading, boundary attribute validation, angular
  quadrature, Fortran-style nodal basis functions, and integration caches are in
  place.
- CIS kinetic sweeps use cached dense LU factors per angle and element.
- CIS iteration, residual calculation, Tecplot output, Fourier reference output,
  and MFEM ParaView output are wired into `callaway_mfem`.
- GSIS acceleration now includes high-order source moments, local macroscopic
  matrix factors, trace coupling tensors, cached sparse trace assembly, GMRES
  or Eigen/SparseLU trace solve, local reconstruction, VDF correction, and a
  `GsisIterationDriver` selected by `gsis.enabled: true`.
- `config/control.gsis_smoke.yaml` provides a lightweight one-step GSIS run for
  executable-level regression testing.
- The migration-time Fortran/C++ comparison scripts, stored baseline outputs,
  and comparison data have been removed from the active tree. `callaway_mfem`
  still supports `--output-prefix` and `--write-output` for reproducible C++
  numerical experiments.

The remaining numerical work is broader boundary-condition support and future
parallelization beyond the validated serial/OpenMP path.

## 8. Implementation Phases

### Phase 0: Fortran Baseline and Regression Data

Tasks:

- Record exact compiler, flags, input file, mesh, and output files used for baseline runs.
- Run at least one CIS case and one GSIS case if the Fortran toolchain and MKL/PARDISO are available.
- Save logs, residual histories, runtime, and output fields under `tests/regression/baseline/`.
- Use `Synthetic_Acceleration.f90` as the authoritative GSIS formula.

Acceptance criteria:

- A reproducible Fortran run command exists.
- Baseline files are versioned or documented.
- Known Fortran issues are either frozen as compatibility behavior or corrected with explicit tests.

### Phase 1: CMake and Configuration

Tasks:

- Add a CMake project using `find_package(MFEM CONFIG REQUIRED)`.
- Support `-DMFEM_DIR=/usr/local/lib/cmake/mfem` and `-DMFEM_DIR=/usr/local/mfem-mpi/lib/cmake/mfem`.
- Add options:
  - `CALLAWAY_USE_MPI`
  - `CALLAWAY_USE_OPENMP`
  - `CALLAWAY_USE_MKL_PARDISO`
  - `CALLAWAY_ENABLE_STRICT_FORTRAN_BASIS`
- Use YAML as the primary C++ configuration format. A compatibility importer for the current Fortran `control.in` can be added as a helper, but the solver runtime should consume validated YAML.

Acceptance criteria:

- A minimal executable links against local MFEM 4.8.0.
- A YAML configuration equivalent to the current `control.in` can be loaded into C++ structs.
- If a `control.in` importer is implemented, it must generate the same validated configuration as the YAML file.

### Phase 2: Mesh and Boundary Adapter

Tasks:

- Load `A1_Nx11_Ny11.msh` using MFEM.
- Build face adjacency and local face ids.
- Map boundary physical ids to configured boundary conditions.
- Validate normals, areas, face lengths, and face counts against Fortran calculations.

Acceptance criteria:

- Mesh summary matches the Fortran mesh summary.
- Unit tests verify boundary attribute mapping and face adjacency.

### Phase 3: Angular Quadrature and Moments

Tasks:

- Port `GaussLegendre` or use a tested C++ quadrature generator.
- Port the current `theta/phi` split exactly.
- Implement moment computations for `T`, `qx`, `qy`.

Acceptance criteria:

- `sum(DOMEGA)` is close to `4*pi`.
- `sum(CX*DOMEGA)` and `sum(CY*DOMEGA)` are near zero.
- `sum(CX*CX*DOMEGA)` and `sum(CY*CY*DOMEGA)` match spherical moment expectations within tolerance.
- Moment calculations match Fortran for a synthetic distribution.

### Phase 4: Basis and Integration Cache

Tasks:

- Port Fortran-equivalent nodal basis generation.
- Port or rederive local mass, derivative, triple-product, element-face, face-face, and neighbor-face tensors.
- Alternatively, generate tensors through MFEM `FiniteElement` shape and derivative evaluation and compare with the Fortran-equivalent path.

Acceptance criteria:

- Basis has partition of unity.
- Basis has Kronecker-delta values at reference nodes.
- In strict-compatibility mode, integration tensors match Fortran for `DEG=1,2,3,4` on simple triangles.
- In MFEM-native mode, physical field regressions and conservation checks pass within documented tolerances even if local basis coefficients differ from Fortran.

### Phase 5: CIS Kinetic Solver

Tasks:

- Implement direction-wise sweep ordering.
- Implement local matrix assembly and RHS assembly.
- Implement thermalizing-wall boundary behavior.
- Define, but do not fully implement, non-thermalizing and periodic boundary interfaces.
- Implement local dense solves with precomputed LU.
- Implement residual calculation.

Acceptance criteria:

- One-step VDF update matches Fortran for a frozen initial state.
- Residual history matches Fortran within a defined tolerance for `ACCFLAG=0`.
- The current square conduction case produces matching `T`, `qx`, `qy` fields.

### Phase 6: Output and Reference Solutions

Tasks:

- Write Tecplot `.dat` files matching current field names.
- Add ParaView output through MFEM.
- Move analytical square Fourier solution into a separate reference module.
- Make sample grid dimensions configurable.

Acceptance criteria:

- Output parser can compare C++ and Fortran fields.
- Max and L2 differences are reported automatically.

### Phase 7: GSIS Acceleration

Tasks:

- Port local macroscopic block assembly.
- Port high-order source moment calculation.
- Assemble trace global system in `mfem::SparseMatrix`.
- Implement GMRES/ILU and optional PARDISO solve paths.
- Port local reconstruction and VDF correction.
- Validate limiter behavior.

Acceptance criteria:

- Local macroscopic matrices match Fortran block-by-block.
- Global trace matrix nonzero pattern and values match the `Synthetic_Acceleration.f90` Fortran CSR output on a small mesh when strict-compatibility assembly is enabled.
- GSIS residual history and final fields match Fortran for `ACCFLAG=1`.
- Iteration reduction relative to CIS is reproduced in small-Knudsen cases.

### Phase 8: Performance and Memory Optimization

Tasks:

- Add profiling timers around angular sweep, moment calculation, GSIS source, global solve, and output.
- Replace dense scan compression with direct sparse triplets.
- Add thread-local scratch buffers for local matrices and RHS vectors.
- Tune memory layout for direction-major sweep locality.
- Add optional precomputed local LU factor cache.

Acceptance criteria:

- C++ CIS is not slower than Fortran for the same serial/thread count after local LU caching.
- GSIS matrix assembly avoids dense `Nfaces`-wide row blocks.
- Peak memory is documented for representative mesh/order/angular settings.

### Phase 9: Optional Parallel Version

Tasks:

- Partition angular directions across threads first.
- Evaluate MPI decomposition by spatial mesh only after handling sweep dependencies.
- Consider direction-space MPI parallelism as a simpler first MPI target.
- Move GSIS trace solve to `HypreParMatrix` only after serial GSIS is validated.

Acceptance criteria:

- Serial and parallel results match.
- Parallel speedup is measured for both CIS and GSIS.

## 9. Testing Strategy

### Unit Tests

- Configuration parsing.
- Boundary physical-id mapping.
- Gmsh/MFEM mesh counts and attributes.
- Angular quadrature moments.
- Basis nodal properties.
- Volume and face integration tensors.
- Sweep ordering consistency.
- Dense local solve residuals.
- Sparse trace matrix assembly on a tiny mesh.

### Regression Tests

- Current `control.in` on `A1_Nx11_Ny11.msh`, `ACCFLAG=0`.
- Same mesh and parameters, `ACCFLAG=1`.
- A small `DEG=1`, low-angular-resolution case for fast CI.
- 1D slab-like benchmark from the paper, if a compatible mesh/input is added.
- Boundary-condition tests for thermalizing walls in the first milestone.
- Interface-level tests for non-thermalizing and periodic boundary dispatch, with full numerical tests added when those implementations are enabled.

### Numerical Metrics

- Residual history: relative L2 difference from baseline.
- Final fields: max norm and L2 norm of `T`, `qx`, `qy`.
- Integral conservation checks for energy.
- Effective thermal conductivity where applicable.
- Iteration counts and CPU time.

## 10. Memory and Efficiency Plan

- Use RAII (`std::vector`, `std::unique_ptr`, `mfem::Vector`, `mfem::DenseMatrix`, `mfem::SparseMatrix`) and avoid raw owning pointers.
- Use contiguous flattened arrays for VDF and macro fields.
- Use `std::span`-like views or small accessor classes to avoid copying.
- Precompute direction values, face normal speeds, and sweep orders.
- Cache local LU factors for CIS because the local matrix is iteration-invariant.
- Avoid storing unnecessary high-rank tensors if MFEM shape evaluation is fast enough; keep a compile/runtime option for exact Fortran-style caches.
- Assemble GSIS sparse matrices directly from local block triplets.
- Keep per-thread scratch buffers for RHS vectors and dense matrices.
- Use `mfem::StopWatch` or a small timing wrapper for performance reports.

## 11. Recommended Initial CMake Direction

Use a serial MFEM target first:

```cmake
cmake_minimum_required(VERSION 3.18)
project(callaway_mfem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MFEM CONFIG REQUIRED)

add_executable(callaway_mfem src/main.cpp)
target_link_libraries(callaway_mfem PRIVATE mfem)
target_include_directories(callaway_mfem PRIVATE include)
```

Configure with:

```bash
cmake -S mfem_cpp_refactor -B mfem_cpp_refactor/build \
  -DMFEM_DIR=/usr/local/lib/cmake/mfem
cmake --build mfem_cpp_refactor/build -j
```

For MPI later:

```bash
cmake -S mfem_cpp_refactor -B mfem_cpp_refactor/build-mpi \
  -DMFEM_DIR=/usr/local/mfem-mpi/lib/cmake/mfem \
  -DCALLAWAY_USE_MPI=ON
```

## 12. Resolved Decisions

1. `Synthetic_Acceleration.f90` is the authoritative GSIS implementation.
2. Exact dof-by-dof Fortran reproduction is not mandatory after the initial validation stage. The target is accurate reproduction of solver behavior, fields, convergence trends, and paper experiments.
3. The recommended solver path is portable first: implement the GSIS trace solve with MFEM iterative solvers and preconditioners, then keep MKL PARDISO/SuiteSparse/SuperLU as optional acceleration or parity backends.
4. YAML is the primary configuration format for the C++ project.
5. Non-thermalizing and periodic boundary conditions are not required in the first production milestone. Their interfaces must be designed now so the implementations can be added later without restructuring the solver.

## 13. Final Recommendation

Proceed with the refactor in two layers:

1. A strict reproduction layer that keeps the current DG basis, quadrature, sweep, and GSIS algebra intact, but implements them in modern C++ with MFEM mesh/sparse/output support.
2. A cleanup layer that gradually replaces custom low-level pieces with MFEM-native finite element and solver abstractions when regression tests show no unacceptable numerical drift.

This approach gives the best chance of reproducing the paper experiments and current Fortran results while still gaining MFEM's maintainability, mesh handling, sparse linear algebra, and future parallelization path.
