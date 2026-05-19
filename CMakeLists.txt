cmake_minimum_required(VERSION 3.18)

project(callaway_mfem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CALLAWAY_USE_MPI "Enable the future MPI build path." OFF)
option(CALLAWAY_USE_OPENMP "Enable OpenMP where available." OFF)
option(CALLAWAY_USE_MKL_PARDISO "Enable the optional MKL PARDISO backend." OFF)
option(CALLAWAY_ENABLE_STRICT_FORTRAN_BASIS "Enable strict Fortran-basis compatibility kernels." ON)

find_package(MFEM CONFIG REQUIRED)
find_package(Eigen3 QUIET)

set(CALLAWAY_REFERENCE_MESH
    ${CMAKE_CURRENT_SOURCE_DIR}/reference/FortranCodes/A1_Nx11_Ny11.msh)

add_library(callaway_core
  src/age_basis.cpp
  src/age_geometry.cpp
  src/age_mesh.cpp
  src/age_preprocessor.cpp
  src/angular_quadrature.cpp
  src/config.cpp
  src/dense_solver.cpp
  src/distribution.cpp
  src/geometry/circular_arc.cpp
  src/geometry/nurbs_curve.cpp
  src/geometry/polyline_curve.cpp
  src/integration_cache.cpp
  src/iteration_driver.cpp
  src/kinetic_sweep_solver.cpp
  src/mesh_adapter.cpp
  src/moment_calculator.cpp
  src/nodal_basis.cpp
  src/output_manager.cpp
  src/synthetic_acceleration_solver.cpp
  src/sweep_ordering.cpp
)

target_include_directories(callaway_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(callaway_core
  PUBLIC
    mfem
)

target_compile_definitions(callaway_core
  PUBLIC
    CALLAWAY_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    CALLAWAY_STRICT_FORTRAN_BASIS=$<BOOL:${CALLAWAY_ENABLE_STRICT_FORTRAN_BASIS}>
    CALLAWAY_HAS_EIGEN=$<BOOL:${Eigen3_FOUND}>
)

if(Eigen3_FOUND)
  target_link_libraries(callaway_core PUBLIC Eigen3::Eigen)
endif()

if(CALLAWAY_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(callaway_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(callaway_mfem src/main.cpp)
target_link_libraries(callaway_mfem PRIVATE callaway_core)

enable_testing()

add_executable(test_config_quadrature tests/unit/test_config_quadrature.cpp)
target_link_libraries(test_config_quadrature PRIVATE callaway_core)

add_executable(test_dense_solver tests/unit/test_dense_solver.cpp)
target_link_libraries(test_dense_solver PRIVATE callaway_core)

add_executable(test_basis_integration tests/unit/test_basis_integration.cpp)
target_link_libraries(test_basis_integration PRIVATE callaway_core)

add_executable(test_moments_sweep tests/unit/test_moments_sweep.cpp)
target_link_libraries(test_moments_sweep PRIVATE callaway_core)

add_executable(test_kinetic_sweep tests/unit/test_kinetic_sweep.cpp)
target_link_libraries(test_kinetic_sweep PRIVATE callaway_core)

add_executable(test_cis_iteration tests/unit/test_cis_iteration.cpp)
target_link_libraries(test_cis_iteration PRIVATE callaway_core)

add_executable(test_output_manager tests/unit/test_output_manager.cpp)
target_link_libraries(test_output_manager PRIVATE callaway_core)

add_executable(test_synthetic_acceleration tests/unit/test_synthetic_acceleration.cpp)
target_link_libraries(test_synthetic_acceleration PRIVATE callaway_core)

add_executable(test_boundary_curves tests/unit/test_boundary_curves.cpp)
target_link_libraries(test_boundary_curves PRIVATE callaway_core)

add_executable(test_age_preprocessor tests/unit/test_age_preprocessor.cpp)
target_link_libraries(test_age_preprocessor PRIVATE callaway_core)

add_executable(test_age_basis tests/unit/test_age_basis.cpp)
target_link_libraries(test_age_basis PRIVATE callaway_core)

add_executable(test_age_integration_cache tests/unit/test_age_integration_cache.cpp)
target_link_libraries(test_age_integration_cache PRIVATE callaway_core)

add_executable(test_age_kinetic_sweep tests/unit/test_age_kinetic_sweep.cpp)
target_link_libraries(test_age_kinetic_sweep PRIVATE callaway_core)

add_executable(test_concentric_ring tests/unit/test_concentric_ring.cpp)
target_link_libraries(test_concentric_ring PRIVATE callaway_core)

add_executable(test_polyline_age tests/unit/test_polyline_age.cpp)
target_link_libraries(test_polyline_age PRIVATE callaway_core)

add_executable(test_diffuse_curved_bc tests/unit/test_diffuse_curved_bc.cpp)
target_link_libraries(test_diffuse_curved_bc PRIVATE callaway_core)

add_executable(test_diffuse_straight_bc tests/unit/test_diffuse_straight_bc.cpp)
target_link_libraries(test_diffuse_straight_bc PRIVATE callaway_core)

add_executable(test_rough_boundary tests/unit/test_rough_boundary.cpp)
target_link_libraries(test_rough_boundary PRIVATE callaway_core)

add_executable(test_periodic_bc tests/unit/test_periodic_bc.cpp)
target_link_libraries(test_periodic_bc PRIVATE callaway_core)

add_test(
  NAME config_quadrature_mesh_smoke
  COMMAND test_config_quadrature
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CALLAWAY_REFERENCE_MESH}
)

add_test(
  NAME dense_solver_lu
  COMMAND test_dense_solver
)

add_test(
  NAME basis_integration_cache
  COMMAND test_basis_integration
          ${CALLAWAY_REFERENCE_MESH}
)

add_test(
  NAME moments_and_sweep_ordering
  COMMAND test_moments_sweep
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CALLAWAY_REFERENCE_MESH}
)

add_test(
  NAME kinetic_sweep_equilibrium
  COMMAND test_kinetic_sweep
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CALLAWAY_REFERENCE_MESH}
)

add_test(
  NAME cis_iteration_equilibrium
  COMMAND test_cis_iteration
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CALLAWAY_REFERENCE_MESH}
)

add_test(
  NAME output_manager_equilibrium
  COMMAND test_output_manager
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CALLAWAY_REFERENCE_MESH}
)

add_test(
  NAME synthetic_acceleration_source
  COMMAND test_synthetic_acceleration
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CALLAWAY_REFERENCE_MESH}
)

add_test(
  NAME gsis_iteration_smoke
  COMMAND callaway_mfem
          --config ${CMAKE_CURRENT_SOURCE_DIR}/config/control.gsis_smoke.yaml
          --solve
)

add_test(
  NAME boundary_curves
  COMMAND test_boundary_curves
)

add_test(
  NAME age_preprocessor
  COMMAND test_age_preprocessor
)

add_test(
  NAME age_basis
  COMMAND test_age_basis
)

add_test(
  NAME age_integration_cache
  COMMAND test_age_integration_cache
)

add_test(
  NAME age_kinetic_sweep
  COMMAND test_age_kinetic_sweep
)

add_test(
  NAME concentric_ring
  COMMAND test_concentric_ring
)

add_test(
  NAME polyline_age
  COMMAND test_polyline_age
)

add_test(
  NAME age_ring_executable_smoke
  COMMAND callaway_mfem
          --config ${CMAKE_CURRENT_SOURCE_DIR}/config/control.age_ring.example.yaml
          --solve
          --max-steps 20
)

add_test(
  NAME diffuse_curved_bc
  COMMAND test_diffuse_curved_bc
)

add_test(
  NAME diffuse_straight_bc
  COMMAND test_diffuse_straight_bc
)

add_test(
  NAME rough_boundary
  COMMAND test_rough_boundary
)

add_test(
  NAME periodic_bc
  COMMAND test_periodic_bc
)
