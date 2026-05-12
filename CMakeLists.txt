cmake_minimum_required(VERSION 3.18)

project(callaway_mfem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CALLAWAY_USE_MPI "Enable the future MPI build path." OFF)
option(CALLAWAY_USE_OPENMP "Enable OpenMP where available." OFF)
option(CALLAWAY_USE_MKL_PARDISO "Enable the optional MKL PARDISO backend." OFF)
option(CALLAWAY_ENABLE_STRICT_FORTRAN_BASIS "Enable strict Fortran-basis compatibility kernels." ON)

find_package(MFEM CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_library(callaway_core
  src/angular_quadrature.cpp
  src/config.cpp
  src/dense_solver.cpp
  src/distribution.cpp
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
)

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

add_test(
  NAME config_quadrature_mesh_smoke
  COMMAND test_config_quadrature
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/../A1_Nx11_Ny11.msh
)

add_test(
  NAME dense_solver_lu
  COMMAND test_dense_solver
)

add_test(
  NAME basis_integration_cache
  COMMAND test_basis_integration
          ${CMAKE_CURRENT_SOURCE_DIR}/../A1_Nx11_Ny11.msh
)

add_test(
  NAME moments_and_sweep_ordering
  COMMAND test_moments_sweep
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/../A1_Nx11_Ny11.msh
)

add_test(
  NAME kinetic_sweep_equilibrium
  COMMAND test_kinetic_sweep
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/../A1_Nx11_Ny11.msh
)

add_test(
  NAME cis_iteration_equilibrium
  COMMAND test_cis_iteration
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/../A1_Nx11_Ny11.msh
)

add_test(
  NAME output_manager_equilibrium
  COMMAND test_output_manager
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/../A1_Nx11_Ny11.msh
)

add_test(
  NAME synthetic_acceleration_source
  COMMAND test_synthetic_acceleration
          ${CMAKE_CURRENT_SOURCE_DIR}/config/control.example.yaml
          ${CMAKE_CURRENT_SOURCE_DIR}/../A1_Nx11_Ny11.msh
)

add_test(
  NAME gsis_iteration_smoke
  COMMAND callaway_mfem
          --config ${CMAKE_CURRENT_SOURCE_DIR}/config/control.gsis_smoke.yaml
          --solve
)

add_test(
  NAME tecplot_compare_tool
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_tecplot.py
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/tecplot_small.dat
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/tecplot_small.dat
          --columns T,qx,qy,Nxx,Nxy,Nyy
)

add_test(
  NAME control_in_converter
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/scripts/test_convert_control_in.py
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/convert_control_in.py
          ${CMAKE_CURRENT_SOURCE_DIR}/../control.in
)

add_test(
  NAME fortran_baseline_script_syntax
  COMMAND bash -n ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_fortran_baseline.sh
)

add_test(
  NAME regression_script_syntax
  COMMAND bash -n ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_regression.sh
)

add_test(
  NAME fortran_residual_parser
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/parse_fortran_residual.py
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/fortran_stdout_small.log
)

add_test(
  NAME residual_compare_tool
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_residual.py
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/residual_small.csv
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/residual_small.csv
          --columns residual,mass
)
