add_library(DGSolver DGSolver.cpp "${PROJECT_SOURCE_DIR}/includes/DGSolver/DGSolver.h")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(DGSolver Utility VelocityMesh SpatialMesh Integration MacroProperty OpenMP::OpenMP_CXX)
endif()