add_library(Integration Integration.cpp "${PROJECT_SOURCE_DIR}/includes/Integration/Integration.h")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Integration Utility VelocityMesh SpatialMesh OpenMP::OpenMP_CXX)
endif()