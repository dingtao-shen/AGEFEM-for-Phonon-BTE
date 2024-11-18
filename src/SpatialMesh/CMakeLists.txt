add_library(SpatialMesh SpatialMesh.cpp "${PROJECT_SOURCE_DIR}/includes/SpatialMesh/SpatialMesh.h")

target_link_libraries(SpatialMesh VelocityMesh Utility)