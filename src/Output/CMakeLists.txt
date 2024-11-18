add_library(Output Output.cpp "${PROJECT_SOURCE_DIR}/includes/Output/Output.h")

target_link_libraries(Output Utility SpatialMesh)