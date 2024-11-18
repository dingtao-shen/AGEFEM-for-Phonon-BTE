add_library(MacroProperty MacroProperty.cpp "${PROJECT_SOURCE_DIR}/includes/MacroProperty/MacroProperty.h")

target_link_libraries(MacroProperty Utility VelocityMesh SpatialMesh Integration)