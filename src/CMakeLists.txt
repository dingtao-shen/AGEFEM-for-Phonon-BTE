# find_package(MPI REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/includes" )

add_subdirectory(Utility)
add_subdirectory(VelocityMesh)
add_subdirectory(SpatialMesh)
add_subdirectory(Integration)
add_subdirectory(MacroProperty)
add_subdirectory(DGSolver)
add_subdirectory(Output)

add_executable(AGEFEM_Poros AGEFEM_Poros.cpp)

target_link_libraries(AGEFEM_Poros Utility VelocityMesh SpatialMesh Integration MacroProperty DGSolver Output)