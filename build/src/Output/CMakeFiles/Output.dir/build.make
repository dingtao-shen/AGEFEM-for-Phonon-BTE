# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build

# Include any dependencies generated for this target.
include src/Output/CMakeFiles/Output.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/Output/CMakeFiles/Output.dir/compiler_depend.make

# Include the progress variables for this target.
include src/Output/CMakeFiles/Output.dir/progress.make

# Include the compile flags for this target's objects.
include src/Output/CMakeFiles/Output.dir/flags.make

src/Output/CMakeFiles/Output.dir/Output.cpp.o: src/Output/CMakeFiles/Output.dir/flags.make
src/Output/CMakeFiles/Output.dir/Output.cpp.o: ../src/Output/Output.cpp
src/Output/CMakeFiles/Output.dir/Output.cpp.o: src/Output/CMakeFiles/Output.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/Output/CMakeFiles/Output.dir/Output.cpp.o"
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/Output && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/Output/CMakeFiles/Output.dir/Output.cpp.o -MF CMakeFiles/Output.dir/Output.cpp.o.d -o CMakeFiles/Output.dir/Output.cpp.o -c /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/src/Output/Output.cpp

src/Output/CMakeFiles/Output.dir/Output.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Output.dir/Output.cpp.i"
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/Output && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/src/Output/Output.cpp > CMakeFiles/Output.dir/Output.cpp.i

src/Output/CMakeFiles/Output.dir/Output.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Output.dir/Output.cpp.s"
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/Output && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/src/Output/Output.cpp -o CMakeFiles/Output.dir/Output.cpp.s

# Object files for target Output
Output_OBJECTS = \
"CMakeFiles/Output.dir/Output.cpp.o"

# External object files for target Output
Output_EXTERNAL_OBJECTS =

src/Output/libOutput.a: src/Output/CMakeFiles/Output.dir/Output.cpp.o
src/Output/libOutput.a: src/Output/CMakeFiles/Output.dir/build.make
src/Output/libOutput.a: src/Output/CMakeFiles/Output.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libOutput.a"
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/Output && $(CMAKE_COMMAND) -P CMakeFiles/Output.dir/cmake_clean_target.cmake
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/Output && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Output.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/Output/CMakeFiles/Output.dir/build: src/Output/libOutput.a
.PHONY : src/Output/CMakeFiles/Output.dir/build

src/Output/CMakeFiles/Output.dir/clean:
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/Output && $(CMAKE_COMMAND) -P CMakeFiles/Output.dir/cmake_clean.cmake
.PHONY : src/Output/CMakeFiles/Output.dir/clean

src/Output/CMakeFiles/Output.dir/depend:
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/src/Output /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/Output /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/Output/CMakeFiles/Output.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/Output/CMakeFiles/Output.dir/depend

