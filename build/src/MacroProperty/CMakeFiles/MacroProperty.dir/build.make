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
include src/MacroProperty/CMakeFiles/MacroProperty.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/MacroProperty/CMakeFiles/MacroProperty.dir/compiler_depend.make

# Include the progress variables for this target.
include src/MacroProperty/CMakeFiles/MacroProperty.dir/progress.make

# Include the compile flags for this target's objects.
include src/MacroProperty/CMakeFiles/MacroProperty.dir/flags.make

src/MacroProperty/CMakeFiles/MacroProperty.dir/MacroProperty.cpp.o: src/MacroProperty/CMakeFiles/MacroProperty.dir/flags.make
src/MacroProperty/CMakeFiles/MacroProperty.dir/MacroProperty.cpp.o: ../src/MacroProperty/MacroProperty.cpp
src/MacroProperty/CMakeFiles/MacroProperty.dir/MacroProperty.cpp.o: src/MacroProperty/CMakeFiles/MacroProperty.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/MacroProperty/CMakeFiles/MacroProperty.dir/MacroProperty.cpp.o"
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/MacroProperty && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/MacroProperty/CMakeFiles/MacroProperty.dir/MacroProperty.cpp.o -MF CMakeFiles/MacroProperty.dir/MacroProperty.cpp.o.d -o CMakeFiles/MacroProperty.dir/MacroProperty.cpp.o -c /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/src/MacroProperty/MacroProperty.cpp

src/MacroProperty/CMakeFiles/MacroProperty.dir/MacroProperty.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MacroProperty.dir/MacroProperty.cpp.i"
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/MacroProperty && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/src/MacroProperty/MacroProperty.cpp > CMakeFiles/MacroProperty.dir/MacroProperty.cpp.i

src/MacroProperty/CMakeFiles/MacroProperty.dir/MacroProperty.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MacroProperty.dir/MacroProperty.cpp.s"
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/MacroProperty && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/src/MacroProperty/MacroProperty.cpp -o CMakeFiles/MacroProperty.dir/MacroProperty.cpp.s

# Object files for target MacroProperty
MacroProperty_OBJECTS = \
"CMakeFiles/MacroProperty.dir/MacroProperty.cpp.o"

# External object files for target MacroProperty
MacroProperty_EXTERNAL_OBJECTS =

src/MacroProperty/libMacroProperty.a: src/MacroProperty/CMakeFiles/MacroProperty.dir/MacroProperty.cpp.o
src/MacroProperty/libMacroProperty.a: src/MacroProperty/CMakeFiles/MacroProperty.dir/build.make
src/MacroProperty/libMacroProperty.a: src/MacroProperty/CMakeFiles/MacroProperty.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libMacroProperty.a"
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/MacroProperty && $(CMAKE_COMMAND) -P CMakeFiles/MacroProperty.dir/cmake_clean_target.cmake
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/MacroProperty && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MacroProperty.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/MacroProperty/CMakeFiles/MacroProperty.dir/build: src/MacroProperty/libMacroProperty.a
.PHONY : src/MacroProperty/CMakeFiles/MacroProperty.dir/build

src/MacroProperty/CMakeFiles/MacroProperty.dir/clean:
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/MacroProperty && $(CMAKE_COMMAND) -P CMakeFiles/MacroProperty.dir/cmake_clean.cmake
.PHONY : src/MacroProperty/CMakeFiles/MacroProperty.dir/clean

src/MacroProperty/CMakeFiles/MacroProperty.dir/depend:
	cd /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/src/MacroProperty /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/MacroProperty /home/dtshen/Develop/AGEFEM/AGEFEM-for-Phonon-BTE/build/src/MacroProperty/CMakeFiles/MacroProperty.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/MacroProperty/CMakeFiles/MacroProperty.dir/depend

