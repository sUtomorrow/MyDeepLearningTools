# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/user/clion-2019.2.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/user/clion-2019.2.3/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/pytorch_extension.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pytorch_extension.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pytorch_extension.dir/flags.make

CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.o: CMakeFiles/pytorch_extension.dir/flags.make
CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.o: ../pytorch_extension_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.o -c /home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension/pytorch_extension_main.cpp

CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension/pytorch_extension_main.cpp > CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.i

CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension/pytorch_extension_main.cpp -o CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.s

# Object files for target pytorch_extension
pytorch_extension_OBJECTS = \
"CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.o"

# External object files for target pytorch_extension
pytorch_extension_EXTERNAL_OBJECTS =

pytorch_extension: CMakeFiles/pytorch_extension.dir/pytorch_extension_main.cpp.o
pytorch_extension: CMakeFiles/pytorch_extension.dir/build.make
pytorch_extension: CMakeFiles/pytorch_extension.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pytorch_extension"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pytorch_extension.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pytorch_extension.dir/build: pytorch_extension

.PHONY : CMakeFiles/pytorch_extension.dir/build

CMakeFiles/pytorch_extension.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pytorch_extension.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pytorch_extension.dir/clean

CMakeFiles/pytorch_extension.dir/depend:
	cd /home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension /home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension /home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension/cmake-build-debug /home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension/cmake-build-debug /home/lty/Code/MyDeepLearningTools/pytorch/pytorch_extension/cmake-build-debug/CMakeFiles/pytorch_extension.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pytorch_extension.dir/depend
