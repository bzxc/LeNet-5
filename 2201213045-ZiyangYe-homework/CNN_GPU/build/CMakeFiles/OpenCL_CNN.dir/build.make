# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.26.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.26.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build

# Include any dependencies generated for this target.
include CMakeFiles/OpenCL_CNN.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/OpenCL_CNN.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/OpenCL_CNN.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/OpenCL_CNN.dir/flags.make

CMakeFiles/OpenCL_CNN.dir/backward.cpp.o: CMakeFiles/OpenCL_CNN.dir/flags.make
CMakeFiles/OpenCL_CNN.dir/backward.cpp.o: /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/backward.cpp
CMakeFiles/OpenCL_CNN.dir/backward.cpp.o: CMakeFiles/OpenCL_CNN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/OpenCL_CNN.dir/backward.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OpenCL_CNN.dir/backward.cpp.o -MF CMakeFiles/OpenCL_CNN.dir/backward.cpp.o.d -o CMakeFiles/OpenCL_CNN.dir/backward.cpp.o -c /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/backward.cpp

CMakeFiles/OpenCL_CNN.dir/backward.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenCL_CNN.dir/backward.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/backward.cpp > CMakeFiles/OpenCL_CNN.dir/backward.cpp.i

CMakeFiles/OpenCL_CNN.dir/backward.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenCL_CNN.dir/backward.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/backward.cpp -o CMakeFiles/OpenCL_CNN.dir/backward.cpp.s

CMakeFiles/OpenCL_CNN.dir/bmp.cpp.o: CMakeFiles/OpenCL_CNN.dir/flags.make
CMakeFiles/OpenCL_CNN.dir/bmp.cpp.o: /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/bmp.cpp
CMakeFiles/OpenCL_CNN.dir/bmp.cpp.o: CMakeFiles/OpenCL_CNN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/OpenCL_CNN.dir/bmp.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OpenCL_CNN.dir/bmp.cpp.o -MF CMakeFiles/OpenCL_CNN.dir/bmp.cpp.o.d -o CMakeFiles/OpenCL_CNN.dir/bmp.cpp.o -c /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/bmp.cpp

CMakeFiles/OpenCL_CNN.dir/bmp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenCL_CNN.dir/bmp.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/bmp.cpp > CMakeFiles/OpenCL_CNN.dir/bmp.cpp.i

CMakeFiles/OpenCL_CNN.dir/bmp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenCL_CNN.dir/bmp.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/bmp.cpp -o CMakeFiles/OpenCL_CNN.dir/bmp.cpp.s

CMakeFiles/OpenCL_CNN.dir/cnn.cpp.o: CMakeFiles/OpenCL_CNN.dir/flags.make
CMakeFiles/OpenCL_CNN.dir/cnn.cpp.o: /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/cnn.cpp
CMakeFiles/OpenCL_CNN.dir/cnn.cpp.o: CMakeFiles/OpenCL_CNN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/OpenCL_CNN.dir/cnn.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OpenCL_CNN.dir/cnn.cpp.o -MF CMakeFiles/OpenCL_CNN.dir/cnn.cpp.o.d -o CMakeFiles/OpenCL_CNN.dir/cnn.cpp.o -c /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/cnn.cpp

CMakeFiles/OpenCL_CNN.dir/cnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenCL_CNN.dir/cnn.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/cnn.cpp > CMakeFiles/OpenCL_CNN.dir/cnn.cpp.i

CMakeFiles/OpenCL_CNN.dir/cnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenCL_CNN.dir/cnn.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/cnn.cpp -o CMakeFiles/OpenCL_CNN.dir/cnn.cpp.s

CMakeFiles/OpenCL_CNN.dir/forward.cpp.o: CMakeFiles/OpenCL_CNN.dir/flags.make
CMakeFiles/OpenCL_CNN.dir/forward.cpp.o: /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/forward.cpp
CMakeFiles/OpenCL_CNN.dir/forward.cpp.o: CMakeFiles/OpenCL_CNN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/OpenCL_CNN.dir/forward.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OpenCL_CNN.dir/forward.cpp.o -MF CMakeFiles/OpenCL_CNN.dir/forward.cpp.o.d -o CMakeFiles/OpenCL_CNN.dir/forward.cpp.o -c /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/forward.cpp

CMakeFiles/OpenCL_CNN.dir/forward.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenCL_CNN.dir/forward.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/forward.cpp > CMakeFiles/OpenCL_CNN.dir/forward.cpp.i

CMakeFiles/OpenCL_CNN.dir/forward.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenCL_CNN.dir/forward.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/forward.cpp -o CMakeFiles/OpenCL_CNN.dir/forward.cpp.s

CMakeFiles/OpenCL_CNN.dir/init.cpp.o: CMakeFiles/OpenCL_CNN.dir/flags.make
CMakeFiles/OpenCL_CNN.dir/init.cpp.o: /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/init.cpp
CMakeFiles/OpenCL_CNN.dir/init.cpp.o: CMakeFiles/OpenCL_CNN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/OpenCL_CNN.dir/init.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OpenCL_CNN.dir/init.cpp.o -MF CMakeFiles/OpenCL_CNN.dir/init.cpp.o.d -o CMakeFiles/OpenCL_CNN.dir/init.cpp.o -c /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/init.cpp

CMakeFiles/OpenCL_CNN.dir/init.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenCL_CNN.dir/init.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/init.cpp > CMakeFiles/OpenCL_CNN.dir/init.cpp.i

CMakeFiles/OpenCL_CNN.dir/init.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenCL_CNN.dir/init.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/init.cpp -o CMakeFiles/OpenCL_CNN.dir/init.cpp.s

CMakeFiles/OpenCL_CNN.dir/main.cpp.o: CMakeFiles/OpenCL_CNN.dir/flags.make
CMakeFiles/OpenCL_CNN.dir/main.cpp.o: /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/main.cpp
CMakeFiles/OpenCL_CNN.dir/main.cpp.o: CMakeFiles/OpenCL_CNN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/OpenCL_CNN.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OpenCL_CNN.dir/main.cpp.o -MF CMakeFiles/OpenCL_CNN.dir/main.cpp.o.d -o CMakeFiles/OpenCL_CNN.dir/main.cpp.o -c /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/main.cpp

CMakeFiles/OpenCL_CNN.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenCL_CNN.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/main.cpp > CMakeFiles/OpenCL_CNN.dir/main.cpp.i

CMakeFiles/OpenCL_CNN.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenCL_CNN.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/main.cpp -o CMakeFiles/OpenCL_CNN.dir/main.cpp.s

CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.o: CMakeFiles/OpenCL_CNN.dir/flags.make
CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.o: /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/math_functions.cpp
CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.o: CMakeFiles/OpenCL_CNN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.o -MF CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.o.d -o CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.o -c /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/math_functions.cpp

CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/math_functions.cpp > CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.i

CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/math_functions.cpp -o CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.s

CMakeFiles/OpenCL_CNN.dir/mnist.cpp.o: CMakeFiles/OpenCL_CNN.dir/flags.make
CMakeFiles/OpenCL_CNN.dir/mnist.cpp.o: /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/mnist.cpp
CMakeFiles/OpenCL_CNN.dir/mnist.cpp.o: CMakeFiles/OpenCL_CNN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/OpenCL_CNN.dir/mnist.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OpenCL_CNN.dir/mnist.cpp.o -MF CMakeFiles/OpenCL_CNN.dir/mnist.cpp.o.d -o CMakeFiles/OpenCL_CNN.dir/mnist.cpp.o -c /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/mnist.cpp

CMakeFiles/OpenCL_CNN.dir/mnist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenCL_CNN.dir/mnist.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/mnist.cpp > CMakeFiles/OpenCL_CNN.dir/mnist.cpp.i

CMakeFiles/OpenCL_CNN.dir/mnist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenCL_CNN.dir/mnist.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/mnist.cpp -o CMakeFiles/OpenCL_CNN.dir/mnist.cpp.s

CMakeFiles/OpenCL_CNN.dir/model.cpp.o: CMakeFiles/OpenCL_CNN.dir/flags.make
CMakeFiles/OpenCL_CNN.dir/model.cpp.o: /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/model.cpp
CMakeFiles/OpenCL_CNN.dir/model.cpp.o: CMakeFiles/OpenCL_CNN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/OpenCL_CNN.dir/model.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OpenCL_CNN.dir/model.cpp.o -MF CMakeFiles/OpenCL_CNN.dir/model.cpp.o.d -o CMakeFiles/OpenCL_CNN.dir/model.cpp.o -c /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/model.cpp

CMakeFiles/OpenCL_CNN.dir/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenCL_CNN.dir/model.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/model.cpp > CMakeFiles/OpenCL_CNN.dir/model.cpp.i

CMakeFiles/OpenCL_CNN.dir/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenCL_CNN.dir/model.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/model.cpp -o CMakeFiles/OpenCL_CNN.dir/model.cpp.s

CMakeFiles/OpenCL_CNN.dir/predict.cpp.o: CMakeFiles/OpenCL_CNN.dir/flags.make
CMakeFiles/OpenCL_CNN.dir/predict.cpp.o: /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/predict.cpp
CMakeFiles/OpenCL_CNN.dir/predict.cpp.o: CMakeFiles/OpenCL_CNN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/OpenCL_CNN.dir/predict.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OpenCL_CNN.dir/predict.cpp.o -MF CMakeFiles/OpenCL_CNN.dir/predict.cpp.o.d -o CMakeFiles/OpenCL_CNN.dir/predict.cpp.o -c /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/predict.cpp

CMakeFiles/OpenCL_CNN.dir/predict.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenCL_CNN.dir/predict.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/predict.cpp > CMakeFiles/OpenCL_CNN.dir/predict.cpp.i

CMakeFiles/OpenCL_CNN.dir/predict.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenCL_CNN.dir/predict.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/predict.cpp -o CMakeFiles/OpenCL_CNN.dir/predict.cpp.s

CMakeFiles/OpenCL_CNN.dir/train.cpp.o: CMakeFiles/OpenCL_CNN.dir/flags.make
CMakeFiles/OpenCL_CNN.dir/train.cpp.o: /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/train.cpp
CMakeFiles/OpenCL_CNN.dir/train.cpp.o: CMakeFiles/OpenCL_CNN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/OpenCL_CNN.dir/train.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OpenCL_CNN.dir/train.cpp.o -MF CMakeFiles/OpenCL_CNN.dir/train.cpp.o.d -o CMakeFiles/OpenCL_CNN.dir/train.cpp.o -c /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/train.cpp

CMakeFiles/OpenCL_CNN.dir/train.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/OpenCL_CNN.dir/train.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/train.cpp > CMakeFiles/OpenCL_CNN.dir/train.cpp.i

CMakeFiles/OpenCL_CNN.dir/train.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/OpenCL_CNN.dir/train.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src/train.cpp -o CMakeFiles/OpenCL_CNN.dir/train.cpp.s

# Object files for target OpenCL_CNN
OpenCL_CNN_OBJECTS = \
"CMakeFiles/OpenCL_CNN.dir/backward.cpp.o" \
"CMakeFiles/OpenCL_CNN.dir/bmp.cpp.o" \
"CMakeFiles/OpenCL_CNN.dir/cnn.cpp.o" \
"CMakeFiles/OpenCL_CNN.dir/forward.cpp.o" \
"CMakeFiles/OpenCL_CNN.dir/init.cpp.o" \
"CMakeFiles/OpenCL_CNN.dir/main.cpp.o" \
"CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.o" \
"CMakeFiles/OpenCL_CNN.dir/mnist.cpp.o" \
"CMakeFiles/OpenCL_CNN.dir/model.cpp.o" \
"CMakeFiles/OpenCL_CNN.dir/predict.cpp.o" \
"CMakeFiles/OpenCL_CNN.dir/train.cpp.o"

# External object files for target OpenCL_CNN
OpenCL_CNN_EXTERNAL_OBJECTS =

OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/backward.cpp.o
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/bmp.cpp.o
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/cnn.cpp.o
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/forward.cpp.o
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/init.cpp.o
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/main.cpp.o
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/math_functions.cpp.o
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/mnist.cpp.o
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/model.cpp.o
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/predict.cpp.o
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/train.cpp.o
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/build.make
OpenCL_CNN: CMakeFiles/OpenCL_CNN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking CXX executable OpenCL_CNN"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/OpenCL_CNN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/OpenCL_CNN.dir/build: OpenCL_CNN
.PHONY : CMakeFiles/OpenCL_CNN.dir/build

CMakeFiles/OpenCL_CNN.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/OpenCL_CNN.dir/cmake_clean.cmake
.PHONY : CMakeFiles/OpenCL_CNN.dir/clean

CMakeFiles/OpenCL_CNN.dir/depend:
	cd /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/src /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build /Users/yee/Desktop/MyCareer/Heterogeneou/bigHomework/home/yeziyang/yzy/cnn_gpu/build/CMakeFiles/OpenCL_CNN.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/OpenCL_CNN.dir/depend

