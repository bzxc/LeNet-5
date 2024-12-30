cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(cnn LANGUAGES CXX)

aux_source_directory(. _SOURCES)

# list(APPEND _CXX_FLAGS "-fopenmp" "-march=native")
# list(APPEND _CXX_FLAGS "-march=native")
# set(_CXX_FLAGS "-I/opt/AMDAPP-3.0/include -L/opt/AMDAPP-3.0/lib/x86_64 -lOpenCL")
# set(_CXX_FLAGS "-g")

# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(OpenCL_CNN ${_SOURCES})

target_compile_options(OpenCL_CNN PRIVATE ${_CXX_FLAGS})
target_include_directories (OpenCL_CNN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (OpenCL_CNN ${OpenCL_LIBRARY})