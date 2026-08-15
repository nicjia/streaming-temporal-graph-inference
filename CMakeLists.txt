cmake_minimum_required(VERSION 3.18)
project(high_throughput_graph_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags for modern CPU cache profiling
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

find_package(pybind11 REQUIRED)

include_directories(csrc/include)

# Compile C++ source files
add_library(graph_core STATIC
    csrc/src/memory_arena.cpp
    csrc/src/pcsr_graph.cpp
)

# Build the Pybind11 Python extension module
pybind11_add_module(graph_engine_cpp csrc/bindings/pybind_module.cpp)
target_link_libraries(graph_engine_cpp PRIVATE graph_core)