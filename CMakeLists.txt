cmake_minimum_required(VERSION 3.18)
project(high_throughput_graph_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(ENABLE_NATIVE_ARCH "Compile with -march=native" ON)
option(BUILD_PYTHON_MODULE "Build the Pybind11 extension" ON)

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3)
    # -march=native is not available on every CI runner or cross-compile
    # target, so it is a switch rather than a hard requirement.
    if(ENABLE_NATIVE_ARCH)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
        if(HAS_MARCH_NATIVE)
            add_compile_options(-march=native)
        endif()
    endif()
endif()

include_directories(csrc/include)

# The engine itself: no Python, no domain logic, no third-party dependencies.
add_library(graph_core STATIC
    csrc/src/memory_arena.cpp
    csrc/src/pcsr_graph.cpp
)

# ---- tests -----------------------------------------------------------------
enable_testing()

add_executable(test_pcsr tests/test_pcsr.cpp)
target_link_libraries(test_pcsr PRIVATE graph_core)
add_test(NAME pcsr_unit_tests COMMAND test_pcsr)

add_executable(cpp_perf_test benchmarks/cpp_perf_test.cpp)
target_link_libraries(cpp_perf_test PRIVATE graph_core)

# ---- python extension ------------------------------------------------------
# Optional so the C++ core can be built and tested with no Python toolchain.
if(BUILD_PYTHON_MODULE)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        # Module name must match the Makefile output and `import graph_engine`.
        pybind11_add_module(graph_engine csrc/bindings/pybind_module.cpp)
        target_link_libraries(graph_engine PRIVATE graph_core)
    else()
        message(STATUS "pybind11 not found - skipping the Python extension. "
                       "Install it with: pip install pybind11")
    endif()
endif()
