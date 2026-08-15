CXX = g++
CXXFLAGS = -O3 -Wall -std=c++20 -fPIC -march=native -I csrc/include
LDFLAGS_EXT = -shared -undefined dynamic_lookup

# Dynamically grab Python paths
PYTHON_SUFFIX = $(shell python3-config --extension-suffix)
PYBIND_INCL = $(shell python3 -m pybind11 --includes)

# Core C++ source files
SRC = csrc/src/memory_arena.cpp csrc/src/pcsr_graph.cpp

.PHONY: all ext tests clean

all: ext tests

# Build the Python extension
ext:
	$(CXX) $(CXXFLAGS) $(LDFLAGS_EXT) $(PYBIND_INCL) \
	$(SRC) csrc/bindings/pybind_module.cpp \
	-o graph_engine$(PYTHON_SUFFIX)
	@echo "Python extension built successfully!"

# Build the C++ executables directly into the main folder
tests: test_pcsr perf_test

test_pcsr:
	$(CXX) $(CXXFLAGS) $(SRC) tests/test_pcsr.cpp -o test_pcsr
	@echo "Unit test binary compiled as ./test_pcsr"

perf_test:
	$(CXX) $(CXXFLAGS) $(SRC) benchmarks/cpp_perf_test.cpp -o cpp_perf_test
	@echo "Performance test binary compiled as ./cpp_perf_test"

clean:
	rm -f graph_engine* test_pcsr cpp_perf_test
	@echo "Cleaned up old builds."