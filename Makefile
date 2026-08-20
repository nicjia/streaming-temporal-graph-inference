CXX ?= g++
ARCH_FLAGS ?= -march=native
CXXFLAGS = -O3 -Wall -Wextra -std=c++20 -fPIC $(ARCH_FLAGS) -I csrc/include

# -undefined dynamic_lookup is a macOS linker flag. On Linux it is not
# recognised and the extension fails to link, so the platform is detected
# rather than assumed. ARCH_FLAGS is overridable because -march=native is
# unavailable on some cross-compiled and ARM CI runners.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LDFLAGS_EXT = -shared -undefined dynamic_lookup
else
LDFLAGS_EXT = -shared
endif

# Dynamically grab Python paths
PYTHON = python3
PYTHON_SUFFIX = $(shell $(PYTHON)-config --extension-suffix)
PYBIND_INCL = $(shell $(PYTHON) -m pybind11 --includes)

# Core C++ source files
SRC = csrc/src/memory_arena.cpp csrc/src/pcsr_graph.cpp
HDR = $(wildcard csrc/include/*.hpp)

EXT = graph_engine$(PYTHON_SUFFIX)

.PHONY: all ext tests test bench clean

all: ext tests

# Build the Python extension
ext: $(EXT)

$(EXT): $(SRC) $(HDR) csrc/bindings/pybind_module.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS_EXT) $(PYBIND_INCL) \
	$(SRC) csrc/bindings/pybind_module.cpp \
	-o $(EXT)
	@echo "Python extension built successfully!"

# Build the C++ executables directly into the main folder
tests: test_pcsr cpp_perf_test

test_pcsr: $(SRC) $(HDR) tests/test_pcsr.cpp
	$(CXX) $(CXXFLAGS) $(SRC) tests/test_pcsr.cpp -o test_pcsr
	@echo "Unit test binary compiled as ./test_pcsr"

cpp_perf_test: $(SRC) $(HDR) benchmarks/cpp_perf_test.cpp
	$(CXX) $(CXXFLAGS) $(SRC) benchmarks/cpp_perf_test.cpp -o cpp_perf_test
	@echo "Performance test binary compiled as ./cpp_perf_test"

# Run the C++ unit tests
test: test_pcsr
	./test_pcsr

# Run the C++ insertion benchmark
bench: cpp_perf_test
	./cpp_perf_test

clean:
	rm -f graph_engine*.so test_pcsr cpp_perf_test
	@echo "Cleaned up old builds."
