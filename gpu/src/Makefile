CXX      := g++
# -O3: High optimization
# -march=native: Use all SIMD instructions (AVX/AVX2) on your specific CPU
# -fopenmp: Enable multi-core parallel loops
CXXFLAGS := -O3 -march=native -fopenmp -std=c++17 -Wall -Wextra
CPPFLAGS := -Isrc/cnpy
# -I/usr/include/opencv4
# Paths & Libraries
CNPY_PATH := /home/raghul/Downloads/GPU_LAB_SUBMISSION/gpu/src/cnpy/build
LIBS      := $(shell pkg-config --libs opencv4) $(CNPY_PATH)/libcnpy.a -lz -lpthread

# Files
TARGET    := opt_conv2d_cpu
SRC       := opt_cpu.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)