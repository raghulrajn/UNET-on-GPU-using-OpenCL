# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -I/cnpy `pkg-config --cflags --libs opencv4`
LDFLAGS = -L/cnpy/build -lz
LIBS = cnpy/build/libcnpy.a `pkg-config --cflags --libs opencv4`

# Target executable
TARGET = main

# Source files and objects
SRCS = main.cpp tensor4D.cpp conv2d.cpp timer.cpp
OBJS = $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Linking the target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LIBS) $(LDFLAGS)
	rm -f $(OBJS)

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
