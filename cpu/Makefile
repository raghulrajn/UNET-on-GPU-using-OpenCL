CXX = g++
CXXFLAGS = -std=c++17 -I/cnpy -I/usr/include/eigen3
LDFLAGS = -L/cnpy/build -lz
LIBS = cnpy/build/libcnpy.a `pkg-config --cflags --libs opencv4`

TARGET = unet
SRC = conv2D.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LIBS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

