# cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# project(pytorch)

# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(pytorch pytorch_cpp.cpp)
# target_link_libraries(pytorch "${TORCH_LIBRARIES}")
# set_property(TARGET pytorch PROPERTY CXX_STANDARD 17)

# cmake -DCMAKE_PREFIX_PATH=/home/raghul/Desktop/Conv2D-on-GPU/CPU/pytorch/libtorch ..
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pytorch)

# Find PyTorch
find_package(Torch REQUIRED)
# find_package(CNPY REQUIRED)

# Set additional include and library paths for cnpy, Eigen3, OpenCV
include_directories(/home/raghul/Desktop/Conv2D-on-GPU/CPU/pytorch/cnpy/build /usr/include/eigen3)
link_directories(/home/raghul/Desktop/Conv2D-on-GPU/CPU/pytorch/cnpy/build)

# Set the CXX flags and link with required libraries
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/cnpy -I/usr/include/eigen3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/../cnpy/build")
set(CMAKE_PREFIX_PATH "/home/raghul/Desktop/Conv2D-on-GPU/CPU/pytorch/libtorch")

# Link OpenCV using pkg-config
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CFLAGS}")
link_directories(${OpenCV_LIB_DIR})

# Add the executable and link libraries
add_executable(pytorch pytorch_cpp.cpp)
target_link_libraries(pytorch "${TORCH_LIBRARIES}" cnpy ${OpenCV_LIBRARIES} -lz)
target_link_libraries(pytorch /home/raghul/Desktop/Conv2D-on-GPU/CPU/pytorch/cnpy/build/libcnpy.a z)


# Set the path to cnpy (replace with the actual path to cnpy)
set(CNPY_INCLUDE_DIR "/home/raghul/Desktop/Conv2D-on-GPU/CPU/pytorch/cnpy/build")
set(CNPY_LIBRARY "/home/raghul/Desktop/Conv2D-on-GPU/CPU/pytorch/cnpy/build/libcnpy.a")

# Add executable
# add_executable(nn pytorch_cpp.cpp)

# # Include directories
# target_include_directories(nn PRIVATE
#     ${CNPY_INCLUDE_DIR}
#     ${OpenCV_INCLUDE_DIRS}
#     ${TORCH_INCLUDE_DIRS}
#     /usr/include/eigen3
# )

# # Link libraries
# target_link_libraries(nn
#     PRIVATE
#     ${CNPY_LIBRARY}
#     ${OpenCV_LIBS}
#     ${TORCH_LIBRARIES}
#     -lz
# )

# Set the C++ standard
set_property(TARGET pytorch PROPERTY CXX_STANDARD 17)
