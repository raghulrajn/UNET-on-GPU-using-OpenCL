# Conv2D Implementation with OpenCL

This project demonstrates various convolutional neural network (CNN) operations using OpenCL for GPU acceleration. The operations include convolution, ReLU activation, max pooling, mean calculation, variance calculation, batch normalization, tensor concatenation, upsampling, and center extraction.

## Features

- Convolution
- ReLU Activation
- Max Pooling
- Mean Calculation
- Variance Calculation
- Batch Normalization
- Tensor Concatenation
- Upsampling
- Center Extraction

## Prerequisites

- OpenCL SDK
- OpenCV
- cnpy
- Meson

## Installation

1. Install the required libraries:

    ```sh
    sudo apt-get install opencl-headers ocl-icd-opencl-dev
    sudo apt-get install libopencv-dev
    sudo apt-get install libboost-all-dev
    sudo apt-get install cmake,meson
    ```

2. Clone the repository:

    ```sh
    git clone https://github.com/raghulrajn/OpenCL
    cd gpu
    ```

3. Build the project:

    ```sh
   chmod +x run.sh
   ./run.sh
    ```

## Usage

1. Run the main executable:

    ```sh
    cd build
    ./conv2d
    ```

2. The program will perform various CNN operations and print the performance metrics.

## Code Structure

- [conv2d.cpp](http://_vscodecontentref_/2): Main source file containing the implementation of CNN operations using OpenCL.
- `CMakeLists.txt`: CMake configuration file for building the project.

## Performance comparison
### GPU performance
![gpu](./utils/gpu.png)
