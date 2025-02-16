# Conv2D Implementation with OpenCL

This project demonstrates various convolutional neural network (CNN) operations using OpenCL for GPU acceleration. The operations include convolution, ReLU activation, max pooling, mean calculation, variance calculation, batch normalization, tensor concatenation, upsampling, and center extraction.

## Features

- **Convolution**: Perform 2D convolution on input tensors.
- **ReLU Activation**: Apply ReLU activation function to tensors.
- **Max Pooling**: Perform max pooling on input tensors.
- **Mean Calculation**: Calculate the mean of input tensors.
- **Variance Calculation**: Calculate the variance of input tensors.
- **Batch Normalization**: Apply batch normalization to tensors.
- **Tensor Concatenation**: Concatenate two tensors along the channel dimension.
- **Upsampling**: Upsample input tensors using Bilinear interpolation.
- **Center Extraction**: Extract the center region of input tensors.

## Prerequisites

- OpenCL SDK
- OpenCV
- Boost
- CMake

## Installation

1. Install the required libraries:

    ```sh
    sudo apt-get install opencl-headers ocl-icd-opencl-dev
    sudo apt-get install libopencv-dev
    sudo apt-get install libboost-all-dev
    sudo apt-get install cmake
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
