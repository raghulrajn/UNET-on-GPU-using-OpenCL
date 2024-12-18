// kernel_loader.cpp

#include "kernel_loader.h"
#include "cnpy/cnpy.h"
#include <iostream>
#include <stdexcept>

Eigen::MatrixXf loadKernel(const std::string& filepath) {
    // Load the .npy file
    cnpy::NpyArray arr = cnpy::npy_load(filepath);
    
    if (arr.word_size != sizeof(float)) {
        throw std::runtime_error("Data type mismatch: Expected float data.");
    }

    //kernel has a 2D shape
    size_t rows = arr.shape[0];
    size_t cols = arr.shape[1];

    std::cout<<rows<<" "<<cols<<std::endl;
    
    // Convert the data to an Eigen matrix
    Eigen::MatrixXf kernel(rows, cols);
    float* data = arr.data<float>();
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            kernel(i, j) = data[i * cols + j];
        }
    }

    return kernel;
}
