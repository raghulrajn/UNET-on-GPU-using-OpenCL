// kernel_loader.h

#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <Eigen/Dense>
#include <string>

// Function to load a kernel from a .npy file and return it as an Eigen matrix
Eigen::MatrixXf loadKernel(const std::string& filepath);

#endif // KERNEL_LOADER_H
