#ifndef CONV2D_H
#define CONV2D_H

#include <vector>
#include <iostream>
#include <random>
#include <stdexcept>
#include "tensor4D.h"


class Conv2d {

public:
    // Apply ReLU activation function
    void applyReLU(Tensor4D& tensor);
    void applyMaxPool(Tensor4D& tensor, int poolHeight, int poolWidth, int stride);
    void applyBatchNorm(Tensor4D& tensor, float epsilon);

    Tensor4D convolution_2d(const Tensor4D& input,const Tensor4D& kernel, int stride, int padding);
    Tensor4D convolution_2d(const Tensor4D& input,const std::string& filename, int stride, int padding, bool bias);
    
};


#endif //CONV2D_H