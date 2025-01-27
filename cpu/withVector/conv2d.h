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
    
};


#endif // TENSOR4D_H