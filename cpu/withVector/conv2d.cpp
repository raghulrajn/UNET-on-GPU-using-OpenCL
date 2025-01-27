#include "tensor4D.h"
#include "conv2d.h"


// Apply ReLU activation function
void Conv2d::applyReLU(Tensor4D& tensor) {
    for (int n = 0; n < tensor.getN(); ++n) {
        for (int c = 0; c < tensor.getC(); ++c) {
            for (int h = 0; h < tensor.getH(); ++h) {
                for (int w = 0; w < tensor.getW(); ++w) {
                    if (tensor.at(n, c, h, w) < 0) {
                        tensor.at(n, c, h, w) = 0;
                    }
                }
            }
        }
    }
}

    // Apply Max Pooling
void Conv2d::applyMaxPool(Tensor4D& tensor, int poolHeight, int poolWidth, int stride) {
    int outputHeight = (tensor.getH() - poolHeight) / stride + 1;
    int outputWidth = (tensor.getW() - poolWidth) / stride + 1;

    Tensor4D pooledTensor(tensor.getN(), tensor.getC(), outputHeight, outputWidth);

    for (int n = 0; n < tensor.getN(); ++n) {
        for (int c = 0; c < tensor.getC(); ++c) {
            for (int h = 0; h < outputHeight; ++h) {
                for (int w = 0; w < outputWidth; ++w) {
                    float maxVal = -std::numeric_limits<float>::infinity();
                    for (int ph = 0; ph < poolHeight; ++ph) {
                        for (int pw = 0; pw < poolWidth; ++pw) {
                            int inputH = h * stride + ph;
                            int inputW = w * stride + pw;
                            maxVal = std::max(maxVal, tensor.at(n, c, inputH, inputW));
                        }
                    }
                    pooledTensor.at(n, c, h, w) = maxVal;
                }
            }
        }
    }

    // Update the tensor with the pooled tensor
    tensor = std::move(pooledTensor);
}

// Apply Batch Normalization
void Conv2d::applyBatchNorm(Tensor4D& tensor, float epsilon = 1e-5) {
    for (int c = 0; c < tensor.getC(); ++c) {
        // Calculate mean and variance for the current channel
        float mean = 0.0f;
        float variance = 0.0f;

        // Calculate mean
        for (int n = 0; n < tensor.getN(); ++n) {
            for (int h = 0; h < tensor.getH(); ++h) {
                for (int w = 0; w < tensor.getW(); ++w) {
                    mean += tensor.at(n, c, h, w);
                }
            }
        }
        mean /= (tensor.getN() * tensor.getH() * tensor.getW());

        // Calculate variance
        for (int n = 0; n < tensor.getN(); ++n) {
            for (int h = 0; h < tensor.getH(); ++h) {
                for (int w = 0; w < tensor.getW(); ++w) {
                    variance += std::pow(tensor.at(n, c, h, w) - mean, 2);
                }
            }
        }
        variance /= (tensor.getN() * tensor.getH() * tensor.getW());

        // Apply batch normalization (standardize)
        for (int n = 0; n < tensor.getN(); ++n) {
            for (int h = 0; h < tensor.getH(); ++h) {
                for (int w = 0; w < tensor.getW(); ++w) {
                    tensor.at(n, c, h, w) = (tensor.at(n, c, h, w) - mean) / std::sqrt(variance + epsilon);
                }
            }
        }
    }
}

Tensor4D Conv2d::convolution_2d(const Tensor4D& input,
                    const Tensor4D& kernel,
                    int stride = 1, int padding = 0) {
    int N = input.getN();
    int C = input.getC();
    int H = input.getH();
    int W = input.getW();

    int k_height = kernel.getH();
    int k_width = kernel.getW();

    int out_height = (H - k_height + 2 * padding) / stride + 1;
    int out_width = (W - k_width + 2 * padding) / stride + 1;

    // Resize the output tensor
    Tensor4D output(N, C, out_height, out_width);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < out_height; ++i) {
                for (int j = 0; j < out_width; ++j) {
                    float sum = 0.0f;

                    for (int ki = 0; ki < k_height; ++ki) {
                        for (int kj = 0; kj < k_width; ++kj) {
                            int h = i * stride + ki - padding;
                            int w = j * stride + kj - padding;

                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                sum += input.at(n, c, h, w) * kernel.at(0, 0, ki, kj);
                            }
                        }
                    }

                    output.at(n, c, i, j) = sum;
                }
            }
        }
    }
    return output;
}
