#include "tensor4D.h"
#include "conv2d.h"
#include <cassert>


/**
 * @brief Applies Relu activation to each channel of the input tensor.
 * 
 * The ReLU activation function replaces all negative values in the tensor with zero
 * while leaving all positive values unchanged.
 * 
 * @param tensor The input 4D tensor (N x C x H x W), where:
 *        - `N`: Number of samples in the batch.
 *        - `C`: Number of channels.
 *        - `H`: Height of each sample.
 *        - `W`: Width of each sample.
 * 
 * @note This function modifies the input `tensor` in place.
 */
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

/**
 * @brief Applies Max pooling to each channel of the input tensor.
 * 
 * This function performs Max pooling on a 4D tensor (e.g., a batch of images or feature maps).
 * It computes the maximum value of each poolHeight x poolWidth(eg. 2x2) each channel across all samples
 * and place it in the tensor thus reducing the size of the tensor.
 * 
 * @param tensor The input 4D tensor (N x C x H x W), where:
 *        - `N`: Number of samples in the batch.
 *        - `C`: Number of channels.
 *        - `H`: Height of each sample.
 *        - `W`: Width of each sample.
 * 
 * @param poolHeight Kernel height for the maxpooling. Usually 2
 * 
 * @param poolWidth Kernel Width for the maxpooling. Usually 2
 * 
 * @param stride stride length while apply max pooling
 * @note This function modifies the input `tensor` in place.
 */
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

/**
 * @brief Applies Batch Normalization to each channel of the input tensor.
 * 
 * This function performs Batch Normalization on a 4D tensor (e.g., a batch of images or feature maps).
 * It computes the mean and variance for each channel across all samples and spatial locations,
 * and normalizes the tensor by subtracting the mean and dividing by the square root of the variance.
 * An epsilon value is used to prevent division by zero during the normalization.
 * 
 * @param tensor The input 4D tensor (N x C x H x W), where:
 *        - `N`: Number of samples in the batch.
 *        - `C`: Number of channels.
 *        - `H`: Height of each sample.
 *        - `W`: Width of each sample.
 * 
 * @param epsilon A small constant to avoid division by zero during normalization. Default is 1e-5.
 * 
 * @note This function modifies the input `tensor` in place.
 */
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

Tensor4D Conv2d::convolution_2d(Tensor4D& input,
                     Tensor4D& kernel,
                    int stride = 1, int padding = 0) {
    int N = input.getN();
    int C = input.getC();
    int H = input.getH();
    int W = input.getW();
    if(padding>0){
        input.addPadding(padding, padding);
    }

    int out_channels = kernel.getN();
    int in_channels = kernel.getC();
    int k_height = kernel.getH();
    int k_width = kernel.getW();

    assert(("Input tensor channels and Filter channels are not matching", C==in_channels));

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

/**
 * @brief Performs a 2D convolution operation on a 4D input tensor using the specified kernel and parameters.
 * 
 * This function applies a 2D convolution to a batch of images (or feature maps) stored in a 4D tensor.
 * The convolution is performed using a kernel loaded from the .npy, with adjustable stride,
 * padding, and an option to include bias.
 * 
 * @param input The input tensor, represented as a 4D tensor (batch_size x channels x height x width).
 *        - `N`: Number of samples in the batch.
 *        - `C`: Number of channels.
 *        - `H`: Height of each sample.
 *        - `W`: Width of each sample.
 * 
 * @param filename The path to the file that contains the convolution kernel weights(.npy format).
 * 
 * @param stride The stride value for the convolution operation. Default is 1.
 *        
 * @param padding The padding value to be applied to the input before the convolution. Default is 0.
 *       
 * @param bias A boolean flag indicating whether to include a bias term in the convolution. Default is false.
 *        If set to true, a bias term will be added to the output after applying the convolution.
 * 
 * @return A new Tensor4D containing the result of the 2D convolution.
 *         The resulting tensor has the following dimensions:
 *        - `N`: Number of samples in the batch.
 *        - `C`: Number of Kernels.
 *        - `H`: Height of each sample.
 *        - `W`: Width of each sample.
 * 
 * @note The dimensions of the output tensor are computed as follows:
 *        - `output_height = (input_height + 2 * padding - kernel_height) / stride + 1`
 *        - `output_width = (input_width + 2 * padding - kernel_width) / stride + 1`
 *        If these values are not integers, the dimensions will be adjusted accordingly, or an error will be thrown.
 * 
 * @throws std::invalid_argument If the kernel file cannot be loaded or if the input tensor dimensions are incompatible
 *         with the kernel size and stride.
 */
Tensor4D Conv2d::convolution_2d(Tensor4D& input,std::string &filename,
                    int stride = 1, int padding = 0, bool bias=false) {
    int N = input.getN();
    int C = input.getC();
    int H = input.getH();
    int W = input.getW();

    if(padding>0){
        input.addPadding(padding, padding);
    }

    Tensor4D kernel = Tensor4D::fromNPY(filename+"_weight.npy");

    std::cout<< "Filter dimensions: ";
    kernel.printDimensions();
    std::cout << std::endl;
    
    int out_channels = kernel.getN();
    int in_channels = kernel.getC();
    int k_height = kernel.getH();
    int k_width = kernel.getW();

    assert(("Input tensor channels and Filter channels are not matching", C==in_channels));

    int out_height = (H - k_height + 2 * padding) / stride + 1;
    int out_width = (W - k_width + 2 * padding) / stride + 1;

    // Resize the output tensor
    Tensor4D output(N, out_channels, out_height, out_width);

    for (int n = 0; n < N; ++n) {
        for (int kn = 0; kn < out_channels; ++kn) { // Kernel output channels
            for (int h = 0; h < out_height; ++h) {
                for (int w = 0; w < out_width; ++w) {
                    float sum = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int kh = 0; kh < k_height; ++kh) {
                            for (int kw = 0; kw < k_width; ++kw) {
                                sum += input.at(n,c,h + kh,w + kw) * 
                                        kernel.at(kn,c,kh,kw);
                            }
                        }
                    }
                    output.at(n,kn,h,w) = sum;
                }
            }
        }
    }
    if (bias){
        std::string filepath = filename+"_bias.npy";
        Tensor4D bias = Tensor4D::fromNPY(filepath);
        output.add(bias);
    }
    return output;
}