#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

__kernel void conv2d(__global float* input1,
                     __global float* kernel1,
                     __global float* output1,
                     int N, int C, int H, int W, int OutC,
                     int Kh, int Kw, int stride, int padding)
{
    // Global indices for input channel (c), output height (out_h), and output width (out_w)
    int c = get_global_id(0); // Input channel index
    int out_h = get_global_id(1); // Output height index
    int out_w = get_global_id(2); // Output width index

    // Calculate output height and width based on padding, stride, kernel size
    int outH = (H + 2 * padding - Kh) / stride + 1;
    int outW = (W + 2 * padding - Kw) / stride + 1;

    // Ensure indices are within bounds for the output tensor
    if (c < C && out_h < outH && out_w < outW) {
        // Loop over batch and output channels
        for (int n = 0; n < N; ++n) {
            for (int out_c = 0; out_c < OutC; ++out_c) {
                float sum = 0.0f;

                // Loop over kernel dimensions (Kh, Kw) and input channels (C)
                for (int kh = 0; kh < Kh; ++kh) {
                    for (int kw = 0; kw < Kw; ++kw) {
                        // Calculate the input indices with padding
                        int in_h = out_h * stride + kh - padding;
                        int in_w = out_w * stride + kw - padding;

                        // Check if the input indices are within bounds
                        if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                            // Calculate flattened indices for input and kernel tensors
                            int inputIdx = n * C * H * W + c * H * W + in_h * W + in_w;
                            int kernelIdx = out_c * C * Kh * Kw + c * Kh * Kw + kh * Kw + kw;

                            // Accumulate the convolution result
                            sum += input1[inputIdx] * kernel1[kernelIdx];
                        }
                    }
                }

                // Correct flattened index for output tensor (1D index across all N, OutC, outH, outW)
                int outputIdx = n * OutC * outH * outW + out_c * outH * outW + out_h * outW + out_w;

                // Store the result in the output buffer
                output1[outputIdx] = sum;
            }
        }
    }
}


__kernel void relu_activation(__global float* input,
                               __global float* output,
                               const int totalsize)
{
    // Global index for accessing the input tensor
    int idx = get_global_id(0);
    // Ensure index is within bounds of the input tensor size
    if (idx < totalsize) {
        // Apply ReLU: set negative values to zero, leave positive values unchanged
        output[idx] = (input[idx] > 0) ? input[idx] : 0;
    }
}

__kernel void maxpool(
    __global const float* input,  // Input tensor (N, C, H, W)
    __global float* output,       // Output tensor (N, C, OutH, OutW)
    const int N,                  // Batch size
    const int C,                  // Number of channels
    const int H,                  // Input height
    const int W,                  // Input width
    const int pool_size,          // Pool size (e.g., 2 for 2x2)
    const int stride              // Stride
) {
    // Flattened global index
    int global_id = get_global_id(0);

    // Calculate n, c, out_h, out_w based on global_id
    int out_w = global_id % (W / stride);  // Output width index
    int out_h = (global_id / (W / stride)) % (H / stride);  // Output height index
    int c = (global_id / (W / stride) / (H / stride)) % C; // Channel index
    int n = global_id / (C * (H / stride) * (W / stride));  // Batch index

    // Calculate the starting index of the pooling region in the input
    int start_h = out_h * stride;
    int start_w = out_w * stride;

    // Ensure we don't go out of bounds
    float max_val = -INFINITY;

    // Pool over the window of size pool_size x pool_size
    for (int ph = 0; ph < pool_size; ++ph) {
        for (int pw = 0; pw < pool_size; ++pw) {
            int in_h = start_h + ph;
            int in_w = start_w + pw;
            if (in_h < H && in_w < W) {
                int inputIndex = n * C * H * W + c * H * W + in_h * W + in_w;
                max_val = fmax(max_val, input[inputIndex]);
            }
        }
    }

    // Write the maximum value to the output tensor
    int outputIndex = n * C * (H / stride) * (W / stride) + c * (H / stride) * (W / stride) + out_h * (W / stride) + out_w;
    output[outputIndex] = max_val;
}

__kernel void batchMean(__global const float* input,
                                __global float* mean_output,
                                const int N,  // Batch size
                                const int C,  // Number of channels
                                const int H,  // Input height
                                const int W)  // Input width
{
    int c = get_global_id(0);  // Channel index

    // Ensure we don't exceed the number of channels
    if (c < C) {
        float mean = 0.0f;
        int channel_size = N * H * W;

        // Calculate mean for the current channel (sum across all batches and spatial dimensions)
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = n * C * H * W + c * H * W + h * W + w;
                    mean += input[idx];
                }
            }
        }

        // Store the mean for the current channel
        mean_output[c] = mean / channel_size;
    }
}

__kernel void batchVariance(__global const float* input,
                                    __global const float* mean_input,
                                    __global float* variance_output,
                                    const int N,  // Batch size
                                    const int C,  // Number of channels
                                    const int H,  // Input height
                                    const int W)  // Input width
{
    int c = get_global_id(0);  // Channel index

    // Ensure we don't exceed the number of channels
    if (c < C) {
        float variance = 0.0f;
        int channel_size = N * H * W;

        // Calculate variance for the current channel (sum of squared differences from mean)
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = n * C * H * W + c * H * W + h * W + w;
                    float diff = input[idx] - mean_input[c];
                    variance += diff * diff;
                }
            }
        }
        // Store the variance for the current channel
        variance_output[c] = variance / channel_size;
    }
}

__kernel void batch_norm(__global const float* input,
                         __global const float* gamma,
                         __global const float* beta,
                         __global const float* mean_input,
                         __global const float* variance_input,
                         __global float* output,
                         const int N,  // Batch size
                         const int C,  // Number of channels
                         const int H,  // Input height
                         const int W)  // Input width
{
    int c = get_global_id(0);  // Channel index
    int h = get_global_id(1);  // Height index
    int w = get_global_id(2);  // Width index

    // Ensure we don't exceed the input dimensions
    if (c < C && h < H && w < W) {
        // Get the channel's mean and variance
        float mean = mean_input[c];
        float variance = variance_input[c];
        float inv_std = 1.0f / sqrt(variance + 1e-5f); // epsilon for numerical stability

        // Barrier to synchronize before starting normalization
        barrier(CLK_GLOBAL_MEM_FENCE);

        // Now apply Batch Normalization to each element for the current (n, c, h, w)
        for (int n = 0; n < N; ++n) {
            int idx = n * C * H * W + c * H * W + h * W + w;
            output[idx] = gamma[c] * ((input[idx] - mean) * inv_std) + beta[c];
        }
        // Barrier after processing all batch elements
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}


