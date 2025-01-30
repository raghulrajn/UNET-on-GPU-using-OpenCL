#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

__kernel void conv2d(__global float* input1,
                     __global float* kernel1,
                     __global float* output1,
                     int N,int C,int H,int W,int OutC, int Kh,int Kw,int stride,int padding)
{
    // Global indices for N, output height, and output width
    int n = get_global_id(0); // Batch index
    int out_h = get_global_id(1); // Output height index
    int out_w = get_global_id(2); // Output width index

    // Calculate output height and width based on padding, stride, kernel size
    int outH = (H + 2 * padding - Kh) / stride + 1;
    int outW = (W + 2 * padding - Kw) / stride + 1;

   // Ensure indices are within bounds for the output tensor
    if (n < N && out_h < outH && out_w < outW) {
        // Loop over output channels
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
                        // Loop over input channels (C)
                        for (int c = 0; c < C; ++c) {
                            int inputIdx = n * C * H * W + c * H * W + in_h * W + in_w;
                            int kernelIdx = out_c * C * Kh * Kw + c * Kh * Kw + kh * Kw + kw;
                            sum += input1[inputIdx] * kernel1[kernelIdx];
                        }
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
