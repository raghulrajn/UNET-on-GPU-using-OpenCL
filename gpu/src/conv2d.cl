#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

__kernel void conv2d(__global float* inputTensor,
                     __global float* kernelTensor,
                     __global float* outputTensor,
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
                            sum += inputTensor[inputIdx] * kernelTensor[kernelIdx];
                        }
                    }
                }

                // Correct flattened index for output tensor (1D index across all N, OutC, outH, outW)
                int outputIdx = n * OutC * outH * outW + out_c * outH * outW + out_h * outW + out_w;
                outputTensor[outputIdx] = sum;
            }
        }
    }
}


__kernel void relu_activation(__global float* input,
                               __global float* output,
                               const int totalsize)
{
    int idx = get_global_id(0);
    if (idx < totalsize) {
        //set negative values to zero, leave positive values unchanged
        output[idx] = fmax(0.0f, input[idx]);
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

    // Find max in the window of size pool_size x pool_size
    for (int ph = 0; ph < pool_size; ++ph) {
        for (int pw = 0; pw < pool_size; ++pw) {
            int in_h = start_h + ph;
            int in_w = start_w + pw;
            if (in_h < H && in_w < W) {
                int inputIndex = n * C * H * W + c * H * W + in_h * W + in_w;
                max_val = fmax(max_val, input[inputIndex]); //max value in the 2x2 window is set to output
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
        float inv_std = 1.0f / sqrt(variance_input[c] + 1e-5f); / 1+e-5f is added to prevent division by zero

        // Now apply Batch Normalization to each element for the current (n, c, h, w)
        for (int n = 0; n < N; ++n) {
            int idx = n * C * H * W + c * H * W + h * W + w;
            //normVal = gamma * (inputVal - mean) / sqrt(variance + epsilon) + beta
            output[idx] = gamma[c] * ((input[idx] - mean_input[c]) * inv_std) + beta[c];
        }
    }
}

__kernel void concatenate_tensors(__global const float* tensor1, __global const float* tensor2, __global float* output,
                                   const unsigned int N1, const unsigned int C1, const unsigned int H1, const unsigned int W1, const unsigned int C2) {

    // Global thread index
    int n = get_global_id(0);  // Batch index (N1)
    int c = get_global_id(1);  // Channel index (C3 = C1 + C2)
    int h = get_global_id(2) / W1;  // Height index (H1)
    int w = get_global_id(2) % W1;  // Width index (W1)

    // Ensure the thread indexes are within bounds
    if (n >= N1 || h >= H1 || w >= W1) {
        return;
    }
    if (c < C1) {
        // Copy data from tensor1 (first C1 channels)
        output[((n * C1 + c) * H1 + h) * W1 + w] = tensor1[((n * C1 + c) * H1 + h) * W1 + w];
    } else {
        // Copy data from tensor2 (last C2 channels)
        output[((n * (C1 + C2) + c) * H1 + h) * W1 + w] = tensor2[((n * C2 + (c - C1)) * H1 + h) * W1 + w];
    }
}

__kernel void extract_center(
    __global const float* input,    // Input tensor (flattened 1D)
    __global float* output,         // Output tensor (flattened 1D)
    const int N, const int C, const int H, const int W,
    const int newH, const int newW
) {
    int n = get_global_id(0); // Batch index
    int c = get_global_id(1); // Channel index
    int idx = get_global_id(2); // Flattened index for (h, w)

    if (n >= N || c >= C || idx >= newH * newW) return;

    int h = idx / newW; // Row index in new tensor
    int w = idx % newW; // Column index in new tensor

    // Compute start positions
    int startH = (H - newH) / 2;
    int startW = (W - newW) / 2;

    // Compute corresponding input index
    int input_h = startH + h;
    int input_w = startW + w;

    // Flattened index in input tensor
    int input_index = ((n * C + c) * H + input_h) * W + input_w;
    int output_index = ((n * C + c) * newH + h) * newW + w;

    // Copy value
    output[output_index] = input[input_index];
}

__kernel void upsample_(
    __global const float* input,  // Input tensor (flattened 1D)
    __global float* output,       // Output tensor (flattened 1D)
    const int N, const int C, const int H, const int W,
    const int newH, const int newW
) {
    int n = get_global_id(0); // Batch index
    int c = get_global_id(1); // Channel index
    int idx = get_global_id(2); // Flattened index for (h, w)

    if (n >= N || c >= C || idx >= newH * newW) return;

    int h = idx / newW; // Row index in new tensor
    int w = idx % newW; // Column index in new tensor

    // Compute source position in input
    float scaleH = (float)(H - 1) / (newH - 1); //H = 4, newH = 8, scaleH = 0.5
    float scaleW = (float)(W - 1) / (newW - 1); //W = 4, newW = 8, scaleW = 0.5
    
    float srcH = h * scaleH; //inputHeight is at scale factor times outputHeight
    float srcW = w * scaleW; //inputWidth is at scale factor times outputWidth
    // (h1, w1) topLeft corner
    // (h1, w2) topRight corner
    // (h2, w1) bottomLeft corner
    // (h2, w2) bottomRight corner
    int h1 = (int)srcH;
    int w1 = (int)srcW;
    int h2 = min(h1 + 1, H - 1);
    int w2 = min(w1 + 1, W - 1);

    float dH = srcH - h1; //distance between top left corner and inputHeight
    float dW = srcW - w1;

    //Compute indices in flattened buffer of inputTensor
    int idx11 = (c * H + h1) * W + w1; //topLeft corner
    int idx12 = (c * H + h1) * W + w2; //topRight corner
    int idx21 = (c * H + h2) * W + w1; //bottomLeft corner
    int idx22 = (c * H + h2) * W + w2; //bottomRight corner

    // Bilinear interpolation
    float value = (1 - dH) * (1 - dW) * input[idx11] +
                  (1 - dH) * dW * input[idx12] +
                  dH * (1 - dW) * input[idx21] +
                  dH * dW * input[idx22];

    int output_idx = ((n * C + c) * newH + h) * newW + w;
    output[output_idx] = value;
}
