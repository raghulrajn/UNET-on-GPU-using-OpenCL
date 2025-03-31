/**********************************************************************************************************************
* File name: conv2d.cl
* 
* This program implements Conv2D and other supporting functions to implement UNET on the input image as GPU kernels in OpenCL.
* Project Team:
* Raghul Raj Navaneethakrishnan (3703553)
* Vinay Vasant Thute (3701607)
* Mandar Bhanap (3702680)
***********************************************************************************************************************
*/
#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

/*
Indexing of a row-major 2D matrix : 

eg : 3x3 matrix
10 11 12 13
14 15 16 17
18 19 20 21
Flattened version = [10 11 12 13 14 15 16 17 18 19 20 21]
Flattened index =     0  1  2  3  4  5  6  7  8  9 10 11
Rows = 3, Columns = 4
Element at (i, j) = i * Columns + j where (i = row number, j = column number)
for element at (1, 2) = 1 * 4 + 2 = 6 ; Flattened index = 6; element = 16
Element at (2, 3) = 2 * 4 + 3 = 11 ; Flattened index = 11; element = 19

Similar indexing methodology is expanded to 4D tensor of dimensions (N, C, H, W)
index = n * C * H * W + c * H * W + h * W + w
where n = batch index, c = channel index, h = height index, w = width index
N = batch size, C = number of channels, H = height, W = width
*/

__kernel void conv2d(__global float* inputTensor,
                    __global float* kernelTensor,
                    __global float* outputTensor,
                    int N, int C, int H, int W, int OutC,
                    int Kh, int Kw, int stride, int padding,__global float* biasTensor) {
    // Compute output dimensions
    int outH = (H + 2 * padding - Kh) / stride + 1;
    int outW = (W + 2 * padding - Kw) / stride + 1;

    // Get global indices
    int out_c = get_global_id(0);
    int out_h = get_global_id(1);
    int out_w = get_global_id(2);

    // Bounds check
    if (out_c >= OutC || out_h >= outH || out_w >= outW) return;

    float sum = 0.0f;

    // Convolution computation
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < Kh; ++kh) {
            for (int kw = 0; kw < Kw; ++kw) {
                int in_h = out_h * stride + kh - padding;
                int in_w = out_w * stride + kw - padding;

                if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                    //If total number of elements exceed the integer limit , type cast idx to long
                    int inputIdx  = c * H * W + in_h * W + in_w;
                    int kernelIdx = out_c * C * Kh * Kw + c * Kh * Kw + kh * Kw + kw;

                    sum += inputTensor[inputIdx] * kernelTensor[kernelIdx];
                }
            }
        }
    }
    sum += biasTensor[out_c];
    long outputIdx = (long)out_c * outH * outW + (long)out_h * outW + (long)out_w;
    outputTensor[outputIdx] = sum;
}

__kernel void relu_activation(__global float* input,
                               __global float* output,
                               int totalsize) {
    
    int idx = get_global_id(0);
    if (idx < totalsize) {
        //set negative values to zero, leave positive values unchanged
        output[idx] = fmax(0.0f, input[idx]);
    }
}

__kernel void maxpool(__global const float* input,  // Input tensor (N, C, H, W)
                    __global float* output,       // Output tensor (N, C, OutH, OutW)
                    const int N,                  // Batch size
                    const int C,                  // Number of channels
                    const int H,                  // Input height
                    const int W,                  // Input width
                    const int pool_size,          // Pool size (e.g., 2 for 2x2)
                    const int stride          // Stride (e.g., 2 for 2x2)
                    ) {

    int out_w = get_global_id(2);
    int out_h = get_global_id(1); // Output height index out_height= H/stride
    int c = get_global_id(0);
    
    // Calculate the starting index of the pooling region(2x2) in the input
    int start_h = out_h * stride;
    int start_w = out_w * stride;
    float max_val = -INFINITY;

    // Find max in the window of size pool_size x pool_size
    for (int ph = 0; ph < pool_size; ++ph) {
        for (int pw = 0; pw < pool_size; ++pw) {
            int in_h = start_h + ph;
            int in_w = start_w + pw;
            if (in_h < H && in_w < W) {
                int inputIndex = c * H * W + in_h * W + in_w;
                max_val = fmax(max_val, input[inputIndex]); //max value in the 2x2 window is set to output
            }
        }
    }

    int outputIndex =  c * (H / stride) * (W / stride) + out_h * (W / stride) + out_w;
    output[outputIndex] = max_val;
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
                         const int W){
    int c = get_global_id(0);  // Channel index
    int h = get_global_id(1);  // Height index
    int w = get_global_id(2);  // Width index

    float inv_std = 1.0f / sqrt(variance_input[c] + 1e-5f); // 1+e-5f is added to prevent division by zero
    //normVal = gamma * (inputVal - mean) / sqrt(variance + epsilon) + beta
    int idx =  c * H * W + h * W + w;
    output[idx] = gamma[c] * ((input[idx] - mean_input[c]) * inv_std) + beta[c];
}

__kernel void concatenate_tensors(__global const float* tensor1, __global const float* tensor2, __global float* output,
                                    int N1, int C1, int H1,  int W1, int C2) {

    int n = get_global_id(0);  // Batch index (N1)
    int c = get_global_id(1);  // concatenated channel index (C3 = C1 + C2)
    int h = get_global_id(2) / W1;  // Height index (H1)
    int w = get_global_id(2) % W1;  // Width index (W1)

    if (c < C1) {
        // Copy data from tensor1 (first C1 channels)
        output[((n * C1 + c) * H1 + h) * W1 + w] = tensor1[((n * C1 + c) * H1 + h) * W1 + w];
    } else {
        // Copy data from tensor2 (last C2 channels)
        output[((n * (C1 + C2) + c) * H1 + h) * W1 + w] = tensor2[((n * C2 + (c - C1)) * H1 + h) * W1 + w];
    }
}

__kernel void upsample_(__global const float* input,  __global float* output,   
                        int N,  int C,  int H,  int W, int newH,  int newW) {

    int c = get_global_id(0); // Channel index
    int h = get_global_id(1); // outputHeight index
    int w = get_global_id(2); // outputWidth index
    if (c >= C || h >= newH || w>= newW) return;

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

    // Compute indices in flattened buffer of inputTensor
    int idx11 = (c * H + h1) * W + w1; //topLeft corner
    int idx12 = (c * H + h1) * W + w2; //topRight corner
    int idx21 = (c * H + h2) * W + w1; //bottomLeft corner
    int idx22 = (c * H + h2) * W + w2; //bottomRight corner

    // Bilinear interpolation
    float value = (1 - dH) * (1 - dW) * input[idx11] +
                  (1 - dH) * dW * input[idx12] +
                  dH * (1 - dW) * input[idx21] +
                  dH * dW * input[idx22];

    int output_idx = (c * newH + h) * newW + w;
    output[output_idx] = value;
}

__kernel void sigmoid(__global const float* input, __global float* output, int totalsize) {
    int idx = get_global_id(0); // Total size of tensor (N*C*H*W)
    if (idx < totalsize) 
    {
        //σ(x) = 1/(1+exp(-x))
        output[idx] = 1.0f / (1.0f + exp(-input[idx]));
    }
}

__kernel void conv2dOptimizedwbias(
    __global float* inputTensor,    // Input tensor [N, C, H, W]
    __global float* kernelTensor,   // Kernel tensor [outC, C, Kh, Kw]
    __global float* outputTensor,   // Output tensor [N, outC, outH, outW]
    __global float* biasTensor,     // Bias tensor [outC]
    int N,                          // Batch size
    int C,                          // Input channels
    int H,                          // Input height
    int W,                          // Input width
    int outC,                       // Output channels
    int Kh,                         // Kernel height
    int Kw,                         // Kernel width
    int stride,                     // Stride
    int padding,                    // Padding
    __local float* input_tile)      // Local memory for input tile [C, tileH + Kh - 1, tileW + Kw - 1]
{
    // Get global indices for output element
    int out_c = get_global_id(0);   // Output channel index
    int out_h = get_global_id(1);   // Output height index
    int out_w = get_global_id(2);   // Output width index

    // Check if the work item is within bounds
    if (out_c >= outC || out_h >= H || out_w >= W) return;

    // Get local indices within the work group
    int local_h = get_local_id(1);  // Local height index
    int local_w = get_local_id(2);  // Local width index

    // Get tile dimensions from local work size
    int tileH = get_local_size(1);  // Tile height
    int tileW = get_local_size(2);  // Tile width

    // Compute the starting position of the output tile
    int out_h_start = get_group_id(1) * tileH;
    int out_w_start = get_group_id(2) * tileW;

    // Compute input tile dimensions (including halo region)
    int local_input_h = tileH + Kh - 1;
    int local_input_w = tileW + Kw - 1;

    // Compute starting input indices for the tile
    int in_h_start = out_h_start - padding;
    int in_w_start = out_w_start - padding;

    // Load input tile into local memory
    int num_elements = C * local_input_h * local_input_w;
    int local_id = local_h * tileW + local_w;
    int work_group_size = tileH * tileW;

    for (int i = local_id; i < num_elements; i += work_group_size) {
        int c = i / (local_input_h * local_input_w);
        int lh = (i % (local_input_h * local_input_w)) / local_input_w;
        int lw = i % local_input_w;
        int in_h = in_h_start + lh;
        int in_w = in_w_start + lw;
        int input_idx = 0 * C * H * W + c * H * W + in_h * W + in_w; // N=1, so n=0
        if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
            input_tile[i] = inputTensor[input_idx];
        } else {
            input_tile[i] = 0.0f; // Padding with zeros
        }
    }

    // Synchronize to ensure all local memory is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute the convolution for this output element
    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < Kh; kh++) {
            int in_h_local = local_h + kh;
            for (int kw = 0; kw < Kw; kw++) {
                int in_w_local = local_w + kw;
                int tile_idx = c * local_input_h * local_input_w + in_h_local * local_input_w + in_w_local;
                float input_val = input_tile[tile_idx];
                int kernel_idx = out_c * C * Kh * Kw + c * Kh * Kw + kh * Kw + kw;
                sum += input_val * kernelTensor[kernel_idx];
            }
        }
    }
    // Add bias
    sum += biasTensor[out_c];
    int output_idx = 0 * outC * H * W + out_c * H * W + out_h * W + out_w; // N=1, so n=0
    outputTensor[output_idx] = sum;
}
