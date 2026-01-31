#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>
#include <omp.h>
#include <assert.h>
#include <random>
#include "cnpy/cnpy.h"

class WeightCache {
public:
    std::map<std::string, std::vector<float>> data;

    void preload(const std::string& name, bool is_bn = false) {
        if (!is_bn) {
            data[name + "_weights"] = load("../../pretrainedKernels/" + name + "_weights.npy");
            data[name + "_bias"]    = load("../../pretrainedKernels/" + name + "_bias.npy");
        } else {
            data[name + "_gamma"] = load("../../pretrainedKernels/" + name + "_gamma.npy");
            data[name + "_beta"]  = load("../../pretrainedKernels/" + name + "_beta.npy");
            data[name + "_rmean"] = load("../../pretrainedKernels/" + name + "_rmean.npy");
            data[name + "_rvar"]  = load("../../pretrainedKernels/" + name + "_rvar.npy");
        }
    }

private:
    std::vector<float> load(const std::string& path) {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        float* raw = arr.data<float>();
        return std::vector<float>(raw, raw + arr.num_vals);
    }
};

WeightCache gCache;

void bn_relu_fused_opt(std::vector<float>& input, const std::string& bn_name, int N, int C, int H, int W) {
    const auto& gamma = gCache.data.at(bn_name + "_gamma");
    const auto& beta  = gCache.data.at(bn_name + "_beta");
    const auto& mean  = gCache.data.at(bn_name + "_rmean");
    const auto& var   = gCache.data.at(bn_name + "_rvar");

    int spatial = H * W;
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float scale = gamma[c] / std::sqrt(var[c] + 1e-5f);
            float shift = beta[c] - (mean[c] * scale);
            float* ptr = &input[(n * C + c) * spatial];

            #pragma omp simd
            for (int i = 0; i < spatial; ++i) {
                float val = ptr[i] * scale + shift;
                ptr[i] = (val > 0.0f) ? val : 0.0f;
            }
        }
    }
}

std::vector<float> conv2d_cpu(const std::vector<float>& input,
								const std::string& kernelname,
                              int N, int C, int H, int W, int OutC,
                              int Kh, int Kw, int stride=1, int padding=1) {
	auto start_time = std::chrono::high_resolution_clock::now();
    // Calculate output dimensions
    int outH = (H + 2 * padding - Kh) / stride + 1;
    int outW = (W + 2 * padding - Kw) / stride + 1;
	
	const auto& kernel = gCache.data.at(kernelname + "_weights");
    const auto& bias   = gCache.data.at(kernelname + "_bias");
	
    // Validate output dimensions
    if (outH <= 0 || outW <= 0) {
        throw std::invalid_argument("Output dimensions must be positive");
    }

    // Validate input and kernel sizes
    if (input.size() != static_cast<size_t>(N * C * H * W)) {
        throw std::invalid_argument("Input tensor size does not match expected dimensions");
    }
    if (kernel.size() != (OutC * C * Kh * Kw)) {
		std::cout<<"Kernel size: "<<kernel.size()<<" Expected size: "<<(OutC * C * Kh * Kw)<<std::endl;
		std::cout<<"OutC: "<<OutC<<" C: "<<C<<" Kh: "<<Kh<<" Kw: "<<Kw<<std::endl;
        throw std::invalid_argument("Kernel tensor size does not match expected dimensions");
    }

    // Initialize output vector
    std::vector<float> output(N * OutC * outH * outW);

    // Perform 4D convolution
    for (int n = 0; n < N; ++n) {
        for (int out_c = 0; out_c < OutC; ++out_c) {
            for (int out_h = 0; out_h < outH; ++out_h) {
                for (int out_w = 0; out_w < outW; ++out_w) {
                    float sum = 0.0f;
                    // Sum over input channels and kernel dimensions
                    for (int c = 0; c < C; ++c) {
                        for (int kh = 0; kh < Kh; ++kh) {
                            for (int kw = 0; kw < Kw; ++kw) {
                                // Compute corresponding input position
                                int in_h = out_h * stride + kh - padding;
                                int in_w = out_w * stride + kw - padding;
                                if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                                    // Calculate flattened indices
                                    int input_idx = n * (C * H * W) + c * (H * W) + in_h * W + in_w;
                                    int kernel_idx = out_c * (C * Kh * Kw) + c * (Kh * Kw) + kh * Kw + kw;
                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                                // Out-of-bounds positions are skipped (zero padding)
                            }
                        }
                    }
					sum += bias[out_c];
                    // Calculate output index in row-major order
                    int output_idx = n * (OutC * outH * outW) + out_c * (outH * outW) + out_h * outW + out_w;
                    output[output_idx] = sum;
                }
            }
        }
    }
	auto end_time = std::chrono::high_resolution_clock::now();
	std::cout<<"CONV TIME UNOPTIMIZED - "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()<<" ms\n";
    return output;
}


void conv2d_opt(const std::vector<float>& input, const std::string& name,
                std::vector<float>& output, int N, int C, int H, int W, 
                int OutC, int Kh, int Kw, int stride, int padding) {
    auto start_time = std::chrono::high_resolution_clock::now();
    const auto& kernel = gCache.data.at(name + "_weights");
    const auto& bias   = gCache.data.at(name + "_bias");
    int outH = (H + 2 * padding - Kh) / stride + 1;
    int outW = (W + 2 * padding - Kw) / stride + 1;
    output.assign(N * OutC * outH * outW, 0.0f);

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OutC; ++oc) {
            float b = bias[oc];
            float* out_base = &output[(n * OutC + oc) * outH * outW];
            const float* k_base = &kernel[oc * C * Kh * Kw];

            for (int c = 0; c < C; ++c) {
                const float* in_chan = &input[(n * C + c) * H * W];
                const float* k_chan = &k_base[c * Kh * Kw];

                for (int oh = 0; oh < outH; ++oh) {
                    for (int ow = 0; ow < outW; ++ow) {
                        float sum = 0.0f;
                        for (int kh = 0; kh < Kh; ++kh) {
                            int ih = oh * stride + kh - padding;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < Kw; ++kw) {
                                int iw = ow * stride + kw - padding;
                                if (iw >= 0 && iw < W) {
                                    sum += in_chan[ih * W + iw] * k_chan[kh * Kw + kw];
                                }
                            }
                        }
                        out_base[oh * outW + ow] += sum;
                    }
                }
            }
            for(int i=0; i<outH*outW; ++i) out_base[i] += b;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout<<"CONV TIME OPTIMIZED - "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()<<" ms\n";
}

void upsample_cpu_opt(const std::vector<float>& input, std::vector<float>& output, 
                      int N, int C, int H, int W, int newH, int newW) {
    float scaleH = (newH > 1) ? static_cast<float>(H - 1) / (newH - 1) : 0.0f;
    float scaleW = (newW > 1) ? static_cast<float>(W - 1) / (newW - 1) : 0.0f;
    output.resize(N * C * newH * newW);

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* in_ptr = &input[(n * C + c) * H * W];
            float* out_ptr = &output[(n * C + c) * newH * newW];

            for (int h = 0; h < newH; ++h) {
                float srcH = h * scaleH;
                int h1 = static_cast<int>(srcH);
                int h2 = std::min(h1 + 1, H - 1);
                float dH = srcH - h1;

                for (int w = 0; w < newW; ++w) {
                    float srcW = w * scaleW;
                    int w1 = static_cast<int>(srcW);
                    int w2 = std::min(w1 + 1, W - 1);
                    float dW = srcW - w1;

                    out_ptr[h * newW + w] = 
                        (1-dH)*(1-dW)*in_ptr[h1*W + w1] + (1-dH)*dW*in_ptr[h1*W + w2] +
                        dH*(1-dW)*in_ptr[h2*W + w1] + dH*dW*in_ptr[h2*W + w2];
                }
            }
        }
    }
}

void concat_cpu_opt(const std::vector<float>& t1, const std::vector<float>& t2, 
                    std::vector<float>& out, int N, int C1, int C2, int H, int W) {
    int C3 = C1 + C2;
    int s1 = C1 * H * W;
    int s2 = C2 * H * W;
    int s3 = C3 * H * W;
    out.resize(N * C3 * H * W);

    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        std::copy(t1.begin() + n * s1, t1.begin() + (n + 1) * s1, out.begin() + n * s3);
        std::copy(t2.begin() + n * s2, t2.begin() + (n + 1) * s2, out.begin() + n * s3 + s1);
    }
}

std::vector<float> doubleConvolution_opt(std::vector<float>& input, const std::string& k1, 
                                         const std::string& k2, const std::string& bn1, 
                                         const std::string& bn2, int N, int C, int H, int W, 
                                         int outC1, int outC2) {
    std::vector<float> buffer1;
    conv2d_opt(input, k1, buffer1, N, C, H, W, outC1, 3, 3, 1, 1);
    bn_relu_fused_opt(buffer1, bn1, N, outC1, H, W);
    
    std::vector<float> buffer2;
    conv2d_opt(buffer1, k2, buffer2, N, outC1, H, W, outC2, 3, 3, 1, 1);
    bn_relu_fused_opt(buffer2, bn2, N, outC2, H, W);
    
    return buffer2;
}

std::vector<float> upward_opt(std::vector<float>& input, const std::string& k1, const std::string& k2,
                              const std::string& bn1, const std::string& bn2, std::vector<float>& skip, 
                              int N, int C, int H, int W, int skipC, int outC1, int outC2) {
    std::vector<float> up;
    upsample_cpu_opt(input, up, N, C, H, W, H*2, W*2);
    
    std::vector<float> concated;
    concat_cpu_opt(up, skip, concated, N, C, skipC, H*2, W*2);
    
    return doubleConvolution_opt(concated, k1, k2, bn1, bn2, N, C+skipC, H*2, W*2, outC1, outC2);
}

int main() {
    std::cout << "Loading weights..." << std::endl;
    gCache.preload("conv2d"); gCache.preload("conv2d_1");
    gCache.preload("batch_normalization", true); gCache.preload("batch_normalization_1", true);

    int N=1, C=3, H=224, W=224;
    std::vector<float> input(N * C * H * W, 0.5f);
    auto out0 = conv2d_cpu(input, "conv2d", N, C, H, W, 32, 3,3);
    auto s = std::chrono::high_resolution_clock::now();
    auto out = doubleConvolution_opt(input, "conv2d", "conv2d_1", "batch_normalization", "batch_normalization_1", N, C, H, W, 32, 32);
    auto e = std::chrono::high_resolution_clock::now();
    std::cout << "Optimized block time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e-s).count() << " ms" << std::endl;
    return 0;
}