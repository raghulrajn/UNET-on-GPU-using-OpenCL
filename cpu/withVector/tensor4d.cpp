#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <chrono>
#include "timer.h"

class Tensor4D {
private:
    int N, C, H, W; // Dimensions
    std::vector<std::vector<std::vector<std::vector<float>>>> data;

    // Utility function to check dimensions match
    void checkDimensions(const Tensor4D& other) const {
        if (N != other.N || C != other.C || H != other.H || W != other.W) {
            throw std::invalid_argument("Tensor dimensions do not match for operation.");
        }
     }

public:
    // Constructor
    Tensor4D(int n, int c, int h, int w)
        : N(n), C(c), H(h), W(w),
          data(n, std::vector<std::vector<std::vector<float>>>(
                       c, std::vector<std::vector<float>>(
                               h, std::vector<float>(w)))) {}

    // Getter for dimensions
    int getN() const { return N; }
    int getC() const { return C; }
    int getH() const { return H; }
    int getW() const { return W; }

    // Access element (read/write)
 		float& at(int n, int c, int h, int w) {
        return data.at(n).at(c).at(h).at(w);
    }

    const float& at(int n, int c, int h, int w) const {
        return data.at(n).at(c).at(h).at(w);
    }

    // Print tensor dimensions
    void printDimensions() const {
        std::cout << "Tensor Dimensions: N=" << N
                  << ", C=" << C << ", H=" << H << ", W=" << W << std::endl;
    }
    
      void setRandomValues(float mean, float std) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, std);

        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        data[n][c][h][w] = dist(gen);
    }
    
    void setValue() {
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        data[n][c][h][w] = w;
    }

    // Scalar addition
    Tensor4D scalarAdd(float value) const {
        Tensor4D result(N, C, H, W);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        result.at(n, c, h, w) = this->at(n, c, h, w) + value;
        return result;
    }

    // Scalar subtraction
    Tensor4D scalarSub(float value) const {
        Tensor4D result(N, C, H, W);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        result.at(n, c, h, w) = this->at(n, c, h, w) - value;
        return result;
    }

    // Scalar multiplication
    Tensor4D scalarMul(float value) const {
        Tensor4D result(N, C, H, W);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        result.at(n, c, h, w) = this->at(n, c, h, w) * value;
        return result;
    }

    // Scalar division
    Tensor4D scalarDiv(float value) const {
        if (value == 0.0f) {
            throw std::invalid_argument("Division by zero is not allowed.");
        }
        Tensor4D result(N, C, H, W);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        result.at(n, c, h, w) = this->at(n, c, h, w) / value;
        return result;
    }

    // Tensor addition
    Tensor4D tensorAdd(const Tensor4D& other) const {
        checkDimensions(other);
        Tensor4D result(N, C, H, W);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        result.at(n, c, h, w) = this->at(n, c, h, w) + other.at(n, c, h, w);
        return result;
    }

    // Tensor subtraction
    Tensor4D tensorSub(const Tensor4D& other) const {
        checkDimensions(other);
        Tensor4D result(N, C, H, W);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        result.at(n, c, h, w) = this->at(n, c, h, w) - other.at(n, c, h, w);
        return result;
    }

    // Tensor multiplication (element-wise)
    Tensor4D tensorMul(const Tensor4D& other) const {
        checkDimensions(other);
        Tensor4D result(N, C, H, W);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        result.at(n, c, h, w) = this->at(n, c, h, w) * other.at(n, c, h, w);
        return result;
    }

    // Tensor division (element-wise)
    Tensor4D tensorDiv(const Tensor4D& other) const {
        checkDimensions(other);
        Tensor4D result(N, C, H, W);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w) {
                        if (other.at(n, c, h, w) == 0.0f) {
                            throw std::invalid_argument("Division by zero in tensor division.");
                        }
                        result.at(n, c, h, w) = this->at(n, c, h, w) / other.at(n, c, h, w);
                    }
        return result;
    }
 
    void printAsMatrix() const {
        for (int n = 0; n < N; ++n) {
            std::cout << "Batch " << n + 1 << ":\n";
            for (int c = 0; c < C; ++c) {
                std::cout << "  Channel " << c + 1 << ":\n";
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        std::cout << at(n, c, h, w) << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
            std::cout << "-----------------------------\n";
        }
    }
    
    void getMatrix(int N, int C) const {
         for (int h = 0; h < H; ++h) {
             for (int w = 0; w < W; ++w) {
                  std::cout << at(N, C, h, w) << " ";
                }
                std::cout << "\n";
            }
                std::cout << "\n";
            }
            
};

Tensor4D convolution_2d(const Tensor4D& input,
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


int main() {

    //Tensor4D result = tensor1.scalarAdd(3.0f);
    //result = tensor1.tensorAdd(tensor2);
    
    int N = 1, C = 3, H = 10, W = 10;
    int k_height = 3, k_width = 3;

    // Create input, kernel, and output tensors
    Tensor4D input(N, C, H, W);
    Tensor4D kernel(1, 128, k_height, k_width);
   
    // Set random values for input and kernel
    //input.setRandomValues(0.0f, 256.0f);
    //kernel.setRandomValues(0.0f, 1.0f);
		input.setValue();
		kernel.setValue();
    // Perform convolution
    Timer t1("conv");
    Tensor4D output = convolution_2d(input, kernel, 1, 0);
		t1.stop();
    output.getMatrix(0,0);

    return 0;
}

