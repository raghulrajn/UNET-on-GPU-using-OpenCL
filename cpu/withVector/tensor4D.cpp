#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <chrono>
#include "timer.h"
#include "tensor4D.h"

    // Constructor
Tensor4D::Tensor4D(int n, int c, int h, int w)
    : N(n), C(c), H(h), W(w),
        data(n, std::vector<std::vector<std::vector<float>>>(
                    c, std::vector<std::vector<float>>(
                            h, std::vector<float>(w)))) {}

// Utility function to check dimensions match
void Tensor4D::checkDimensions(const Tensor4D& other) const {
    if (N != other.N || C != other.C || H != other.H || W != other.W) {
        throw std::invalid_argument("Tensor dimensions do not match for operation.");
    }
}

// Getter for dimensions
int Tensor4D::getN() const { return N; }
int Tensor4D::getC() const { return C; }
int Tensor4D::getH() const { return H; }
int Tensor4D::getW() const { return W; }

// Access element (read/write)
float& Tensor4D::at(int n, int c, int h, int w) {
    return data.at(n).at(c).at(h).at(w);
}

const float& Tensor4D::at(int n, int c, int h, int w) const {
    return data.at(n).at(c).at(h).at(w);
}

// Print tensor dimensions
void Tensor4D::printDimensions() const {
    std::cout << "Tensor Dimensions: N=" << N
                << ", C=" << C << ", H=" << H << ", W=" << W << std::endl;
}

void Tensor4D::setRandomValues(float mean, float std) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);

    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    data[n][c][h][w] = dist(gen);
}

void Tensor4D::setValue() {
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    data[n][c][h][w] = w;
}

// Scalar addition
Tensor4D Tensor4D::add(float value) const {
    Tensor4D result(N, C, H, W);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    result.at(n, c, h, w) = this->at(n, c, h, w) + value;
    return result;
}

// Scalar subtraction
Tensor4D Tensor4D::subtract(float value) const {
    Tensor4D result(N, C, H, W);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    result.at(n, c, h, w) = this->at(n, c, h, w) - value;
    return result;
}

// Scalar multiplication
Tensor4D Tensor4D::multiply(float value) const {
    Tensor4D result(N, C, H, W);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    result.at(n, c, h, w) = this->at(n, c, h, w) * value;
    return result;
}

// Scalar division
Tensor4D Tensor4D::divide(float value) const {
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
Tensor4D Tensor4D::add(const Tensor4D& other) const {
    Tensor4D::checkDimensions(other);
    Tensor4D result(N, C, H, W);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    result.at(n, c, h, w) = this->at(n, c, h, w) + other.at(n, c, h, w);
    return result;
}

// Tensor subtraction
Tensor4D Tensor4D::subtract(const Tensor4D& other) const {
    Tensor4D::checkDimensions(other);
    Tensor4D result(N, C, H, W);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    result.at(n, c, h, w) = this->at(n, c, h, w) - other.at(n, c, h, w);
    return result;
}

// Tensor multiplication (element-wise)
Tensor4D Tensor4D::multiply(const Tensor4D& other) const {
    Tensor4D::checkDimensions(other);
    Tensor4D result(N, C, H, W);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    result.at(n, c, h, w) = this->at(n, c, h, w) * other.at(n, c, h, w);
    return result;
}

// Tensor division (element-wise)
Tensor4D Tensor4D::divide(const Tensor4D& other) const {
    Tensor4D::checkDimensions(other);
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

void Tensor4D::printAsMatrix() const {
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

void Tensor4D::getMatrix(int N, int C) const {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                std::cout << at(N, C, h, w) << " ";
            }
            std::cout << "\n";
        }
            std::cout << "\n";
        }
void Tensor4D::addPadding(int padHeight, int padWidth) {
    // New dimensions after padding
    int newH = H + 2 * padHeight;
    int newW = W + 2 * padWidth;

    // Resize data to the new dimensions
    std::vector<std::vector<std::vector<std::vector<float>>>> newData(
        N, std::vector<std::vector<std::vector<float>>>(
                C, std::vector<std::vector<float>>(
                        newH, std::vector<float>(newW, 0.0f))));

    // Copy original data into the new padded tensor
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    newData[n][c][h + padHeight][w + padWidth] = data[n][c][h][w];
                }
            }
        }
    }

    // Update data with padded tensor
    data = std::move(newData);
    H = newH;
    W = newW;
}  

Tensor4D Tensor4D::concatAlongChannels(const Tensor4D& other) const {
    // Ensure dimensions match except for the channel dimension
    if (N != other.N || H != other.H || W != other.W) {
        throw std::invalid_argument("Tensors cannot be concatenated due to dimension mismatch.");
    }

    // Create a new tensor with updated channel dimension
    Tensor4D result(N, C + other.C, H, W);

    // Copy the current tensor's data
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    result.data[n][c][h][w] = data[n][c][h][w];
                }
            }
        }
    }

    // Copy the other tensor's data
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < other.C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    result.data[n][C + c][h][w] = other.data[n][c][h][w];
                }
            }
        }
    }

    return result;
}

Tensor4D Tensor4D::upsample(int newH, int newW) const {
    // Scaling factors
    float scaleH = static_cast<float>(H) / newH;
    float scaleW = static_cast<float>(W) / newW;

    // Create a new tensor with updated dimensions
    Tensor4D result(N, C, newH, newW);

    // Perform bilinear interpolation
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < newH; ++h) {
                for (int w = 0; w < newW; ++w) {
                    // Compute original coordinates
                    float origH = h * scaleH;
                    float origW = w * scaleW;

                    int h0 = static_cast<int>(origH);
                    int w0 = static_cast<int>(origW);

                    int h1 = std::min(h0 + 1, H - 1);
                    int w1 = std::min(w0 + 1, W - 1);

                    float hLerp = origH - h0;
                    float wLerp = origW - w0;

                    // Perform bilinear interpolation
                    result.data[n][c][h][w] =
                        (1 - hLerp) * (1 - wLerp) * data[n][c][h0][w0] +
                        (1 - hLerp) * wLerp * data[n][c][h0][w1] +
                        hLerp * (1 - wLerp) * data[n][c][h1][w0] +
                        hLerp * wLerp * data[n][c][h1][w1];
                }
            }
        }
    }

    return result;
}

