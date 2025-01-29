#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <chrono>
#include "timer.h"
#include "tensor4D.h"
#include "cnpy/cnpy.h"
#include <opencv2/opencv.hpp>

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
    std::cout << "("<< N
                << ", " << C << ", " << H << ", " << W <<")"<< std::endl;
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

Tensor4D Tensor4D::extract(int newH, int newW) {
        int startH = (H - newH) / 2;
        int startW = (W - newW) / 2;
        
        Tensor4D centerTensor(N, C, newH, newW);
        
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < newH; ++h) {
                    for (int w = 0; w < newW; ++w) {
                        centerTensor.data[n][c][h][w] = data[n][c][startH + h][startW + w];
                    }
                }
            }
        }
        return centerTensor;
    }

Tensor4D Tensor4D::fromNPY(const std::string &filename) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    float* raw_data = arr.data<float>();
    int n,c,h,w;
    if (arr.shape.size() == 1) {
        // For 1D array (e.g., 128, which could be interpreted as (1, 1, 1, 128))
        n = 1;
        c = 1;
        h = 1;
        w = arr.shape[0];
    } else if (arr.shape.size() == 2) {
        // For 2D array, assuming it is a (1, 1, 64, 64) shape
        n = 1;
        c = 1;
        h = arr.shape[0];
        w = arr.shape[1];
    } else if (arr.shape.size() == 4) {
        // For 4D array, it is in the format (n, c, h, w)
        n = arr.shape[0];
        c = arr.shape[1];
        h = arr.shape[2];
        w = arr.shape[3];
    } 
    else {
        throw std::runtime_error("Unsupported kernel shape.");
    }
    Tensor4D kernel(n,c,h,w);
    float* data = arr.data<float>();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < c; ++j) {
            for (int x = 0; x < h; ++x) {
                for (int y = 0; y < w; ++y) {
                    kernel.at(i, j,x, y) = data[(i * c * h * w) + (j * h * w) + (x * w) + y];
                }
            }
        }
    }
    return kernel;
}

Tensor4D Tensor4D::fromJPG(const std::string &filename) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    int h = img.rows;
    int w = img.cols;
    Tensor4D tensor(1, 3, h, w);
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                tensor.data[0][c][i][j] = img.at<cv::Vec3b>(i, j)[c] / 255.0f;
            }
        }
    }
    return tensor;
}

void Tensor4D::saveAsJPG(const std::string &filename) {
    cv::Mat img(H, W, CV_8UC3);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            img.at<cv::Vec3b>(i, j) = cv::Vec3b(
                static_cast<unsigned char>(data[0][2][i][j] * 255),
                static_cast<unsigned char>(data[0][1][i][j] * 255),
                static_cast<unsigned char>(data[0][0][i][j] * 255)
            );
        }
    }
    cv::imwrite(filename, img);
}

