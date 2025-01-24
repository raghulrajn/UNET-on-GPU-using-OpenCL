#ifndef TENSOR4D_H
#define TENSOR4D_H

#include <vector>
#include <iostream>
#include <random>
#include <stdexcept>

class Tensor4D {
private:
    int N, C, H, W;
    std::vector<std::vector<std::vector<std::vector<float>>>> data;

    // Helper function to check dimensions for operations
    void checkDimensions(const Tensor4D& other) const;

public:
    // Constructor
    Tensor4D(int n, int c, int h, int w);

    // Accessor methods
    int getN() const;
    int getC() const;
    int getH() const;
    int getW() const;

    // Element access
    float& value(int n, int c, int h, int w);
    const float& value(int n, int c, int h, int w) const;

    // Utility methods
    void printDimensions() const;
    void setRandomValues(float mean, float std);
    void printAsMatrix() const;

    // Arithmetic operations
    Tensor4D add(const Tensor4D& other) const;
    Tensor4D subtract(const Tensor4D& other) const;
    Tensor4D multiply(const Tensor4D& other) const;
    Tensor4D divide(const Tensor4D& other) const;

    // Scalar operations
    Tensor4D add(float scalar) const;
    Tensor4D subtract(float scalar) const;
    Tensor4D multiply(float scalar) const;
    Tensor4D divide(float scalar) const;
};

#endif // TENSOR4D_H
