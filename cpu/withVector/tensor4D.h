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
    /**
     * @brief Constructs a Tensor4D object with specified dimensions and data.
     * 
     * This constructor initializes a 4D tensor with the given batch size (`N`), number of channels (`C`),
     * height (`H`), and width (`W`). It also allocates a 4D data structure to hold the tensor values, represented
     * as a nested vector of floats. The data is initially empty, and the tensor dimensions are set according to the
     * provided values.
     * 
     * @param N The number of samples (batch size) in the tensor.
     * @param C The number of channels in each sample.
     * @param H The height (number of rows) of each sample.
     * @param W The width (number of columns) of each sample.
     * 
     * @note The data structure is allocated as a 4D vector with dimensions `(N x C x H x W)` to hold the tensor values.
     *       All values are initially uninitialized and need to be set manually or through further methods.
     */
    Tensor4D(int n, int c, int h, int w);

    // Accessor methods
    int getN() const;
    int getC() const;
    int getH() const;
    int getW() const;

    // Element access
    float& at(int n, int c, int h, int w);
    const float& at(int n, int c, int h, int w) const;

    // Utility methods
    void printDimensions() const;
    void setRandomValues(float mean, float std);
    void printAsMatrix() const;
    void setValue();
    void addPadding(int padHeight, int padWidth);
    void getMatrix(int N, int C) const; 
    void saveAsJPG(const std::string &filename);

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

    /**
     * @brief Concatenates two 4D tensors along the channel dimension.
     * 
     * This function concatenates the input tensor (`this`) and another tensor (`other`) along the channels dimension (axis 1).
     * The tensors must have the same batch size, height, and width. The resulting tensor will contain the channels from both tensors,
     * maintaining their respective batch size, height, and width.
     * 
     * @param other The second 4D tensor to concatenate with the current tensor. The tensors must have matching dimensions for:
     *        - `N` (batch size),
     *        - `H` (height),
     *        - `W` (width).
     *        The `C` (channel dimension) of the resulting tensor will be the sum of the channels of the two input tensors.
     * 
     * @return A new 4D tensor with the concatenated channels. The resulting tensor will have the following shape:
     *         - `N`: Same as the batch size of the input tensors.
     *         - `H`: Same as the height of the input tensors.
     *         - `W`: Same as the width of the input tensors.
     *         - `C`: The sum of the channels of both tensors (`C1 + C2`), where `C1` is the number of channels in `this` tensor
     *           and `C2` is the number of channels in the `other` tensor.
     * 
     * @note The tensors must have the same height, width, and batch size. If the dimensions do not match, an exception or error
     *       may be thrown.
     */
    Tensor4D concatAlongChannels(const Tensor4D& other) const;

    /**
     * @brief Upsamples the input tensor to a new height and width using bilinear interpolation.
     * 
     * This function resizes the input 4D tensor (batch of images or feature maps) to the specified
     * `newH` (new height) and `newW` (new width) dimensions using bilinear interpolation. 
     * Bilinear interpolation is a method of interpolation that uses the closest four pixels to 
     * compute the new value, making it suitable for image resizing tasks.
     * 
     * @param newH The target height (in pixels) for the upsampled tensor.
     * @param newW The target width (in pixels) for the upsampled tensor.
     * 
     * @return A new 4D tensor with the upsampled height and width, while maintaining the same batch size and number of channels.
     *         The resulting tensor has the shape `(N x C x newH x newW)`, where:
     *         - `N`: Number of samples in the batch.
     *         - `C`: Number of channels.
     *         - `newH`: The upsampled height.
     *         - `newW`: The upsampled width.
     * 
     * @note The upsampling is performed using bilinear interpolation, which smoothly resizes the input while considering
     *       neighboring pixels. The function does not modify the original tensor and returns a new tensor with the resized dimensions.
     */
    Tensor4D upsample(int newH, int newW) const;
    Tensor4D extract(int newH, int newW);
    /**
     * @brief Loads a `.npy` file and converts its data into a Tensor4D object.
     * 
     * This function reads the contents of a `.npy` file, which contains a multi-dimensional array, and 
     * converts it into a `Tensor4D` object. The `.npy` file should represent a 4D tensor with the correct
     * dimensions (N x C x H x W), where:
     *  - `N` is the batch size,
     *  - `C` is the number of channels,
     *  - `H` is the height,
     *  - `W` is the width.
     * 
     * The `npy_load` function will parse the file, extract the data and converted to `Tensor4D`
     * 
     * @param filename The path to the `.npy` file that contains the tensor data.
     * 
     * @return `Tensor4D` loaded from .npy file
     * 
     * @note The `.npy` file must be formatted correctly as a 4D tensor, with the appropriate dimensions for
     *       batch size, channels, height, and width. If the file format is incompatible or the dimensions do not
     *       match, an exception may be thrown.
     */
    static Tensor4D fromNPY(const std::string &filename);

     /**
     * @brief Loads a `.jpg` file and converts its data into a Tensor4D object.
     * 
     * This function reads the contents of a `.jpg` file, which contains a multi-dimensional array, and 
     * converts it into a `Tensor4D` object. The `.jpg` file should represent a 4D tensor with the correct
     * dimensions (N x C x H x W), where:
     *  - `N` is the batch size,
     *  - `C` is the number of channels,
     *  - `H` is the height,
     *  - `W` is the width.
     * 
     * The `fromJPG` function will parse the file, extract the data and converted to `Tensor4D`
     * 
     * @param filename The path to the `.jpg`
     * 
     * @return `Tensor4D` loaded from .jpg 
     */
    static Tensor4D fromJPG(const std::string &filename);


};

#endif 
