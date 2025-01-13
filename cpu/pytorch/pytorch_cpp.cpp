#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include "cnpy/cnpy.h"
#include <string>
#include <chrono>

// Function to load kernel weights from a .npy file
// torch::Tensor loadKernelFromNpy(const std::string& file_path) {
//     cnpy::NpyArray array = cnpy::npy_load(file_path);
//     if (array.word_size != sizeof(float)) {
//         throw std::runtime_error("Kernel data must be of type float.");
//     }
//     std::vector<size_t> shape = array.shape;
//     if (shape.size() != 4) {
//         throw std::runtime_error("Kernel shape must be 4D (out_channels, in_channels, height, width).");
//     }
//     // Create a tensor from the kernel data
//     auto kernel = torch::from_blob(array.data<float>(), 
//                                     {static_cast<int64_t>(shape[0]), 
//                                     static_cast<int64_t>(shape[1]), 
//                                     static_cast<int64_t>(shape[2]), 
//                                     static_cast<int64_t>(shape[3])}, 
//                                     torch::kFloat32)
//                                     .clone();
//     return kernel;
// }

// Function to perform Conv -> ReLU -> BatchNorm
// torch::Tensor conv_relu_bn(torch::Tensor x, const std::string& file_path, int stride, int padding, bool add_bias= false) {
//     // torch::Tensor kernel = loadKernelFromNpy(file_path);

//     auto conv = torch::nn::functional::conv2d(
//         x,
//         kernel,
//         torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(padding));

//     auto relu = torch::relu(conv);
//     auto bn = torch::nn::functional::batch_norm(
//         relu,
//         torch::randn(kernel.sizes()[0]), // Running mean
//         torch::randn(kernel.sizes()[0]), // Running variance
//         torch::nn::functional::BatchNormFuncOptions().eps(1e-5).momentum(0.1).training(true));
//     return bn;
// }
 
// Functopn of Conv->relu->batch_norm (tensor, input, outchannel, inchannel, filer, stride, padding, add_bias)
torch::Tensor conv_relu_bn(torch::Tensor x, int inchannel, int outchannel, int kernel_size, int stride, int padding, bool add_bias= false) {
    // torch::Tensor kernel = loadKernelFromNpy(file_path);
    torch::Tensor kernel = torch::rand({outchannel, inchannel,kernel_size, kernel_size});
    // std::cout<<x.sizes()<<std::endl;
    // std::cout<<kernel.sizes()<<std::endl;
    auto conv = torch::nn::functional::conv2d(
        x,
        kernel,
        torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(padding));

    auto relu = torch::relu(conv);
    auto bn = torch::nn::functional::batch_norm(
        relu,
        torch::randn(kernel.sizes()[0]), // Running mean
        torch::randn(kernel.sizes()[0]), // Running variance
        torch::nn::functional::BatchNormFuncOptions().eps(1e-5).momentum(0.1).training(true));
    return bn;
}


// Function for MaxPooling
torch::Tensor max_pooling(torch::Tensor x, int kernel_size, int stride) {
    torch::nn::functional::MaxPool2dFuncOptions options(2);
    options.stride(2).padding(1);  // setting stride and padding
    return torch::nn::functional::max_pool2d(x, options);
}

// Function for Upsampling
torch::Tensor upsample(torch::Tensor x, int scale_factor) {

    return torch::nn::functional::interpolate(x,
                                torch::nn::functional::InterpolateFuncOptions()
                                .mode(torch::kBilinear)
                                .size(std::vector<int64_t>({x.size(2) * scale_factor, x.size(3) * scale_factor})));
}

// Function for Cropping
torch::Tensor crop_tensor(torch::Tensor x, int target_height, int target_width) {
    int h = x.size(2);
    int w = x.size(3);
    if (h < target_height || w < target_width) {
        throw std::runtime_error("Cannot crop: target size is larger than the tensor size.");
    }
    int start_h = (h - target_height) / 2;
    int start_w = (w - target_width) / 2;
    std::cout<<start_h<<"  "<<start_w<<std::endl;
    return x.index({"...", start_h, start_h + target_height, start_w, start_w + target_width});
}

// Function to concatenate tensors
torch::Tensor concatenate_tensors(torch::Tensor x1, torch::Tensor x2) {
    if (x1.size(2) != x2.size(2) || x1.size(3) != x2.size(3)) {
        throw std::runtime_error("Cannot concatenate tensors: dimensions must match.");
    }
    return torch::cat({x1, x2}, 1); // Concatenate along the channel dimension
}

// Function to save tensor as an image
// void save_tensor_as_image(torch::Tensor tensor, const std::string& file_name) {
//     tensor = tensor.squeeze(0).squeeze(0); // Remove batch and channel dimensions
//     tensor = tensor.clamp(0, 1).mul(255).to(torch::kU8); // Scale to [0, 255] and convert to uint8
//     cv::Mat image(tensor.size(0), tensor.size(1), CV_8UC1, tensor.data_ptr());
//     cv::imwrite(file_name, image);
// }

// torch::Tensor readRGBImageAsTensor(const std::string& image_path) {
//     // Load the image in RGB format
//     cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
//     if (image.empty()) {
//         throw std::runtime_error("Failed to load image: " + image_path);
//     }

//     // Convert the image to floating-point and normalize to [0, 1]
//     image.convertTo(image, CV_32F, 1.0 / 255.0);

//     // Convert BGR (OpenCV default) to RGB
//     cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

//     // Create a tensor from the image (Height, Width, Channels -> Channels, Height, Width)
//     auto tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kFloat32);
//     tensor = tensor.permute({2, 0, 1}).clone(); // Change to CxHxW format

//     return tensor.unsqueeze(0); // Add a batch dimension (1, C, H, W)
// }


class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;

public:
    Timer(const std::string& function_name = "Function") : name(function_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
    }

    void stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << name << " - " << duration.count()/1000 << " s" << std::endl;
    }
};



// Main function
int main(int argc, char** argv) {
    // if (argc < 2) {
    //     std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    //     return -1;
    // }

    std::string image_path = "utils/image.jpg";

    // torch::Tensor input_tensor = readRGBImageAsTensor(image_path);
    torch::Tensor input_tensor = torch::rand({1,3,572, 572});
   
    Timer t1("conv1");
    auto conv1 = conv_relu_bn(input_tensor, 3, 64, 3, 1, 0, true);
    // auto pool1 = max_pooling(conv1, 2, 2);
    conv1 = conv1.contiguous();
    t1.stop();
    std::cout<<"conv1 "<<conv1.sizes()<<std::endl;

    Timer t2("conv2");
    auto conv2 = conv_relu_bn(conv1, 64, 128, 3, 1, 0, true);
    std::cout<<"conv2 "<<conv2.sizes()<<std::endl;
    // auto pool2 = max_pooling(conv2, 2, 2);
    t2.stop();
   
    Timer t4("crop and concat");
    // auto cropped = crop_tensor(conv2, 284, 284);
    // std::cout<<"cropped "<<cropped.sizes()<<std::endl;
    auto concat = concatenate_tensors(conv2, conv2);
    std::cout<<concat.sizes()<<std::endl;
    t4.stop();

    return 0;
}
