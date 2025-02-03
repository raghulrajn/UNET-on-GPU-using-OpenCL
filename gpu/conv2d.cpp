// includes
#include <stdio.h>
#include <Core/Assert.hpp>
#include <Core/Image.hpp>
#include <Core/Time.hpp>
#include <OpenCL/Device.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/cl-patched.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdlib>  // for std::abs (integer)
#include <ctime>
#include <random>
#include <chrono>
class GPUInit {

	private:
		cl::Context context;
		cl::CommandQueue queue;
		cl::Program program;
		cl::Device device;

		cl::Kernel convKernel;
		cl::Kernel reluKernel;
		cl::Kernel maxpoolKernel;

		cl::Buffer pipeBuffer;
		std::vector<cl::Device> devices;

	public:
	GPUInit(){
		context = cl::Context(CL_DEVICE_TYPE_GPU);
		device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		devices.push_back(device);
		queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

		// Load the source code
		extern unsigned char conv2d_cl[];
		extern unsigned int conv2d_cl_len;
		cl::Program program(context, std::string((const char*)conv2d_cl, conv2d_cl_len));
		OpenCL::buildProgram(program, devices);
		std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size()<< " devices" << std::endl;
		try {
			convKernel = cl::Kernel(program, "conv2d");
			reluKernel = cl::Kernel(program, "relu_activation");
			maxpoolKernel = cl::Kernel(program, "maxpool");
		} catch (OpenCL::Error &e) {
			std::cerr << "Error creating kernel: " << e.what() << std::endl;
			throw std::runtime_error("Failed to create kernel.");
		}

	}

	void getGpuDetails(){
		std::cout << "Device: "                   << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "Max Work Group Size: "      << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
		std::cout << "Max Compute Units: "        << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
		std::cout << "Global Memory Size: "       << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/1000000 << " MB" << std::endl;
		std::cout << "Local Memory Size: "        << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << " bytes" << std::endl;
		std::cout << "Max Work Item Dimensions: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
		auto maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
		std::cout << "Max Work Item Sizes: ";
		for (const auto& size : maxWorkItemSizes) {
			std::cout << size << " ";
		}
		std::cout<<"\n";
	}

	std::vector<float> convolution(std::vector<float> &input, std::vector<float> &kernel, int N, int C, int H, int W, int OutC, int Kh, int Kw, int stride, int padding){
		int outH = (H + 2 * padding - Kh) / stride + 1;
		int outW = (W + 2 * padding - Kw) / stride + 1;

		std::vector<float> output(N*OutC*outH* outW);
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*input.size(), input.data());
		cl::Buffer kernelBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*kernel.size(), kernel.data());
		cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*output.size());
		convKernel.setArg(0, inputBuffer);
		convKernel.setArg(1, kernelBuffer);
		convKernel.setArg(2, outputBuffer);
		convKernel.setArg(3, N); // Batch size
		convKernel.setArg(4, C); // Number of input channels
		convKernel.setArg(5, H); // Height of the input
		convKernel.setArg(6, W); // Width of the input
		convKernel.setArg(7, OutC); // Number of output channels
		convKernel.setArg(8, Kh); // Kernel height
		convKernel.setArg(9, Kw); // Kernel width
		convKernel.setArg(10, stride); // Stride
		convKernel.setArg(11, padding); // Padding
        // Set up work sizes (output size)
		cl::Event convevent;
		cl::NDRange global(C, outH, outW); // (width * height * output channels, batch size)
		queue.enqueueNDRangeKernel(convKernel, cl::NullRange, global, cl::NDRange(),NULL,&convevent);
        // Read the output back to the host
		pipeBuffer = outputBuffer;
		//queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float)*output.size(), output.data(),NULL,&convevent);
		queue.finish();

		Core::TimeSpan timeGpu = OpenCL::getElapsedTime(convevent);
		std::cout<<"Conv GPU time is "<<timeGpu<<std::endl;
		return output;
	}

	std::vector<float> relu(std::vector<float> &input, int N, int C, int H, int W){
		std::vector<float> output(N*C*H*W);
		//cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
		cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * input.size());
		reluKernel.setArg(0, pipeBuffer);  // input tensor
		//reluKernel.setArg(0, inputBuffer);  // input tensor
		reluKernel.setArg(1, outputBuffer); // output tensor
		reluKernel.setArg(2, N*C*H*W); // size of the input tensor

		// Define the global and local work sizes
		cl::NDRange global(N*C*H*W);  // Global work size, size of input tensor
		cl::NDRange local(256);            // Local work size, can be adjusted based on hardware
		// Enqueue the kernel for execution
		cl::Event reluevent;
		queue.enqueueNDRangeKernel(reluKernel, cl::NullRange, global, local,NULL,&reluevent);
		queue.finish();
		//queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * input.size(), output.data(),NULL,&reluevent);
		pipeBuffer = outputBuffer;
		Core::TimeSpan timeGpu = OpenCL::getElapsedTime(reluevent);
		std::cout<<"Relu GPU time is "<<timeGpu<<std::endl;
		return output;
	}

	std::vector<float> maxpool(std::vector<float> &input, int N, int C, int H, int W, int pool_size, int stride=2){

		int outH = (H - pool_size) / stride + 1;
		int outW = (W - pool_size) / stride + 1;
		std::vector<float> output(N*C*outH*outW);
		//cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
		cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());

		maxpoolKernel.setArg(0, pipeBuffer);
		maxpoolKernel.setArg(1, outputBuffer);
		maxpoolKernel.setArg(2, N); // Batch size
		maxpoolKernel.setArg(3, C); // Number of channels
		maxpoolKernel.setArg(4, H); // Input height
		maxpoolKernel.setArg(5, W); // Input width
		maxpoolKernel.setArg(6, pool_size); // Pool size
		maxpoolKernel.setArg(7, stride); // Stride

		// Set up work sizes (output size)
		cl::NDRange global(N * C * outH * outW); // (Batch * Channels * Output Height * Output Width)
		cl::NDRange local(1, 1); // Workgroup size (could be optimized further)

		// Enqueue kernel for execution
		cl::Event maxpoolevent[3];
		queue.enqueueNDRangeKernel(maxpoolKernel, cl::NullRange, global, local, NULL,&maxpoolevent[1]);
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * output.size(), output.data(),NULL,&maxpoolevent[2]);
		queue.finish();
		 Core::TimeSpan timeGpu = OpenCL::getElapsedTime(maxpoolevent[1]);
		std::cout<<"Maxpool GPU time is "<<timeGpu<<std::endl;
		Core::TimeSpan timeGpu2 = OpenCL::getElapsedTime(maxpoolevent[2]);
		std::cout<<"copying maxpool GPU time is "<<timeGpu2<<std::endl;
		return output;
	}

};

void convolution_2d_flattened(const std::vector<float>& flattened_input,
                              const std::vector<float>& kernel,
                              std::vector<float>& flattened_output,
                              int N, int C, int H, int W,
                              int k_height, int k_width,
                              int stride = 1, int padding = 0) {

    // Reshape the flattened input back to 4D tensor (N, C, H, W)
    std::vector<std::vector<std::vector<std::vector<float>>>> input(N, std::vector<std::vector<std::vector<float>>>(C, std::vector<std::vector<float>> (H, std::vector<float>(W))));

    int index = 0;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    input[n][c][h][w] = flattened_input[index++];
                }
            }
        }
    }

    // Calculate the dimensions of the output tensor
    int out_height = (H - k_height + 2 * padding) / stride + 1;
    int out_width = (W - k_width + 2 * padding) / stride + 1;

    // Flattened output tensor
    flattened_output.resize(N * C * out_height * out_width);

    // Perform the convolution operation
    index = 0; // To keep track of the flattened output index
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < out_height; ++i) {
                for (int j = 0; j < out_width; ++j) {
                    float sum = 0.0f;

                    // Perform the convolution operation for each kernel element
                    for (int ki = 0; ki < k_height; ++ki) {
                        for (int kj = 0; kj < k_width; ++kj) {
                            int h = i * stride + ki - padding;
                            int w = j * stride + kj - padding;

                            // Check if the indices are within bounds
                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                sum += input[n][c][h][w] * kernel[ki * k_width + kj];
                            }
                        }
                    }

                    // Store the result in the flattened output
                    flattened_output[index++] = sum;
                }
            }
        }
    }
}

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
        std::cout << name << " - " << duration.count()<< " ms" << std::endl;
    }
};

int main() {
    GPUInit gpu = GPUInit();
	gpu.getGpuDetails();

	std::random_device rd;
    std::mt19937 gen(rd());

    // Define the distribution range (0, 256)
    std::uniform_real_distribution<float> dis(0.0f, 256.0f);
	std::uniform_real_distribution<float> ker(0.0f, 10.0f);

    // Define input dimensions
    int N = 1;     // Batch size
    int C = 3;     // Channels in the input
    int H = 256;    // Height of the input
    int W = 256;    // Width of the input
    int Kh = 3;    // Kernel height
    int Kw = 3;    // Kernel width
    int OutC = 128; // Output channels
    int stride = 1; // Stride
    int padding = 0; // Padding

    // Create buffers for input and kernel
    std::vector<float> inputTensor(N * C * H * W, 2.0f);
    std::vector<float> kernelTensor(OutC * C * Kh * Kw, 0.5f);
    std::vector<float> outputTensor = gpu.convolution(inputTensor, kernelTensor, N, C, H, W, OutC, Kh, Kw, stride, padding);

	std::vector<float> flattened_output;

    // Perform the convolution
	Timer t1("cpu");
    convolution_2d_flattened(inputTensor, kernelTensor, flattened_output, N, C, H, W, Kh, Kw);
	t1.stop();
	int count=0;
	int outH = ((H + 2 * padding - Kh) / stride) + 1;
	int outW = ((W + 2 * padding - Kw) / stride) + 1;
	std::vector<float> reluTensor = gpu.relu(outputTensor, N, OutC, outH, outW);
	std::vector<float> maxpoolTensor = gpu.maxpool(reluTensor, N, OutC, outH, outW,2);


	for(auto i=0; i<flattened_output.size();i++){
		//std::cout<<outputTensor[i]<<" "<<flattened_output[i]<<std::endl;
		if(std::abs(outputTensor[i]-flattened_output[i])>0.0005){
			++count;
		}
	}
	std::cout<<"Unequal values: "<<count<<std::endl;
	std::cout<<"Total values: "<<N*OutC*outH*outW<<std::endl;
	std::cout<<"% = "<<((float)count/float(N*OutC*outH*outW))*100<<std::endl;

}
