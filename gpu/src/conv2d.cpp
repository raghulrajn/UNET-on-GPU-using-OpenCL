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
#include <opencv2/opencv.hpp>
class GPUInit {

	private:
		cl::Context context;
		cl::CommandQueue queue;
		cl::Program program;
		cl::Device device;

		cl::Kernel convKernel;
		cl::Kernel reluKernel;
		cl::Kernel maxpoolKernel;
		cl::Kernel meanKernel;
		cl::Kernel varianceKernel;
		cl::Kernel upsampleKernel;
		cl::Kernel batchnormKernel;
		cl::Kernel concatKernel;
		cl::Kernel extractKernel;

		cl::Buffer pipeBuffer;
	
		std::vector<cl::Device> devices;

	public:
	int _outH, _outW,_outC;
	

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
			convKernel      = cl::Kernel(program, "conv2d");
			reluKernel      = cl::Kernel(program, "relu_activation");
			maxpoolKernel   = cl::Kernel(program, "maxpool");
			meanKernel      = cl::Kernel(program, "batchMean");
			varianceKernel  = cl::Kernel(program, "batchVariance");
			batchnormKernel = cl::Kernel(program, "batch_norm");
			concatKernel    = cl::Kernel(program, "concatenate_tensors");
			extractKernel   = cl::Kernel(program, "extract_center");
			upsampleKernel  = cl::Kernel(program, "upsample");


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
	double convOp , convCopy ;
	double reluOp, reluCopy;
	double maxpoolOp, maxpoolCopy;
	double bnOp, bnCopy;
	double concatOp, concatCopy;
	double extOp, extCopy;
	double upsampleOp, upsampleCopy;
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
		cl::Event event[2];
		cl::NDRange global(C, outH, outW);
		queue.enqueueNDRangeKernel(convKernel, cl::NullRange, global, cl::NDRange(),NULL,&event[0]);

		pipeBuffer = outputBuffer;
		// Read the output back to the host
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float)*output.size(), output.data(),NULL,&event[1]);
		queue.finish();
		_outC = OutC;
		_outH = outH;
		_outW = outW;
		convOp = convOp + OpenCL::getElapsedTime(event[0]).getMilliseconds();
		convCopy = convCopy+ OpenCL::getElapsedTime(event[1]).getMilliseconds();
		return output;
	}

	std::vector<float> relu(std::vector<float> &input, int N, int C, int H, int W){
		std::vector<float> output(N*C*H*W);
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
		cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * input.size());
		// reluKernel.setArg(0, pipeBuffer);  // input tensor
		reluKernel.setArg(0, inputBuffer);  // input tensor
		reluKernel.setArg(1, outputBuffer); // output tensor
		reluKernel.setArg(2, N*C*H*W); // size of the input tensor

		// Define the global and local work sizes
		cl::NDRange global(N*C*H*W);  // Global work size, size of input tensor
		// Enqueue the kernel for execution
		cl::Event event[2];
		queue.enqueueNDRangeKernel(reluKernel, cl::NullRange, global, cl::NDRange(),NULL,&event[0]);
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * input.size(), output.data(),NULL,&event[1]);
		queue.finish();
		pipeBuffer = outputBuffer;
		reluOp += OpenCL::getElapsedTime(event[0]).getMilliseconds();
		reluCopy += OpenCL::getElapsedTime(event[1]).getMilliseconds();
		
		_outC = C;
		_outH = H;
		_outW = W;
		return output;
	}

	std::vector<float> maxpool(std::vector<float> &input, int N, int C, int H, int W, int pool_size, int stride=2){

		int outH = (H - pool_size) / stride + 1;
		int outW = (W - pool_size) / stride + 1;
		std::vector<float> output(N*C*outH*outW);
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
		cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());

		maxpoolKernel.setArg(0, inputBuffer);
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
		cl::Event event[2];
		queue.enqueueNDRangeKernel(maxpoolKernel, cl::NullRange, global, local, NULL,&event[0]);
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * output.size(), output.data(),NULL,&event[1]);
		queue.finish();
		maxpoolOp += OpenCL::getElapsedTime(event[0]).getMilliseconds();
		maxpoolCopy += OpenCL::getElapsedTime(event[1]).getMilliseconds();
		_outC = C;
		_outH = H;
		_outW = W;
		return output;
	}

	 std::vector<float> mean(std::vector<float>& input, int N, int C, int H, int W){
		std::vector<float> mean(C, 0.0f);
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
		cl::Buffer meanBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * mean.size());
		meanKernel.setArg(0, inputBuffer);
		meanKernel.setArg(1, meanBuffer);
		meanKernel.setArg(2, N);
		meanKernel.setArg(3, C);
		meanKernel.setArg(4, H);
		meanKernel.setArg(5, W);

		cl::NDRange global(C);
		cl::NDRange local(1, 1, 1);  // Workgroup size (1x1x1)

        cl::Event meanEvent;
        queue.enqueueNDRangeKernel(meanKernel, cl::NullRange, global, cl::NDRange(), nullptr, &meanEvent);
        queue.enqueueReadBuffer(meanBuffer, CL_TRUE, 0, sizeof(float) * mean.size(), mean.data());
		queue.finish();
        return mean;
	}

	 std::vector<float> variance(std::vector<float>& input,std::vector<float>& mean, int N, int C, int H, int W){
		std::vector<float> variance(C, 0.0f);
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
		cl::Buffer meanBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * mean.size(), mean.data());
		cl::Buffer varianceBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * variance.size());
		varianceKernel.setArg(0, inputBuffer);
		varianceKernel.setArg(1, meanBuffer);
		varianceKernel.setArg(2, varianceBuffer);
		varianceKernel.setArg(3, N);
		varianceKernel.setArg(4, C);
		varianceKernel.setArg(5, H);
		varianceKernel.setArg(6, W);

		cl::NDRange global(C);
		cl::NDRange local(1, 1, 1);  // Workgroup size (1x1x1)

        cl::Event vEvent;
        queue.enqueueNDRangeKernel(varianceKernel, cl::NullRange, global, cl::NDRange(), nullptr, &vEvent);
        queue.enqueueReadBuffer(varianceBuffer, CL_TRUE, 0, sizeof(float) * variance.size(), variance.data());
		queue.finish();
        return variance;
	}
	// Batch Normalization
    std::vector<float> batchnorm(std::vector<float>& input, int N, int C, int H, int W) {
        // Output tensor
		std::vector<float>gamma(C,1.0f);
		std::vector<float>beta(C, 0.0f);
        std::vector<float> output(N * C * H * W);
		std::vector<float> mean(C, 0.0f);
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
		cl::Buffer meanBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * mean.size());
		meanKernel.setArg(0, inputBuffer);
		meanKernel.setArg(1, meanBuffer);
		meanKernel.setArg(2, N);
		meanKernel.setArg(3, C);
		meanKernel.setArg(4, H);
		meanKernel.setArg(5, W);

		cl::NDRange global(C);
		cl::NDRange local(1, 1, 1);  // Workgroup size (1x1x1)

        cl::Event event[2];
        queue.enqueueNDRangeKernel(meanKernel, cl::NullRange, global, cl::NDRange(), nullptr, &event[0]);

		std::vector<float> variance(C, 0.0f);
		cl::Buffer varianceBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * variance.size());
		varianceKernel.setArg(0, inputBuffer);
		varianceKernel.setArg(1, meanBuffer);
		varianceKernel.setArg(2, varianceBuffer);
		varianceKernel.setArg(3, N);
		varianceKernel.setArg(4, C);
		varianceKernel.setArg(5, H);
		varianceKernel.setArg(6, W);

		cl::NDRange vglobal(C);

        queue.enqueueNDRangeKernel(varianceKernel, cl::NullRange, vglobal, cl::NDRange(), nullptr, &event[0]);

        cl::Buffer gammaBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * gamma.size(), gamma.data());
        cl::Buffer betaBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * beta.size(), beta.data());
		cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());

		// Set kernel arguments
        batchnormKernel.setArg(0, inputBuffer);
        batchnormKernel.setArg(1, gammaBuffer);
        batchnormKernel.setArg(2, betaBuffer);
        batchnormKernel.setArg(3, meanBuffer);
        batchnormKernel.setArg(4, varianceBuffer);
        batchnormKernel.setArg(5, outputBuffer);
        batchnormKernel.setArg(6, N);
        batchnormKernel.setArg(7, C);
		batchnormKernel.setArg(8, H);
		batchnormKernel.setArg(9, W);

		// Launch kernel
        cl::NDRange bglobal(C, H , W);

        queue.enqueueNDRangeKernel(batchnormKernel, cl::NullRange, bglobal, cl::NDRange(), nullptr, &event[0]);
        queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * output.size(), output.data(),NULL, &event[1]);
		queue.finish();

		bnOp += OpenCL::getElapsedTime(event[0]).getMilliseconds();
		bnCopy += OpenCL::getElapsedTime(event[1]).getMilliseconds();
		_outC = C;
		_outH = H;
		_outW = W;
        return output;
    }

	std::vector<float> concat(std::vector<float> &input1,std::vector<float> &input2, int N, int C1,int C2, int H, int W){
		
		std::vector<float> output(N * (C1+C2) * H * W, 0.0f);
		cl::Buffer buffer_tensor1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input1.size() * sizeof(float), input1.data());
		cl::Buffer buffer_tensor2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input2.size() * sizeof(float), input2.data());
		cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, output.size() * sizeof(float));

		// Set kernel arguments
		concatKernel.setArg(0, buffer_tensor1);
		concatKernel.setArg(1, buffer_tensor2);
		concatKernel.setArg(2, buffer_output);
		concatKernel.setArg(3, N);
		concatKernel.setArg(4, C1);
		concatKernel.setArg(5, H);
		concatKernel.setArg(6, W);
		concatKernel.setArg(7, C2);
		int C3 = C1+C2;
		// Define global and local work sizes
		cl::NDRange global(N, C3, H * W); // Global work size
		cl::NDRange local(1, 1, 1);          // Local work size (one per element)
		
		cl::Event event[2];
		// Run the kernel
		queue.enqueueNDRangeKernel(concatKernel, cl::NullRange, global, local,NULL, &event[0]);
		// Retrieve the result
		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, output.size() * sizeof(float), output.data(), NULL, &event[1]);
		queue.finish();
		concatOp += OpenCL::getElapsedTime(event[0]).getMilliseconds();
		concatCopy += OpenCL::getElapsedTime(event[1]).getMilliseconds();
		_outC = C1+C2;
		_outH = H;
		_outW = W;
		return output;

		}

	std::vector<float> extract(std::vector<float> &input, int N, int C,int H, int W, int newH, int newW){
	
		std::vector<float> output(N * C * newH * newW, 0.0f);
		cl::Buffer buffer_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), input.data());
		cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, output.size() * sizeof(float));

		// Set kernel arguments
		extractKernel.setArg(0, buffer_input);
		extractKernel.setArg(1, buffer_output);
		extractKernel.setArg(2, N);
		extractKernel.setArg(3, C);
		extractKernel.setArg(4, H);
		extractKernel.setArg(5, W);
		extractKernel.setArg(6, newH);
		extractKernel.setArg(7, newW);

		// Define global and local work sizes
		cl::NDRange global(N, C, newH * newW); // Global work size
		cl::NDRange local(1, 1, 1);          // Local work size (one per element)
		cl::Event event[2];
		// Run the kernel
		queue.enqueueNDRangeKernel(extractKernel, cl::NullRange, global, local, NULL, &event[0]);
		// Retrieve the result
		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, output.size() * sizeof(float), output.data(), NULL, &event[1]);
		// queue.finish();
		extOp += OpenCL::getElapsedTime(event[0]).getMilliseconds();
		extCopy += OpenCL::getElapsedTime(event[1]).getMilliseconds();
		_outC = C;
		_outH = newH;
		_outW = newW;
		return output;

		}

	std::vector<float> upsample(std::vector<float> &input, int N, int C,int H, int W, int newH, int newW){

		std::vector<float> output(N * C * newH * newW, 0.0f);
		cl::Buffer buffer_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), input.data());
		cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, output.size() * sizeof(float));
		
		upsampleKernel.setArg(0, buffer_input);
		upsampleKernel.setArg(1, buffer_output);
		upsampleKernel.setArg(2, N);
		upsampleKernel.setArg(3, C);
		upsampleKernel.setArg(4, H);
		upsampleKernel.setArg(5, W);
		upsampleKernel.setArg(6, newH);
		upsampleKernel.setArg(7, newW);

		// Define global and local work sizes
		cl::Event event[2];
		cl::NDRange global(N, C, newH * newW); // Global work size
		cl::NDRange local(1, 1, 1);          // Local work size (one per element)

		// Run the kernel
		queue.enqueueNDRangeKernel(upsampleKernel, cl::NullRange, global, local, NULL, &event[0]);
		// Retrieve the result
		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, output.size() * sizeof(float), output.data(), NULL,&event[1]);
		upsampleOp += OpenCL::getElapsedTime(event[0]).getMilliseconds();
		upsampleCopy += OpenCL::getElapsedTime(event[1]).getMilliseconds();
		_outC = C;
		_outH = newH;
		_outW = newW;
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

std::vector<float> readImage(const std::string filepath){
	 cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        exit(1);
    }
    // Ensure the image is in RGB format (OpenCV loads in BGR by default)
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Flatten the image into a 1D array
    std::vector<float> flatArray;
    flatArray.reserve(image.rows * image.cols * image.channels());

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            flatArray.push_back(pixel[0]);  // Red channel
            flatArray.push_back(pixel[1]);  // Green channel
            flatArray.push_back(pixel[2]);  // Blue channel
        }
    }
    return flatArray;
}

void printTensor(std::vector<float> input, int N, int C, int H, int W, std::string name){
	std::cout<<"<--------------"<<name<<"-------------->"<<std::endl;
	for(int n=0; n<N; n++){
		for(int c=0; c<C;c++){
			std::cout<<"channel "<<c<<std::endl;
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					std::cout<<input[n * C * H * W + c * H * W + h * W + w]<<" ";
				}
				std::cout<<"\n";
			}
			std::cout<<"\n";
		}
	}
}

void compareTensors(std::vector<float> arr1, std::vector<float> arr2, float tolerance = 0.0005){
	int count=0;
	if(arr1.size() != arr2.size()){
		std::cout<<"[ERROR] Comparing Unequal vectors"<<std::endl;
		return;
	}
	for(auto i=0; i<arr1.size();i++){
		if(std::abs(arr1[i]-arr2[i])>tolerance){
			++count;
		}
	}
	std::cout<<"Number of incorrect values = "<<count<<std::endl;
	std::cout<<"Total values: "<<arr1.size()<<std::endl;
	std::cout<<"% of error = "<<((float)count/float(arr1.size()))*100<<std::endl;
}

// std::vector<float> loadKernelfromNPY(const std::string &filename) {
//     cnpy::NpyArray arr = cnpy::npy_load(filename);
//     float* raw_data = arr.data<float>();
//     int n,c,h,w;
//     if (arr.shape.size() == 1) {
//         // For 1D array (e.g., 128, which could be interpreted as (1, 1, 1, 128))
//         n = 1;
//         c = 1;
//         h = 1;
//         w = arr.shape[0];
//     } else if (arr.shape.size() == 2) {
//         // For 2D array, assuming it is a (1, 1, 64, 64) shape
//         n = 1;
//         c = 1;
//         h = arr.shape[0];
//         w = arr.shape[1];
//     } else if (arr.shape.size() == 4) {
//         // For 4D array, it is in the format (n, c, h, w)
//         n = arr.shape[0];
//         c = arr.shape[1];
//         h = arr.shape[2];
//         w = arr.shape[3];
//     } 
//     else {
//         throw std::runtime_error("Unsupported kernel shape.");
//     }

//     size_t total_size = n * c * h * w;
//     std::vector<float> kernel(raw_data, raw_data + total_size);
//     return kernel;
// }

void saveFlattenedAsJPG(const std::vector<float> &flattened, int N, int C, int H, int W, const std::string &filename) {
    if (C != 3) {
        throw std::runtime_error("Only RGB images (C=3) are supported.");
    }
    
    // OpenCV image matrix
    cv::Mat img(H, W, CV_8UC3);

    // Convert from flattened 1D to 3D RGB image
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            int index = (i * W + j);  // Row-major order
            img.at<cv::Vec3b>(i, j) = cv::Vec3b(
                static_cast<unsigned char>(flattened[index + 2 * H * W] * 255),  // R
                static_cast<unsigned char>(flattened[index + 1 * H * W] * 255),  // G
                static_cast<unsigned char>(flattened[index] * 255)              // B
            );
        }
    }

    // Save as JPG
    if (!cv::imwrite(filename, img)) {
        throw std::runtime_error("Failed to save image.");
    }
}

void printPerformace(GPUInit gpu){
	std::stringstream str1;
	str1 << std::setiosflags(std::ios::left) << std::setw(20) << "Functionality";
	str1 << std::setiosflags(std::ios::right);
	str1 << " " << std::setw(9) << "| CpuTime |";
	str1 << " " << std::setw(9) << "GPU executionTime |";
	str1 << " " << std::setw(9) << "GPU dataTransferTime |";
	str1 << " " << std::setw(9) << "GPU Total Time |";
	str1 << " " << std::setw(9) << "Speedup |";
	std::cout << str1.str() << std::endl;

	std::stringstream str;
	str << std::setiosflags(std::ios::left) << std::setw(20) << "Convolution";
	str << std::setiosflags(std::ios::right);
	str << " " << std::setw(10) << "XX";
	str << " " << std::setw(12) << gpu.convOp;
	str << " " << std::setw(15) << gpu.convCopy;
	str << " " << std::setw(20) << (gpu.convOp+gpu.convCopy);
	str << " " << std::setw(15) << "XX";
	std::cout << str.str() << std::endl;

	std::stringstream relu;
	relu << std::setiosflags(std::ios::left) << std::setw(20) << "ReLU";
	relu << std::setiosflags(std::ios::right);
	relu << " " << std::setw(10) << "XX";
	relu << " " << std::setw(12) << gpu.reluOp;
	relu << " " << std::setw(15) << gpu.reluCopy;
	relu << " " << std::setw(20) << (gpu.reluOp+gpu.reluCopy);
	relu << " " << std::setw(15) << "XX";
	std::cout << relu.str() << std::endl;

	std::stringstream maxpool;
	maxpool << std::setiosflags(std::ios::left) << std::setw(20) << "Maxpool";
	maxpool << std::setiosflags(std::ios::right);
	maxpool << " " << std::setw(10) << "XX";
	maxpool << " " << std::setw(12) << gpu.maxpoolOp;
	maxpool << " " << std::setw(15) << gpu.maxpoolCopy;
	maxpool << " " << std::setw(20) << (gpu.maxpoolOp+gpu.maxpoolCopy);
	maxpool << " " << std::setw(15) << "XX";
	std::cout << maxpool.str() << std::endl;

	std::stringstream bn;
	bn << std::setiosflags(std::ios::left) << std::setw(20) << "batchNorm";
	bn << std::setiosflags(std::ios::right);
	bn << " " << std::setw(10) << "XX";
	bn << " " << std::setw(12) << gpu.bnOp;
	bn << " " << std::setw(15) << gpu.bnCopy;
	bn << " " << std::setw(20) << (gpu.bnOp+gpu.bnCopy);
	bn << " " << std::setw(15) << "XX";
	std::cout << bn.str() << std::endl;

	std::stringstream upsample;
	upsample << std::setiosflags(std::ios::left) << std::setw(20) << "Upsampling";
	upsample << std::setiosflags(std::ios::right);
	upsample << " " << std::setw(10) << "XX";
	upsample << " " << std::setw(12) << gpu.upsampleOp;
	upsample << " " << std::setw(15) << gpu.upsampleCopy;
	upsample << " " << std::setw(20) << (gpu.upsampleOp+gpu.upsampleCopy);
	upsample << " " << std::setw(15) << "XX";
	std::cout << upsample.str() << std::endl;

}

int main() {
    GPUInit gpu = GPUInit();
	gpu.getGpuDetails();

	std::random_device rd;
    std::mt19937 gen(rd());

    // Define the distribution range (0, 256)
    std::uniform_real_distribution<float> dis(0.0f, 256.0f);
	std::uniform_real_distribution<float> ker(0.0f, 10.0f);

    // Define input dimensions
    // int N = 1;     // Batch size
    // int C = 3;     // Channels in the input
    // int H = 576;    // Height of the input
    // int W = 576;    // Width of the input
    // int Kh = 3;    // Kernel height
    // int Kw = 3;    // Kernel width
    // int OutC = 64; // Output channels
    // int stride = 1; // Stride
    // int padding = 0; // Padding

    // Create buffers for input and kernel
    //std::vector<float> inputTensor = readImage("/zhome/navanerj/Documents/Conv2d/src/sample.jpg");
    // std::vector<float> inputTensor(N * C * H * W, 1.0f);//2

	// for(auto i=0; i<inputTensor.size();i++){
	// 	inputTensor[i] = i+1;
	// }

    // std::vector<float> kernelTensor(OutC * C * Kh * Kw, 0.5f);
    // std::vector<float> outputTensor = gpu.convolution(inputTensor, kernelTensor, N, C, H, W, OutC, Kh, Kw, stride, padding);
	// std::vector<float> flattened_output;
    // // Perform the convolution
	// Timer t1("cpu");
    // convolution_2d_flattened(inputTensor, kernelTensor, flattened_output, N, C, H, W, Kh, Kw);
	// t1.stop();

	// int outH = ((H + 2 * padding - Kh) / stride) + 1;
	// int outW = ((W + 2 * padding - Kw) / stride) + 1;
	// std::vector<float> reluTensor = gpu.relu(outputTensor, N, OutC, outH, outW);
	// std::vector<float> maxpoolTensor = gpu.maxpool(reluTensor, N, OutC, outH, outW,2);
	// std::vector<float> meanTensor = gpu.mean(inputTensor, N, C, H, W);
	// std::vector<float> varianceTensor = gpu.variance(inputTensor, meanTensor,N, C, H, W);
	// std::vector<float> batchNormTensor = gpu.batchnorm(inputTensor,N, C, H, W);

    int N = 1; // Number of batches
    int C = 3; // Number of channels
    int H = 5; // Height
    int W = 5; // Width

	int outC = 3;
	int inC = C;
	int kh = 3;
	int kw = 3;
	int stride = 1;
	int padding = 0;
	std::vector<float> kernel(outC*inC*kh*kw); 
	std::vector<float> tensor1(N*C*H*W);
	std::vector<float> tensor2(N*C*H*W);
	
	for (size_t i = 0; i < kernel.size(); ++i) {
        kernel[i] = 1;
    }

	
	// for (size_t i = 0; i < input.size(); ++i) {
    //     input[i] = dis(gen);
    // }
	for(auto i=0; i<tensor1.size();i++){
		tensor1[i] = i+1;
		tensor2[i] = 1;
	}

	std::vector<float> outputTensor = gpu.convolution(tensor1, kernel, N, C, H, W, outC, kh, kw, stride, padding);
	std::vector<float> reluTensor = gpu.relu(outputTensor, N, gpu._outC, gpu._outH, gpu._outW);
	std::vector<float> maxpoolTensor = gpu.maxpool(outputTensor, N, gpu._outC, gpu._outH, gpu._outW,2);
	std::vector<float> meanTensor = gpu.mean(tensor1, N, C, H, W);
	std::vector<float> varianceTensor = gpu.variance(tensor1, meanTensor,N, C, H, W);
	std::vector<float> batchNormTensor = gpu.batchnorm(tensor1,N, C, H, W);
	//sample input gpu.concat(tensor1, tensor2, N, C1,C2,H, W) eg. (tensor, tensor2, 1, 3, 3,20, 20) -> returns (1,6,20,20)
	std::vector<float> concatTensor = gpu.concat(tensor1,tensor2, N, C,C, H, W);
	//printTensor(concatTensor,N, C+C, H, W, "concat");
	                                     //sample input gpu.upsample(inputtensor, N, C, H, W, newH, newW) eg. (tensor, 1, 3, 10, 10, 20, 20) ->returns (1,3,20,20)
	std::vector<float> upsampledTensor = gpu.upsample(tensor1, N,C, H, W, H*2, W*2);
	//printTensor(upsampledTensor,N, C, H*2, W*2,"upsampling");
	 //sample input gpu.extract(inputtensor, N, C, H, W, newH, newW) eg. (tensor, 1, 3, 20, 20, 10, 10) -> return (1,3,10,10)
	std::vector<float> centerTensor = gpu.extract(tensor1, N, C, H, W, H/2, W/2);
	//printTensor(centerTensor,N, C, H/2, W/2,"center");
	printPerformace(gpu);
}

