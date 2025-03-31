/**********************************************************************************************************************
* File name: conv2d.cpp
* 
* This program implements Conv2D and other supporting functions to implement UNET on the input image.
* This file contains CPU implemetation and Host code for GPU implementation of UNET.
* Functions in this file are used to perform convolution, relu, maxpool, batchnorm, concat, upsample and sigmoid operations on the input image.
* UNET is a convolutional neural network architecture for image segmentation.UNET includes an encoder-decoder structure with skip connections
* between the encoder and decoder paths. The encoder path captures context and the decoder path enables precise localization.
* 
* This UNET is implemented on both CPU and GPU and the performance speedup is displayed to the user.
* The output images are stored in the results directory.
* 
* Project Team:
* Raghul Raj Navaneethakrishnan (3703553)
***********************************************************************************************************************
*/


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
#include "cnpy/cnpy.h"
#include <algorithm>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
// Global variables to store the time taken by each operation on CPU
long long unet_cpu_time = 0;
long long cpu_double_conv = 0;
long long convTime = 0;
long long batchNormTime = 0;
long long reluTime = 0;
long long maxpoolTime = 0;
long long upsampleTime = 0;
long long concatTime = 0;

/**
 * @brief GPUInit class for initializing GPU parameters such as context, command queue, Kernels etc..
 * 
 * This class provides all the mentioned functionalities to initialize the GPU and run the kernels on the GPU.
 * Methods of this class are used to perform convolution, relu, maxpool, batchnorm, concat, upsample and sigmoid operations on the GPU.
 * class also provides methods to load kernel weights and biases from .npy files.
 */
class GPUInit {
	private:
		cl::Context context;
		cl::CommandQueue queue;
		cl::Program program;
		cl::Device device;

		cl::Kernel convKernel;
		cl::Kernel convKernelOptimizedwbias;
		cl::Kernel reluKernel;
		cl::Kernel maxpoolKernel;
		cl::Kernel upsampleKernel;
		cl::Kernel batchnormKernel;
		cl::Kernel concatKernel;
		cl::Kernel sigmoidKernel;
	
		std::vector<cl::Device> devices;

		// Function to determine local size based on global size and device limits
		// This function calculates the optimal local work group size for the given global work sizes (global_x, global_y, global_z)
		// and the device's maximum work group size. Returns find the largest possible local size that divides the global size
		// evenly while respecting the device's maximum work group size constraints.
		cl::NDRange determineLocalSize(const cl::Device& device, size_t global_x, size_t global_y, size_t global_z) {
			size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
			size_t max_local = std::sqrt(maxWorkGroupSize);
			for (int i = std::min({max_local, global_y, global_z}); i >= 1; --i) {
				if (global_y % i == 0 && global_z % i == 0) {
					return cl::NDRange(1, i, i);
				}
			}
			return cl::NDRange(1, 1, 1);
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
		/**
		 * @brief Loads kernel weights or biases from a .npy file.
		 * 
		 * This function reads a NumPy .npy file containing kernel weights or biases
		 * and converts the data into a 1D vector of floats. function supports
		 * 1D, 2D, and 4D arrays and reshapes them into a flat vector.
		 * 
		 * @param filename name of the file (without extension) to load. function
		 *                 assumes the file is located in the "../pretrainedKernels/" directory
		 *                 and appends the ".npy" extension automatically.
		 * 
		 * @return A std::vector<float> containing the kernel weights or biases in a flat format.
		 * 
		 * @throws std::runtime_error If the kernel shape is unsupported or if the file cannot be loaded.
		 * 
		 * @details
		 * - For 1D arrays, the shape is interpreted as (1, 1, 1, size).
		 * - For 2D arrays, the shape is interpreted as (1, 1, height, width).
		 * - For 4D arrays, the shape is interpreted as (n, c, h, w).
		 * - function calculates the total size of the array and copies the data into a vector.
		 */
		std::vector<float> loadKernelfromNPY(const std::string &filename) {
			std::string fullPath = "../pretrainedKernels/" + filename + ".npy";
			cnpy::NpyArray arr = cnpy::npy_load(fullPath);
			if (arr.word_size != sizeof(float)) {
				throw std::runtime_error("Unsupported data type. Only float32 is supported for Kernel and bias.");
			}
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
		
			size_t total_size = n * c * h * w;
			std::vector<float> kernel(raw_data, raw_data + total_size);
			return kernel;
		}

	public:
	int _outH, _outW,_outC;
	

	GPUInit(){
		//Constructor to initialize the GPU, Context, device queue and load the GPU kernels//
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
			convKernelOptimizedwbias = cl::Kernel(program, "conv2dOptimizedwbias");
			reluKernel      = cl::Kernel(program, "relu_activation");
			maxpoolKernel   = cl::Kernel(program, "maxpool");
			batchnormKernel = cl::Kernel(program, "batch_norm");
			concatKernel    = cl::Kernel(program, "concatenate_tensors");
			upsampleKernel  = cl::Kernel(program, "upsample_");
			sigmoidKernel  = cl::Kernel(program, "sigmoid");

		} catch (OpenCL::Error &e) {
			std::cerr << "Error creating kernel: " << e.what() << std::endl;
			throw std::runtime_error("Failed to create kernel.");
		}
	}

	void getGpuDetails(){
		std::cout << "Device: "                   << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "Max Work Group Size: "      << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
		std::cout << "Max Compute Units: "        << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
		std::cout << "Global Memory Size: "       << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/ 1048576 << " MiB" << std::endl;
		std::cout << "Local Memory Size: "        << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << " bytes" << std::endl;
		std::cout << "Max Work Item Dimensions: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
		std::cout << "Max work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
		auto maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
		std::cout << "Max Work Item Sizes: ";
		for (const auto& size : maxWorkItemSizes) {
			std::cout << size << " ";
		}
		std::cout<<"\n";
	}
	//To measure the time taken by each operation on GPU
	double convOp=0 , convCopy=0 ;
	double convTiledOp=0 , convTiledCopy=0 ;
	double reluOp=0, reluCopy=0;
	double maxpoolOp=0, maxpoolCopy=0;
	double bnOp=0, bnCopy=0;
	double concatOp=0, concatCopy=0;
	double upsampleOp=0, upsampleCopy=0;
	
	//Methods to print the dimensions of the current output tensor of GPU operation
	//After every operation the output tensor dimensions are updated as _outC, _outH, _outW
	//These methods can be used to print the dimensions of the output tensor
	void printdims(std::string name="Output Dims")
	{
		std::cout<<name<<": "<<_outC<<" "<<_outH<<" "<<_outW<<std::endl;
	}
	
	/**
	 * @brief Naive Convolution kernel for 2D Convolution
	 * 
	 * This function accepts a 4D input tensor (N, C, H, W) and file name of the kernel weights, biases and performs a 2D convolution
	 * Tensor of shape (N, C, H, W) is convolved with kernel of shape (OutC, C, Kh, Kw) with stride and padding. 
	 * stride=1 and padding=1 by default, makes input Height and width same as output height and width.
	 * Input(N,C,H,W) is convolved with kernel(OutC, C, Kh, Kw) to produce output(N, OutC, H, W)
	 * @param input input tensor in NCHW format
	 * @param filename name of the pretained kernel and bias (without extension .npy)
	 * @param N batch size
	 * @param C number of input channels
	 * @param H height of the input
	 * @param W width of the input
	 * @param OutC number of output channels
	 * @param Kh kernel height
	 * @param Kw kernel width
	 * @param stride stride 
	 * @param padding padding 
	 * @return A std::vector<float> convolution output
	 * 
	 */
	std::vector<float> convolution(std::vector<float> &input,const std::string &filename,int N, int C, int H, int W, int OutC, int Kh, int Kw, int stride=1, int padding=1 ){
		int outH = (H + 2 * padding - Kh) / stride + 1;
		int outW = (W + 2 * padding - Kw) / stride + 1;

		std::vector<float> output(N*OutC*outH* outW);
		std::vector<float> kernel = loadKernelfromNPY(filename+"_weights"); //load pretrained weights from .npy file with shape (OutC, C, Kh, Kw)
		std::vector<float> bias = loadKernelfromNPY(filename+"_bias"); //load pretrained bias from .npy file with shape (OutC)
		if(kernel.size() != OutC*C*Kh*Kw){
			throw std::runtime_error("Size of kernel should be equal to the number of output channels");
		}

		if(bias.size() != OutC){
			throw std::runtime_error("Size of bias should be equal to the number of output channels");
		}
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*input.size(), input.data());
		cl::Buffer kernelBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*kernel.size(), kernel.data());
		cl::Buffer biasBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*bias.size(), bias.data());
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
		convKernel.setArg(12, biasBuffer);
        
		cl::Event event[2]; // Create events to measure time
		cl::NDRange global(OutC, outH, outW);
		queue.enqueueNDRangeKernel(convKernel, cl::NullRange, global, cl::NDRange(),NULL,&event[0]);
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float)*output.size(), output.data(),NULL,&event[1]);
		queue.finish();

		_outC = OutC;
		_outH = outH;
		_outW = outW;
		convOp = convOp + OpenCL::getElapsedTime(event[0]).getMilliseconds();
		convCopy = convCopy+ OpenCL::getElapsedTime(event[1]).getMilliseconds();
		return output;
	}

	/*Convolution with kernel tennsor is passed as input parameter for testing purposes*/
	std::vector<float> convolution(std::vector<float> &input,std::vector<float> &kernel,int N, int C, int H, int W, int OutC, int Kh, int Kw, int stride=1, int padding=1 ){
		int outH = (H + 2 * padding - Kh) / stride + 1;
		int outW = (W + 2 * padding - Kw) / stride + 1;

		std::vector<float> output(N*OutC*outH* outW);
		std::vector<float> bias(OutC,0.001f);
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*input.size(), input.data());
		cl::Buffer kernelBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*kernel.size(), kernel.data());
		cl::Buffer biasBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*kernel.size(), bias.data());
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
		convKernel.setArg(12, biasBuffer);
        // Set up work sizes (output size)
		cl::Event event[2];
		cl::NDRange global(OutC, outH, outW);
		queue.enqueueNDRangeKernel(convKernel, cl::NullRange, global, cl::NDRange(),NULL,&event[0]);

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

	/**
	@brief Convolution optimized with tiling
	*
	inputTensor and Kernel tensor are loaded into local memory for faster access and then convolution is performed
	This reduces the global memory access and helps in faster computation. 
	* Input(N,C,H,W) is convolved with kernel(OutC, C, Kh, Kw) to produce output(N, OutC, H, W)
	* @param input input tensor in NCHW format
	* @param filename name of the pretained kernel and bias (without extension .npy)
	* @param N batch size
	* @param C number of input channels
	* @param H height of the input
	* @param W width of the input
	* @param OutC number of output channels
	* @param Kh kernel height
	* @param Kw kernel width
	* @param stride stride 
	* @param padding padding 
	* @return A std::vector<float> convolution output
	* 
	*/
	std::vector<float> convolutionOptimizedwbias(std::vector<float> &input, const std::string &filename,int N, int C, int H, int W, int OutC, int Kh=3, int Kw=3, int stride=1, int padding=1 ){
		int outH = (H + 2 * padding - Kh) / stride + 1;
		int outW = (W + 2 * padding - Kw) / stride + 1;

		std::vector<float> output(N*OutC*outH* outW);
		std::vector<float> kernel = loadKernelfromNPY(filename+"_weights");
		std::vector<float> bias = loadKernelfromNPY(filename+"_bias");

		if(kernel.size() != OutC*C*Kh*Kw){
			throw std::runtime_error("Size of kernel should be equal to the number of output channels");
		}

		if(bias.size() != OutC){
			throw std::runtime_error("Size of bias should be equal to the number of output channels");
		}
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*input.size(), input.data());
		cl::Buffer kernelBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*kernel.size(), kernel.data());
		cl::Buffer biasBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*bias.size(), bias.data());
		cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*output.size());

		cl::NDRange global_size(OutC, outH, outW);
		cl::NDRange local_size = determineLocalSize(device, N * OutC, outH, outW); //global size and local size must be perfectly divisible and should be within the hardware limits
		size_t LOCAL_SIZE_H = local_size[1];
		size_t LOCAL_SIZE_W = local_size[2];
		int local_input_h = (LOCAL_SIZE_H - 1) * stride + Kh;
		int local_input_w = (LOCAL_SIZE_W - 1) * stride + Kw;
		const size_t local_mem_size = C * local_input_h * local_input_w * sizeof(float); // 3 * 18 * 18 * 4 bytes
		convKernelOptimizedwbias.setArg(0, inputBuffer);
		convKernelOptimizedwbias.setArg(1, kernelBuffer);
		convKernelOptimizedwbias.setArg(2, outputBuffer);
		convKernelOptimizedwbias.setArg(3, biasBuffer);
		convKernelOptimizedwbias.setArg(4, N); // Batch size
		convKernelOptimizedwbias.setArg(5, C); // Number of input channels
		convKernelOptimizedwbias.setArg(6, H); // Height of the input
		convKernelOptimizedwbias.setArg(7, W); // Width of the input
		convKernelOptimizedwbias.setArg(8, OutC); // Number of output channels
		convKernelOptimizedwbias.setArg(9, Kh); // Kernel height
		convKernelOptimizedwbias.setArg(10, Kw); // Kernel width
		convKernelOptimizedwbias.setArg(11, stride); // Stride
		convKernelOptimizedwbias.setArg(12, padding); // Padding
		convKernelOptimizedwbias.setArg(13, cl::Local(local_mem_size)); // Local memory for input_tile
		
		cl::Event event[2];
		queue.enqueueNDRangeKernel(convKernelOptimizedwbias, cl::NullRange, global_size, local_size,NULL,&event[0]);
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float)*output.size(), output.data(),NULL,&event[1]);
		queue.finish();
		
		_outC = OutC;
		_outH = outH;
		_outW = outW;
		convTiledOp = convTiledOp + OpenCL::getElapsedTime(event[0]).getMilliseconds();
		convTiledCopy = convTiledCopy+ OpenCL::getElapsedTime(event[1]).getMilliseconds();
		return output;
	}

	std::vector<float> convolutionOptimizedwbias(std::vector<float> &input, std::vector<float> &kernel,int N, int C, int H, int W, int OutC, int Kh=3, int Kw=3, int stride=1, int padding=1 ){
		int outH = (H + 2 * padding - Kh) / stride + 1;
		int outW = (W + 2 * padding - Kw) / stride + 1;

		std::vector<float> output(N*OutC*outH* outW);
		std::vector<float> bias(OutC,0.001f);	
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*input.size(), input.data());
		cl::Buffer kernelBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*kernel.size(), kernel.data());
		cl::Buffer biasBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*bias.size(), bias.data());
		cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*output.size());
		
		cl::NDRange global_size(OutC, outH, outW);
		cl::NDRange local_size = determineLocalSize(device, N * OutC, outH, outW); //global size and local size must be perfectly divisible and should be within the hardware limits
		size_t LOCAL_SIZE_H = local_size[1];
		size_t LOCAL_SIZE_W = local_size[2];
		int local_input_h = (LOCAL_SIZE_H - 1) * stride + Kh;
		int local_input_w = (LOCAL_SIZE_W - 1) * stride + Kw;
		const size_t local_mem_size = C * local_input_h * local_input_w * sizeof(float); // 3 * 18 * 18 * 4 bytes
		convKernelOptimizedwbias.setArg(0, inputBuffer);
		convKernelOptimizedwbias.setArg(1, kernelBuffer);
		convKernelOptimizedwbias.setArg(2, outputBuffer);
		convKernelOptimizedwbias.setArg(3, biasBuffer);
		convKernelOptimizedwbias.setArg(4, N); // Batch size
		convKernelOptimizedwbias.setArg(5, C); // Number of input channels
		convKernelOptimizedwbias.setArg(6, H); // Height of the input
		convKernelOptimizedwbias.setArg(7, W); // Width of the input
		convKernelOptimizedwbias.setArg(8, OutC); // Number of output channels
		convKernelOptimizedwbias.setArg(9, Kh); // Kernel height
		convKernelOptimizedwbias.setArg(10, Kw); // Kernel width
		convKernelOptimizedwbias.setArg(11, stride); // Stride
		convKernelOptimizedwbias.setArg(12, padding); // Padding
		convKernelOptimizedwbias.setArg(13, cl::Local(local_mem_size)); // Local memory for input_tile
		
		cl::Event event[2];
		queue.enqueueNDRangeKernel(convKernelOptimizedwbias, cl::NullRange, global_size, local_size,NULL,&event[0]);
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float)*output.size(), output.data(),NULL,&event[1]);
		queue.finish();
		
		_outC = OutC;
		_outH = outH;
		_outW = outW;
		convTiledOp = convTiledOp + OpenCL::getElapsedTime(event[0]).getMilliseconds();
		convTiledCopy = convTiledCopy+ OpenCL::getElapsedTime(event[1]).getMilliseconds();
		return output;
	}

	/**
	 * @brief Relu activation function
	 * 
	 * This function accepts a 4D input tensor (N, C, H, W) and performs a ReLU activation function on the input tensor. 
	 * If the input value is less than 0, it is set to 0 else same value is set.(output = max(0, input))
	 * @param input input tensor in NCHW format
	 * @param filename name of the file (without extension) to load.
	 * @param N batch size
	 * @param C number of input channels
	 * @param H height of the input
	 * @param W width of the input
	 * @return A std::vector<float> relu activated output
	 * 
	 */
	std::vector<float> relu(std::vector<float> &input, int N, int C, int H, int W){
		std::vector<float> output(N*C*H*W);
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
		cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * input.size());

		reluKernel.setArg(0, inputBuffer);  
		reluKernel.setArg(1, outputBuffer);
		reluKernel.setArg(2, N*C*H*W);

		cl::NDRange global(C*H*W);  //size of input tensor
		cl::Event event[2];
		queue.enqueueNDRangeKernel(reluKernel, cl::NullRange, global, cl::NDRange(),NULL,&event[0]);
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * output.size(), output.data(),NULL,&event[1]);
		queue.finish();

		reluOp += OpenCL::getElapsedTime(event[0]).getMilliseconds();
		reluCopy += OpenCL::getElapsedTime(event[1]).getMilliseconds();
		
		_outC = C; //global output shape is updated
		_outH = H;
		_outW = W;
		return output;
	}

	/**
	 * @brief Maxpool operation
	 * 
	 * This function accepts a 4D input tensor (N, C, H, W) and performs a maxpool operation on the input tensor.
	 * function calculates the output tensor using the maxpool operation with a window size of 2x2 and stride of 2 by default.alignas
	 * Size of the output tensor is N, C, H/2, W/2. For each cell in output tensor is the maximum value of the 2x2 window in the input tensor strided by 2.
	 * @param input input tensor in NCHW format
	 * @param N batch size
	 * @param C number of input channels
	 * @param H height of the input
	 * @param W width of the input
	 * @param pool_size size of the maxpool window default= 2
	 * @param stride stride of the maxpool operation default = 2
	 * @return A std::vector<float> maxpooled output
	 * 
	 */
	std::vector<float> maxpool(std::vector<float> &input, int N, int C, int H, int W, int pool_size=2, int stride=2){

		int outH = (H - pool_size) / stride + 1; //outH = H/2
		int outW = (W - pool_size) / stride + 1; //outW = W/2

		std::vector<float> output(N*C*outH*outW);
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
		cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());

		maxpoolKernel.setArg(0, inputBuffer);
		maxpoolKernel.setArg(1, outputBuffer);
		maxpoolKernel.setArg(2, N); // Batch size
		maxpoolKernel.setArg(3, C); // Number of channels
		maxpoolKernel.setArg(4, H); // Input height
		maxpoolKernel.setArg(5, W); // Input width
		maxpoolKernel.setArg(6, pool_size); // Pool size is 2x2 by default
		maxpoolKernel.setArg(7, stride); // Stride of maxpool is set to 2 by default

		cl::NDRange global(C ,outH ,outW); 

		cl::Event event[2];
		queue.enqueueNDRangeKernel(maxpoolKernel, cl::NullRange, global, cl::NDRange(), NULL,&event[0]);
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * output.size(), output.data(),NULL,&event[1]);
		queue.finish();

		maxpoolOp += OpenCL::getElapsedTime(event[0]).getMilliseconds();
		maxpoolCopy += OpenCL::getElapsedTime(event[1]).getMilliseconds();
		
		_outC = C;
		_outH = outH; //global output shape is updated
		_outW = outW;
		return output;
	}

	/**
	 * @brief Batchnorm Inference
	 * 
	 * This function accepts a 4D input tensor (N, C, H, W) and file name of gamma, beta, running mean, running variance and performs a batch normalization operation.
	 * Size of gamma, beta, mean, variance is equal to the number of input channels.
	 * For all the values in a channel, batch normalization is performed using the formula
	 * value = gamma * (input - mean) / sqrt(variance + epsilon) + beta
	 * @param input input tensor in NCHW format
	 * @param filename Filename for the gamma, beta, mean, variance only prefix is needed
	 * @param N batch size
	 * @param C number of input channels
	 * @param H height of the input
	 * @param W width of the input
	 * @param OutC number of output channels
	 * @param Kh kernel height
	 * @param Kw kernel width
	 * @param stride stride of the convolution
	 * @param padding padding of the convolution
	 * @return A std::vector<float> normalized output
	 * 
	 */
	std::vector<float> batchnormInference(std::vector<float>& input,const std::string &filename, int N, int C, int H, int W) {
        // loading gamma, beta, running mean and running variance from pre-trained kernels each of shape (C)
		std::vector<float>gamma = loadKernelfromNPY(filename+"_gamma");
		std::vector<float>beta = loadKernelfromNPY(filename+"_beta");
		std::vector<float> mean = loadKernelfromNPY(filename+"_rmean");
		std::vector<float> variance = loadKernelfromNPY(filename+"_rvar");
        
		if(gamma.size() != C || beta.size() != C || mean.size() != C || variance.size() != C){
			std::cout<<"gamma- "<<gamma.size()<<" beta- "<<beta.size()<<" mean-"<<mean.size()<<" variance- "<<variance.size()<<std::endl;
			throw std::runtime_error("Size of gamma, beta, mean, variance should be equal to the number of input channels");
		}
		// Output tensor
		
		std::vector<float> output(N * C * H * W);
		
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
		
		cl::Buffer varianceBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * variance.size(), variance.data());
		cl::Buffer meanBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * mean.size(), mean.data());
        cl::Buffer gammaBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * gamma.size(), gamma.data());
        cl::Buffer betaBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * beta.size(), beta.data());
		cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());

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

        cl::NDRange bglobal(C, H , W);
		cl::Event event[2];
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
	
	std::vector<float> batchnormInference(std::vector<float>& input,std::vector<float>& gamma,
											std::vector<float>& beta,std::vector<float>& mean,std::vector<float>& variance, int N, int C, int H, int W) {
        // Output tensor
        std::vector<float> output(N * C * H * W);
		
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.size(), input.data());
		
		cl::Buffer varianceBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * variance.size(), variance.data());
		cl::Buffer meanBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * mean.size(), mean.data());
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

        cl::NDRange bglobal(C, H , W);
		cl::Event event[2];
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

	/**
	 * @brief Concatenation of two tensors
	 * 
	 * This function accepts two 4D input tensors: Tensor1(N, C1, H, W) and Tensor2(N, C2, H, W) and concatenates them along the channel dimension.
	 * output tensor will have the shape (N, C1+C2, H, W). In the output tensor, Tensor1 is placed first and then Tensor2 is placed.
	 * from channel 0 to C1-1, Tensor1 values are placed and from C1 to C1+C2-1, Tensor2 values are placed.
	 * H and W of both the input tensors should be same.
	 * @param input1 first input tensor in NCHW format
	 * @param input2 second input tensor in NCHW format
	 * @param N batch size
	 * @param C1 number of input channels in the first tensor
	 * @param C2 number of input channels in the second tensor
	 * @param H height of the input
	 * @param W width of the input
	 * @return A std::vector<float> concatenated output
	 * 
	 */
	std::vector<float> concat(std::vector<float> &input1,std::vector<float> &input2, int N, int C1,int C2, int H, int W){
		if(input1.size() != N*C1*H*W || input2.size() != N*C2*H*W){
			throw std::runtime_error("Input tensor dimensions do not match, ensure Tensor dimeensions match given N, C1, C2, H, W");
		}

		if(input1.size()/C1 != H*W || input2.size()/C2 != H*W){
			throw std::runtime_error("Input tensor dimensions do not match, H & W both tensors should be the same");
		}
		
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

		cl::NDRange global(N, C3, H * W); // Global work size
		cl::NDRange local(1, 1, 1);
		
		cl::Event event[2];
		queue.enqueueNDRangeKernel(concatKernel, cl::NullRange, global, local,NULL, &event[0]);
		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, output.size() * sizeof(float), output.data(), NULL, &event[1]);
		queue.finish();
		concatOp += OpenCL::getElapsedTime(event[0]).getMilliseconds();
		concatCopy += OpenCL::getElapsedTime(event[1]).getMilliseconds();
		_outC = C1+C2;
		_outH = H;
		_outW = W;
		return output;
	}

	std::vector<float> upsample(std::vector<float> &input, int N, int C,int H, int W, int newH, int newW){
		if(newH%H!=0 || newW%W!=0){
			throw std::runtime_error("Upsampling factor should be a multiple of the input tensor dimensions");
		}
		if(newH<H || newW<W){
			throw std::runtime_error("Upsampling factor should be greater than the input tensor dimensions eg. H = 64, newH = 128");
		}
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

		cl::Event event[2];
		cl::NDRange global(C, newH , newW); 

		queue.enqueueNDRangeKernel(upsampleKernel, cl::NullRange, global, cl::NDRange(), NULL, &event[0]);
		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, output.size() * sizeof(float), output.data(), NULL,&event[1]);
		queue.finish();
		upsampleOp += OpenCL::getElapsedTime(event[0]).getMilliseconds();
		upsampleCopy += OpenCL::getElapsedTime(event[1]).getMilliseconds();
		_outC = C;
		_outH = newH;
		_outW = newW;
		return output;

	}

	std::vector<float> sigmoidActivation(std::vector<float> &input, int N, int C,int H, int W){

		std::vector<float> output(N * C * H * W, 0.0f);
		cl::Buffer buffer_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), input.data());
		cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, output.size() * sizeof(float));
		
		sigmoidKernel.setArg(0, buffer_input);
		sigmoidKernel.setArg(1, buffer_output);
		sigmoidKernel.setArg(2, C*H*W);

		cl::Event event[2];
		cl::NDRange global(C*H*W); // Global work size
		
		queue.enqueueNDRangeKernel(sigmoidKernel, cl::NullRange, global, cl::NDRange(), NULL, &event[0]);
		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, output.size() * sizeof(float), output.data(), NULL,&event[1]);
		queue.finish();
		return output;
	}
};
//-----------------------------------------------------------------------------Timer Class--------------------------------------------------------------------------------------------
/**
 * @brief Timer class for measuring the execution time of a code block.
 * 
 * This class provides a simple way to measure the execution time of a code block.
 * timer starts when the object is created and stops when the object goes out of scope.
 * total duration can be obtained using the get_total_duration() method.
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;
	long long& total_duration_ms;

public:
    Timer(long long& duration_ref,const std::string& function_name = "Function") : name(function_name), total_duration_ms(duration_ref) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
    }

    void stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
		total_duration_ms += duration.count();
        //std::cout << name << " - " << duration.count()<< " ms" << std::endl;
    }
	long long get_total_duration() const {
        return total_duration_ms;
    }
};

std::vector<float> loadKernelfromNPY(const std::string &filename) {
	std::string fullPath = "../pretrainedKernels/" + filename + ".npy";
	cnpy::NpyArray arr = cnpy::npy_load(fullPath);
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

	size_t total_size = n * c * h * w;
	std::vector<float> kernel(raw_data, raw_data + total_size);
	return kernel;
}

/**
 * @brief Load image from NPY file
 * 
 * This function loads a 4D image tensor from a NPY file. The expected shape of the image is (n, c, h, w).
 * This is impemented to avoid the confusion of image shapes either RGB or BGR or (N, H, W, W) or (N, C, H, W)
 * For testing purposes, Image is always loaded in the npy format with shape (N, C, H, W) where N=1, C=3, H=224, W=224
 * 
 * @param filename The name of the NPY file to load.
 * @return A std::vector<float> containing the loaded image data.
 */
std::vector<float> loadImagefromNPY(const std::string &filename) {
	cnpy::NpyArray arr = cnpy::npy_load(filename);
	float* raw_data = arr.data<float>();
	int n,c,h,w;
	if (arr.shape.size() == 4) {
		// For 4D array, it is in the format (n, c, h, w)
		n = arr.shape[0];
		c = arr.shape[1];
		h = arr.shape[2];
		w = arr.shape[3];
	} 
	else {
		throw std::runtime_error("Unsupported image shape.");
	}
	size_t total_size = n * c * h * w;
	std::vector<float> kernel(raw_data, raw_data + total_size);
	return kernel;
}

/**
 * @brief Performs the double convolution operation in the U-Net architecture.
 * This function performs the double convolution operation in the U-Net architecture, which includes:
 * conv1 -> batchnorm1 ->relu1 ->conv2 ->batchnorm2 -> relu2
 * For infenrence, Batchnorm uses the gamma, beta, running mean and running variance from the pretrained kernel
 * @param gpu GPUInit object used for GPU operations.
 * @param input input tensor in NCHW format (batch size, channels, height, width).
 * @param kernel1 filename prefix for the first convolutional kernel weights and biases.
 * @param kernel2 filename prefix for the second convolutional kernel weights and biases.
 * @param bn1 filename prefix for the first batch normalization parameters (gamma, beta, mean, variance).
 * @param bn2 filename prefix for the second batch normalization parameters (gamma, beta, mean, variance).
 * @param N batch size.
 * @param C number of input channels.
 * @param H height of the input tensor.
 * @param W width of the input tensor.
 * @param outC1 number of output channels for the first convolutional layer.
 * @param outC2 number of output channels for the second convolutional layer.
 * 
 * @return A std::vector<float> containing the output tensor after the downward operation.
 * 
 * @details
 * - function applied double convolution operating with same padding, which maintains the input H, W equals to output H, W
 * - Two convolutional layers are applied sequentially, each followed by batch normalization and ReLU activation.
 * - output tensor has the same spatial dimensions as the input tensor but with the specified number of output channels.
 */
std::vector<float> doubleConvolution(GPUInit& gpu, std::vector<float>& input, const std::string& kernel1,const std::string& kernel2, const std::string& bn1,const std::string& bn2,int N, int C, int H, int W, int outC1,int outC2){
	std::vector<float> outputTensor     = gpu.convolution(input, kernel1, N, C, H, W, outC1, 3, 3, 1, 1); //kernel size 3x3, stride 1, padding 1
	std::vector<float> batchNormTensor  = gpu.batchnormInference(outputTensor,bn1,N, gpu._outC, gpu._outH, gpu._outW);
	std::vector<float> reluTensor       = gpu.relu(batchNormTensor, N, gpu._outC, gpu._outH, gpu._outW);
	std::vector<float> outputTensor2    = gpu.convolution(reluTensor, kernel2, N, gpu._outC, gpu._outH, gpu._outW, outC2, 3, 3, 1, 1);
	std::vector<float> batchNormTensor2 = gpu.batchnormInference(outputTensor2,bn2,N, gpu._outC, gpu._outH, gpu._outW);
	std::vector<float> reluTensor2      = gpu.relu(batchNormTensor2, N, gpu._outC, gpu._outH, gpu._outW);
	return reluTensor2;
}

/**
 * @brief Performs the upward operation in the U-Net architecture.
 * 
 * This function performs the upward operation in the U-Net architecture, which includes:
 * - Upsampling the input tensor using nearest-neighbor interpolation.
 * - Concatenating the upsampled tensor with a skip connection tensor from the encoder path.
 * - Applying two consecutive convolutional layers with batch normalization and ReLU activation.
 * 
 * @param gpu GPUInit object used for GPU operations.
 * @param input input tensor in NCHW format (batch size, channels, height, width).
 * @param kernel1 filename prefix for the first convolutional kernel weights and biases.
 * @param kernel2 filename prefix for the second convolutional kernel weights and biases.
 * @param bn1 filename prefix for the first batch normalization parameters (gamma, beta, mean, variance).
 * @param bn2 filename prefix for the second batch normalization parameters (gamma, beta, mean, variance).
 * @param concatTensor tensor from the encoder path to be concatenated with the upsampled tensor.
 * @param N batch size.
 * @param C number of input channels.
 * @param H height of the input tensor.
 * @param W width of the input tensor.
 * @param concatChannel number of channels in the concatenation tensor.
 * @param outC1 number of output channels for the first convolutional layer.
 * @param outC2 number of output channels for the second convolutional layer.
 * 
 * @return A std::vector<float> containing the output tensor after the upward operation.
 * 
 * @details
 * - function first upsamples the input tensor to double its spatial dimensions using nearest-neighbor interpolation.
 * - upsampled tensor is concatenated with the skip connection tensor along the channel dimension.
 * - Two convolutional layers are applied sequentially, each followed by batch normalization and ReLU activation.
 * - output tensor has the same spatial dimensions as the concatenated tensor but with the specified number of output channels.
 */
std::vector<float> upward(GPUInit& gpu, std::vector<float>& input, const std::string& kernel1,const std::string& kernel2,const std::string& bn1,const std::string& bn2,std::vector<float>& concatTensor, int N, int C, int H, int W,int concatChannel, int outC1,int outC2){
	std::vector<float> upsampleTensor = gpu.upsample(input, N, C, H, W, H*2, W*2);
	std::vector<float> concatedTensor = gpu.concat(upsampleTensor, concatTensor, N, gpu._outC, concatChannel, gpu._outH, gpu._outW);
	std::vector<float> outputTensor = doubleConvolution(gpu,concatedTensor, kernel1, kernel2,bn1,bn2, N, gpu._outC, gpu._outH, gpu._outW, outC1, outC2);
	return outputTensor;
}
//-----------------------------------------------------------------------------CPU Implementation--------------------------------------------------------------------------------------------
//With reference to naive 1D convolution https://gist.github.com/cuongvng/cef5620a5a44971bdb019494188c27ab
std::vector<float> conv2d_cpu(const std::vector<float>& input,
                              const std::vector<float>& kernel,
                              int N, int C, int H, int W, int OutC,
                              int Kh, int Kw, int stride=1, int padding=1) {
    // Calculate output dimensions
    int outH = (H + 2 * padding - Kh) / stride + 1;
    int outW = (W + 2 * padding - Kw) / stride + 1;
	std::vector<float> bias(OutC, 0.001f);
    // Validate output dimensions
    if (outH <= 0 || outW <= 0) {
        throw std::invalid_argument("Output dimensions must be positive");
    }

    // Validate input and kernel sizes
    if (input.size() != static_cast<size_t>(N * C * H * W)) {
        throw std::invalid_argument("Input tensor size does not match expected dimensions");
    }
    if (kernel.size() != static_cast<size_t>(OutC * C * Kh * Kw)) {
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
                                int in_h = out_h * stride + kh - padding;
                                int in_w = out_w * stride + kw - padding;
                                if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
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

    return output;
}

std::vector<float> conv2d_cpu(const std::vector<float>& input,
								const std::string& kernelname,
                              int N, int C, int H, int W, int OutC,
                              int Kh, int Kw, int stride=1, int padding=1) {
	auto start_time = std::chrono::high_resolution_clock::now();
    // Calculate output dimensions
    int outH = (H + 2 * padding - Kh) / stride + 1;
    int outW = (W + 2 * padding - Kw) / stride + 1;
	
	std::vector<float> kernel = loadKernelfromNPY(kernelname+"_weights");
	std::vector<float> bias = loadKernelfromNPY(kernelname+"_bias");
	
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
	convTime += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    return output;
}

std::vector<float> batch_norm_cpu(const std::vector<float>& input,const std::string& filename,int N,  int C, int H, int W) 
{
	auto start_time = std::chrono::high_resolution_clock::now();
	const std::vector<float> gamma = loadKernelfromNPY(filename+"_gamma");
	const std::vector<float> beta = loadKernelfromNPY(filename+"_beta");
	const std::vector<float> mean = loadKernelfromNPY(filename+"_rmean");
	const std::vector<float> variance = loadKernelfromNPY(filename+"_rvar");
	size_t expected_size = static_cast<size_t>(N * C * H * W);
	assert(input.size() == expected_size && "Input size does not match N * C * H * W");
	assert(gamma.size() == static_cast<size_t>(C) && "Gamma size does not match C");
	assert(beta.size() == static_cast<size_t>(C) && "Beta size does not match C");
	assert(mean.size() == static_cast<size_t>(C) && "Mean size does not match C");
	assert(variance.size() == static_cast<size_t>(C) && "Variance size does not match C");

	// Initialize output vector
	std::vector<float> output(expected_size);
	
	// Batch normalization computation
	for (int c = 0; c < C; ++c) {
	// Precompute inverse standard deviation for the channel
	float mean_val = mean[c];
	float inv_std = 1.0f / std::sqrt(variance[c] + 1e-5f); // epsilon for stability
	float gamma_val = gamma[c];
	float beta_val = beta[c];

	for (int n = 0; n < N; ++n) {
		for (int h = 0; h < H; ++h) {
			for (int w = 0; w < W; ++w) {
				int idx = n * C * H * W + c * H * W + h * W + w;
				float normalized = (input[idx] - mean_val) * inv_std;
				output[idx] = gamma_val * normalized + beta_val;
				}
			}
		}
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	batchNormTime += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	return output;
}

std::vector<float> batch_norm_cpu(const std::vector<float>& input,std::vector<float>& gamma,
											std::vector<float>& beta,std::vector<float>& mean,std::vector<float>& variance,int N,  int C, int H, int W) 
{
	size_t expected_size = static_cast<size_t>(N * C * H * W);
	assert(input.size() == expected_size && "Input size does not match N * C * H * W");
	assert(gamma.size() == static_cast<size_t>(C) && "Gamma size does not match C");
	assert(beta.size() == static_cast<size_t>(C) && "Beta size does not match C");
	assert(mean.size() == static_cast<size_t>(C) && "Mean size does not match C");
	assert(variance.size() == static_cast<size_t>(C) && "Variance size does not match C");

	// Initialize output vector
	std::vector<float> output(expected_size);

	// Batch normalization computation
	for (int c = 0; c < C; ++c) {
	// Precompute inverse standard deviation for the channel
	float mean_val = mean[c];
	float inv_std = 1.0f / std::sqrt(variance[c] + 1e-5f); // epsilon for stability
	float gamma_val = gamma[c];
	float beta_val = beta[c];

	for (int n = 0; n < N; ++n) {
		for (int h = 0; h < H; ++h) {
			for (int w = 0; w < W; ++w) {
				int idx = n * C * H * W + c * H * W + h * W + w;
				float normalized = (input[idx] - mean_val) * inv_std;
				output[idx] = gamma_val * normalized + beta_val;
				}
			}
		}
	}
	return output;
}
std::vector<float> relu_cpu(const std::vector<float>& input,int N,  int C, int H,  int W)  {
	auto start_time = std::chrono::high_resolution_clock::now();
	
	size_t expected_size = static_cast<size_t>(N * C * H * W);
	assert(input.size() == expected_size && "Input size does not match N * C * H * W");
	std::vector<float> output(expected_size);
	
	for (size_t i = 0; i < expected_size; ++i) {
		output[i] = std::max(0.0f, input[i]);
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	reluTime += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	return output;
}

std::vector<float> maxpool_cpu(const std::vector<float>& input, int N, int C, int H, int W, int pool_size=2, int stride=2) {
	auto start_time = std::chrono::high_resolution_clock::now();

	int outH = H / stride;  // Output height (integer division)
	int outW = W / stride;  // Output width (integer division)

	// Validate input dimensions
	size_t input_size = static_cast<size_t>(N * C * H * W);
	assert(input.size() == input_size && "Input size does not match N * C * H * W");
	assert(outH > 0 && outW > 0 && "Output dimensions must be positive");
	assert(H >= pool_size && W >= pool_size && "Pool size must not exceed input dimensions");

	// Initialize output vector
	size_t output_size = static_cast<size_t>(N * C * outH * outW);
	std::vector<float> output(output_size);
	
	// Perform max pooling
	for (int n = 0; n < N; ++n) {
		for (int c = 0; c < C; ++c) {
			for (int out_h = 0; out_h < outH; ++out_h) {
				for (int out_w = 0; out_w < outW; ++out_w) {
					int start_h = out_h * stride;
					int start_w = out_w * stride;
					float max_val = -std::numeric_limits<float>::infinity();

					// Pool over the window of size pool_size x pool_size
					for (int ph = 0; ph < pool_size; ++ph) {
						for (int pw = 0; pw < pool_size; ++pw) {
							int in_h = start_h + ph;
							int in_w = start_w + pw;
							if (in_h < H && in_w < W) {
								int input_idx = n * C * H * W + c * H * W + in_h * W + in_w;
								max_val = std::max(max_val, input[input_idx]);
							}
						}
					}
					// Write the maximum value to the output
					int output_idx = n * C * outH * outW + c * outH * outW + out_h * outW + out_w;
					output[output_idx] = max_val;
				}
			}
		}
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	maxpoolTime += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	return output;
}

std::vector<float> upsample_cpu(const std::vector<float>& input,int N, int C, int H,  int W, int newH, int newW) {
	// Validate input dimensions
	size_t input_size = static_cast<size_t>(N * C * H * W);
	assert(input.size() == input_size && "Input size does not match N * C * H * W");
	assert(newH > 0 && newW > 0 && "Output dimensions must be positive");
	assert(H > 0 && W > 0 && "Input dimensions must be positive");

	// Initialize output vector
	size_t output_size = static_cast<size_t>(N * C * newH * newW);
	std::vector<float> output(output_size);
	auto start_time = std::chrono::high_resolution_clock::now();
	//with reference to https://gist.github.com/folkertdev/6b930c7a7856e36dcad0a72a03e66716 - Bilinear interpolation
	for (int n = 0; n < N; ++n) {
		for (int c = 0; c < C; ++c) {
			for (int h = 0; h < newH; ++h) {
				for (int w = 0; w < newW; ++w) {

					float scaleH = static_cast<float>(H - 1) / (newH - 1);
					float scaleW = static_cast<float>(W - 1) / (newW - 1);

					float srcH = h * scaleH;
					float srcW = w * scaleW;

					int h1 = static_cast<int>(srcH);
					int w1 = static_cast<int>(srcW);
					int h2 = std::min(h1 + 1, H - 1);
					int w2 = std::min(w1 + 1, W - 1);

					float dH = srcH - h1;
					float dW = srcW - w1;
					
					// Compute indices in flattened buffer
					int idx11 = ((n * C + c) * H + h1) * W + w1; // Top-left
					int idx12 = ((n * C + c) * H + h1) * W + w2; // Top-right
					int idx21 = ((n * C + c) * H + h2) * W + w1; // Bottom-left
					int idx22 = ((n * C + c) * H + h2) * W + w2; // Bottom-right

					// Bilinear interpolation
					float value = (1 - dH) * (1 - dW) * input[idx11] +  
						(1 - dH) * dW * input[idx12] +        
						dH * (1 - dW) * input[idx21] +        
						dH * dW * input[idx22];               

					int output_idx = ((n * C + c) * newH + h) * newW + w;
					output[output_idx] = value;
				}
			}
		}
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	upsampleTime += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
	return output;
}

std::vector<float> concat_cpu(const std::vector<float>& tensor1,const std::vector<float>& tensor2, int N1, int C1, int H1, int W1, int C2) {
    // Validate input sizes (NCHW format)
	Timer co(concatTime);
    size_t tensor1_size = static_cast<size_t>(N1 * C1 * H1 * W1);
    size_t tensor2_size = static_cast<size_t>(N1 * C2 * H1 * W1);
    assert(tensor1.size() == tensor1_size && "tensor1 size must match N1 * C1 * H1 * W1");
    assert(tensor2.size() == tensor2_size && "tensor2 size must match N1 * C2 * H1 * W1");

    // Output tensor has C1 + C2 channels
    int C3 = C1 + C2;
    size_t output_size = static_cast<size_t>(N1 * C3 * H1 * W1);
    std::vector<float> output(output_size);

    // Loop over all dimensions
    for (int n = 0; n < N1; ++n) {
        for (int c = 0; c < C3; ++c) {
            for (int h = 0; h < H1; ++h) {
                for (int w = 0; w < W1; ++w) {
                    int output_idx = ((n * C3 + c) * H1 + h) * W1 + w;
                    if (c < C1) {
                        // Copy from tensor1 (first C1 channels)
                        int tensor1_idx = ((n * C1 + c) * H1 + h) * W1 + w;
                        output[output_idx] = tensor1[tensor1_idx];
                    } else {
                        // Copy from tensor2 (last C2 channels)
                        int tensor2_idx = ((n * C2 + (c - C1)) * H1 + h) * W1 + w;
                        output[output_idx] = tensor2[tensor2_idx];
                    }
                }
            }
        }
    }
	co.stop();
    return output;
}

std::vector<float> sigmoidActivation_cpu(std::vector<float> &input, int N, int C,int H, int W){
	std::vector<float> output(N * C * H * W, 0.0f);
	for(int n=0; n<N; n++){
		for(int c=0; c<C;c++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					output[n * C * H * W + c * H * W + h * W + w] = 1/(1+exp(-input[n * C * H * W + c * H * W + h * W + w]));
				}
			}
		}
	}
	return output;
}

std::vector<float> doubleConvolution_cpu(std::vector<float>& input, const std::string& kernel1,const std::string& kernel2, const std::string& bn1,const std::string& bn2,int N, int C, int H, int W, int outC1,int outC2){

	std::vector<float> outputTensor     = conv2d_cpu(input, kernel1, N, C, H, W, outC1, 3, 3, 1, 1);
	std::vector<float> batchNormTensor  = batch_norm_cpu(outputTensor,bn1,N, outC1, H, W);
	std::vector<float> reluTensor       = relu_cpu(batchNormTensor, N, outC1, H, W);
	std::vector<float> outputTensor2    = conv2d_cpu(reluTensor, kernel2, N, outC1, H, W, outC2, 3, 3, 1, 1);
	std::vector<float> batchNormTensor2 = batch_norm_cpu(outputTensor2,bn2,N, outC2, H, W);
	std::vector<float> reluTensor2      = relu_cpu(batchNormTensor2, N, outC2, H, W);
	return reluTensor2;
}

std::vector<float> upward_cpu(std::vector<float>& input, const std::string& kernel1,const std::string& kernel2,const std::string& bn1,const std::string& bn2,std::vector<float>& concatTensor, int N, int C, int H, int W,int concatChannel, int outC1,int outC2){
	std::vector<float> upsampleTensor = upsample_cpu(input, N, C, H, W, H*2, W*2);
	std::vector<float> concatedTensor = concat_cpu(upsampleTensor, concatTensor, N, C, H*2, W*2,concatChannel);
	std::vector<float> outputTensor = doubleConvolution_cpu(concatedTensor, kernel1, kernel2,bn1,bn2, N, C+concatChannel, H*2, W*2, outC1, outC2);
	return outputTensor;
}

std::vector<float> spatial_dropout_2d_cpu(const std::vector<float>& input,float dropout_rate,int N, int C, int H, int W) {

	// Validate input size (NCHW format)
	size_t expected_size = static_cast<size_t>(N * C * H * W);
	assert(input.size() == expected_size && "Input size must match N * C * H * W");

	// Initialize output as a copy of input
	std::vector<float> output = input;

	// Random number generation
	std::random_device rd;
	std::mt19937 gen(rd());
	std::bernoulli_distribution dist(1.0f - dropout_rate); // P(keep) = 1 - dropout_rate

	// Scaling factor for kept channels
	float scale = 1.0f / (1.0f - dropout_rate);
	// Apply dropout and scaling per channel
	int spatial_size = H * W;  // Number of elements per channel
	for (int n = 0; n < N; ++n) {
		for (int c = 0; c < C; ++c) {
			bool keep = dist(gen); // Randomly decide: keep (true) or drop (false)
			float factor = keep ? scale : 0.0f; // Scale if kept, zero if dropped
			// Apply to all spatial positions in this channel for this batch
			int channel_start = n * C * H * W + c * H * W;
			for (int i = 0; i < spatial_size; ++i) {
				output[channel_start + i] *= factor;
				}
			}
		}
	return output;
}


//----------------------------------------------------------------Supporting Function for additional operations ...............................................//
void getSysInfo() {
    FILE *fp = popen("grep 'model name' /proc/cpuinfo | head -n 1", "r");
    if (fp == nullptr) {
        std::cerr << "Failed to run command." << std::endl;
        return;
    }
    char buffer[256];
    if (fgets(buffer, sizeof(buffer), fp) != nullptr) {
        // Find the position of the first colon (:) and skip it
        char* cpuName = strstr(buffer, ":");
        if (cpuName != nullptr) {
            cpuName++;
            while (*cpuName == ' ') {
                cpuName++;
            }
            std::cout << "CPU: " << cpuName << std::endl;
        }
    } else {
        std::cerr << "Error reading the output." << std::endl;
    }
    pclose(fp);

	FILE *fp1 = popen("lspci | grep VGA", "r");
    if (fp1 == nullptr) {
        std::cerr << "Failed to run command." << std::endl;
        return;
    }
    char buffer1[256];
    if (fgets(buffer1, sizeof(buffer1), fp1) != nullptr) {
        // Find the position of the first colon (:) and skip it
        char* gpuName = strstr(buffer1, ": ");
        if (gpuName != nullptr) {
            gpuName++;
            while (*gpuName == ' ') {
                gpuName++;
            }
            std::cout << "GPU: " << gpuName << std::endl;
        }
    } else {
        std::cerr << "Error reading the output." << std::endl;
    }
    pclose(fp1);
}

void saveTensorToNpy(const std::string& filename,std::vector<float>& data, int N, int C, int H, int W) {
	std::string fullPath = "./npy/" + filename + ".npy";
    std::vector<size_t> shape = {N, C, H, W};
    cnpy::npy_save(fullPath, &data[0], shape, "w");
}

template <typename T>
void printVector(const std::vector<T>& vec) {
    std::cout << "[ ";
    for (T val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]\n";
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

void compareTensors(std::vector<float>& arr1, std::vector<float>& arr2, float tolerance = 0.00005){
	int count=0;
	if(arr1.size() != arr2.size()){
		std::cout<<arr1.size()<<" compared with "<<arr2.size()<<std::endl;
		std::cout<<"[ERROR] Comparing Unequal vectors"<<std::endl;
		return;
	}
	for (size_t i = 0; i < arr1.size(); ++i) {
        // Calculate absolute difference
        float diff = std::abs(arr1[i] - arr2[i]);
		if(diff>tolerance){
            ++count;
        }
    }
	std::cout<<"Number of incorrect values = "<<count<<std::endl;
	std::cout<<"Total values: "<<arr1.size()<<std::endl;
	std::cout<<"% of error = "<<((float)count/float(arr1.size()))*100<<std::endl;
}

void saveAsImage(const std::vector<float> &flattened, int N, int C, int H, int W, const std::string &filename) {
    cv::Mat img(H, W, CV_8UC1);
	std::string fullPath = "./results/" + filename+".png"; ;
	// Populate the image from the single-channel flattened array
	for (int i = 0; i < H; ++i) {
		for (int j = 0; j < W; ++j) {
			int index = i * W + j;  // Row-major order
			img.at<unsigned char>(i, j) = static_cast<unsigned char>(flattened[index] * 255.0f);
		}
	}
	if (!cv::imwrite(fullPath, img)) {
		throw std::runtime_error("Failed to save image.");
	}
}

void printPerformace(GPUInit gpu){
	std::stringstream str1;
	str1 << std::setiosflags(std::ios::left) << std::setw(20) << "Functionality";
	str1 << std::setiosflags(std::ios::right);
	str1 << " " << std::setw(9) << "| CpuTime(ms) |";
	str1 << " " << std::setw(9) << "GPU executionTime(ms) |";
	str1 << " " << std::setw(9) << "GPU dataTransferTime(ms) |";
	str1 << " " << std::setw(9) << "GPU Total Time(ms) |";
	str1 << " " << std::setw(9) << "Speedup% |";
	std::cout << str1.str() << std::endl;
	std::stringstream str;
	str << std::setiosflags(std::ios::left) << std::setw(20) << "Convolution";
	str << std::setiosflags(std::ios::right);
	str << " " << std::setw(10) << convTime;
	str << " " << std::setw(15) << gpu.convOp;
	str << " " << std::setw(20) << gpu.convCopy;
	str << " " << std::setw(25) << (gpu.convOp+gpu.convCopy);
	str << " " << std::setw(21) << 100 *convTime/(gpu.convOp+gpu.convCopy);
	std::cout << str.str() << std::endl;

	std::stringstream relu;
	relu << std::setiosflags(std::ios::left) << std::setw(20) << "ReLU";
	relu << std::setiosflags(std::ios::right);
	relu << " " << std::setw(10) << reluTime;
	relu << " " << std::setw(15) << gpu.reluOp;
	relu << " " << std::setw(20) << gpu.reluCopy;
	relu << " " << std::setw(25) << (gpu.reluOp+gpu.reluCopy);
	relu << " " << std::setw(21) << 100 *reluTime/(gpu.reluOp+gpu.reluCopy);
	std::cout << relu.str() << std::endl;

	std::stringstream maxpool;
	maxpool << std::setiosflags(std::ios::left) << std::setw(20) << "Maxpool";
	maxpool << std::setiosflags(std::ios::right);
	maxpool << " " << std::setw(10) << maxpoolTime;
	maxpool << " " << std::setw(15) << gpu.maxpoolOp;
	maxpool << " " << std::setw(20) << gpu.maxpoolCopy;
	maxpool << " " << std::setw(25) << (gpu.maxpoolOp+gpu.maxpoolCopy);
	maxpool << " " << std::setw(21) << 100 *maxpoolTime/(gpu.maxpoolOp+gpu.maxpoolCopy);
	std::cout << maxpool.str() << std::endl;

	std::stringstream bn;
	bn << std::setiosflags(std::ios::left) << std::setw(20) << "batchNorm";
	bn << std::setiosflags(std::ios::right);
	bn << " " << std::setw(10) << batchNormTime;
	bn << " " << std::setw(15) << gpu.bnOp;
	bn << " " << std::setw(20) << gpu.bnCopy;
	bn << " " << std::setw(25) << (gpu.bnOp+gpu.bnCopy);
	bn << " " << std::setw(21) << 100 *batchNormTime/(gpu.bnOp+gpu.bnCopy);
	std::cout << bn.str() << std::endl;

	std::stringstream upsample;
	upsample << std::setiosflags(std::ios::left) << std::setw(20) << "Upsampling";
	upsample << std::setiosflags(std::ios::right);
	upsample << " " << std::setw(10) << upsampleTime;
	upsample << " " << std::setw(15) << gpu.upsampleOp;
	upsample << " " << std::setw(20) << gpu.upsampleCopy;
	upsample << " " << std::setw(25) << (gpu.upsampleOp+gpu.upsampleCopy);
	upsample << " " << std::setw(21) <<100 * upsampleTime/(gpu.upsampleOp+gpu.upsampleCopy);
	std::cout << upsample.str() << std::endl;
	std::cout << "<------------------------------------------------------------------------------------->"<< std::endl;

}

void printGPUPerformace(GPUInit gpu){
	std::stringstream str1;
	str1 << std::setiosflags(std::ios::left) << std::setw(20) << "Functionality";
	str1 << std::setiosflags(std::ios::right);
	str1 << " " << std::setw(9) << "GPU executionTime(ms) |";
	str1 << " " << std::setw(9) << "GPU dataTransferTime(ms) |";
	str1 << " " << std::setw(9) << "GPU Total Time(ms) |";
	std::cout << str1.str() << std::endl;
	std::stringstream str;
	
	str << std::setiosflags(std::ios::left) << std::setw(20) << "Convolution";
	str << std::setiosflags(std::ios::right);
	str << " " << std::setw(15) << gpu.convOp;
	str << " " << std::setw(20) << gpu.convCopy;
	str << " " << std::setw(25) << (gpu.convOp+gpu.convCopy);
	std::cout << str.str() << std::endl;
	
	std::stringstream ct;
	ct << std::setiosflags(std::ios::left) << std::setw(20) << "ConvolutionTiled";
	ct << std::setiosflags(std::ios::right);
	ct << " " << std::setw(15) << gpu.convTiledOp;
	ct << " " << std::setw(20) << gpu.convTiledCopy;
	ct << " " << std::setw(25) << (gpu.convTiledOp+gpu.convTiledCopy);
	std::cout << ct.str() << std::endl;

	std::stringstream relu;
	relu << std::setiosflags(std::ios::left) << std::setw(20) << "ReLU";
	relu << std::setiosflags(std::ios::right);
	relu << " " << std::setw(15) << gpu.reluOp;
	relu << " " << std::setw(20) << gpu.reluCopy;
	relu << " " << std::setw(25) << (gpu.reluOp+gpu.reluCopy);
	std::cout << relu.str() << std::endl;

	std::stringstream maxpool;
	maxpool << std::setiosflags(std::ios::left) << std::setw(20) << "Maxpool";
	maxpool << std::setiosflags(std::ios::right);
	maxpool << " " << std::setw(15) << gpu.maxpoolOp;
	maxpool << " " << std::setw(20) << gpu.maxpoolCopy;
	maxpool << " " << std::setw(25) << (gpu.maxpoolOp+gpu.maxpoolCopy);
	std::cout << maxpool.str() << std::endl;

	std::stringstream bn;
	bn << std::setiosflags(std::ios::left) << std::setw(20) << "batchNorm";
	bn << std::setiosflags(std::ios::right);
	bn << " " << std::setw(15) << gpu.bnOp;
	bn << " " << std::setw(20) << gpu.bnCopy;
	bn << " " << std::setw(25) << (gpu.bnOp+gpu.bnCopy);
	std::cout << bn.str() << std::endl;

	std::stringstream upsample;
	upsample << std::setiosflags(std::ios::left) << std::setw(20) << "Upsampling";
	upsample << std::setiosflags(std::ios::right);
	upsample << " " << std::setw(15) << gpu.upsampleOp;
	upsample << " " << std::setw(20) << gpu.upsampleCopy;
	upsample << " " << std::setw(25) << (gpu.upsampleOp+gpu.upsampleCopy);
	std::cout << upsample.str() << std::endl;
	std::cout << "<------------------------------------------------------------------------------------->"<< std::endl;

}

std::string getBaseFilename(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    std::string filename = (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);
    
    size_t lastDot = filename.find_last_of(".");
    return (lastDot == std::string::npos) ? filename : filename.substr(0, lastDot);
}
int main(int argc, char* argv[]) {
	getSysInfo();
	if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image.npy>" << std::endl;
        return 1;
    }
	std::string imageFilename = argv[1];
	std::string baseName = getBaseFilename(imageFilename);
    std::string cpu_output = "cpu_" + baseName;
	std::string gpu_output = "gpu_" + baseName;
	//-----------------------------------------------------------CPU & GPU Output for sample inputs -----------------------------------------------------------//
	{
		GPUInit _gpu = GPUInit();
		_gpu.getGpuDetails();
		int n = 1;
		int c = 1;
		int h = 5;
		int w = 5;
		int out_c = 1;
		int kh = 3;
		int kw = 3;
		std::vector<float> input(n * c * h * w);
		std::vector<float> input2(n * c * h * w);
		std::vector<float> filter(out_c * c * kh * kw);
		//sample gamma, beta, mean, variance values are set for testing purpose
		std::vector<float> gamma(out_c,0.001f); 
		std::vector<float> beta(out_c,0.001f);
		std::vector<float> mean(out_c,0.005f);
		std::vector<float> variance(out_c,0.005f);
		//sample values are set for input tensors and filter tensor in row-major format
		for(auto i=0; i<input.size();i++){
			input[i] = i+1;
		}
		for(auto i=0; i<input2.size();i++){
			input2[i] = 10;
		}
		for(auto i=0; i<filter.size();i++){
			filter[i] = i+1;
		}

		std::vector<float> outputcpuTensor  = conv2d_cpu(input, filter, n, c, h, w, out_c, kh, kw,1,1);
		std::vector<float> batchNormcpuTensor  = batch_norm_cpu(outputcpuTensor,gamma, beta, mean, variance,n, out_c, h, w);
		std::vector<float> relucpuTensor = relu_cpu(batchNormcpuTensor,n, out_c, h, w);
		std::vector<float> maxpoolcpuTensor   = maxpool_cpu(relucpuTensor, n, out_c, h, w);
		std::vector<float> upsamplecpuTensor = upsample_cpu(maxpoolcpuTensor,n , out_c, h/2, w/2, h*2, w*2);
		
		std::vector<float> outputgpuTensor  = _gpu.convolution(input, filter, n, c, h, w, out_c, kh, kw,1,1);
		std::vector<float> batchNormgpuTensor  = _gpu.batchnormInference(outputgpuTensor,gamma, beta, mean, variance,n, out_c, h, w);
		std::vector<float> relugpuTensor = _gpu.relu(batchNormgpuTensor, n, out_c, h, w);
		std::vector<float> maxpoolgpuTensor   = _gpu.maxpool(relugpuTensor, n, out_c, h, w);
		std::vector<float> upsamplegpuTensor = _gpu.upsample(maxpoolgpuTensor,n, out_c, h/2, w/2, h*2, w*2);
		
		printTensor(outputcpuTensor,n, out_c, h, w,"CPU Convolution");
		printTensor(outputgpuTensor,n, out_c, h, w,"GPU Convolution");
		printTensor(batchNormcpuTensor,n, out_c, h, w,"CPU BatchNorm");
		printTensor(batchNormgpuTensor,n, out_c, h, w,"GPU BatchNorm");
		printTensor(relucpuTensor,n, out_c, h, w,"CPU ReLU");
		printTensor(relugpuTensor,n, out_c, h, w,"GPU ReLU");
		printTensor(maxpoolcpuTensor,n, out_c, 2, 2,"CPU Maxpool");
		printTensor(maxpoolgpuTensor,n, out_c, 2, 2,"GPU Maxpool");
		printTensor(upsamplecpuTensor,n, out_c, 5,5,"CPU Upsample");
		printTensor(upsamplegpuTensor,n, out_c, 5,5,"GPU Upsample");
	}

	//-----------------------------------------------------------CPU vs GPU Comparison -----------------------------------------------------------//

	{
		GPUInit gpu = GPUInit();
		int N = 1; // Number of batches
		int C = 3; // Number of channels
		int H = 224; // Height
		int W = 224; // Width
		int outC = 32;
		int kh = 3;
		int kw = 3;
		std::vector<float> gamma(outC,0.001f); 
		std::vector<float> beta(outC,0.001f);
		std::vector<float> mean(outC,0.005f);
		std::vector<float> variance(outC,0.005f);
		std::vector<float> filter(outC * C * kh * kw); //(outC, C, kh, kw)
		std::vector<float> inputTensor(N * C * H * W); //(N,C,H,W)
		for(auto i=0; i<inputTensor.size();i++){
			inputTensor[i] = 0.05;
		}
		for(auto i=0; i<filter.size();i++){
			filter[i] = 0.005;
		}

		//CPU convolution, BathchNorm, ReLU, Maxpool, Upsample
		getSysInfo();
		Timer t1(cpu_double_conv,"CPU Operations");
		std::vector<float> cpu_convolution  = conv2d_cpu(inputTensor, filter, N, C, H, W, outC,  kh, kw);
		std::vector<float> cpu_batchnorm  = batch_norm_cpu(cpu_convolution,gamma,beta, mean, variance,N, outC, H, W);
		std::vector<float> cpu_relu = relu_cpu(inputTensor,N, C, H, W);
		std::vector<float> cpu_maxpool = maxpool_cpu(inputTensor, N, C, H, W);
		std::vector<float> cpu_upsample = upsample_cpu(cpu_maxpool, N, C, 112,112, H, W);
		t1.stop();
			
		//GPU convolution, BathchNorm, ReLU, Maxpool, Upsample
		std::vector<float> gpu_convolution = gpu.convolution(inputTensor, filter, N, C, H, W, outC, kh, kw);
		std::vector<float> gpu_batchnorm   = gpu.batchnormInference(gpu_convolution, gamma, beta,mean, variance,N, outC, H, W);
		std::vector<float> gpu_relu        = gpu.relu(inputTensor, N, C, H, W);
		std::vector<float> gpu_maxpool     = gpu.maxpool(inputTensor,N, C, H, W,2);
		std::vector<float> gpu_upsample    = gpu.upsample(gpu_maxpool, N, C, 112, 112, H, W);
		saveTensorToNpy("cpu_convolution",cpu_convolution,N, outC, H, W);
		saveTensorToNpy("gpu_convolution",gpu_convolution,N, outC, H, W);
		saveTensorToNpy("cpu_batchnorm",cpu_batchnorm,N, outC, H, W);
		saveTensorToNpy("gpu_batchnorm",gpu_batchnorm,N, outC, H, W);
		saveTensorToNpy("cpu_relu",cpu_relu,N, C, H, W);
		saveTensorToNpy("gpu_relu",gpu_relu,N, C, H, W);
		saveTensorToNpy("cpu_maxpool",cpu_maxpool,N, C, H/2, W/2);
		saveTensorToNpy("gpu_maxpool",gpu_maxpool,N, C, H/2, W/2);
		saveTensorToNpy("cpu_upsample",cpu_upsample,N, C, H, W);
		saveTensorToNpy("gpu_upsample",gpu_upsample,N, C, H, W);
	
		std::vector<float> gpu_convolutionTiled   = gpu.convolutionOptimizedwbias(inputTensor, filter, N, C, H, W, outC,3,3);
		std::cout<<"<-----------------GPU Naive convolution vs Convolution Tiled comparison----------------->"<<std::endl;
		compareTensors(gpu_convolution,gpu_convolutionTiled);
		std::cout<<"<------------------------------------------------------------------>"<<std::endl;
		saveTensorToNpy("gpu_convolutionTiled",gpu_convolutionTiled,N, outC, H, W);
		printGPUPerformace(gpu);
	}

	{
		convTime = 0;
		batchNormTime = 0;
		reluTime = 0;
		maxpoolTime = 0;
		upsampleTime = 0;
		getSysInfo();
		std::cout<<"Running UNET in CPU"<<std::endl;
		Timer t1(unet_cpu_time,"UNET CPU Time");
		int N = 1; // Number of batches
		int C = 3; // Number of channels
		int H = 224; // Height
		int W = 224; // Width
		int outC = 32;
		int kh = 3;
		int kw = 3;
		std::vector<float> inputTensor = loadImagefromNPY(imageFilename);
		std::vector<float> conv_224 = doubleConvolution_cpu(inputTensor,
																	"conv2d",
																	"conv2d_1",
																	"batch_normalization",
																	"batch_normalization_1",N,C,H,W,32,32);
		std::vector<int> conv224Dim = {N, outC, H, W};
		//Timer mp1(maxpoolTime);
		std::vector<float> pool_112 = maxpool_cpu(conv_224, N, outC, H, W, 2,2);
		//mp1.stop();
		H = H/2;
		W = W/2;
		std::vector<int> pool112Dim = {N, outC, H, W};
		std::vector<float> conv_112 = doubleConvolution_cpu(pool_112,
															"conv2d_2",
															"conv2d_3",
															"batch_normalization_2",
															"batch_normalization_3",N,outC,H,W,64,64);
			
		std::vector<int> conv112Dim = {N, outC*2, H, W};
		//Timer mp2(maxpoolTime);
		std::vector<float> pool_56 = maxpool_cpu(conv_112, N, outC*2, H, W, 2);
		//mp2.stop();
		H = H/2;
		W = W/2;
		std::vector<int> pool56Dim = {N, outC*2, H, W};
		std::vector<float> conv_56 = doubleConvolution_cpu(pool_56,
															"conv2d_4",
															"conv2d_5",
															"batch_normalization_4",
															"batch_normalization_5",N,outC*2,H,W,128,128);
		std::vector<int> conv56Dim = {N, outC*4, H, W};
		//Timer mp3(maxpoolTime);
		std::vector<float> pool_28 = maxpool_cpu(conv_56, N, outC*4, H, W, 2);
		//mp3.stop();
		H = H/2;
		W = W/2;
		std::vector<int> pool28Dim = {N, outC*4, H, W};
		std::vector<float> conv_28 = doubleConvolution_cpu(pool_28,
															"conv2d_6",
															"conv2d_7",
															"batch_normalization_6",
															"batch_normalization_7",N,outC*4,H,W,256,256);
		std::vector<int> conv28Dim = {N, outC*8, H, W};
		//Timer mp4(maxpoolTime);
		std::vector<float> pool_14 =maxpool_cpu(conv_28, N, outC*8, H, W, 2);
		//mp4.stop();
		H = H/2;
		W = W/2;
		std::vector<int> pool14Dim = {N, outC*8, H, W};
		std::vector<float> conv_14 = doubleConvolution_cpu(pool_14,
															"conv2d_8",
															"conv2d_9",
															"batch_normalization_8",
															"batch_normalization_9",N,outC*8,H,W,512,512);
		std::vector<int> conv14Dim = {N, outC*16, H, W};
		Timer mp5(maxpoolTime);
		std::vector<float> pool_7 = maxpool_cpu(conv_14, N, outC*16, H, W, 2);
		mp5.stop();
		H = H/2;
		W = W/2;
		std::vector<int> pool7Dim = {N, outC*16, H, W};
		std::vector<float> conv_7 = doubleConvolution_cpu(pool_7,
															"conv2d_10",
															"conv2d_11",
															"batch_normalization_10",
															"batch_normalization_11",N,outC*16,H,W,1024,1024);
		std::vector<int> conv7Dim = {N, outC*32, H, W};
		std::vector<float> up_14 = upward_cpu(conv_7,
												"conv2d_12",
												"conv2d_13",
												"batch_normalization_12",
												"batch_normalization_13",conv_14, N, conv7Dim[1], conv7Dim[2], conv7Dim[3],512,512,512);
		H = H*2;
		W = W*2;
		std::vector<int> up14Dim = {N, outC*16, H, W};
		std::vector<float> up_28 = upward_cpu(up_14,
											"conv2d_14",
											"conv2d_15",
											"batch_normalization_14",
											"batch_normalization_15",conv_28,N, up14Dim[1], up14Dim[2], up14Dim[3],256,256,256);
		H = H*2;
		W = W*2;
		std::vector<int> up28Dim = {N, outC*8, H, W};
		std::vector<float> up_56 = upward_cpu(up_28,
											"conv2d_16",
											"conv2d_17",
											"batch_normalization_16",
											"batch_normalization_17",conv_56,N, up28Dim[1], up28Dim[2], up28Dim[3],128,128,128);
		H = H*2;
		W = W*2;
		std::vector<int> up56Dim = {N, outC*4, H, W};
		std::vector<float> up_112 = upward_cpu(up_56,
											"conv2d_18",
											"conv2d_19",
											"batch_normalization_18",
											"batch_normalization_19",conv_112,N, up56Dim[1], up56Dim[2], up56Dim[3],64,64,64);
		H = H*2;
		W = W*2;
		std::vector<int> up112Dim = {N, outC*2, H, W};
		std::vector<float> up_224 = upward_cpu(up_112,
											"conv2d_20",
											"conv2d_21",
											"batch_normalization_20",
											"batch_normalization_21",conv_224,N, up112Dim[1], up112Dim[2], up112Dim[3],32,32,32);
		H = H*2;
		W = W*2;
		std::vector<int> up224Dim = {N, outC, H, W};
		std::vector<float> dropout = spatial_dropout_2d_cpu(up_224, 0.2, N, outC, H, W);
		std::vector<float> output = conv2d_cpu(dropout, "conv2d_22", N, outC, H, W, 1, 1, 1, 1, 0);
		std::vector<int> dim2 = {N, outC, H, W};
		std::vector<float> segmentedImagecpu = sigmoidActivation_cpu(output, N, 1, 224, 224);
		t1.stop();
		saveAsImage(segmentedImagecpu,1,1,224,224,cpu_output);
		saveTensorToNpy(cpu_output,segmentedImagecpu,N, 1, 224, 224);
		
		std::cout<<"UNET CPU Done"<<std::endl;
		std::cout<<"UNET CPU Convolution "<<convTime<<" ms"<<std::endl;
		std::cout<<"UNET CPU BatchNorm   "<<batchNormTime<<" ms"<<std::endl;
		std::cout<<"UNET CPU ReLU        "<<reluTime<<" ms"<<std::endl;
		std::cout<<"UNET CPU Maxpool     "<<maxpoolTime<<" ms"<<std::endl;
		std::cout<<"UNET CPU Upsample    "<<upsampleTime<<" ms"<<std::endl;
	}

	//-----------------------------------------------------------UNET Implementation------------------------------------------------------------//
	getSysInfo();
	std::cout<<"Running UNET in GPU"<<std::endl;
	GPUInit gpu = GPUInit();
	int N = 1; // Number of batches
    int C = 3; // Number of channels
    int H = 224; // Height
    int W = 224; // Width
	int outC = 32;
	int kh = 3;
	int kw = 3;
	std::vector<float> inputTensor = loadImagefromNPY(imageFilename);
	std::vector<float> conv_224 = doubleConvolution(gpu,inputTensor,
															"conv2d",
															"conv2d_1",
															"batch_normalization",
															"batch_normalization_1",N,C,H,W,32,32);
	std::vector<int> conv224Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("conv_224");
	std::vector<float> pool_112 = gpu.maxpool(conv_224, N, gpu._outC, gpu._outH, gpu._outW, 2);
	std::vector<int> pool112Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("pool_112");
	std::vector<float> conv_112 = doubleConvolution(gpu,pool_112,
															"conv2d_2",
															"conv2d_3",
															"batch_normalization_2",
															"batch_normalization_3",N,gpu._outC,gpu._outH,gpu._outW,64,64);
															
	std::vector<int> conv112Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("conv_112");
	std::vector<float> pool_56 = gpu.maxpool(conv_112, N, gpu._outC, gpu._outH, gpu._outW, 2);
	std::vector<int> pool56Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("pool_56");
	std::vector<float> conv_56 = doubleConvolution(gpu,pool_56,
															"conv2d_4",
															"conv2d_5",
															"batch_normalization_4",
															"batch_normalization_5",N,gpu._outC,gpu._outH,gpu._outW,128,128);
															
	std::vector<int> conv56Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("conv_56");
	std::vector<float> pool_28 = gpu.maxpool(conv_56, N, gpu._outC, gpu._outH, gpu._outW, 2);
	std::vector<int> pool28Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	gpu.printdims("pool_28");
	std::vector<float> conv_28 = doubleConvolution(gpu,pool_28,
															"conv2d_6",
															"conv2d_7",
															"batch_normalization_6",
															"batch_normalization_7",N,gpu._outC,gpu._outH,gpu._outW,256,256);

	std::vector<int> conv28Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("conv_28");
	std::vector<float> pool_14 = gpu.maxpool(conv_28, N, gpu._outC, gpu._outH, gpu._outW, 2);
	std::vector<int> pool14Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("pool_14");
	std::vector<float> conv_14 = doubleConvolution(gpu,pool_14,
															"conv2d_8",
															"conv2d_9",
															"batch_normalization_8",
															"batch_normalization_9",N,gpu._outC,gpu._outH,gpu._outW,512,512);

	std::vector<int> conv14Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("conv_14");
	std::vector<float> pool_7 = gpu.maxpool(conv_14, N, gpu._outC, gpu._outH, gpu._outW, 2);
	std::vector<int> pool7Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("pool_7");
	std::vector<float> conv_7 = doubleConvolution(gpu,pool_7,
															"conv2d_10",
															"conv2d_11",
															"batch_normalization_10",
															"batch_normalization_11",N,gpu._outC,gpu._outH,gpu._outW,1024,1024);
	std::vector<int> conv7Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("conv_7");
	std::vector<float> up_14 = upward(gpu,conv_7,
										"conv2d_12",
										"conv2d_13",
										"batch_normalization_12",
										"batch_normalization_13",conv_14,N, conv7Dim[1], conv7Dim[2], conv7Dim[3],512,512,512);
										
	std::vector<int> up14Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("up_14");
	std::vector<float> up_28 = upward(gpu,up_14,
										"conv2d_14",
										"conv2d_15",
										"batch_normalization_14",
										"batch_normalization_15",conv_28,N, up14Dim[1], up14Dim[2], up14Dim[3],256,256,256);
										
	std::vector<int> up28Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("up_28");
	std::vector<float> up_56 = upward(gpu,up_28,
										"conv2d_16",
										"conv2d_17",
										"batch_normalization_16",
										"batch_normalization_17",conv_56,N, up28Dim[1], up28Dim[2], up28Dim[3],128,128,128);
										
	std::vector<int> up56Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("up_56");
	std::vector<float> up_112 = upward(gpu,up_56,
										"conv2d_18",
										"conv2d_19",
										"batch_normalization_18",
										"batch_normalization_19",conv_112,N, up56Dim[1], up56Dim[2], up56Dim[3],64,64,64);
	std::vector<int> up112Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("up_112");
	std::vector<float> up_224 = upward(gpu,up_112,
										"conv2d_20",
										"conv2d_21",
										"batch_normalization_20",
										"batch_normalization_21",conv_224,N, up112Dim[1], up112Dim[2], up112Dim[3],32,32,32);
	std::vector<int> up224Dim = {N, gpu._outC, gpu._outH, gpu._outW};
	//gpu.printdims("up_224");
	std::vector<float> dropout = spatial_dropout_2d_cpu(up_224, 0.2, N, gpu._outC, gpu._outH, gpu._outW);
	std::vector<float> output = gpu.convolution(dropout, "conv2d_22", N, gpu._outC, gpu._outH, gpu._outW, 1, 1, 1, 1, 0);
	//gpu.printdims("output");
	std::vector<float> segmentedImagegpu = gpu.sigmoidActivation(output, N, 1, 224, 224);
	std::cout<<"UNET completed in GPU"<<std::endl;
	//gpu.printdims("segmentedImage");
	saveAsImage(segmentedImagegpu,1,1,224,224,gpu_output);
	saveTensorToNpy(gpu_output,segmentedImagegpu,N, 1, 224, 224);
	printPerformace(gpu);
}
