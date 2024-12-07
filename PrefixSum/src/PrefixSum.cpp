//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 6: Prefix sum (Scan)
//////////////////////////////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
void prefixSumHost(const std::vector<cl_int>& h_input,
                   std::vector<cl_int>& h_output) {
  if (h_input.size() == 0) return;
  cl_int sum = h_input[0];
  h_output[0] = sum;
  for (std::size_t i = 1; i < h_input.size(); i++) {
    sum += h_input[i];
    h_output[i] = sum;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "OpenCL Exercise 6: Prefix Sum" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  // Create a context
  cl::Context context(CL_DEVICE_TYPE_GPU);
cl::Event event;
  // Get a device of the context
  int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
  std::cout << "Using device " << deviceNr << " / "
            << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
  ASSERT(deviceNr > 0);
  ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
  cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];

  std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "Max Work Group Size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
        std::cout << "Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        std::cout << "Global Memory Size: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << " bytes" << std::endl;
        std::cout << "Local Memory Size: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << " bytes" << std::endl;

        // Additional useful information
        std::cout << "Max Work Item Dimensions: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;

        auto maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
        std::cout << "Max Work Item Sizes: ";
        for (const auto& size : maxWorkItemSizes) {
            std::cout << size << " ";
        }
        std::cout << std::endl;
  std::vector<cl::Device> devices;
  devices.push_back(device);
  OpenCL::printDeviceInfo(std::cout, device);

  // Create a command queue
  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  // Declare some values
  std::size_t wgSize = 256;  // Number of work items per work group
  std::size_t count = wgSize * wgSize * wgSize;  // Number of values

  std::size_t size = count * sizeof(cl_int);

  // Load the source code
  extern unsigned char OpenCLExercise6_PrefixSum_cl[];
  extern unsigned int OpenCLExercise6_PrefixSum_cl_len;
  cl::Program program(context,
                      std::string((const char*)OpenCLExercise6_PrefixSum_cl,
                                  OpenCLExercise6_PrefixSum_cl_len));
  // Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
  // This will pass the value of wgSize as a preprocessor constant "WG_SIZE" to the OpenCL C compiler
  OpenCL::buildProgram(program, devices,
                       "-DWG_SIZE=" + boost::lexical_cast<std::string>(wgSize));

  // Allocate space for output data from CPU and GPU on the host
  std::vector<cl_int> h_input(count);
  std::vector<cl_int> h_outputCpu(count);
  std::vector<cl_int> h_temp1(wgSize * wgSize);
  std::vector<cl_int> h_temp2(wgSize);
  std::vector<cl_int> h_outputGpu(count);

  // Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
  memset(h_input.data(), 255, size);
  memset(h_temp1.data(), 255, wgSize * wgSize * sizeof(cl_int));
  memset(h_temp2.data(), 255, wgSize * sizeof(cl_int));
  memset(h_outputCpu.data(), 255, size);
  memset(h_outputGpu.data(), 255, size);
  //////// Generate input data ////////////////////////////////
  // Use random input data
  //for (std::size_t i = 0; i < count; i++) h_input[i] = rand() % 100 - 40;
  // Or: Use consecutive integer numbers as data

	for (std::size_t i = 0; i < count; i++)
		h_input[i] = i;
  std::cout<<h_input[count-1]<<"\n";
  // Do calculation on the host side
    prefixSumHost(h_input, h_outputCpu);

    cl::Kernel prefixSumKernel(program, "prefixSum");
    cl::Kernel blockAddKernel(program, "blockAdd");

    const int WORK_GROUP_SIZE = 256;
    const int NUM_ELEMENTS = WORK_GROUP_SIZE * WORK_GROUP_SIZE * WORK_GROUP_SIZE;

    cl::Buffer d_input(context, CL_MEM_READ_WRITE, sizeof(int) * NUM_ELEMENTS);
    cl::Buffer d_output(context, CL_MEM_READ_WRITE, sizeof(int) * NUM_ELEMENTS);
    cl::Buffer d_temp1(context, CL_MEM_READ_WRITE, sizeof(int) * WORK_GROUP_SIZE * WORK_GROUP_SIZE);
    cl::Buffer d_temp2(context, CL_MEM_READ_WRITE, sizeof(int) * WORK_GROUP_SIZE);

    queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
    queue.enqueueWriteBuffer(d_temp1, true, 0, sizeof(int) * WORK_GROUP_SIZE * WORK_GROUP_SIZE, h_temp1.data());

    // 1. First prefix sum kernel launch
    prefixSumKernel.setArg(0, d_input);
    prefixSumKernel.setArg(1, d_output);
    prefixSumKernel.setArg(2, d_temp1);
    prefixSumKernel.setArg(3, NUM_ELEMENTS);
    queue.enqueueNDRangeKernel(prefixSumKernel, cl::NullRange, cl::NDRange(NUM_ELEMENTS), cl::NDRange(WORK_GROUP_SIZE));

    // 2. Second prefix sum kernel launch
    prefixSumKernel.setArg(0, d_temp1);
    prefixSumKernel.setArg(1, d_temp1);
    prefixSumKernel.setArg(2, d_temp2);
    prefixSumKernel.setArg(3, WORK_GROUP_SIZE);
    queue.enqueueNDRangeKernel(prefixSumKernel, cl::NullRange, cl::NDRange(256 * WORK_GROUP_SIZE), cl::NDRange(WORK_GROUP_SIZE));

    // 3. Third prefix sum kernel launch
    prefixSumKernel.setArg(0, d_temp2);
    prefixSumKernel.setArg(1, d_temp2);
    prefixSumKernel.setArg(2, cl::Buffer()); // No need for block sums
    prefixSumKernel.setArg(3, 256);
    queue.enqueueNDRangeKernel(prefixSumKernel, cl::NullRange, cl::NDRange(WORK_GROUP_SIZE), cl::NDRange(WORK_GROUP_SIZE));

    // 4. First block add kernel launch
    blockAddKernel.setArg(0, d_temp1);
    blockAddKernel.setArg(1, d_temp2);
    blockAddKernel.setArg(2, WORK_GROUP_SIZE);
    queue.enqueueNDRangeKernel(blockAddKernel, cl::NullRange, cl::NDRange(256 * WORK_GROUP_SIZE), cl::NDRange(WORK_GROUP_SIZE));

    // 5. Second block add kernel launch
    blockAddKernel.setArg(0, d_output);
    blockAddKernel.setArg(1, d_temp1);
    blockAddKernel.setArg(2, NUM_ELEMENTS);
    queue.enqueueNDRangeKernel(blockAddKernel, cl::NullRange, cl::NDRange(NUM_ELEMENTS), cl::NDRange(WORK_GROUP_SIZE));

    queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, &event);
    queue.finish();

  // Check whether results are correct
  std::size_t errorCount = 0;
  for (size_t i = 0; i < count; i = i + 1) {
    if (h_outputCpu[i] != h_outputGpu[i]) {
      if (errorCount < 15)
        std::cout << "Result at " << i << " is incorrect: GPU value is "
                  << h_outputGpu[i] << ", CPU value is " << h_outputCpu[i]
                  << std::endl;
      else if (errorCount == 15)
        std::cout << "..." << std::endl;
      errorCount++;
    }
  }
  if (errorCount != 0) {
    std::cout << "Found " << errorCount << " incorrect results" << std::endl;
    return 1;
  }

  std::cout << "Success" << std::endl;

  return 0;
}
