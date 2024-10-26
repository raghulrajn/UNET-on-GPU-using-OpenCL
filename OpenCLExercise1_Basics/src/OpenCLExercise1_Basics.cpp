//////////////////////////////////////////////////////////////////////////////
// OpenCL exercise 1: Basics
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <OpenCL/Device.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/cl-patched.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
void calculateHost(const std::vector<float>& h_input,
                   std::vector<float>& h_output) {
  for (std::size_t i = 0; i < h_output.size(); i++)
    h_output[i] = std::cos(h_input[i]);
}

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "OpenCL Exercise 1: Basics" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
  cl::Event event;
  // Create a context
  cl::Context context(CL_DEVICE_TYPE_GPU);

  // Get the first device of the context
  std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size()
            << " devices" << std::endl;
  cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
  std::vector<cl::Device> devices;
  devices.push_back(device);
  OpenCL::printDeviceInfo(std::cout, device);

  // Create a command queue
  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  // Load the source code
  extern unsigned char OpenCLExercise1_Basics_cl[];
  extern unsigned int OpenCLExercise1_Basics_cl_len;
  cl::Program program(context,
                      std::string((const char*)OpenCLExercise1_Basics_cl,
                                  OpenCLExercise1_Basics_cl_len));
  // Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
  OpenCL::buildProgram(program, devices);

  // Create a kernel object
  cl::Kernel kernel1(program, "kernel1");

  // Declare some values
  std::size_t wgSize = 128;  // Number of work items per work group
  std::size_t count =
      wgSize * 1000000;  // Overall number of work items = Number of elements
  std::size_t size = count * sizeof(float);  // Size of data in bytes

  // Allocate space for input data and for output data from CPU and GPU on the host
  std::vector<float> h_input(count);
  std::vector<float> h_outputCpu(count);
  std::vector<float> h_outputGpu(count);

  // Allocate space for input and output data on the device
  cl::Buffer d_input(context, CL_MEM_READ_WRITE, sizeof(float)*count);
  cl::Buffer d_output(context, CL_MEM_READ_WRITE, sizeof(float)*count);

  // Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
  memset(h_input.data(), 255, size);
  memset(h_outputCpu.data(), 255, size);
  memset(h_outputGpu.data(), 255, size);
  //TODO



  // Initialize input data with more or less random values
  for (std::size_t i = 0; i < count; i++) h_input[i] = ((i * 1009) % 31) * 0.1;

  // Do calculation on the host side
  Core::TimeSpan time1 = Core::getCurrentTime();
  calculateHost(h_input, h_outputCpu);
  Core::TimeSpan time2 = Core::getCurrentTime();

  // Copy input data to device
  //TODO: enqueueWriteBuffer()
    queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, &event);

  // Launch kernel on the device
  //TODO
    kernel1.setArg<cl::Buffer>(0, d_input);
    kernel1.setArg<cl::Buffer>(1, d_output);
    queue.enqueueNDRangeKernel(kernel1, 0, count, wgSize);

    // Copy output data back to host
  //TODO: enqueueReadBuffer()
    queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(),NULL, &event);
  // Print performance data
    queue.finish();
  //for (std::size_t i = 0; i < count; i++) {
    //std::cout << "Result for " << i << " GPU value is "<< h_outputGpu[i] << ", CPU value is " << //h_outputCpu[i]<< std::endl;}

  Core::TimeSpan time = OpenCL::getElapsedTime(event);
  std::cout <<"GPU Time :"<< time << std::endl;

  Core::TimeSpan timeCpu = time2 - time1;
  std::cout <<"CPU Time :"<< timeCpu << std::endl;
  //std::size_t errorCount = 0;
  //for (std::size_t i = 0; i < count; i++) {
    // Allow small differences between CPU and GPU results (due to different rounding behavior)
    //if (!(std::abs(h_outputCpu[i] - h_outputGpu[i]) <= 1e-5)) {
      //if (errorCount < 15)
        //std::cout << "Result for " << i << " is incorrect: GPU value is "
          //        << h_outputGpu[i] << ", CPU value is " << h_outputCpu[i]
            //      << std::endl;
      //else if (errorCount == 15)
        //std::cout << "..." << std::endl;
      //errorCount++;
    //}
  //}
  //if (errorCount != 0) {
    //std::cout << "Found " << errorCount << " incorrect results" << std::endl;
    //return 1;
  //}

  std::cout << "Success" << std::endl;

  return 0;
}
