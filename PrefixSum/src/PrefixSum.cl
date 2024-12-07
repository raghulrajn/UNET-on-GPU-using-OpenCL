#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif
#define LOCAL_SIZE 256

// Prefix Sum Kernel
__kernel void prefixSum(__global int* input, __global int* output, __global int* blockSums, int n) {
    __local int localData[LOCAL_SIZE];
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wid = get_group_id(0);

    // Load input data to local memory
    if (gid < n) {
        localData[lid] = input[gid];
    } else {
        localData[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform prefix sum in local memory
    for (int offset = 1; offset < LOCAL_SIZE; offset *= 2) {
        int temp = 0;
        if (lid >= offset) {
            temp = localData[lid - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid >= offset) {
            localData[lid] += temp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results to global memory
    if (gid < n) {
        output[gid] = localData[lid];
    }

    // Write block sum to global memory
    if (lid == LOCAL_SIZE - 1) {
        blockSums[wid] = localData[lid];
    }
}

// Block Add Kernel
__kernel void blockAdd(__global int* input, __global int* blockSums, int n) {
    int gid = get_global_id(0);
    int wid = get_group_id(0);

    if (gid < n) {
        input[gid] += blockSums[wid];
    }
}
