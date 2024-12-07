#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

__kernel void matrixMulKernel2(__global float *A,
                               __global float *B,
                               __global float *d_outputC,
                               size_t countAX_BY,
                                size_t countAY,
                               size_t countBX) {

  size_t i = get_global_id(0);  //loop in the x-direction
  size_t j = get_global_id(1); //loop in the y-direction
      float sum = 0;
      for (size_t k = 0; k < countAX_BY; k++) {
        float a = A[k + j * countAX_BY];
        float b = B[i + k * countBX];
        sum += a * b;
      }
      d_outputC[i + j * countBX] = sum;
    }


#define blksz 8
__kernel void matrixMulKernel1(__global float *A,
                               __global float *B,
                               __global float *d_outputC,
                               size_t countAX_BY,
                               __local float *aTile,
                               __local float *bTile) {
 int row = get_global_id(1);
 int col = get_global_id(0);

 int iblk = get_group_id(0);
 int jblk = get_group_id(1);

 float sum = 0.0f;

 int x = get_local_id(0);
 int y = get_local_id(1);

 int NUM_BLK = countAX_BY/blksz;

 int Abase = iblk*countAX_BY*blksz;
 int Bbase = jblk*blksz;

 int Ainc = blksz;
 int Binc = blksz*countAX_BY;

 int kblk, kloc;

 for(kblk=0;kblk<NUM_BLK;kblk++){
   aTile[y*blksz +x] = A[Abase + y*countAX_BY + x];
   bTile[y*blksz +x] = B[Bbase + y*countAX_BY + x];
  barrier(CLK_LOCAL_MEM_FENCE);

  #pragma unroll
  for(kloc=0;kloc<blksz;kloc++){
    sum+= aTile[y*blksz + kloc]*bTile[kloc*blksz + x];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  Abase += Ainc;
  Bbase += Binc;
}
d_outputC[row * countAX_BY + col] = sum;

}


// The preprocessor constant WG_SIZE will contain the size of a work group in X/Y-direction

//__attribute__((reqd_work_group_size(WG_SIZE, WG_SIZE, 1))) __kernel void
//matrixMulKernel2(/*...*/) {
  //TODO
//}
