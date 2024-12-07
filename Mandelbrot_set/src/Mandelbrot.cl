#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

//TODO
__kernel void mandelbrotKernel(__global int* d_output,
                               const int niter,
                                const float xmin,
                               const float xmax,
                                const float ymin,
                                const float ymax,
                                size_t countX,
                                size_t countY) {
    size_t i = get_global_id(0);  //loop in the x-direction
    float xc = xmin + (xmax - xmin) / (countX - 1) * i;  //xc=real(c)
    size_t j = get_global_id(1); //loop in the y-direction
      float yc = ymin + (ymax - ymin) / (countY - 1) * j;  //yc=imag(c)
      float x = 0.0;                                       //x=real(z_k)
      float y = 0.0;                                       //y=imag(z_k)
      for (size_t k = 0; k < niter; k = k + 1) {           //iteration loop
        float tempx = x * x - y * y + xc;                  //z_{n+1}=(z_n)^2+c;
        y = 2 * x * y + yc;
        x = tempx;
        float r2 = x * x + y * y;          //r2=|z_k|^2
        if ((r2 > 4) || k == niter - 1) {  //divergence condition
          d_output[i + j * countX] = k;
          break;
        }
      }
}
