#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

__kernel void sobelKernel1(/*...*/) {}

__kernel void sobelKernel2(/*...*/) {}

__kernel void sobelKernel3(__read_only image2d_t inputImage,
                           __write_only image2d_t outputImage,sampler_t sampler) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    // Sobel kernels
    const int Gx[3][3] = { {-1, 0, 1},
                           {-2, 0, 2},
                           {-1, 0, 1} };
    const int Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    // Initialize gradients
    int gradientX = 0;
    int gradientY = 0;

    int width = get_image_width(inputImage);
    int height = get_image_height(inputImage);

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int2 sampleCoord = (int2)(coord.x + i, coord.y + j);

            // Boundary check
            if (sampleCoord.x >= 0 && sampleCoord.x < width &&
                sampleCoord.y >= 0 && sampleCoord.y < height) {
                float pixel = read_imagef(inputImage, sampler, sampleCoord).x;
            printf("%f\n", pixel);
            gradientX += pixel * Gx[i + 1][j + 1];
                gradientY += pixel * Gy[i + 1][j + 1];
            }
        }
    }

    // Calculate gradient magnitude
    float magnitude = sqrt((float)(gradientX * gradientX + gradientY * gradientY));

magnitude = clamp(magnitude / 255.0f, 0.0f, 1.0f);

    write_imagef(outputImage, coord, (float4)(magnitude, magnitude, magnitude, 1.0f));}
