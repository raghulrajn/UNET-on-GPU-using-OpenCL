#include "tensor4D.h"
#include "conv2d.h"
#include "timer.h"

int main(){

    {

    int N = 1, C = 3, H = 576, W = 576;
    int k_height = 3, k_width = 3;

    // Create input, kernel, and output tensors
    Tensor4D input(N, C, H, W);
    Tensor4D input2(N, C, H, W);
    Tensor4D x(N, 1, 10, 10);
    Tensor4D kernel(1, 128, k_height, k_width);
   
    // Set random values for input and kernel
    //input.setRandomValues(0.0f, 256.0f);
    //kernel.setRandomValues(0.0f, 1.0f);
	input.setValue();
	kernel.setValue();
    x.setValue();
		
	input.addPadding(1,1);
	
    Conv2d nn = Conv2d();
    Timer t1("conv");
    Tensor4D output1 = nn.convolution_2d(input, kernel, 1, 0);
	t1.stop();
   
    Timer t2("relu");
    nn.applyReLU(output1);
    t2.stop();

    Timer t3("maxpool");
    nn.applyMaxPool(output1, 2, 2, 2);
    t3.stop();
   
    Timer t4("batchNorm");
    nn.applyBatchNorm(output1, 1e-5);
    t4.stop();

    Timer t5("upsample");
    Tensor4D up = x.upsample(x.getH()*2,x.getW()*2);
    t5.stop();

    up.printDimensions();

    Timer t6("concat");
    Tensor4D c = x.concatAlongChannels(x);
    t6.stop();
    return 0;
}

}