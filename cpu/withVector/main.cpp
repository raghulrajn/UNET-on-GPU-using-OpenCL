#include "tensor4D.h"
#include "conv2d.h"
#include "timer.h"
#include <iomanip>
int main(){
    std::string kernelPath = "/home/raghul/Desktop/conv2d/CPU/kernels/inc_double_conv_0";
    Tensor4D input_image = Tensor4D::fromJPG("utils/image.jpg");
    std::cout<< "Input image dimensions: ";
    input_image.printDimensions();
    std::cout << std::endl;
    
    Conv2d nn = Conv2d();
    Timer t1("convolution");
    Tensor4D conv1 = nn.convolution_2d(input_image, kernelPath,1,0,false);
    // conv1.printDimensions();
    t1.stop();

    Timer t2("relu");
    nn.applyReLU(conv1);
    t2.stop();

    Timer t3("maxpool");
    nn.applyMaxPool(conv1, 2, 2, 2);
    t3.stop();
    // conv1.printDimensions();
   
    Timer t4("batchNorm");
    nn.applyBatchNorm(conv1, 1e-5);
    t4.stop();

    Timer t5("upsample");
    Tensor4D up = conv1.upsample(conv1.getH()*2,conv1.getW()*2);
    t5.stop();

    // up.printDimensions();

    Timer t6("concat");
    Tensor4D c = conv1.concatAlongChannels(conv1);
    t6.stop();

    return 0;
}
