#include<iostream>
#include<vector>
#include "tensor4d.cpp"
#include<Eigen/Dense>
#include<cmath>
#include<random>
#include <stdexcept>

class Conv2D{

    private:
        bool checkDimension(Tensor4D t){
            int h = t.dimension(2);
            int w = t.dimension(3);
            if((h%2 == 0 && w%2 ==0) && h == w) return true;

            else if((h%2 != 0 || w%2 !=0)) return false;

            else if (h !=w) return false;

            else return false;
        }

        Tensor4D padTensor(Tensor4D inputTensor, int padding){
            int dim = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h = inputTensor.dimension(2);
            int w = inputTensor.dimension(3);

            int h_out = h+padding*2;
            int w_out = w+padding*2;

            Tensor4D outputTensor(dim, channels, h_out, w_out);

            const int h_in = h;
            const int w_in = w;
            
            //  Eigen::MatrixXf newMatrix = Eigen::MatrixXd::Zero(h+padding*2, w+padding*2);
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < channels; j++){
                    Eigen::MatrixXf newMatrix = Eigen::MatrixXf::Zero(h+padding*2, w+padding*2);
                    newMatrix.block(padding, padding, h,w) = inputTensor(i,j);
                    outputTensor(i,j) = newMatrix;
                    // newMatrix.block(padding, padding, h,w) = inputTensor(i,j);
                }
            }


            return outputTensor;
        }

        // Shrink the matrix from the top left cornner if the padding is >0 and the matrix is having odd dimensions
        Tensor4D shrinkMatrix(Tensor4D inputTensor){
            int dim = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h = inputTensor.dimension(2);
            int w = inputTensor.dimension(3);
            std::cout<<h<<"  "<<w<<std::endl;
            if (h==w && h%2 != 0 && w%2!=0){
                h = h-1;
                w = w-1;
            }
            else{
                int minEven = std::min(h,w) & ~1;
                h = minEven;
                w = minEven;
            }
            assert(h%2==0);
            assert(w%2==0);
            Tensor4D outputTensor(dim, channels, h, w);

            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < channels; j++){
                    outputTensor(i,j) = inputTensor(i,j);
                    }
                }
            return outputTensor;
        }
        
        // Kernel initilization using kaiming for fixed kernel size (number of filters, kernel size)
        std::vector<Eigen::MatrixXf> kernelInitialization(int dim, int kernel_size){
            
            std::vector<Eigen::MatrixXf> filters;
            filters.resize(dim);
            // std::cout<<"resized"<<"\n";
            for(int i=0; i<dim;++i)
            {
                float std = sqrt(2/(float) kernel_size);
                std::random_device rd;
                std::mt19937 gen(rd()); 
                std::normal_distribution<float> dist(0.0f, std); 
                Eigen::MatrixXf fil(3,3);
                for (int col = 0; col < kernel_size; col++) {
                    for (int row = 0; row < kernel_size; row++) {
                        // fil(row, col) = dist(gen);
                        fil(row, col) = 1;
                    }
                }
                // std::cout<<"i = "<<i<<"\n";
                filters[i] = fil;
            }
            // std::cout<<"out of for loop"<<"\n";
            return filters;
        }

        template <typename matrixBlock>
        double dot(matrixBlock m1, Eigen::MatrixXf m2){
            double result=0;
            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++){
                    result += m1(i,j)*m2(i,j);
                }
            }
            return result;
        }
        
        
    public:
        Tensor4D conv2d(Tensor4D inputTensor,int outputdim, const int kernel_size=3, int stride=1, int padding=0, bool bias=false){
            int dim      = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h_in     = inputTensor.dimension(2);
            int w_in     = inputTensor.dimension(3);
            const int k_z = kernel_size; 

            int h_out = floor(((h_in + 2*padding - (kernel_size-1)-1)/stride) +1);
            int w_out = floor(((w_in + 2*padding - (kernel_size-1)-1)/stride) +1);
            int row_stride = 0;
            int col_stride = 0;
            // std::cout<<"kernel size "<<kernel_size<<"\n";
            std::vector<Eigen::MatrixXf> filters = kernelInitialization(outputdim, kernel_size);
            
            Tensor4D outputTensor(dim, channels, h_out, w_out);

            if(checkDimension(inputTensor))
            {
                for (int i = 0; i < dim; i++) 
                {
                    // std::cout<<"dim = "<<i<<"\n";
                    int rows = 0;
                    int cols = 0;
                    for (int j = 0; j < channels; j++) 
                    {
                    // std::cout<<"channel = "<<j<<"\n";
                    while(rows<h_in - (kernel_size-1))
                    {
                        while(cols<w_in-(kernel_size-1))
                        {
                            // std::cout<<"rows = "<<rows<<" cols = "<<cols<<"\n";
                            double d_p = dot(inputTensor(i,j).block<3,3>(rows,cols), filters[dim]);
                            cols+=stride;
                            outputTensor(i, j)(row_stride,col_stride) = d_p;
                            col_stride++;
                        }
                        rows+=stride;
                        cols=0;
                        row_stride++;
                        col_stride = 0;
                    }
                    }
                }

            }

        return outputTensor;
        }

        Tensor4D maxpool2d(Tensor4D inputTensor, int kernel_size=2, int stride=2, int padding=0, bool ceil_mode=false){
            int dim = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            const int conv_h = inputTensor.dimension(2);
            const int conv_w = inputTensor.dimension(3);
            if (padding>(int)kernel_size/2){
                throw std::invalid_argument( "padding should be at most half of effective kernel size");
            }

            if(padding>0){
                inputTensor = padTensor(inputTensor, padding);
            }

            inputTensor = shrinkMatrix(inputTensor);

            int h_in     = inputTensor.dimension(2);
            int w_in     = inputTensor.dimension(3);

            int h_out = floor(((conv_h + 2*padding - (kernel_size-1)-1)/stride) +1);
            int w_out = floor(((conv_w + 2*padding - (kernel_size-1)-1)/stride) +1);
            int row_stride = 0;
            int col_stride = 0;
            
            Tensor4D outputTensor(dim, channels, h_out, w_out);

            for (int i = 0; i < dim; i++) 
            {
                // std::cout<<"dim = "<<i<<"\n";
                int rows = 0;
                int cols = 0;
                for (int j = 0; j < channels; j++) 
                {
                // std::cout<<"channel = "<<j<<"\n";
                while(rows<h_in - (kernel_size-1))
                {
                    while(cols<w_in-(kernel_size-1))
                    {
                        float d_p = inputTensor(i,j).block<2,2>(rows,cols).maxCoeff();
                        cols+=stride;
                        outputTensor(i, j)(row_stride,col_stride) = d_p;
                        col_stride++;
                    }
                    rows+=stride;
                    cols=0;
                    row_stride++;
                    col_stride = 0;
                }
                }
            }
        return outputTensor;
        }
};


void getShape(Tensor4D tensor4d){
        int dim_n = tensor4d.dimension(0);
        int channels_n = tensor4d.dimension(1);
        int h_n = tensor4d.dimension(2);
        int w_n = tensor4d.dimension(3);

        std::cout<<"("<<dim_n<<","<<channels_n<<","<<h_n<<","<<w_n<<")"<<std::endl;
    }

int main(){
    int N = 1;  // Batch size
    int C = 1;  // Channels
    int H = 10;  // Height
    int W = 10;  // Width

    Tensor4D input_tensor1(N, C, H, W);
    Tensor4D input_tensor2(N, C, H, W);

    // Initialize one of the matrices with random values
    input_tensor1(0, 0) = Eigen::MatrixXf::Random(H, W);
    input_tensor2(0, 0) = Eigen::MatrixXf::Random(H, W);

    input_tensor2.print();

    Conv2D nn = Conv2D();

    Tensor4D output = nn.conv2d(input_tensor2, 2);

    std::cout<<"After Convolution \n";
    output.print();

    Tensor4D output1 = nn.maxpool2d(output,2,2,1);
    std::cout<<"After Maxpooling \n";
    output1.print();
    return 0;
}