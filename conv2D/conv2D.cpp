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

        // Adding padding vector to the matrix with odd dimensions. The padding vector is initilialized to zero
        Tensor4D paddingforOdd(Tensor4D inputTensor, int padding){
            int dim = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h = inputTensor.dimension(2);
            int w = inputTensor.dimension(3);

            if((padding>0) && (h%2)!=0 && (w%2)==0){
                Eigen::VectorXd zerovec(h+padding);
                zerovec.setZero();
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < channels; j++){
                        inputTensor(i,j).conservativeResize(h+padding,w);
                        inputTensor(i,j).row(inputTensor(i,j).rows()-1)= zerovec;
                    }
                }
            }

            if((padding>0) && (h%2)==0 && (w%2)!=0){
                Eigen::VectorXd zerovec(w+padding);
                zerovec.setZero();
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < channels; j++){
                        inputTensor(i,j).conservativeResize(h,w+padding);
                        inputTensor(i,j).col(inputTensor(i,j).cols()-1)= zerovec;
                    }
                }
            }

            if((padding>0) && (h%2)!=0 && (w%2)!=0 && (h==w)){
                Eigen::VectorXd zerovec(w+padding);
                zerovec.setZero();
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < channels; j++){
                        inputTensor(i,j).conservativeResize(h+padding,w+padding);
                        inputTensor(i,j).row(inputTensor(i,j).rows()-1)= zerovec;
                        inputTensor(i,j).col(inputTensor(i,j).cols()-1)= zerovec;
                    }
                }
            }

            return inputTensor;
        }

        Tensor4D paddingforEven(Tensor4D inputTensor, int padding){
            int dim = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h = inputTensor.dimension(2);
            int w = inputTensor.dimension(3);
            Eigen::Matrix3d mat3x3;
            mat3x3 << 1, 2, 3,
              4, 5, 6,
              7, 8, 9;

            //  Eigen::MatrixXf newMatrix = Eigen::MatrixXd::Zero(h+padding*2, w+padding*2);
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < channels; j++){
                    Eigen::MatrixXf newMatrix = Eigen::MatrixXf::Zero(h+padding*2, w+padding*2);
                    newMatrix.block<h,w>(1,1) = inputTensor(i,j);
                    // inputTensor(i,j).conservativeResize(h+padding*2, w+padding*2);
                    // inputTensor(i,j).row(inputTensor(i,j).rows()-1)= zerovec;
                }
            }

            return inputTensor;
        }

        // Shrink the matrix from the top left cornner if the padding is >0 and the matrix is having odd dimensions
        Tensor4D shrinkMatrix(Tensor4D inputTensor){
            int dim = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h = inputTensor.dimension(2);
            int w = inputTensor.dimension(3);

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

            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < channels; j++){
                    inputTensor(i,j).conservativeResize(h,w);
                    }
                }
            return inputTensor;
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

        template <typename T>
        double dot(T m1, Eigen::MatrixXf m2){
            double result=0;
            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++){
                    result += m1(i,j)*m2(i,j);
                }
            }
            return result;
        }

    public:
        Tensor4D conv2d(Tensor4D inputTensor,int outputdim, int kernel_size=3, int stride=1, int padding=0, bool bias=false){
            int dim      = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h_in     = inputTensor.dimension(2);
            int w_in     = inputTensor.dimension(3);

            int h_out = floor(((h_in + 2*padding - kernel_size)/stride) +1);
            int w_out = floor(((w_in + 2*padding - kernel_size)/stride) +1);
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

        Tensor4D maxpool(Tensor4D inputTensor, int kernel_size=2, int stride=1, int padding=0, bool ceil_mode=false){
            int dim = inputTensor.dimension(0);
            int channels = inputTensor.dimension(1);
            int h = inputTensor.dimension(2);
            int w = inputTensor.dimension(3);
            if (padding>(int)kernel_size/2){
                throw std::invalid_argument( "padding should be at most half of effective kernel size");
            }

            if((padding>0) && ((h%2 != 0) || (w%2 != 0))){
                inputTensor = paddingforOdd(inputTensor, padding);
            }

            if((padding==0) && ((h%2 != 0) || (w%2 != 0))){
                inputTensor = shrinkMatrix(inputTensor);
            }

            if((padding==0) && (h%2 == 0) && (w%2 == 0)){
                inputTensor = paddingforEven(inputTensor, padding);
            }

            int h_in     = inputTensor.dimension(2);
            int w_in     = inputTensor.dimension(3);

            int h_out = floor(((h_in + 2*padding - kernel_size)/stride) +1);
            int w_out = floor(((w_in + 2*padding - kernel_size)/stride) +1);
            int row_stride = 0;
            int col_stride = 0;
            
            Tensor4D outputTensor(dim, channels, h_out, w_out);

            


        }
};

int main(){
    int N = 1;  // Batch size
    int C = 1;  // Channels
    int H = 10;  // Height
    int W = 10;  // Width

    Tensor4D input_tensor1(N, C, H, W);
    Tensor4D input_tensor2(N, C, H, W);
    std::cout<<"before"<<"\n";
    // Initialize one of the matrices with random values
    input_tensor1(0, 0) = Eigen::MatrixXf::Random(H, W);
    input_tensor2(0, 0) = Eigen::MatrixXf::Random(H, W);
    std::cout<<"conv2d"<<"\n";
    Conv2D nn = Conv2D();
    std::cout<<"flag"<<"\n";
    Tensor4D output = nn.conv2d(input_tensor2, 2);

    input_tensor2.print();
    output.print();
    return 0;
}