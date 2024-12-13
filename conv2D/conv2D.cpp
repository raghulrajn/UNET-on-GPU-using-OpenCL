#include<iostream>
#include<vector>
#include "tensor4d.cpp"
#include<Eigen/Dense>
#include<cmath>
#include<random>

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

        // Kernel initilization using kaiming for fixed kernel size (number of filters, kernel size)
        std::vector<Eigen::MatrixXf> kernelInitialization(int dim, int kernel_size){
            
            std::vector<Eigen::MatrixXf> filters;
            filters.resize(dim);
            std::cout<<"resized"<<"\n";
            for(int i=0; i<dim;++i)
            {
                float std = sqrt(2/(float) kernel_size);
                // std::cout<<"1"<<"\n";
                std::random_device rd;
                // std::cout<<"2"<<"\n";
                std::mt19937 gen(rd()); 
                // std::cout<<"3"<<"\n";
                std::normal_distribution<float> dist(0.0f, std); 
                // std::cout<<"4"<<"\n";
                Eigen::MatrixXf fil(3,3);
                for (int col = 0; col < kernel_size; col++) {
                    for (int row = 0; row < kernel_size; row++) {
                        // fil(row, col) = dist(gen);
                        
                        fil(row, col) = 1;
                    }
                }
                std::cout<<"i = "<<i<<"\n";
                filters[i] = fil;

            }
            std::cout<<"out of for loop"<<"\n";
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
            std::cout<<"kernel size "<<kernel_size<<"\n";
            std::vector<Eigen::MatrixXf> filters = kernelInitialization(outputdim, kernel_size);
            
            Tensor4D outputTensor(dim, channels, h_out, w_out);

            if(checkDimension(inputTensor))
            {
                for (int i = 0; i < dim; i++) 
                {
                    std::cout<<"dim = "<<i<<"\n";
                    int rows = 0;
                    int cols = 0;
                    for (int j = 0; j < channels; j++) 
                    {
                        std::cout<<"channel = "<<j<<"\n";
                    while(rows<h_in - (kernel_size-1))
                    {
                        while(cols<w_in-(kernel_size-1))
                        {
                            std::cout<<"rows = "<<rows<<" cols = "<<cols<<"\n";
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