# Conv2D implementation in CPU

### Features

Implemented 4D initialisation of tensor
 - Tensor-Tensor addition, Subtraction, Multiplication, Division
 - Tensor-Scalar addition, Subtraction, Multiplication, Division
 
 Implemented Conv2D functions
 - Convolution 
 - Maxpooling
 - Upsampling
 - ReLU
 - BatchNormalisation
 - LoadKernelfromModel
 - LoadBiasfromModel

### Installation required libs

```console
sudo apt-get install libeigen3-dev
sudo apt-get install -y libopencv-dev
pip install -r requirements.txt
```

### To save [pre-trained kernels](https://github.com/milesial/Pytorch-UNet)
- Download the .pth model from the above link
- run getKernels.py to save all the kernel as .npy file. Give path to .pth file in getKernels.py

```
git clone https://github.com/rogersce/cnpy.git
cd cnpy
mkdir -p build
cd build
cmake ..
make
```

### To read [pre-trained kernels](https://github.com/milesial/Pytorch-UNet)

```
git clone https://github.com/rogersce/cnpy.git
cd cnpy
mkdir -p build
cd build
cmake ..
make
```
Export cnpy to build path
```
export LD_LIBRARY_PATH=cnpy/build:$LD_LIBRARY_PATH
chmod +r cnpy/build/libcnpy.a
```
Copy cnpy folder to withVector folder

### To compile
```
cd withVector
make
```

### Link to documentation
[Link](https://docs.google.com/document/d/1qH3mKdrBO7R1P-sYfWqf-k0hHxdn1PYL75xTPcQsFmI/)
