#Convolution sample with pytorch. Output is saved as .npy file can be used to compare with CPU/GPU output
import numpy as np
import torch
import torch.nn.functional as F
import cv2
eps = 1e-5

outC = 32  # Output channels
C = 3      # Input channels
kh, kw = 3, 3  # Kernel height and width
bias_tensor = torch.full((outC,), 0.001, dtype=torch.float32)   # Shape: (outC,)
gamma_tensor = torch.full((outC,), 0.001, dtype=torch.float32)  # Shape: (outC,)
beta_tensor = torch.full((outC,), 0.001, dtype=torch.float32)   # Shape: (outC,)
rmean_tensor = torch.full((outC,), 0.005, dtype=torch.float32)  # Shape: (outC,)
rvar_tensor = torch.full((outC,), 0.005, dtype=torch.float32)   # Shape: (outC,)
weights_tensor = torch.full((outC, C, kh, kw), 0.005, dtype=torch.float32)  # Shape: (outC, C, kh, kw)

print("Weights shape:", weights_tensor.shape)
print("Bias shape:", bias_tensor.shape)
print("Gamma shape:", gamma_tensor.shape)
print("Beta shape:", beta_tensor.shape)
print("Running mean shape:", rmean_tensor.shape)
print("Running var shape:", rvar_tensor.shape)

inputShape = (1, C, 224, 224) #(N,C,H,W)
input_tensor = torch.full(inputShape, 0.05, dtype=torch.float32)
print("Image tensor shape:", input_tensor.shape)
# Apply convolution
conv_result = F.conv2d(
    input=input_tensor,
    weight=weights_tensor,
    bias=None,
    stride=1,
    padding=1,
)

conv_result = conv_result + bias_tensor.view(1, -1, 1, 1)
print("Pytorch Convolution result shape:", conv_result.shape)

# Normalization: (x - mean) / sqrt(var + eps) * gamma + beta
x_norm = (conv_result - rmean_tensor.view(1, -1, 1, 1)) / torch.sqrt(rvar_tensor.view(1, -1, 1, 1) + eps)
x_bn = x_norm * gamma_tensor.view(1, -1, 1, 1) + beta_tensor.view(1, -1, 1, 1)
print("Pytorch batchnorm result shape:", x_bn.shape)
# ReLU activation
x_relu = F.relu(input_tensor)
print("Pytorch ReLU result shape:", x_relu.shape)
# MaxPooling (2x2 kernel with stride 2)
x_maxpool = F.max_pool2d(input_tensor, kernel_size=2, stride=2)
print("Pytorch x_maxpool result shape:", x_maxpool.shape)
# Upsampling (bilinear interpolation)
x_upsampled = F.interpolate(x_maxpool, scale_factor=2, mode='bilinear', align_corners=True)
print("Pytorch Upsample result shape:", x_upsampled.shape)

# Convert to NumPy and save for comparison
conv_result_np = conv_result.detach().numpy()
x_bn_np = x_bn.detach().numpy()
x_relu_np = x_relu.detach().numpy()
x_maxpool_np = x_maxpool.detach().numpy()
x_upsampled_np = x_upsampled.detach().numpy()

np.save('npy/pytorch_convolution.npy', conv_result_np)
np.save('npy/pytorch_batchnorm.npy', x_bn_np)
np.save('npy/pytorch_relu.npy', x_relu_np)
np.save('npy/pytorch_maxpool.npy', x_maxpool_np)
np.save('npy/pytorch_upsample.npy', x_upsampled_np)