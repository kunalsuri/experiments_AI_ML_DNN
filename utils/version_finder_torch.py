import torch

# PyTorch
# To find the version of the PyTorch installed in your system
# Latest version of the PyTorch can be downloaded via : https://pytorch.org/

print("Torch Version is : " + torch.__version__)

# CUDA
# Check if CUDA is availabe | To verify if the correct version of the PyTorch has been installed
# CUDA 2.3.1 RC : https://dev-discuss.pytorch.org/t/pytorch-release-2-3-1-final-rc-is-available/2126

print("Torch Version for CUDA available : ") 
print(torch.cuda.is_available())

print("Torch Example") 
x = torch.rand(5, 3)
print(x)