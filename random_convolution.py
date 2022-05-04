import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2, ceil, floor

def sample_unit_kernel(size: tuple) -> torch.Tensor:
    """Sampler kernel $K \in \mathbb{R}^{c \times k \times k}$ that
    satisphies the constraint $\sum_{h=1}^{c} \sum_{i=1}^{d} \sum_{j=1}^{d} K_{h, i, j}^{(1) 2}=1$

    Args:
        size(tuple) size of the kernel (c, d1, d2)

    Returns:
        unit_kernel(torch.Tensor): kernel
    """
    kernel = torch.rand(size)
    unit_kernel = kernel / torch.sum(kernel)
    return unit_kernel

def generate_base_kernel_size(kernel_size: tuple, N:int):
    """Generate a list of kernel sizes for base slicer and even input size
    
    Args:
        input_size(tuple): input size (c, d1, d2)
        N(int): number of kernels
    
    Returns:
        kernel_sizes(list): list of kernel sizes
    """
    c, d = kernel_size[0], kernel_size[1]
    sizes = [(1, c, int(d/2 + 1), int(d/2 + 1))]
    for i in range(2,N):
        sizes.append((1, 1, int(d*2**-i + 1), int(d*2**-i + 1)))
    sizes.append((1, 1, int(d/(2**(N-1))), int(d/(2**(N-1)))))
    return sizes

class Conv_Base_Slicer(nn.Module):
    """Base slicer for convolutional sliced Wasserstein

    Attributes:
        input_size(tuple): input size (c, d1, d2)
        N(int): number of kernels
    """
    def __init__(self, input_size: tuple, N:int):
        super(Conv_Base_Slicer, self).__init__()
        
        self.input_size = input_size
        self.N = N
        self.kernel_sizes = generate_base_kernel_size(input_size, N)
        self.kernels = [sample_unit_kernel(size) for size in self.kernel_sizes]

    def forward(self, x):
        for filt in self.kernels:
            x = F.conv2d(x, filt)
        return x

def generate_kernel_size(kernel_size: tuple, N:int):
    """Generate a list of kernel sizes for stride and 
    dilatation slicer, works only for even input size
    
    Args:
        input_size(tuple): input size (c, d1, d2)
        N(int): number of kernels
    
    Returns:
        kernel_sizes(list): list of kernel sizes
    """
    c, d = kernel_size[0], kernel_size[1]
    sizes = [(1, c, 2, 2)]
    for i in range(2,N):
        sizes.append((1, 1, 2, 2))
    sizes.append((1, 1, int(d/(2**(N-1))), int(d/(2**(N-1)))))
    return sizes

class Conv_Stride_Slicer(nn.Module):
    """Stride slicer for convolutional sliced Wasserstein

    Attributes:
        input_size(tuple): input size (c, d1, d2)
        N(int): number of kernels
    """
    def __init__(self, input_size: tuple, N:int):
        super(Conv_Stride_Slicer, self).__init__()
        
        self.input_size = input_size
        self.N = N
        self.kernel_sizes = generate_kernel_size(input_size, N)
        self.kernels = [sample_unit_kernel(size) for size in self.kernel_sizes]

    def forward(self, x):
        for filt in self.kernels[:-1]:
            x = F.conv2d(x, filt, stride=2)
        x = F.conv2d(x, self.kernels[-1], stride=1)
        return x

class Conv_Dilatation_Slicer(nn.Module):
    """Stride slicer for convolutional sliced Wasserstein

    Attributes:
        input_size(tuple): input size (c, d1, d2)
        N(int): number of kernels
    """
    def __init__(self, input_size: tuple, N:int):
        super(Conv_Dilatation_Slicer, self).__init__()
        
        self.input_size = input_size
        self.N = N
        self.kernel_sizes = generate_kernel_size(input_size, N)
        self.kernels = [sample_unit_kernel(size) for size in self.kernel_sizes]

    def forward(self, x):
        for filt in self.kernels[:-1]:
            x = F.conv2d(x, filt, stride=1, dilation=3)
        x = F.conv2d(x, self.kernels[-1])
        return x

class Conv_Sliced_Wasserstein(nn.Module):
    """Convolutional sliced Wasserstein
    
    Attributes:
        input_size(tuple): input size (c, d1, d2)
        N(int): number of kernels
        L(int): number of sliced wasserstein layers
    """
    def __init__(self, input_size: tuple, N:int, L=int, type="base"):
        super(Conv_Sliced_Wasserstein, self).__init__()
        if type not in ["base", "stride", "dilatation"]:
            raise ValueError("type must be base, stride or dilatation")

        self.input_size = input_size
        self.N = N
        self.L = L
        if type == "base":
            self.conv_slicer = [Conv_Base_Slicer(input_size, N) for i in range(L)]
        elif type == "stride":
            self.conv_slicer = [Conv_Stride_Slicer(input_size, N) for i in range(L)]
        elif type == "dilatation":
            self.conv_slicer = [Conv_Dilatation_Slicer(input_size, N) for i in range(L)]

    def forward(self, x):
        out = []
        for filt in self.conv_slicer:
            out.append(filt.forward(x))
        out = torch.cat(out, dim=1).squeeze()
        return out

def wasserstein_distance(mu:torch.Tensor, nu:torch.Tensor, p=2):
    """
    Sliced Wasserstein distance between encoded samples and distribution samples

    Args:
        mu (torch.Tensor): tensor of samples from measure mu
        nu (torch.Tensor): tensor of samples from measure nu
        p (int): power of distance metric

    Return:
        torch.Tensor: Tensor of wasserstein distances of size (num_projections, 1)
    """
    wasserstein_distance = (torch.sort(mu, dim=1).values -
                            torch.sort(nu, dim=1).values)

    wasserstein_distance = torch.pow(wasserstein_distance, p)
    return wasserstein_distance.mean(1)