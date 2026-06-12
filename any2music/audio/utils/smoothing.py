import typing as tp

import torch
import torch.nn.functional as F

def get_gaussian_kernel1d(kernel_size: int, sigma: float, device: tp.Union[torch.device, str]):
    """Generates a 1D Gaussian kernel."""
    # Create a coordinate grid centered at 0
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32, device=device)
    
    # Calculate the Gaussian distribution
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    
    # Normalize the kernel so the sum is 1 (preserves audio volume)
    kernel = kernel / kernel.sum()
    return kernel

def apply_gaussian_smoothing(audio: torch.Tensor, kernel_size: int = 21, sigma: float = 1.0):
    """
    Applies Gaussian smoothing to a 1D audio tensor.
    Expected input shape: (Batch, Channels, Time)
    """
    B, C, T = audio.shape
    
    # Get the kernel
    kernel = get_gaussian_kernel1d(kernel_size, sigma, device=audio.device)
    
    # Reshape kernel for conv1d: (out_channels, in_channels/groups, kernel_size)
    # We repeat it for each channel so we can process stereo/mono dynamically
    kernel = kernel.view(1, 1, -1).repeat(C, 1, 1)
    
    # Calculate padding to keep the audio length exactly the same
    padding = kernel_size // 2
    
    # Apply depthwise convolution (groups=C) to filter each channel independently
    smoothed_audio = F.conv1d(audio, kernel, padding=padding, groups=C)
    
    return smoothed_audio