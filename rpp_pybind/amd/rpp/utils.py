# Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.
# MIT License
# Mukesh/rpp/rpp_pybind/amd/rpp/utils.py

"""
RPP Utility Functions
====================

Image loading, decoding, and conversion utilities using TurboJPEG.
Matches C++ test suite image loading for pixel-accurate testing.
"""

import numpy as np
import torch

try:
    from turbojpeg import TurboJPEG, TJPF_GRAY, TJPF_RGB
    TURBOJPEG_AVAILABLE = True
except ImportError:
    TURBOJPEG_AVAILABLE = False
    print("Warning: PyTurboJPEG not installed. Install with: pip install PyTurboJPEG")


def load_image(image_path, grayscale=False, device='cpu', apply_padding=True):
    """
    Load single JPEG image using TurboJPEG decoder.
    
    Args:
        image_path: Path to JPEG image file
        device: 'cpu' or 'cuda'
        apply_padding: Apply width padding to multiple of 8 (matches C++ tests)
        grayscale: If True, load as grayscale (PLN1), otherwise RGB (PKD3/PLN3)
    
    Returns:
        PyTorch tensor in NCHW format (1, C, H, W)
        C=1 for grayscale, C=3 for RGB
        If apply_padding=True, W is padded to (W/8)*8 + 8
    """
    if not TURBOJPEG_AVAILABLE:
        raise RuntimeError("PyTurboJPEG not installed. Run: pip install PyTurboJPEG")
    
    # Read and decode JPEG
    with open(image_path, 'rb') as f:
        jpeg_data = f.read()
    
    jpeg = TurboJPEG()
    
    if grayscale:
        # Decode as grayscale
        gray_array = jpeg.decode(jpeg_data, pixel_format=TJPF_GRAY)
        # Handle both 2D (H, W) and 3D (H, W, 1) grayscale arrays
        if len(gray_array.shape) == 3:
            height, width, _ = gray_array.shape
            gray_array = gray_array[:, :, 0]  # Take first channel
        else:
            height, width = gray_array.shape
        
        # Apply C++ test suite padding pattern
        if apply_padding:
            padded_width = (width // 8) * 8 + 8
            
            if padded_width > width:
                # Create padded array
                padded_array = np.zeros((height, padded_width), dtype=np.uint8)
                # Copy original image
                padded_array[:, :width] = gray_array
                # Replicate last column for padding
                padded_array[:, width:] = gray_array[:, -1:]
                gray_array = padded_array
        
        # Convert to PyTorch tensor (H, W) -> (1, H, W)
        tensor = torch.from_numpy(gray_array).unsqueeze(0).float()
    else:
        # Decode as RGB
        bgr_array = jpeg.decode(jpeg_data)  # Returns BGR (H, W, C)
        rgb_array = bgr_array[:, :, ::-1]   # Convert to RGB
        
        height, width, channels = rgb_array.shape
        
        # Apply C++ test suite padding pattern
        if apply_padding:
            padded_width = (width // 8) * 8 + 8
            
            if padded_width > width:
                # Create padded array
                padded_array = np.zeros((height, padded_width, channels), dtype=np.uint8)
                # Copy original image
                padded_array[:, :width, :] = rgb_array
                # Replicate last column for padding (matches C++ behavior)
                padded_array[:, width:, :] = rgb_array[:, -1:, :]
                rgb_array = padded_array
        
        # Convert to PyTorch tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float()
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    if device == 'cuda':
        tensor = tensor.cuda()
    
    return tensor


def decode_jpeg_bytes(jpeg_bytes):
    """
    Decode JPEG bytes to numpy array using TurboJPEG.
    
    Args:
        jpeg_bytes: JPEG file content as bytes
    
    Returns:
        numpy array (H, W, C) in RGB format, uint8
    """
    if not TURBOJPEG_AVAILABLE:
        raise RuntimeError("PyTurboJPEG not installed")
    
    jpeg = TurboJPEG()
    bgr_array = jpeg.decode(jpeg_bytes)
    rgb_array = bgr_array[:, :, ::-1]  # BGR -> RGB
    
    return rgb_array


def numpy_to_tensor(np_array, layout='NCHW', device='cpu'):
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        np_array: numpy array (H, W, C) or (B, H, W, C)
        layout: 'NCHW' or 'NHWC'
        device: 'cpu' or 'cuda'
    
    Returns:
        PyTorch tensor in specified layout
    """
    tensor = torch.from_numpy(np_array).float()
    
    if layout == 'NCHW':
        if tensor.ndim == 3:  # (H, W, C) -> (C, H, W)
            tensor = tensor.permute(2, 0, 1)
        elif tensor.ndim == 4:  # (B, H, W, C) -> (B, C, H, W)
            tensor = tensor.permute(0, 3, 1, 2)
    
    if device == 'cuda':
        tensor = tensor.cuda()
    
    return tensor


def tensor_to_numpy(tensor, layout='NCHW'):
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor
        layout: 'NCHW' or 'NHWC'
    
    Returns:
        numpy array uint8
    """
    # Move to CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert to numpy
    np_array = tensor.numpy()
    
    # Convert layout if needed
    if layout == 'NCHW':
        if np_array.ndim == 3:  # (C, H, W) -> (H, W, C)
            np_array = np.transpose(np_array, (1, 2, 0))
        elif np_array.ndim == 4:  # (B, C, H, W) -> (B, H, W, C)
            np_array = np.transpose(np_array, (0, 2, 3, 1))
    
    # Clip to valid range and convert to uint8
    np_array = np.clip(np_array, 0, 255).astype(np.uint8)
    
    return np_array


def save_image(tensor, output_path):
    """
    Save PyTorch tensor as JPEG image.
    
    Args:
        tensor: PyTorch tensor (C, H, W) or (1, C, H, W)
        output_path: Output JPEG file path
    """
    if not TURBOJPEG_AVAILABLE:
        raise RuntimeError("PyTurboJPEG not installed")
    
    # Remove batch dimension if present
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    
    # Convert to numpy (H, W, C) uint8
    np_array = tensor_to_numpy(tensor, layout='NCHW')
    
    # Convert RGB to BGR for JPEG encoding
    bgr_array = np_array[:, :, ::-1]
    
    # Encode with TurboJPEG
    jpeg = TurboJPEG()
    jpeg_bytes = jpeg.encode(bgr_array, quality=95)
    
    # Write to file
    with open(output_path, 'wb') as f:
        f.write(jpeg_bytes)


def create_test_batch(batch_size, height=224, width=224, channels=3, device='cpu'):
    """
    Create random test image batch for testing.
    
    Args:
        batch_size: Number of images
        height: Image height
        width: Image width
        channels: Number of channels (3 for RGB)
        device: 'cpu' or 'cuda'
    
    Returns:
        Random PyTorch tensor (B, C, H, W), values 0-255
    """
    tensor = torch.rand(batch_size, channels, height, width) * 255.0
    
    if device == 'cuda':
        tensor = tensor.cuda()
    
    return tensor


__all__ = [
    'load_image',
    'load_images',
    'decode_jpeg_bytes',
    'numpy_to_tensor',
    'tensor_to_numpy',
    'save_image',
    'create_test_batch'
]
