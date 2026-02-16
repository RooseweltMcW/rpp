# Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.
# MIT License
# Mukesh/rpp/rpp_pybind/amd/rpp/fn.py

"""
RPP Augmentation Functions
===========================

High-level wrapper functions for RPP augmentations.
Similar to rocAL's fn.py pattern.
"""

# Simple imports - like rocAL
import rpp_pybind
from rpp_pybind.amd.rpp.rpp_types import get_default_backend, HOST, HIP
from rpp_pybind.amd.rpp.rpp_types import RpptLayout
from rpp_pybind.amd.rpp.layout_utils import convert_nchw_to_nhwc, convert_nhwc_to_nchw, get_layout_name
import ctypes
import torch

# Direct access to C++ functions
_brightness = rpp_pybind.brightness
_gamma_correction = rpp_pybind.gamma_correction
_contrast = rpp_pybind.contrast
_hue = rpp_pybind.hue
_flip = rpp_pybind.flip
_resize = rpp_pybind.resize
_rotate = rpp_pybind.rotate
_crop = rpp_pybind.crop
_vignette = rpp_pybind.vignette
_pixelate = rpp_pybind.pixelate
rppCreate = rpp_pybind.rppCreate
rppDestroy = rpp_pybind.rppDestroy

def _prepare_tensor_and_backend(tensor, backend):
    """Prepare tensor for RPP operations"""   
    if backend is None:
        backend = get_default_backend()

    # Get backend integer consistently
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    # Validate tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got {tensor.dim()}D")

    # Ensure contiguous
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # Ensure correct device
    if backend_int == 1:
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        torch.cuda.synchronize()
    else:  # HOST
        if tensor.is_cuda:
            tensor = tensor.cpu()
    
    return tensor, backend_int

# Wrapper functions - 10 augmentations from different groups

# Color Augmentations (4)
def brightness(images, alpha=1.0, beta=0.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Adjust image brightness - Auto-detects layout from tensor shape
    
    Args:
        images: Input tensor
        alpha: Brightness multiplier
        beta: Brightness offset  
        roi_widths: List of actual image widths
        roi_heights: List of actual image heights
        input_layout: Ignored - auto-detected from tensor
        output_layout: Ignored - auto-detected from tensor
        backend: RppBackend
    
    Returns:
        Augmented images tensor
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()

    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    if not images.is_contiguous():
        images = images.contiguous()
    
    if images.dtype != torch.uint8:
        images = (images.clamp(0, 255)).to(torch.uint8)
    
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()

    # Create output with same shape - C++ auto-detects layouts
    output = torch.zeros_like(images).contiguous()

    batch_size = images.shape[0]
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
 
    handle = rppCreate(batch_size, backend_int)

    alpha_array = [alpha] * batch_size
    beta_array = [beta] * batch_size
    
    _brightness(images, output, alpha_array, beta_array, roi_widths, roi_heights, handle, backend_int)

    rppDestroy(handle, backend_int)
    
    return output


def gamma_correction(images, gamma=1.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Apply gamma correction - Auto-detects layout from tensor shape
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()

    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()

    batch_size = images.shape[0]
    output = torch.zeros_like(images).contiguous()

    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size

    handle = rppCreate(batch_size, backend_int)
    gamma_array = [gamma] * batch_size
    _gamma_correction(images, output, gamma_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output

# __all__ = [
#     # Color augmentations
#     'brightness',
#     'gamma_correction'
# ]

def contrast(images, contrast_factor=1.0, contrast_center=128.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Adjust image contrast - Auto-detects layout from tensor shape
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.empty_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    
    contrast_factor_array = [contrast_factor] * batch_size
    contrast_center_array = [contrast_center] * batch_size
    
    _contrast(images, output, contrast_factor_array, contrast_center_array, roi_widths, roi_heights, handle, backend_int)
    
    rppDestroy(handle, backend_int)
    
    return output


def hue(images, hue_shift=0.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Adjust image hue - Auto-detects layout from tensor shape
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.empty_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    hue_shift_array = [hue_shift] * batch_size
    _hue(images, output, hue_shift_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


# Geometric Augmentations (4)
def flip(images, horizontal=False, vertical=False, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Flip images - Auto-detects layout from tensor shape
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.empty_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    
    # Convert bool to list
    if isinstance(horizontal, bool):
        horizontal = [int(horizontal)] * batch_size
    if isinstance(vertical, bool):
        vertical = [int(vertical)] * batch_size
    
    _flip(images, output, horizontal, vertical, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


def resize(images, width, height, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Resize images - Creates output with resize dimensions
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.empty_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    
    width_array = [width] * batch_size
    height_array = [height] * batch_size
    
    _resize(images, output, width_array, height_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


def rotate(images, angle=0.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Rotate images - Auto-detects layout from tensor shape
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.empty_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    angle_array = [angle] * batch_size
    _rotate(images, output, angle_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


def crop(images, x1, y1, crop_width, crop_height, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Crop images - Creates output with crop dimensions, C++ auto-detects layouts
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    device = images.device
    
    # Detect layout from input shape
    is_nchw = (images.shape[1] <= 3)
    channels = images.shape[1] if is_nchw else images.shape[3]
    
    # Create output maintaining same layout as input
    if is_nchw:
        output = torch.empty(batch_size, channels, crop_height, crop_width, dtype=images.dtype, device=device)
    else:
        output = torch.empty(batch_size, crop_height, crop_width, channels, dtype=images.dtype, device=device)
    
    handle = rppCreate(batch_size, backend_int)
    
    # Convert scalars to lists
    x1_array = [x1] * batch_size if isinstance(x1, (int, float)) else x1
    y1_array = [y1] * batch_size if isinstance(y1, (int, float)) else y1
    width_array = [crop_width] * batch_size if isinstance(crop_width, (int, float)) else crop_width
    height_array = [crop_height] * batch_size if isinstance(crop_height, (int, float)) else crop_height
    
    _crop(images, output, x1_array, y1_array, width_array, height_array, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


# Effects Augmentations (2)
def vignette(images, intensity=0.5, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Apply vignette effect - Auto-detects layout from tensor shape
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.empty_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    intensity_array = [intensity] * batch_size
    _vignette(images, output, intensity_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


def pixelate(images, pixelation_percentage=50.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Apply pixelate effect - Auto-detects layout from tensor shape
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    
    # Create output tensor based on layout
    if input_layout == output_layout:
        output = torch.empty_like(images)
    else:
        if input_layout == 'NCHW' and output_layout == 'NHWC':
            b, c, h, w = images.shape
            output = torch.zeros(b, h, w, c, dtype=images.dtype, device=images.device)
        elif input_layout == 'NHWC' and output_layout == 'NCHW':
            b, h, w, c = images.shape
            output = torch.zeros(b, c, h, w, dtype=images.dtype, device=images.device)
        else:
            output = torch.empty_like(images)
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if input_layout == 'NCHW' else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if input_layout == 'NCHW' else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    
    angle_array = [angle] * batch_size
    
    _rotate(images, output, angle_array, roi_widths, roi_heights, handle, backend_int)
    
    rppDestroy(handle, backend_int)
    
    return output


def crop(images, x1, y1, crop_width, crop_height, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Crop images to specified region.
    
    Args:
        images: Input tensor
        x1: Top-left x coordinate
        y1: Top-left y coordinate  
        crop_width: Width of crop region
        crop_height: Height of crop region
        roi_widths: Ignored (for compatibility)
        roi_heights: Ignored (for compatibility)
        input_layout: 'NCHW' or 'NHWC'
        output_layout: 'NCHW' or 'NHWC'
        backend: RppBackend
    
    Returns:
        Cropped images
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    device = images.device
    
    # Determine channels based on input layout
    if input_layout == 'NCHW':
        channels = images.shape[1]
    else:  # NHWC
        channels = images.shape[3]
    
    # Create output with correct shape based on output layout
    if output_layout == 'NCHW':
        output = torch.empty(batch_size, channels, crop_height, crop_width, dtype=images.dtype, device=device)
    else:  # NHWC
        output = torch.empty(batch_size, crop_height, crop_width, channels, dtype=images.dtype, device=device)
    
    handle = rppCreate(batch_size, backend=backend)
    
    # Convert scalars to lists
    x1_array = [x1] * batch_size if isinstance(x1, (int, float)) else x1
    y1_array = [y1] * batch_size if isinstance(y1, (int, float)) else y1
    width_array = [crop_width] * batch_size if isinstance(crop_width, (int, float)) else crop_width
    height_array = [crop_height] * batch_size if isinstance(crop_height, (int, float)) else crop_height
    
    _crop(images, output, x1_array, y1_array, width_array, height_array, handle, backend)
    
    rppDestroy(handle, backend)
    
    return output


# Effects Augmentations (2)
def vignette(images, intensity=0.5, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Apply vignette effect to images.
    
    Args:
        images: Input tensor
        intensity: Vignette intensity (0.0 to 1.0)
        roi_widths: List of image widths
        roi_heights: List of image heights
        input_layout: 'NCHW' or 'NHWC'
        output_layout: 'NCHW' or 'NHWC'
        backend: RppBackend
    
    Returns:
        Images with vignette effect
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    
    # Create output tensor based on layout
    if input_layout == output_layout:
        output = torch.empty_like(images)
    else:
        if input_layout == 'NCHW' and output_layout == 'NHWC':
            b, c, h, w = images.shape
            output = torch.zeros(b, h, w, c, dtype=images.dtype, device=images.device)
        elif input_layout == 'NHWC' and output_layout == 'NCHW':
            b, h, w, c = images.shape
            output = torch.zeros(b, c, h, w, dtype=images.dtype, device=images.device)
        else:
            output = torch.empty_like(images)
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if input_layout == 'NCHW' else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if input_layout == 'NCHW' else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    
    intensity_array = [intensity] * batch_size
    
    _vignette(images, output, intensity_array, roi_widths, roi_heights, handle, backend_int)
    
    rppDestroy(handle, backend_int)
    
    return output


def pixelate(images, pixelation_percentage=50.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Apply pixelate effect to images.
    
    Args:
        images: Input tensor
        pixelation_percentage: Pixelation level (0-100)
        roi_widths: List of image widths
        roi_heights: List of image heights
        input_layout: 'NCHW' or 'NHWC'
        output_layout: 'NCHW' or 'NHWC'
        backend: RppBackend
    
    Returns:
        Pixelated images
    """
    import torch
    
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    if images.dtype != torch.uint8:
        images = images.clamp(0, 255).to(torch.uint8)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    
    # Create output tensor based on layout
    if input_layout == output_layout:
        output = torch.empty_like(images)
    else:
        if input_layout == 'NCHW' and output_layout == 'NHWC':
            b, c, h, w = images.shape
            output = torch.zeros(b, h, w, c, dtype=images.dtype, device=images.device)
        elif input_layout == 'NHWC' and output_layout == 'NCHW':
            b, h, w, c = images.shape
            output = torch.zeros(b, c, h, w, dtype=images.dtype, device=images.device)
        else:
            output = torch.empty_like(images)
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if input_layout == 'NCHW' else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if input_layout == 'NCHW' else images.shape[1]] * batch_size
    
    # Create scratch buffer
    if input_layout == 'NCHW':
        scratch_size = batch_size * images.shape[1] * images.shape[2] * images.shape[3]
    else:  # NHWC
        scratch_size = batch_size * images.shape[1] * images.shape[2] * images.shape[3]
    scratch = torch.empty(scratch_size, dtype=torch.float32, device=images.device)
    
    handle = rppCreate(batch_size, backend_int)
    
    _pixelate(images, output, scratch, pixelation_percentage, roi_widths, roi_heights, handle, backend_int)
    
    rppDestroy(handle, backend_int)
    
    return output


__all__ = [
    # Color augmentations
    'brightness',
    'gamma_correction', 
    'contrast',
    'hue',
    # Geometric augmentations
    'flip',
    'resize',
    'rotate',
    'crop',
    # Effects augmentations
    'vignette',
    'pixelate'
]
