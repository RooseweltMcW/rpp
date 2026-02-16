"""
Layout conversion utilities for RPP tensors
"""
import torch

def convert_nchw_to_nhwc(tensor):
    """
    Convert tensor from NCHW to NHWC format.
    
    Args:
        tensor: Input tensor (B, C, H, W)
    
    Returns:
        Converted tensor (B, H, W, C)
    """
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.dim()}D")
    
    # Permute dimensions: NCHW (0,1,2,3) → NHWC (0,2,3,1)
    return tensor.permute(0, 2, 3, 1).contiguous()


def convert_nhwc_to_nchw(tensor):
    """
    Convert tensor from NHWC to NCHW format.
    
    Args:
        tensor: Input tensor (B, H, W, C)
    
    Returns:
        Converted tensor (B, C, H, W)
    """
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.dim()}D")
    
    # Permute dimensions: NHWC (0,1,2,3) → NCHW (0,3,1,2)
    return tensor.permute(0, 3, 1, 2).contiguous()


def get_layout_name(layout_enum):
    """Get string name for layout enum"""
    from rpp_pybind.amd.rpp.rpp_types import RpptLayout
    
    if layout_enum == RpptLayout.NCHW:
        return "PKD3"
    elif layout_enum == RpptLayout.NHWC:
        return "PLN3"
    else:
        return f"LAYOUT_{layout_enum}"


__all__ = ['convert_nchw_to_nhwc', 'convert_nhwc_to_nchw', 'get_layout_name']
