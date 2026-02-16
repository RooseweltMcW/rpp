"""
RPP Test Suite
==============

Unified test suite with combined Unit/QA testing per augmentation.
- Unit mode: Apply augmentation and save image
- QA mode: Apply augmentation and compare with reference tensor
- Performance mode: Time measurements
- All mode: Run all three test types

Usage:
    python test_suite.py --mode UNIT --backend HOST
    python test_suite.py --mode QA --backend HOST
    python test_suite.py --mode PERF --backend HIP
    python test_suite.py --mode ALL --backend HOST
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

print(f"Loading from: {SCRIPT_DIR}")

# Import RPP modules
import rpp_pybind.amd.rpp.fn as fn
import rpp_pybind.amd.rpp.utils as util
from rpp_pybind.amd.rpp.rpp_types import (
    is_gpu_available, get_default_backend, HOST, HIP
)

print("✓ All RPP modules loaded successfully\n")

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class TestConfig:
    """Global test configuration"""
    
    def __init__(self):
        # Directories
        self.TEST_IMAGES_DIR = "../test_suite/TEST_IMAGES/three_images_mixed_src1"
        self.REFERENCE_DIR = "../test_suite/REFERENCE_OUTPUT"
        
        # Test settings
        self.TOLERANCE = 1  # pixel difference tolerance for QA
        
        # Test image paths and specs
        self.TEST_IMAGES = [
            "1_img50x50.jpg",
            "2_img100x100.jpg", 
            "3_img150x150.jpg"
        ]
        
        # Image dimensions (actual sizes)
        self.IMAGE_SPECS = [
            (50, 50),   # Image 0
            (100, 100), # Image 1
            (150, 150)  # Image 2
        ]
        
        # Layout variants to test
        self.LAYOUT_VARIANTS = [
            ('PKD3', 'PKD3'),  # NCHW → NCHW
            ('PKD3', 'PLN3'),  # NCHW → NHWC
            ('PLN3', 'PLN3'),  # NHWC → NHWC
            ('PLN3', 'PKD3'),  # NHWC → NCHW
            ('PLN1', 'PLN1'),  # NCHW → NCHW
        ]

        # Reference batch dimensions
        self.BATCH_HEIGHT = 150
        self.BATCH_WIDTH = 152
        self.BATCH_SIZE = 3
        
        # Augmentation parameters (matching C++ test suite)
        # NOTE: crop and resize dimensions are calculated per-image (not hardcoded)
        self.AUGMENTATION_PARAMS = {
            'brightness': {'alpha': 1.75, 'beta': 50.0},
            'gamma_correction': {'gamma': 1.9},
            'contrast': {'contrast_factor': 2.96, 'contrast_center': 128.0},
            'flip': {'horizontal': True, 'vertical': False},
            'resize': {},  # Calculated per image (width/2, height/2)
            'crop': {'x1': 10, 'y1': 10},  # crop_width and crop_height calculated per image
            'hue': {'hue_shift': 60.0},
            'rotate': {'angle': 50.0},
            'vignette': {'intensity': 6.0},
            'pixelate': {'pixelation_percentage': 87.5}
        }
        
        # Timestamp for output directories
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def get_output_dir(self, backend_name, mode):
        """Get output directory based on backend and mode"""
        return f"OUTPUT_IMAGES_{backend_name}_{self.timestamp}"


# =============================================================================
# UNIFIED TEST CLASS
# =============================================================================

class UnifiedTestSuite:
    """Unified test suite with combined Unit/QA/Performance testing"""
    
    def __init__(self, backend, mode="ALL", case_list=None, preserve_output=0):
        self.backend = backend
        self.backend_name = "HIP" if backend == HIP else "HOST"
        self.mode = mode.upper()  # UNIT, QA, PERF, or ALL
        self.case_list = case_list  # List of case numbers to run (None = all)
        self.preserve_output = preserve_output  # 0=delete previous, 1=preserve
        self.config = TestConfig()
        self.results = {
            'unit': [],
            'qa': [],
            'perf': []
        }
        
        # Case number to test function mapping
        self.case_to_test_map = {
            0: ('brightness', self.test_brightness),
            1: ('gamma_correction', self.test_gamma_correction),
            4: ('contrast', self.test_contrast),
            5: ('pixelate', self.test_pixelate),
            20: ('flip', self.test_flip),
            21: ('resize', self.test_resize),
            23: ('rotate', self.test_rotate),
            37: ('crop', self.test_crop),
            42: ('hue', self.test_hue),
            46: ('vignette', self.test_vignette)
        }
        
        # Clean up previous output folders if preserve_output=0
        if self.mode in ["UNIT", "ALL"] and self.preserve_output == 0:
            self._cleanup_previous_outputs()
        
        # Setup directories
        if self.mode in ["UNIT", "ALL"]:
            self.unit_output_dir = self.config.get_output_dir(self.backend_name, "UNIT")
            os.makedirs(self.unit_output_dir, exist_ok=True)
            print(f"Unit Output Directory: {self.unit_output_dir}")
        
        # Load test images paths
        self.test_images = [
            os.path.join(self.config.TEST_IMAGES_DIR, img) 
            for img in self.config.TEST_IMAGES
        ]
        
        print(f"Backend: {self.backend_name}")
        print(f"Mode: {self.mode}")
        print(f"Test Images: {len(self.test_images)}")
        print("-" * 70)
    
    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    
    def _cleanup_previous_outputs(self):
        """Delete previous output folders for current backend"""
        import shutil
        import glob
        
        # Pattern to match output directories for this backend
        pattern = f"OUTPUT_IMAGES_{self.backend_name}_*"
        
        # Find and delete matching directories
        deleted_count = 0
        for dir_path in glob.glob(pattern):
            if os.path.isdir(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Warning: Could not delete {dir_path}: {e}")
        
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} previous output folder(s)")
    
    def _save_output_image(self, tensor, augmentation_name, image_name, img_idx, layout_variant=None):
        """Save output image to filesystem (Unit mode)"""
        try:
            # Create augmentation-specific directory
            aug_output_dir = os.path.join(self.unit_output_dir, augmentation_name)
            
            # If layout_variant is provided, create subdirectory for it
            if layout_variant:
                aug_output_dir = os.path.join(aug_output_dir, layout_variant)
            
            os.makedirs(aug_output_dir, exist_ok=True)
            
            # Convert tensor to numpy
            if hasattr(tensor, 'cpu'):
                tensor_np = tensor.cpu().numpy()
            else:
                tensor_np = np.array(tensor)
            
            # Handle different tensor formats
            if len(tensor_np.shape) == 4:  # 4D tensor (B, C, H, W) or (B, H, W, C)
                output_single = tensor_np[0]  # Extract first image
                
                # Check if NCHW or NHWC by examining channel dimension
                if output_single.shape[0] <= 3 and output_single.shape[0] >= 1:  # Likely NCHW (C, H, W)
                    if output_single.shape[0] == 1:  # Grayscale (1, H, W)
                        output_hwc = output_single[0]  # Extract to (H, W)
                    else:  # RGB (3, H, W)
                        output_hwc = np.transpose(output_single, (1, 2, 0))  # (H, W, 3)
                else:  # Likely NHWC (H, W, C)
                    output_hwc = output_single  # Already in HWC format
                    
            elif len(tensor_np.shape) == 3:
                # Check if CHW or HWC
                if tensor_np.shape[0] <= 3 and tensor_np.shape[0] >= 1:  # Likely CHW
                    if tensor_np.shape[0] == 1:  # Grayscale (1, H, W)
                        output_hwc = tensor_np[0]  # Extract to (H, W)
                    else:  # RGB (3, H, W)
                        output_hwc = np.transpose(tensor_np, (1, 2, 0))  # (H, W, 3)
                else:  # Likely HWC already
                    output_hwc = tensor_np
            else:
                output_hwc = tensor_np
            
            # Ensure uint8 type
            if output_hwc.dtype != np.uint8:
                output_hwc = np.clip(output_hwc, 0, 255).astype(np.uint8)
            
            # For resize and crop, the output dimensions may differ from input
            # Don't crop the output - save the full tensor as-is
            if augmentation_name not in ['resize', 'crop']:
                # Get actual dimensions and crop before saving (for non-resize/crop augmentations)
                actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
                if len(output_hwc.shape) == 3:  # RGB
                    output_hwc = output_hwc[:actual_h, :actual_w, :]
                else:  # Grayscale
                    output_hwc = output_hwc[:actual_h, :actual_w]
            
            # Save image
            output_path = os.path.join(aug_output_dir, image_name)
            img = Image.fromarray(output_hwc)
            img.save(output_path)
            return True
            
        except Exception as e:
            print(f"    ✗ Failed to save: {e}")
            return False
    
    def _extract_from_batch_nhwc(self, batch_data, img_idx, extract_h=None, extract_w=None):
        """
        Extract individual image from NHWC batch reference data.
        
        Args:
            batch_data: Reference batch data
            img_idx: Image index in batch
            extract_h: Height to extract (None = use IMAGE_SPECS)
            extract_w: Width to extract (None = use IMAGE_SPECS)
        
        Batch structure (273,600 bytes total):
        - Each image slot: 150×152×3
        """
        # Calculate offsets based on actual storage layout
        slot_height = 150  # Padded height for all slots
        slot_width = 152   # Common width
        slot_size = slot_height * slot_width * 3
        
        # Calculate the offset for the requested image
        offset = img_idx * slot_size
        
        # Extract the image slot
        img_slot = batch_data[offset:offset + slot_size]
        
        # Reshape to HWC
        img_slot_reshaped = img_slot.reshape(slot_height, slot_width, 3)
        
        # Get dimensions to extract
        if extract_h is None or extract_w is None:
            # Use actual image dimensions from IMAGE_SPECS
            actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
        else:
            # Use provided dimensions (for resize/crop cases)
            actual_h, actual_w = extract_h, extract_w
        
        # Extract only the valid region (top-left corner)
        ref_roi = img_slot_reshaped[:actual_h, :actual_w, :]
        
        return ref_roi
    
    def _extract_from_batch_pln1(self, batch_data, img_idx):
        """
        Extract individual grayscale image from PLN1 batch reference data.
        
        PLN1 structure: Grayscale images stored in NCHW format (C=1)
        The PLN1 data starts after RGB data in the same file.
        RGB data size: 150 × 152 × 3 × 3 (batch_size) = 205,200 bytes
        Each PLN1 slot: 150 × 152 × 1 = 22,800 bytes
        """
        slot_height = 150
        slot_width = 152
        rgb_slot_size = slot_height * slot_width * 3  # RGB slot size
        pln1_slot_size = slot_height * slot_width # PLN1 slot size (1 channel)
        
        # PLN1 data starts after all RGB data
        pln1_offset_start = rgb_slot_size * 3  # 3 RGB images
        
        # Calculate the offset for the requested grayscale image
        offset = pln1_offset_start + (img_idx * pln1_slot_size)
        
        # Extract the image slot
        img_slot = batch_data[offset:offset + pln1_slot_size]
        
        # Reshape to HW
        img_slot_reshaped = img_slot.reshape(slot_height, slot_width)
        
        # Get actual dimensions and extract valid region
        actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
        
        # Extract only the valid region (top-left corner)
        ref_roi = img_slot_reshaped[:actual_h, :actual_w]
        
        return ref_roi

    
    def _compare_with_reference(self, output_tensor, ref_data, img_idx, is_grayscale=False):
        """
        Compare output tensor with reference from batch.
        
        Returns: (passed, stats_dict)
        """
        try:
            # Convert output to numpy HWC
            if hasattr(output_tensor, 'cpu'):
                output_np = output_tensor.cpu().numpy()
            else:
                output_np = np.array(output_tensor)
            
            # Handle tensor format conversion
            if is_grayscale:
                # Grayscale: NCHW (1, 1, H, W) -> (H, W)
                if len(output_np.shape) == 4:
                    output_hw = output_np[0, 0, :, :]  # Extract (H, W)
                elif len(output_np.shape) == 3:
                    output_hw = output_np[0, :, :]  # Extract (H, W)
                else:
                    output_hw = output_np
            else:
                # RGB: Convert to HWC
                if len(output_np.shape) == 4:  # NCHW or NHWC
                    output_single = output_np[0]
                    if output_single.shape[0] == 3:  # NCHW (C, H, W)
                        output_hwc = np.transpose(output_single, (1, 2, 0))
                    else:  # NHWC (H, W, C)
                        output_hwc = output_single
                elif len(output_np.shape) == 3 and output_np.shape[0] == 3:  # CHW
                    output_hwc = np.transpose(output_np, (1, 2, 0))
                else:
                    output_hwc = output_np
            
            # Get actual dimensions for this image
            actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
            
            # Extract ROI from output (remove any padding)
            if is_grayscale:
                output_roi = output_hw[:actual_h, :actual_w]
                # Extract reference for grayscale
                ref_roi = self._extract_from_batch_pln1(ref_data, img_idx)
            else:
                output_roi = output_hwc[:actual_h, :actual_w, :]
                # Extract reference from batch
                ref_roi = self._extract_from_batch_nhwc(ref_data, img_idx)
            
            # Verify shapes match
            if output_roi.shape != ref_roi.shape:
                return False, {
                    "error": f"Shape mismatch: output {output_roi.shape} vs ref {ref_roi.shape}"
                }

            # Calculate differences
            diff = output_roi.astype(np.int16) - ref_roi.astype(np.int16)
            abs_diff = np.abs(diff)
            
            # Statistics
            max_diff = int(abs_diff.max())
            mismatched = np.sum(abs_diff > self.config.TOLERANCE)
            total_pixels = output_roi.size
            
            stats = {
                "max_diff": max_diff,
                "mismatched_pixels": int(mismatched),
                "total_pixels": int(total_pixels),
                "match_percentage": 100.0 * (total_pixels - mismatched) / total_pixels
            }
            
            # Pass if all pixels within tolerance
            passed = max_diff <= self.config.TOLERANCE
            
            # Only print detailed debug info for failing cases
            if not passed:
                print(f"\n  ⚠️  QA FAILURE DEBUG:")
                print(f"     Output shape: {output_roi.shape}")
                print(f"     Reference shape: {ref_roi.shape}")
                print(f"     Max pixel difference: {max_diff}")
                print(f"     Output sample (top-left 5x5):\n{output_roi.astype(np.int16)[:5, :5]}")
                print(f"     Reference sample (top-left 5x5):\n{ref_roi.astype(np.int16)[:5, :5]}")
                print(f"     Difference sample (top-left 5x5):\n{abs_diff[:5, :5]}\n")
            
            return passed, stats
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _convert_layout(self, tensor, from_layout, to_layout):
        """Convert tensor between layouts (NCHW <-> NHWC)"""
        from rpp_pybind.amd.rpp.layout_utils import convert_nchw_to_nhwc, convert_nhwc_to_nchw
        
        if from_layout == to_layout:
            return tensor
        
        # NCHW to NHWC conversion
        if from_layout == 'NCHW' and to_layout == 'NHWC':
            return convert_nchw_to_nhwc(tensor)
        # NHWC to NCHW conversion
        elif from_layout == 'NHWC' and to_layout == 'NCHW':
            return convert_nhwc_to_nchw(tensor)
        else:
            return tensor
    
    def _run_augmentation_test(self, aug_name, aug_function, aug_params, ref_file_suffix=""):
        """
        Generic function to run augmentation test across all layout variants.
        
        Args:
            aug_name: Name of augmentation (e.g., 'brightness')
            aug_function: Function to call (e.g., fn.brightness)
            aug_params: Dictionary of augmentation-specific parameters
            ref_file_suffix: Optional suffix for reference file (e.g., '_interpolationTypeBicubic')
        """
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        # Load reference ONCE (before loop)
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR, aug_name,
                f"{aug_name}_u8_Tensor{ref_file_suffix}.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes\n")
        
        # Track overall success
        overall_success_count = 0
        overall_total = 0

        # Loop through all layout variants
        for input_layout, output_layout in self.config.LAYOUT_VARIANTS:
            variant_name = f"{input_layout}-{output_layout}"
            print(f"  Testing variant: {variant_name}")
            
            variant_success = 0
            grayscale = (input_layout.upper() == 'PLN1')

            # Process each test image
            for idx, img_path in enumerate(self.test_images):
                image_name = os.path.basename(img_path)
                if self.mode in ["QA", "ALL"]:
                    overall_total += 1

                try:
                    # Load image - always returns NCHW (PLN3 format)
                    image = util.load_image(img_path, grayscale=grayscale, device=device)

                    # Convert to required input layout
                    # PKD3 = NHWC (channels last), PLN3 = NCHW (channels first)
                    if input_layout == 'PKD3':
                        image = self._convert_layout(image, 'NCHW', 'NHWC')
                        input_layout_str = 'NHWC'
                    else:  # PLN3 or PLN1
                        input_layout_str = 'NCHW'
                    
                    # Set output layout
                    output_layout_str = 'NHWC' if output_layout == 'PKD3' else 'NCHW'

                    # Get ROI
                    actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                    roi_widths = [actual_w]
                    roi_heights = [actual_h]
                    
                    # Call augmentation function
                    output = aug_function(
                        image,
                        roi_widths=roi_widths,
                        roi_heights=roi_heights,
                        input_layout=input_layout_str,
                        output_layout=output_layout_str,
                        backend=self.backend,
                        **aug_params
                    )
                    
                    # UNIT MODE: Save output
                    # For saving, ALWAYS convert to PKD3 (NHWC) format
                    if self.mode in ["UNIT", "ALL"]:
                        # Check output shape to determine if conversion needed
                        if len(output.shape) == 4:
                            if output.shape[1] == 3:  # PLN3 (NCHW) - needs conversion
                                output_for_save = self._convert_layout(output, 'NCHW', 'NHWC')
                            else:  # Already NHWC (PKD3)
                                output_for_save = output
                        else:
                            output_for_save = output
                        
                        if self._save_output_image(output_for_save, aug_name, image_name, idx, layout_variant=variant_name):
                            print(f"    ✓ {image_name} ({variant_name}): SAVED")
                            if self.mode == "UNIT":
                                variant_success += 1
                        else:
                            print(f"    ✗ {image_name} ({variant_name}): SAVE FAILED")
                    
                    # QA MODE: Compare with reference (all variants including grayscale)
                    # Reference is ALWAYS in PKD3 (NHWC) format for RGB, PLN1 (NCHW) for grayscale
                    # Convert ANY output to PKD3/PLN1 before comparison
                    if self.mode in ["QA", "ALL"] and ref_data is not None:
                        if grayscale:
                            # Grayscale: output is already in NCHW format
                            output_for_qa = output
                        else:
                            # RGB: Determine current output layout from tensor shape
                            # If 2nd dim is 3 -> PLN3 (NCHW), if last dim is 3 -> PKD3 (NHWC)
                            if len(output.shape) == 4:
                                if output.shape[1] == 3:  # PLN3 (NCHW)
                                    # Convert PLN3 to PKD3 for comparison
                                    output_for_qa = self._convert_layout(output, 'NCHW', 'NHWC')
                                elif output.shape[3] == 3:  # PKD3 (NHWC)
                                    # Already in PKD3 format
                                    output_for_qa = output
                                else:
                                    output_for_qa = output
                            else:
                                output_for_qa = output

                        passed, stats = self._compare_with_reference(output_for_qa, ref_data, idx, is_grayscale=grayscale)
                        
                        if passed:
                            print(f"    ✓ {variant_name}/{image_name}: QA PASS")
                            variant_success += 1
                            overall_success_count += 1
                        else:
                            # Check if error occurred or just pixel mismatch
                            if "error" in stats:
                                print(f"    ✗ {variant_name}/{image_name}: QA FAIL - {stats['error']}")
                            else:
                                print(f"    ✗ {variant_name}/{image_name}: QA FAIL (diff={stats['max_diff']})")
                                print(f"      (tolerance: {self.config.TOLERANCE})")
                                print(f"      Mismatched: {stats['mismatched_pixels']}/{stats['total_pixels']} ({100-stats['match_percentage']:.2f}%)")
                    
                except Exception as e:
                    print(f"    ✗ {image_name} ({variant_name}): ERROR → {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"  {variant_name}: {variant_success}/{len(self.test_images)}\n")
        
        # Report results
        if self.mode == "QA":
            success = overall_success_count == overall_total
            status = f"PASSED {overall_success_count}/{overall_total} comparisons"
            print(f"\n  OVERALL: {overall_success_count}/{overall_total} {'✓' if success else '✗'}")
            self.results['qa'].append((aug_name, success))
        elif self.mode == "UNIT":
            # For UNIT mode, count all saved images including PLN1
            success = True  # If we got here without exceptions
            self.results['unit'].append((aug_name, success))
        else:  # ALL
            success = overall_success_count == overall_total
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success

    # =========================================================================
    # AUGMENTATION FUNCTIONS (Combined Unit + QA)
    # =========================================================================
    
    def test_brightness(self):
        """Brightness augmentation test - All layout variants"""
        params = self.config.AUGMENTATION_PARAMS['brightness']
        print(f"  [brightness] (alpha={params['alpha']}, beta={params['beta']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('brightness', fn.brightness, params)
    
    def test_gamma_correction(self):
        """Gamma correction augmentation test - All layout variants"""
        params = self.config.AUGMENTATION_PARAMS['gamma_correction']
        print(f"  [gamma_correction] (gamma={params['gamma']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('gamma_correction', fn.gamma_correction, params)
    
    def test_flip(self):
        """Flip augmentation test - All layout variants"""
        params = self.config.AUGMENTATION_PARAMS['flip']
        print(f"  [flip] (horizontal={params['horizontal']}, vertical={params['vertical']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('flip', fn.flip, params)
    
    def test_resize(self):
        """Resize augmentation test - Per-image dimensions (width/2, height/2)"""
        print(f"  [resize] (per-image: width/2, height/2)")
        print("  " + "-" * 50)
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR, 'resize',
                f"resize_u8_Tensor_interpolationTypeBicubic.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes\n")
        
        overall_success_count = 0
        overall_total = 0

        for input_layout, output_layout in self.config.LAYOUT_VARIANTS:
            variant_name = f"{input_layout}-{output_layout}"
            print(f"  Testing variant: {variant_name}")
            variant_success = 0
            grayscale = (input_layout.upper() == 'PLN1')

            for idx, img_path in enumerate(self.test_images):
                image_name = os.path.basename(img_path)
                if self.mode in ["QA", "ALL"] and not grayscale:
                    overall_total += 1

                try:
                    image = util.load_image(img_path, grayscale=grayscale, device=device)
                    if input_layout == 'PKD3':
                        image = self._convert_layout(image, 'NCHW', 'NHWC')
                        input_layout_str = 'NHWC'
                    else:  # PLN3 or PLN1
                        input_layout_str = 'NCHW'
                    
                    # Set output layout
                    output_layout_str = 'NHWC' if output_layout == 'PKD3' else 'NCHW'
                    
                    # Calculate resize dimensions per image (half of original)
                    actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                    resize_width = actual_w // 2
                    resize_height = actual_h // 2
                    
                    output = fn.resize(
                        image,
                        width=resize_width,
                        height=resize_height,
                        roi_widths=[actual_w],
                        roi_heights=[actual_h],
                        input_layout=input_layout_str,
                        output_layout=output_layout_str,
                        backend=self.backend
                    )
                    
                    if self.mode in ["UNIT", "ALL"]:
                        # Debug: print output shape before conversion
                        print(f"      DEBUG: output shape before conversion: {output.shape}")
                        if output_layout == 'PLN3':
                            output_for_save = self._convert_layout(output, 'NCHW', 'NHWC')
                            print(f"      DEBUG: output shape after conversion: {output_for_save.shape}")
                        else:
                            output_for_save = output
                        
                        if self._save_output_image(output_for_save, 'resize', image_name, idx, layout_variant=variant_name):
                            print(f"    ✓ {image_name} ({variant_name}): SAVED (resized to {resize_width}x{resize_height})")
                            if self.mode == "UNIT":
                                variant_success += 1
                    
                    if self.mode in ["QA", "ALL"] and ref_data is not None and not grayscale:
                        if output_layout == 'PLN3':
                            output_for_qa = self._convert_layout(output, 'NCHW', 'NHWC')
                        else:
                            output_for_qa = output

                        passed, stats = self._compare_with_reference(output_for_qa, ref_data, idx, is_grayscale=False)
                        
                        if passed:
                            print(f"    ✓ {variant_name}/{image_name}: QA PASS")
                            variant_success += 1
                            overall_success_count += 1
                        else:
                            if "error" in stats:
                                print(f"    ✗ {variant_name}/{image_name}: QA FAIL - {stats['error']}")
                            else:
                                print(f"    ✗ {variant_name}/{image_name}: QA FAIL (diff={stats['max_diff']})")
                    
                except Exception as e:
                    print(f"    ✗ {image_name} ({variant_name}): ERROR → {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"  {variant_name}: {variant_success}/{len(self.test_images)}\n")
        
        if self.mode == "QA":
            success = overall_success_count == overall_total
            self.results['qa'].append(('resize', success))
        elif self.mode == "UNIT":
            self.results['unit'].append(('resize', True))
        else:
            self.results['unit'].append(('resize', True))
            self.results['qa'].append(('resize', overall_success_count == overall_total))
        
        return True
    
    def test_crop(self):
        """Crop augmentation test - Per-image dimensions (width/2, height/2)"""
        print(f"  [crop] (x1=10, y1=10, per-image: width/2, height/2)")
        print("  " + "-" * 50)
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR, 'crop',
                f"crop_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes\n")
        
        overall_success_count = 0
        overall_total = 0

        for input_layout, output_layout in self.config.LAYOUT_VARIANTS:
            variant_name = f"{input_layout}-{output_layout}"
            print(f"  Testing variant: {variant_name}")
            variant_success = 0
            grayscale = (input_layout.upper() == 'PLN1')

            for idx, img_path in enumerate(self.test_images):
                image_name = os.path.basename(img_path)
                if self.mode in ["QA", "ALL"] and not grayscale:
                    overall_total += 1

                try:
                    image = util.load_image(img_path, grayscale=grayscale, device=device)
                    if input_layout == 'PKD3':
                        image = self._convert_layout(image, 'NCHW', 'NHWC')
                        input_layout_str = 'NHWC'
                    else:  # PLN3 or PLN1
                        input_layout_str = 'NCHW'
                    
                    # Set output layout
                    output_layout_str = 'NHWC' if output_layout == 'PKD3' else 'NCHW'
                    
                    # Calculate crop dimensions per image (half of original)
                    actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                    crop_width = actual_w // 2
                    crop_height = actual_h // 2
                    
                    output = fn.crop(
                        image,
                        x1=10,
                        y1=10,
                        crop_width=crop_width,
                        crop_height=crop_height,
                        input_layout=input_layout_str,
                        output_layout=output_layout_str,
                        backend=self.backend
                    )
                    
                    if self.mode in ["UNIT", "ALL"]:
                        if output_layout == 'PLN3':
                            output_for_save = self._convert_layout(output, 'NCHW', 'NHWC')
                        else:
                            output_for_save = output
                        
                        if self._save_output_image(output_for_save, 'crop', image_name, idx, layout_variant=variant_name):
                            print(f"    ✓ {image_name} ({variant_name}): SAVED (cropped to {crop_width}x{crop_height})")
                            if self.mode == "UNIT":
                                variant_success += 1
                    
                    if self.mode in ["QA", "ALL"] and ref_data is not None and not grayscale:
                        if output_layout == 'PLN3':
                            output_for_qa = self._convert_layout(output, 'NCHW', 'NHWC')
                        else:
                            output_for_qa = output

                        passed, stats = self._compare_with_reference(output_for_qa, ref_data, idx, is_grayscale=False)
                        
                        if passed:
                            print(f"    ✓ {variant_name}/{image_name}: QA PASS")
                            variant_success += 1
                            overall_success_count += 1
                        else:
                            if "error" in stats:
                                print(f"    ✗ {variant_name}/{image_name}: QA FAIL - {stats['error']}")
                            else:
                                print(f"    ✗ {variant_name}/{image_name}: QA FAIL (diff={stats['max_diff']})")
                    
                except Exception as e:
                    print(f"    ✗ {image_name} ({variant_name}): ERROR → {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"  {variant_name}: {variant_success}/{len(self.test_images)}\n")
        
        if self.mode == "QA":
            success = overall_success_count == overall_total
            self.results['qa'].append(('crop', success))
        elif self.mode == "UNIT":
            self.results['unit'].append(('crop', True))
        else:
            self.results['unit'].append(('crop', True))
            self.results['qa'].append(('crop', overall_success_count == overall_total))
        
        return True
    
    def test_hue(self):
        """Hue augmentation test - All layout variants"""
        params = self.config.AUGMENTATION_PARAMS['hue']
        print(f"  [hue] (hue_shift={params['hue_shift']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('hue', fn.hue, params)
    
    def test_rotate(self):
        """Rotate augmentation test - All layout variants"""
        params = self.config.AUGMENTATION_PARAMS['rotate']
        print(f"  [rotate] (angle={params['angle']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('rotate', fn.rotate, params, ref_file_suffix='_interpolationTypeBilinear')
    
    def test_contrast(self):
        """Contrast augmentation test - All layout variants"""
        params = self.config.AUGMENTATION_PARAMS['contrast']
        print(f"  [contrast] (factor={params['contrast_factor']}, center={params['contrast_center']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('contrast', fn.contrast, params)
    
    def test_vignette(self):
        """Vignette augmentation test - All layout variants"""
        params = self.config.AUGMENTATION_PARAMS['vignette']
        print(f"  [vignette] (intensity={params['intensity']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('vignette', fn.vignette, params)
    
    def test_contrast(self):
        """Contrast augmentation test - All layout variants"""
        params = self.config.AUGMENTATION_PARAMS['contrast']
        print(f"  [contrast] (factor={params['contrast_factor']}, center={params['contrast_center']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('contrast', fn.contrast, params)
    
    def test_pixelate(self):
        """Pixelate augmentation test - All layout variants"""
        params = self.config.AUGMENTATION_PARAMS['pixelate']
        print(f"  [pixelate] (percentage={params['pixelation_percentage']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('pixelate', fn.pixelate, params)
    
    # =========================================================================
    # TEST RUNNERS
    # =========================================================================
    
    def _get_filtered_tests(self):
        """Get test functions based on case_list filter"""
        if self.case_list is None:
            # Run all tests in order
            return [
                (0, 'brightness', self.test_brightness),
                (1, 'gamma_correction', self.test_gamma_correction),
                (20, 'flip', self.test_flip),
                (21, 'resize', self.test_resize),
                (37, 'crop', self.test_crop),
                (42, 'hue', self.test_hue),
                (23, 'rotate', self.test_rotate),
                (4, 'contrast', self.test_contrast),
                (46, 'vignette', self.test_vignette),
                (5, 'pixelate', self.test_pixelate)
            ]
        else:
            # Filter tests based on selected cases
            filtered = []
            for case_num in self.case_list:
                if case_num in self.case_to_test_map:
                    name, test_func = self.case_to_test_map[case_num]
                    filtered.append((case_num, name, test_func))
            return filtered
    
    def run_unit_tests(self):
        """Run augmentations in Unit mode"""
        print(f"\n{'='*70}")
        print(f"UNIT TESTS - Image Generation ({self.backend_name})")
        print(f"{'='*70}\n")
        
        tests = self._get_filtered_tests()
        
        for case_num, name, test_func in tests:
            try:
                test_func()
                print("-" * 70)
            except Exception as e:
                print(f"ERROR in {test_func.__name__}: {e}")
                print("-" * 70)
    
    def run_qa_tests(self):
        """Run augmentations in QA mode"""
        print(f"\n{'='*70}")
        print(f"QA TESTS - Reference Comparison ({self.backend_name})")
        print(f"{'='*70}")
        print(f"Tolerance: ±{self.config.TOLERANCE} pixel values\n")
        
        tests = self._get_filtered_tests()
        
        for case_num, name, test_func in tests:
            try:
                test_func()
                print("-" * 70)
            except Exception as e:
                print(f"ERROR in {test_func.__name__}: {e}")
                print("-" * 70)
    
    def run_performance_tests(self):
        """Run performance tests - timing measurements for augmentations"""
        print(f"\n{'='*70}")
        print(f"PERFORMANCE TESTS ({self.backend_name})")
        print(f"{'='*70}\n")
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        num_iterations = 100
        warmup_iterations = 5
        
        # Test image for performance
        test_image = util.load_image(self.test_images[1], device=device)  # 100x100
        
        # Get parameters from config
        params = self.config.AUGMENTATION_PARAMS
        
        # Map case numbers to performance test functions
        perf_test_map = {
            0: ('brightness', lambda: fn.brightness(test_image, alpha=params['brightness']['alpha'], 
                                                beta=params['brightness']['beta'], backend=self.backend)),
            1: ('gamma_correction', lambda: fn.gamma_correction(test_image, gamma=params['gamma_correction']['gamma'], 
                                                            backend=self.backend)),
            20: ('flip', lambda: fn.flip(test_image, horizontal=params['flip']['horizontal'], 
                                   vertical=params['flip']['vertical'], backend=self.backend)),
            21: ('resize', lambda: fn.resize(test_image, width=params['resize']['width'], 
                                        height=params['resize']['height'], backend=self.backend)),
            37: ('crop', lambda: fn.crop(test_image, x1=params['crop']['x1'], y1=params['crop']['y1'], 
                                    crop_width=params['crop']['crop_width'], 
                                    crop_height=params['crop']['crop_height'], backend=self.backend)),
            42: ('hue', lambda: fn.hue(test_image, hue_shift=params['hue']['hue_shift'], backend=self.backend)),
            23: ('rotate', lambda: fn.rotate(test_image, angle=params['rotate']['angle'], backend=self.backend)),
            4: ('contrast', lambda: fn.contrast(test_image, contrast_factor=params['contrast']['contrast_factor'], 
                                            contrast_center=params['contrast']['contrast_center'], backend=self.backend)),
            46: ('vignette', lambda: fn.vignette(test_image, intensity=params['vignette']['intensity'], backend=self.backend)),
            5: ('pixelate', lambda: fn.pixelate(test_image, pixelation_percentage=params['pixelate']['pixelation_percentage'], 
                                            backend=self.backend))
        }
        
        # Filter based on selected cases
        if self.case_list is None:
            # Run all tests
            perf_tests = [(k, v[0], v[1]) for k, v in perf_test_map.items()]
        else:
            # Run only selected tests
            perf_tests = [(k, perf_test_map[k][0], perf_test_map[k][1]) 
                         for k in self.case_list if k in perf_test_map]
        
        for i, (case_num, func_name, func_call) in enumerate(perf_tests, 1):
            print(f"  [Case {case_num}] Testing {func_name}...", end=" ")
            
            # Check if function is available
            if not hasattr(fn, func_name):
                print("SKIP (function not available)")
                self.results['perf'].append((func_name, None))
                continue
            
            try:
                # Warmup
                for _ in range(warmup_iterations):
                    _ = func_call()
                    if self.backend == HIP:
                        torch.cuda.synchronize()
                
                # Timing
                times = []
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = func_call()
                    if self.backend == HIP:
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # Convert to ms
                
                avg_time = np.mean(times)
                min_time = np.min(times)
                max_time = np.max(times)
                
                print(f"Avg: {avg_time:.2f}ms, Min: {min_time:.2f}ms, Max: {max_time:.2f}ms")
                
                self.results['perf'].append((func_name, {
                    'avg': avg_time,
                    'min': min_time,
                    'max': max_time
                }))
                
            except Exception as e:
                print(f"ERROR: {e}")
                self.results['perf'].append((func_name, None))
    
    def run_all(self):
        """Run tests based on mode"""
        if self.mode == "UNIT":
            self.run_unit_tests()
        elif self.mode == "QA":
            self.run_qa_tests()
        elif self.mode == "PERF":
            self.run_performance_tests()
        elif self.mode == "ALL":
            self.run_unit_tests()
            self.run_qa_tests()
            self.run_performance_tests()
        else:
            print(f"ERROR: Invalid mode '{self.mode}'")
            return False
        
        # Print summary
        self._print_summary()
        return True
    
    def _print_summary(self):
        """Print test summary"""
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY ({self.mode} mode)")
        print(f"{'='*70}")
        
        if self.mode in ["UNIT", "ALL"] and self.results['unit']:
            passed = sum(1 for _, r in self.results['unit'] if r is True)
            failed = sum(1 for _, r in self.results['unit'] if r is False)
            skipped = sum(1 for _, r in self.results['unit'] if r is None)
            print(f"\nUNIT TESTS:")
            print(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
        
        if self.mode in ["QA", "ALL"] and self.results['qa']:
            passed = sum(1 for _, r in self.results['qa'] if r is True)
            failed = sum(1 for _, r in self.results['qa'] if r is False)
            skipped = sum(1 for _, r in self.results['qa'] if r is None)
            print(f"\nQA TESTS:")
            print(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
        
        if self.mode in ["PERF", "ALL"] and self.results['perf']:
            valid = sum(1 for _, r in self.results['perf'] if r is not None)
            print(f"\nPERFORMANCE TESTS:")
            print(f"  Completed: {valid}/{len(self.results['perf'])}")
            
            if valid > 0:
                # Show average timing across all functions
                all_avg_times = [r['avg'] for _, r in self.results['perf'] if r is not None]
                if all_avg_times:
                    print(f"  Overall average: {np.mean(all_avg_times):.2f}ms")
        
        print(f"{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RPP Test Suite - Complete Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unit testing only
  python test_suite.py --mode UNIT --backend HOST
  
  # QA testing only
  python test_suite.py --mode QA --backend HOST
  
  # Performance testing only
  python test_suite.py --mode PERF --backend HIP
  
  # Run all tests
  python test_suite.py --mode ALL --backend HOST
  
  # Run specific cases by number
  python test_suite.py --mode QA --backend HOST --cases 0,1,2
  
  # Run specific cases by name
  python test_suite.py --mode UNIT --backend HOST --cases brightness,gamma_correction,flip
  
  # Mix numbers and names
  python test_suite.py --mode ALL --backend HIP --cases 0,brightness,2,flip
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['UNIT', 'QA', 'PERF', 'ALL'],
                       default='ALL', 
                       help='Test mode to run')
    parser.add_argument('--backend', 
                       choices=['HOST', 'HIP'],
                       required=True, 
                       help='Backend to test')
    parser.add_argument('--case_list',
                       type=str,
                       default=None,
                       nargs='+',
                       help='Comma-separated list of test cases to run (numbers 0-9 or names). If not specified, all cases run.')
    parser.add_argument('--preserve_output',
                       type=int,
                       choices=[0, 1],
                       default=0,
                       help='Preserve previous output folders. 0=delete previous outputs (default), 1=preserve all outputs')
    
    args = parser.parse_args()
    
    # Determine backend
    backend = HIP if args.backend == 'HIP' else HOST
    backend_name = args.backend
    
    # Check GPU availability for HIP backend
    if backend == HIP and not is_gpu_available():
        print(f"ERROR: HIP backend requested but GPU not available")
        return 1
    
    # Parse cases argument
    case_list = None
    if args.case_list:
        # Case name to index mapping
        case_map = {
            'brightness': 0,
            'gamma_correction': 1,
            'contrast': 4,
            'pixelate': 5,
            'flip': 20,
            'resize': 21,
            'rotate': 23,
            'crop': 37,
            'hue': 42,
            'vignette': 46
        }
        
        # Supported case list based on case_map values
        supportedCaseList = list(case_map.values())
        
        case_list = set()
        case_items = args.case_list
        
        for item in case_items:
            item = item.strip()
            # Try to parse as number
            if item.isdigit():
                case_num = int(item)
                if case_num in supportedCaseList:
                    case_list.add(case_num)
                else:
                    print(f"WARNING: Case number {case_num} not supported (valid: {sorted(supportedCaseList)}), ignoring")
            # Try to parse as name
            elif item in case_map:
                case_list.add(case_map[item])
            else:
                print(f"WARNING: Unknown case '{item}', ignoring")
        
        if not case_list:
            print("ERROR: No valid cases specified")
            return 1
        
        case_list = sorted(list(case_list))
        
        # Create reverse mapping for display (case number -> name)
        reverse_case_map = {v: k for k, v in case_map.items()}
    
    # Print header
    print("\n" + "="*70)
    print("RPP TEST SUITE - COMPLETE VERSION")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Backend: {backend_name}")
    print(f"GPU Available: {is_gpu_available()}")
    if case_list:
        # Use reverse_case_map to get proper case name display
        reverse_case_map = {v: k for k, v in case_map.items()}
        selected_names = [f"{i}:{reverse_case_map[i]}" for i in case_list]
        print(f"Selected Cases: {', '.join(selected_names)}")
    else:
        print(f"Selected Cases: All")
    print("="*70)
    
    # Run tests
    try:
        test_suite = UnifiedTestSuite(backend, args.mode, case_list, args.preserve_output)
        success = test_suite.run_all()
        return 0 if success else 1
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
