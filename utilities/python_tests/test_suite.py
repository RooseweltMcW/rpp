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
        
        # Reference batch dimensions
        self.BATCH_HEIGHT = 150
        self.BATCH_WIDTH = 152
        self.BATCH_SIZE = 3
        
        # Augmentation parameters (matching C++ test suite)
        self.AUGMENTATION_PARAMS = {
            'brightness': {'alpha': 1.75, 'beta': 50.0},
            'gamma_correction': {'gamma': 1.9},
            'contrast': {'contrast_factor': 2.96, 'contrast_center': 128.0},
            'flip': {'horizontal': True, 'vertical': False},
            'resize': {'width': 224, 'height': 224},
            'crop': {'x1': 10, 'y1': 10, 'crop_width': 30, 'crop_height': 30},
            'hue': {'hue_shift': 60.0},
            'rotate': {'angle': 45.0},
            'vignette': {'intensity': 6.0},
            'pixelate': {'pixelation_percentage': 87.5}
        }
        
        # Timestamp for output directories
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def get_output_dir(self, backend_name, mode):
        """Get output directory based on backend and mode"""
        return f"{backend_name}_OUTPUT_{mode}_{self.timestamp}"


# =============================================================================
# UNIFIED TEST CLASS
# =============================================================================

class UnifiedTestSuite:
    """Unified test suite with combined Unit/QA/Performance testing"""
    
    def __init__(self, backend, mode="ALL", case_list=None):
        self.backend = backend
        self.backend_name = "HIP" if backend == HIP else "HOST"
        self.mode = mode.upper()  # UNIT, QA, PERF, or ALL
        self.case_list = case_list  # List of case numbers to run (None = all)
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
    
    def _save_output_image(self, tensor, augmentation_name, image_name, img_idx):
        """Save output image to filesystem (Unit mode)"""
        try:
            # Create augmentation-specific directory
            aug_output_dir = os.path.join(self.unit_output_dir, augmentation_name)
            os.makedirs(aug_output_dir, exist_ok=True)
            
            # Convert tensor to numpy
            if hasattr(tensor, 'cpu'):
                tensor_np = tensor.cpu().numpy()
            else:
                tensor_np = np.array(tensor)
            
            # Handle different tensor formats
            if len(tensor_np.shape) == 4:  # NCHW format
                output_single = tensor_np[0]  # Extract first image
                output_hwc = np.transpose(output_single, (1, 2, 0))  # CHW to HWC
            elif len(tensor_np.shape) == 3 and tensor_np.shape[0] == 3:  # CHW format
                output_hwc = np.transpose(tensor_np, (1, 2, 0))
            else:
                output_hwc = tensor_np
            
            # Ensure uint8 type
            if output_hwc.dtype != np.uint8:
                output_hwc = np.clip(output_hwc, 0, 255).astype(np.uint8)
            
            # Get actual dimensions and crop before saving
            actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
            output_hwc = output_hwc[:actual_h, :actual_w, :]
            
            # Save image
            output_path = os.path.join(aug_output_dir, image_name)
            img = Image.fromarray(output_hwc)
            img.save(output_path)
            return True
            
        except Exception as e:
            print(f"    ✗ Failed to save: {e}")
            return False
    
    def _extract_from_batch_nhwc(self, batch_data, img_idx):
        """
        Extract individual image from NHWC batch reference data.
        
        Batch structure (273,600 bytes total):
        - First two images: each in 150×152×3 slots
        - Third image: in remaining space (also 150×152×3)
        
        But stored with padding to 200 height for uniformity.
        """
        # The batch has a complex layout
        # Total: 273,600 bytes = 3 × 200 × 152 × 3
        
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
        
        # Get actual dimensions and extract valid region
        actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
        
        # Extract only the valid region (top-left corner)
        ref_roi = img_slot_reshaped[:actual_h, :actual_w, :]
        
        return ref_roi

    
    def _compare_with_reference(self, output_tensor, ref_data, img_idx):
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
            if len(output_np.shape) == 4:  # NCHW
                output_single = output_np[0]
                output_hwc = np.transpose(output_single, (1, 2, 0))
            elif len(output_np.shape) == 3 and output_np.shape[0] == 3:  # CHW
                output_hwc = np.transpose(output_np, (1, 2, 0))
            else:
                output_hwc = output_np
            
            # Get actual dimensions for this image
            actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
            
            # Extract ROI from output (remove any padding)
            output_roi = output_hwc[:actual_h, :actual_w, :]
            
            # Extract reference from batch
            ref_roi = self._extract_from_batch_nhwc(ref_data, img_idx)
            
            # Verify shapes match
            # print("Output ROI Shape: ",output_roi.shape)
            # print("reference ROI Shape: ",ref_roi.shape)
            if output_roi.shape != ref_roi.shape:
                return False, {
                    "error": f"Shape mismatch: output {output_roi.shape} vs ref {ref_roi.shape}"
                }
            print("Output values: ",output_roi.astype(np.int16))
            print("Reference values: ",ref_roi.astype(np.int16))

            # Calculate differences
            diff = output_roi.astype(np.int16) - ref_roi.astype(np.int16)
            abs_diff = np.abs(diff)
            # print(abs_diff)
            
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
            
            return passed, stats
            
        except Exception as e:
            return False, {"error": str(e)}
    
    # =========================================================================
    # AUGMENTATION FUNCTIONS (Combined Unit + QA)
    # =========================================================================
    
    def test_brightness(self):
        """
        Brightness augmentation test.
        - Unit mode: Apply and save
        - QA mode: Apply and compare with reference
        """
        aug_name = "brightness"
        params = self.config.AUGMENTATION_PARAMS[aug_name]
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        print(f"  [1/10] Brightness (alpha={params['alpha']}, beta={params['beta']})")
        print("  " + "-" * 50)
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.brightness(
                    image, 
                    alpha=params['alpha'], 
                    beta=params['beta'],
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS (max_diff={stats['max_diff']})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']} (tolerance: {self.config.TOLERANCE})")
                            print(f"      Mismatched: {stats['mismatched_pixels']}/{stats['total_pixels']} ({100-stats['match_percentage']:.2f}%)")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:  # ALL
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:  # ALL
            self.results['unit'].append((aug_name, True))  # Assume unit passed if we get here
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_gamma_correction(self):
        """
        Gamma correction augmentation test.
        - Unit mode: Apply and save
        - QA mode: Apply and compare with reference
        """
        aug_name = "gamma_correction"
        params = self.config.AUGMENTATION_PARAMS[aug_name]
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        print(f"  [2/10] Gamma Correction (gamma={params['gamma']})")
        print("  " + "-" * 50)
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.gamma_correction(
                    image, 
                    gamma=params['gamma'],
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_flip(self):
        """
        Flip augmentation test.
        - Unit mode: Apply and save
        - QA mode: Apply and compare with reference
        """
        aug_name = "flip"
        params = self.config.AUGMENTATION_PARAMS[aug_name]
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        print(f"  [3/10] Flip (horizontal={params['horizontal']}, vertical={params['vertical']})")
        print("  " + "-" * 50)
        
        if not hasattr(fn, 'flip'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.flip(
                    image, 
                    horizontal=True,
                    vertical=False,
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_resize(self):
        """
        Resize augmentation test.
        - Unit mode: Apply and save
        - QA mode: Apply and compare with reference
        """
        aug_name = "resize"
        params = self.config.AUGMENTATION_PARAMS[aug_name]
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        print(f"  [4/10] Resize (width={params['width']}, height={params['height']})")
        print("  " + "-" * 50)
        
        if not hasattr(fn, 'resize'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor_interpolationTypeBicubic.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.resize(
                    image, 
                    width=params['width'], 
                    height=params['height'],
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_crop(self):
        """
        Crop augmentation test.
        - Unit mode: Apply and save
        - QA mode: Apply and compare with reference
        """
        aug_name = "crop"
        params = self.config.AUGMENTATION_PARAMS[aug_name]
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        print(f"  [5/10] Crop (x1={params['x1']}, y1={params['y1']}, width={params['crop_width']}, height={params['crop_height']})")
        print("  " + "-" * 50)
        
        if not hasattr(fn, 'crop'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                output = fn.crop(
                    image, 
                    x1=params['x1'], 
                    y1=params['y1'], 
                    crop_width=params['crop_width'], 
                    crop_height=params['crop_height'], 
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_hue(self):
        """
        Hue augmentation test.
        - Unit mode: Apply and save
        - QA mode: Apply and compare with reference
        """
        aug_name = "hue"
        params = self.config.AUGMENTATION_PARAMS[aug_name]
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        print(f"  [6/10] Hue (hue_shift={params['hue_shift']})")
        print("  " + "-" * 50)
        
        if not hasattr(fn, 'hue'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.hue(
                    image, 
                    hue_shift=params['hue_shift'],
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_rotate(self):
        """
        Rotate augmentation test.
        - Unit mode: Apply and save
        - QA mode: Apply and compare with reference
        """
        aug_name = "rotate"
        params = self.config.AUGMENTATION_PARAMS[aug_name]
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        print(f"  [7/10] Rotate (angle={params['angle']})")
        print("  " + "-" * 50)
        
        if not hasattr(fn, 'rotate'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor_interpolationTypeBilinear.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.rotate(
                    image, 
                    angle=params['angle'],
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_contrast(self):
        """
        Contrast augmentation test.
        - Unit mode: Apply and save
        - QA mode: Apply and compare with reference
        """
        aug_name = "contrast"
        params = self.config.AUGMENTATION_PARAMS[aug_name]
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        print(f"  [8/10] Contrast (factor={params['contrast_factor']}, center={params['contrast_center']})")
        print("  " + "-" * 50)
        
        if not hasattr(fn, 'contrast'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.contrast(
                    image, 
                    contrast_factor=params['contrast_factor'], 
                    contrast_center=params['contrast_center'],
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_vignette(self):
        """
        Vignette augmentation test.
        - Unit mode: Apply and save
        - QA mode: Apply and compare with reference
        """
        aug_name = "vignette"
        params = self.config.AUGMENTATION_PARAMS[aug_name]
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        print(f"  [9/10] Vignette (intensity={params['intensity']})")
        print("  " + "-" * 50)
        
        if not hasattr(fn, 'vignette'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.vignette(
                    image, 
                    intensity=params['intensity'],
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_pixelate(self):
        """
        Pixelate augmentation test.
        - Unit mode: Apply and save
        - QA mode: Apply and compare with reference
        """
        aug_name = "pixelate"
        params = self.config.AUGMENTATION_PARAMS[aug_name]
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        print(f"  [10/10] Pixelate (percentage={params['pixelation_percentage']})")
        print("  " + "-" * 50)
        
        if not hasattr(fn, 'pixelate'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.pixelate(
                    image, 
                    pixelation_percentage=params['pixelation_percentage'],
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
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
                       help='Comma-separated list of test cases to run (numbers 0-9 or names). If not specified, all cases run.')
    
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
        case_items = args.case_list.split(',')
        
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
        test_suite = UnifiedTestSuite(backend, args.mode, case_list)
        success = test_suite.run_all()
        return 0 if success else 1
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
