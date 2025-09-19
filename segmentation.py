import os
import cv2
import numpy as np
import torch
import time
from pathlib import Path
from functools import lru_cache

class Segmenter:
    def __init__(self, output_dir="data/output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Segmenter using device: {self.device}")

        # Cache for models
        self._models = {}
        
        # Cache for results
        self._result_cache = {}

    @lru_cache(maxsize=2)
    def _load_sam_model(self, model_type="vit_h"):
        """Load and cache SAM model"""
        if model_type in self._models:
            return self._models[model_type]
            
        print(f"Loading SAM model: {model_type}")
        start_time = time.time()
        
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError("Please install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")
        
        # Select the appropriate checkpoint based on model_type
        if model_type == "vit_h":
            checkpoint = "sam_vit_h_4b8939.pth"
        elif model_type == "vit_l":
            checkpoint = "sam_vit_l_0b3195.pth"
        elif model_type == "vit_b":
            checkpoint = "sam_vit_b_01ec64.pth"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}. Please download from https://github.com/facebookresearch/segment-anything#model-checkpoints")
        
        # Load the model
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(self.device)
        
        # Create mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100
        )
        
        loading_time = time.time() - start_time
        print(f"SAM model loaded in {loading_time:.2f} seconds")
        
        self._models[model_type] = mask_generator
        return mask_generator

    def segment_image(self, img_path, use_cache=True, model_type="vit_b"):
        """
        Segment an image using SAM (Segment Anything Model)
        
        Args:
            img_path: Path to the input image
            use_cache: Whether to use cached results if available
            model_type: SAM model type ('vit_h', 'vit_l', or 'vit_b')
            
        Returns:
            Dict with paths to output files and segmentation masks
        """
        img_name = Path(img_path).stem
        cache_key = f"{img_path}_{model_type}"
        
        # Check cache
        if use_cache:
            # Check memory cache
            if cache_key in self._result_cache:
                print(f"Using cached segmentation for {img_name}")
                return self._result_cache[cache_key]
                
            # Check file cache
            masks_path = os.path.join(self.output_dir, f"{img_name}_masks.jpg")
            segmented_path = os.path.join(self.output_dir, f"{img_name}_segmented.jpg")
            if os.path.exists(masks_path) and os.path.exists(segmented_path):
                print(f"Loading cached segmentation files for {img_name}")
                segmented_img = cv2.imread(segmented_path)
                result = {
                    "masks": masks_path,
                    "segmented": segmented_path,
                    "segmented_img": segmented_img
                }
                self._result_cache[cache_key] = result
                return result
        
        # Load the image
        print(f"Segmenting image: {img_path}")
        start_time = time.time()
        
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load SAM model based on hardware capabilities if not specified
        if model_type == "auto":
            model_type = self.optimize_model_selection()
            print(f"Automatically selected model: {model_type}")
        
        # Get the mask generator
        mask_generator = self._load_sam_model(model_type)
        
        # Generate masks
        print("Generating masks...")
        masks = mask_generator.generate(img_rgb)
        print(f"Generated {len(masks)} masks")
        
        # Sort masks by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Visualize segmentation
        def show_anns(anns):
            if len(anns) == 0:
                return np.zeros_like(img)[:, :, 0]
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
            mask_all = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]), dtype=np.uint8)
            img_overlay = img.copy()
            
            # Use a color palette for mask visualization
            colors = [
                [255, 0, 0], [0, 255, 0], [0, 0, 255],
                [255, 255, 0], [255, 0, 255], [0, 255, 255],
                [128, 0, 0], [0, 128, 0], [0, 0, 128],
                [128, 128, 0], [128, 0, 128], [0, 128, 128]
            ]
            
            for i, ann in enumerate(sorted_anns):
                color_idx = i % len(colors)
                m = ann['segmentation']
                mask_all[m] = i + 1
                img_overlay[m] = img_overlay[m] * 0.5 + np.array(colors[color_idx]) * 0.5
                
            return mask_all, img_overlay
        
        mask_all, img_overlay = show_anns(masks)
        
        # Save results
        masks_path = os.path.join(self.output_dir, f"{img_name}_masks.jpg")
        segmented_path = os.path.join(self.output_dir, f"{img_name}_segmented.jpg")
        
        cv2.imwrite(masks_path, mask_all)
        cv2.imwrite(segmented_path, img_overlay)
        
        processing_time = time.time() - start_time
        print(f"Segmentation completed in {processing_time:.2f} seconds")
        
        result = {
            "masks": masks_path,
            "segmented": segmented_path,
            "segmented_img": img_overlay,
            "masks_data": mask_all,
            "raw_masks": masks
        }
        
        # Cache the result
        self._result_cache[cache_key] = result
        
        return result

    def optimize_model_selection(self):
        """
        Choose the optimal SAM model based on available hardware
        
        Returns:
            String with recommended model type
        """
        if not torch.cuda.is_available():
            # CPU-only system: use smallest model
            return "vit_b"
        
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
        
        if gpu_memory >= 12:
            # High-end GPU: use largest model
            return "vit_h"
        elif gpu_memory >= 8:
            # Mid-range GPU: use large model
            return "vit_l"
        else:
            # Low-end GPU: use base model
            return "vit_b"
        
    def segment_objects(self, img_path, method="maskrcnn", use_cache=True):
        """
        Segment objects in an image using the specified method
        
        Args:
            img_path: Path to the input image
            method: Segmentation method ('maskrcnn' or 'yolo')
            use_cache: Whether to use cached results if available
            
        Returns:
            Dict with paths to output files and segmentation masks
        """
        print(f"Segmenting objects using {method} method")
        
        if method == "maskrcnn" or method == "sam":
            # Use SAM or MaskRCNN for segmentation
            model_type = "vit_b"  # Default model type for SAM
            return self.segment_image(img_path, use_cache=use_cache, model_type=model_type)
        elif method == "yolo":
            # Import YOLODetector only when needed
            try:
                from yolo_detector import YOLODetector
                detector = YOLODetector(output_dir=self.output_dir)
                return detector.detect_and_segment(img_path, use_cache=use_cache)
            except ImportError:
                print("YOLO detector not available, falling back to SAM")
                return self.segment_image(img_path, use_cache=use_cache)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")