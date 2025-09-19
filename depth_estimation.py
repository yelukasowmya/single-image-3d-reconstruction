import os
import numpy as np
import torch
import cv2
from pathlib import Path
import time
from functools import lru_cache

class DepthEstimator:
    def __init__(self, output_dir="data/output/depth"):
        self.output_dir = output_dir
        self.model_cache = {}  # Cache for loaded models
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Depth Estimator using device: {self.device}")
        
        # Optional: Initialize model in constructor for faster first inference
        # self._load_midas_model("DPT_Large")

    @lru_cache(maxsize=3)  # Cache the 3 most recent model types
    def _load_midas_model(self, model_type):
        """Load and cache MiDaS model"""
        if model_type in self.model_cache:
            return self.model_cache[model_type]
            
        print(f"Loading MiDaS model: {model_type}")
        start_time = time.time()
        
        # Import here to avoid loading unnecessary dependencies
        # if model won't be used
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        midas_processor = AutoImageProcessor.from_pretrained(f"vinvino02/midas-{model_type}")
        midas_model = AutoModelForDepthEstimation.from_pretrained(f"vinvino02/midas-{model_type}")
        
        # Move model to appropriate device
        midas_model.to(self.device)
        
        loading_time = time.time() - start_time
        print(f"Model loaded in {loading_time:.2f} seconds")
        
        # Cache the model
        self.model_cache[model_type] = (midas_model, midas_processor)
        return midas_model, midas_processor

    def generate_depth_map(self, input_img, model_type="DPT_Large", use_mixed_precision=True, use_cache=True):
        """
        Generate a depth map using MiDaS
        
        Args:
            input_img: Path to image or numpy array
            model_type: MiDaS model type ('DPT_Large', 'DPT_Hybrid', or 'MiDaS_small')
            use_mixed_precision: Whether to use FP16 for faster inference when supported
            use_cache: Whether to use cached depth maps if available
            
        Returns:
            Dict with depth map and paths to output files
        """
        # Load input image if path was provided
        if isinstance(input_img, str) or isinstance(input_img, Path):
            img_path = input_img
            img_name = Path(img_path).stem
            input_img = cv2.imread(str(img_path))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        else:
            img_name = "masked_image"
        
        depth_output_path = os.path.join(self.output_dir, f"{img_name}_depth.jpg")
        depth_colored_path = os.path.join(self.output_dir, f"{img_name}_depth_colored.jpg")
        
        # Check if cached results exist
        if use_cache and os.path.exists(depth_output_path) and os.path.exists(depth_colored_path):
            print(f"Using cached depth map for {img_name}")
            depth_map = cv2.imread(depth_output_path, cv2.IMREAD_UNCHANGED)
            return {
                "depth": depth_colored_path,
                "depth_data": depth_map,
                "depth_raw": depth_output_path
            }
        
        # Process the image to generate a depth map
        print(f"Generating depth map using MiDaS {model_type}...")
        start_time = time.time()
        
        # Load the model
        midas_model, midas_processor = self._load_midas_model(model_type)
        
        # Prepare the image
        height, width = input_img.shape[:2]
        
        # Preprocess the image
        inputs = midas_processor(images=input_img, return_tensors="pt")
        
        # Move input to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference with mixed precision if requested and supported
        if use_mixed_precision and self.device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = midas_model(**inputs)
                    prediction = outputs.predicted_depth
        else:
            with torch.no_grad():
                outputs = midas_model(**inputs)
                prediction = outputs.predicted_depth
        
        # Convert the prediction to a numpy array and resize to original dimensions
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map, (width, height))
        
        # Normalize the depth map for visualization (0-255)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        depth_map_normalized *= 255
        depth_map_normalized = depth_map_normalized.astype(np.uint8)
        
        # Generate colored depth map for visualization
        depth_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        # Save output files
        cv2.imwrite(depth_output_path, depth_map_normalized)
        cv2.imwrite(depth_colored_path, cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
        
        processing_time = time.time() - start_time
        print(f"Depth map generated in {processing_time:.2f} seconds")
        
        return {
            "depth": depth_colored_path,
            "depth_data": depth_map,
            "depth_raw": depth_output_path
        }
    
    def optimize_model_selection(self):
        """
        Choose the optimal model based on available hardware
        
        Returns:
            String with recommended model type
        """
        if not torch.cuda.is_available():
            # CPU-only system: use smallest model
            return "MiDaS_small"
        
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
        
        if gpu_memory >= 8:
            # High-end GPU: use largest model
            return "DPT_Large"
        elif gpu_memory >= 4:
            # Mid-range GPU: use hybrid model
            return "DPT_Hybrid"
        else:
            # Low-end GPU: use small model
            return "MiDaS_small"