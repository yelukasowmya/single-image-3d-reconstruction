import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import trimesh
from skimage import measure
from scipy import ndimage
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from ultralytics import YOLO
from view_synthesizer import ViewSynthesizer
import open3d as o3d

class EvaluationMetrics:
    """
    Class for evaluating 3D reconstruction quality
    """
    @staticmethod
    def chamfer_distance(points_a, points_b):
        """
        Compute Chamfer Distance between two point clouds
        
        Args:
            points_a: First point cloud (N x 3)
            points_b: Second point cloud (M x 3)
            
        Returns:
            Chamfer distance (bidirectional mean of closest point distances)
        """
        # Find nearest neighbors from A to B
        tree_b = cKDTree(points_b)
        dist_a_to_b, _ = tree_b.query(points_a)
        
        # Find nearest neighbors from B to A
        tree_a = cKDTree(points_a)
        dist_b_to_a, _ = tree_a.query(points_b)
        
        # Calculate bidirectional Chamfer distance
        chamfer_dist = np.mean(dist_a_to_b) + np.mean(dist_b_to_a)
        
        return chamfer_dist
    
    @staticmethod
    def hausdorff_distance(points_a, points_b):
        """
        Compute Hausdorff Distance between two point clouds
        
        Args:
            points_a: First point cloud (N x 3)
            points_b: Second point cloud (M x 3)
            
        Returns:
            Hausdorff distance
        """
        # Compute directed Hausdorff distances
        forward_hausdorff = directed_hausdorff(points_a, points_b)[0]
        backward_hausdorff = directed_hausdorff(points_b, points_a)[0]
        
        # Take maximum of the two directed distances
        hausdorff_dist = max(forward_hausdorff, backward_hausdorff)
        
        return hausdorff_dist
    
    @staticmethod
    def normal_consistency(mesh_a, mesh_b, samples=10000):
        """
        Compute normal consistency between two meshes
        
        Args:
            mesh_a: First mesh (trimesh.Trimesh)
            mesh_b: Second mesh (trimesh.Trimesh)
            samples: Number of sample points
            
        Returns:
            Normal consistency score (1 is perfect, 0 is worst)
        """
        # Sample points and normals from both meshes
        points_a, face_idx_a = mesh_a.sample(samples, return_index=True)
        normals_a = mesh_a.face_normals[face_idx_a]
        
        points_b, face_idx_b = mesh_b.sample(samples, return_index=True)
        normals_b = mesh_b.face_normals[face_idx_b]
        
        # Find nearest points in mesh B for each point in mesh A
        tree_b = cKDTree(points_b)
        _, idx_b = tree_b.query(points_a)
        
        # Get corresponding normals
        corresponding_normals_b = normals_b[idx_b]
        
        # Compute dot product of normalized normals
        normals_a_normalized = normals_a / np.linalg.norm(normals_a, axis=1, keepdims=True)
        normals_b_normalized = corresponding_normals_b / np.linalg.norm(corresponding_normals_b, axis=1, keepdims=True)
        
        # Compute absolute dot product (normal consistency)
        dot_products = np.abs(np.sum(normals_a_normalized * normals_b_normalized, axis=1))
        normal_consistency = np.mean(dot_products)
        
        return normal_consistency
    
    @staticmethod
    def f_score(points_a, points_b, threshold=0.01):
        """
        Compute F-score between two point clouds
        
        Args:
            points_a: First point cloud (N x 3)
            points_b: Second point cloud (M x 3)
            threshold: Distance threshold for counting matches
            
        Returns:
            F-score (harmonic mean of precision and recall)
        """
        # Find nearest neighbors from A to B
        tree_b = cKDTree(points_b)
        dist_a_to_b, _ = tree_b.query(points_a)
        
        # Find nearest neighbors from B to A
        tree_a = cKDTree(points_a)
        dist_b_to_a, _ = tree_a.query(points_b)
        
        # Calculate precision and recall
        precision = np.mean(dist_a_to_b < threshold)
        recall = np.mean(dist_b_to_a < threshold)
        
        # Calculate F-score
        if precision + recall > 0:
            f_score = 2 * precision * recall / (precision + recall)
        else:
            f_score = 0.0
            
        return f_score, precision, recall
    
    @staticmethod
    def iou_voxel(voxel_grid_a, voxel_grid_b, threshold=0.5):
        """
        Compute IoU between two voxel grids
        
        Args:
            voxel_grid_a: First voxel grid (binary)
            voxel_grid_b: Second voxel grid (binary)
            threshold: Threshold for binary conversion
            
        Returns:
            IoU score
        """
        # Convert to binary if not already
        binary_a = voxel_grid_a > threshold
        binary_b = voxel_grid_b > threshold
        
        # Calculate intersection and union
        intersection = np.logical_and(binary_a, binary_b).sum()
        union = np.logical_or(binary_a, binary_b).sum()
        
        # Calculate IoU
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
            
        return iou
    
    @staticmethod
    def depth_accuracy(predicted_depth, ground_truth_depth, mask=None):
        """
        Compute depth map accuracy metrics
        
        Args:
            predicted_depth: Predicted depth map
            ground_truth_depth: Ground truth depth map
            mask: Optional mask for evaluation on specific regions
            
        Returns:
            Dictionary with depth accuracy metrics
        """
        if mask is not None:
            valid_mask = mask > 0
            pred = predicted_depth[valid_mask]
            gt = ground_truth_depth[valid_mask]
        else:
            pred = predicted_depth.flatten()
            gt = ground_truth_depth.flatten()
            
        # Calculate error metrics
        abs_rel = np.mean(np.abs(pred - gt) / gt)
        sq_rel = np.mean(((pred - gt) ** 2) / gt)
        rmse = np.sqrt(mean_squared_error(gt, pred))
        
        return {
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse
        }
    
    @staticmethod
    def mesh_self_consistency(mesh):
        """
        Evaluate mesh quality metrics without ground truth
        
        Args:
            mesh: Trimesh object
            
        Returns:
            Dictionary with mesh quality metrics
        """
        # Check for watertightness
        watertight = mesh.is_watertight
        
        # Calculate manifold edges ratio
        if mesh.edges_unique.shape[0] > 0:
            manifold_edges = np.mean(mesh.edges_unique_length > 0)
        else:
            manifold_edges = 0.0
        
        # Calculate mesh surface area and volume
        surface_area = mesh.area
        volume = 0
        if watertight:
            volume = mesh.volume
            
        # Count mesh components
        connected_components = len(mesh.split())
        
        # Edge length statistics
        if len(mesh.edges_unique_length) > 0:
            edge_lengths = mesh.edges_unique_length
            mean_edge_length = np.mean(edge_lengths)
            std_edge_length = np.std(edge_lengths)
        else:
            mean_edge_length = 0
            std_edge_length = 0
        
        return {
            "watertight": watertight,
            "manifold_edges_ratio": manifold_edges,
            "surface_area": surface_area,
            "volume": volume,
            "connected_components": connected_components,
            "mean_edge_length": mean_edge_length,
            "std_edge_length": std_edge_length
        }

class Pipeline3D:
    def __init__(self, input_dir="data/input", output_dir="data/output"):
        """
        Initialize the 3D reconstruction pipeline
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.depth_dir = os.path.join(output_dir, "depth")
        self.voxels_dir = os.path.join(output_dir, "voxels")
        self.meshes_dir = os.path.join(output_dir, "meshes")
        self.views_dir = os.path.join(output_dir, "views")  # New directory for synthetic views
        self.eval_dir = os.path.join(output_dir, "evaluation")  # New directory for evaluation results
        
        # Create output directories
        for directory in [output_dir, self.depth_dir, self.voxels_dir, self.meshes_dir, self.views_dir, self.eval_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Initialize device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize view synthesizer
        self.view_synthesizer = ViewSynthesizer(device=self.device)
    
    def segment_objects(self, img_path, method="maskrcnn"):
        """
        Segment objects in an image using Mask R-CNN or YOLO
        
        Args:
            img_path: Path to the input image
            method: Segmentation method ('maskrcnn' or 'yolo')
            
        Returns:
            Dict with paths to output files
        """
        img_name = Path(img_path).stem
        
        if method == "maskrcnn":
            return self._segment_with_maskrcnn(img_path, img_name)
        elif method == "yolo":
            return self._segment_with_yolo(img_path, img_name)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def _segment_with_maskrcnn(self, img_path, img_name):
        """
        Segment objects using Mask R-CNN
        """
        print(f"Segmenting objects in {img_path} using Mask R-CNN...")
        
        # Load image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image_rgb).to(self.device)
        
        # Load Mask R-CNN model
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model.to(self.device)
        model.eval()
        
        # Perform inference
        with torch.no_grad():
            prediction = model([image_tensor])[0]
        
        # Extract masks, boxes, scores, labels
        masks = prediction["masks"].cpu().numpy()
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        
        # Create masked image (for all objects)
        mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        segmented_image = image.copy()
        
        conf_threshold = 0.5
        for i, score in enumerate(scores):
            if score > conf_threshold:
                mask = masks[i, 0]
                mask_binary = (mask > 0.5).astype(np.uint8)
                mask_combined = np.maximum(mask_combined, mask_binary)
                
                # Add colored mask to segmented image
                color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                segmented_image[mask_binary > 0] = segmented_image[mask_binary > 0] * 0.7 + color * 0.3
                
                # Draw bounding box
                box = boxes[i].astype(np.int32)
                cv2.rectangle(segmented_image, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)
                
                # Add label
                label_id = labels[i]
                label_text = f"Class {label_id}: {score:.2f}"
                cv2.putText(segmented_image, label_text, (box[0], box[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
        
        # Save outputs
        detected_path = os.path.join(self.output_dir, f"{img_name}_detected.jpg")
        mask_path = os.path.join(self.output_dir, f"{img_name}_masks.jpg")
        segmented_path = os.path.join(self.output_dir, f"{img_name}_segmented.jpg")
        
        # Save detected image with bounding boxes
        cv2.imwrite(detected_path, segmented_image)
        
        # Save mask
        cv2.imwrite(mask_path, mask_combined * 255)
        
        # Save segmented image (original with colored masks)
        segmented_vis = image.copy()
        segmented_vis[mask_combined > 0, 2] = 255  # Highlight in red
        cv2.imwrite(segmented_path, segmented_vis)
        
        print(f"Segmentation results saved to {self.output_dir}")
        return {
            "detected": detected_path,
            "masks": mask_path,
            "segmented": segmented_path
        }
    
    def _segment_with_yolo(self, img_path, img_name):
        """
        Segment objects using YOLOv8
        """
        print(f"Detecting objects in {img_path} using YOLOv8...")
        
        # Load YOLO model
        model = YOLO("yolov8n.pt")  # Use nano model by default
        
        # Run inference
        results = model(img_path, conf=0.25)
        
        # Get the first result
        result = results[0]
        
        # Load the original image to draw on
        image = cv2.imread(img_path)
        
        # Process results
        detected_objects = []
        
        # Create empty mask for the detected objects
        mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for box in result.boxes:
            # Get box coordinates, confidence and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green color
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add label with class name and confidence
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Create a simple mask from the bounding box (since YOLOv8 doesn't provide masks)
            cv2.rectangle(mask_combined, (x1, y1), (x2, y2), 255, -1)  # Filled rectangle
            
            detected_objects.append({
                'class': cls_name,
                'confidence': conf,
                'box': (x1, y1, x2, y2)
            })
        
        # Save results
        detected_path = os.path.join(self.output_dir, f"{img_name}_yolo.jpg")
        mask_path = os.path.join(self.output_dir, f"{img_name}_masks.jpg")
        
        cv2.imwrite(detected_path, image)
        cv2.imwrite(mask_path, mask_combined)
        
        print(f"YOLO detected {len(detected_objects)} objects in {img_path}")
        print(f"Results saved to {detected_path}")
        
        return {
            "detected": detected_path,
            "masks": mask_path
        }
    
    def generate_depth_map(self, img_path, model_type="DPT_Large"):
        """
        Generate depth map from an image using MiDaS
        
        Args:
            img_path: Path to the input image (usually a mask)
            model_type: MiDaS model type ('DPT_Large', 'DPT_Hybrid', or 'MiDaS_small')
            
        Returns:
            Dict with paths to output files
        """
        print(f"Generating depth map for {img_path} using MiDaS {model_type}...")
        
        img_name = Path(img_path).stem
        
        # Initialize MiDaS model
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(self.device)
        midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply input transform
        input_batch = transform(img).to(self.device)
        
        # Prediction
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Create colored depth map
        colored_depth_map = plt.cm.jet(depth_map)[:, :, :3]
        colored_depth_map = (colored_depth_map * 255).astype(np.uint8)
        
        # Save results
        depth_path = os.path.join(self.depth_dir, f"{img_name}_depth.jpg")
        colored_depth_path = os.path.join(self.depth_dir, f"{img_name}_depth_colored.jpg")
        
        # Save grayscale depth map
        cv2.imwrite(depth_path, (depth_map * 255).astype(np.uint8))
        
        # Save colored depth map
        cv2.imwrite(colored_depth_path, cv2.cvtColor(colored_depth_map, cv2.COLOR_RGB2BGR))
        
        print(f"Depth map generated for {img_path}")
        print(f"Saved to {depth_path} and {colored_depth_path}")
        
        return {
            "depth": depth_path,
            "colored_depth": colored_depth_path,
            "depth_data": depth_map
        }
    
    def generate_synthetic_views(self, image_path, depth_path, mask_path, num_views=8):
        """
        Generate synthetic views of the object from different angles
        
        Args:
            image_path: Path to the original image
            depth_path: Path to the depth map
            mask_path: Path to the object mask
            num_views: Number of views to generate
            
        Returns:
            Dictionary with paths to synthetic views
        """
        print(f"\n--- Generating Synthetic Views ({num_views} views) ---")
        
        img_name = Path(image_path).stem
        output_dir = os.path.join(self.views_dir, img_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load input data
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Generate synthetic views
        synthetic_views = self.view_synthesizer.generate_synthetic_views(
            image, depth, mask, num_views=num_views
        )
        
        # Save views
        output_paths = self.view_synthesizer.save_synthetic_views(
            synthetic_views, output_dir, img_name
        )
        
        print(f"Generated {len(synthetic_views)} synthetic views from {image_path}")
        
        return output_paths
    
    def create_multi_view_voxel_grid(self, original_image_path, original_depth_path, mask_path, synthetic_views, resolution=128):
        """
        Create a voxel grid using multiple views for better 3D reconstruction
        
        Args:
            original_image_path: Path to the original image
            original_depth_path: Path to the original depth map
            mask_path: Path to the object mask
            synthetic_views: Dictionary with paths to synthetic views
            resolution: Voxel grid resolution
            
        Returns:
            3D voxel grid as numpy array
        """
        print(f"\n--- Creating Multi-View Voxel Grid ---")
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
        
        # Load original depth map
        original_depth = self.load_depth_map(original_depth_path)
        
        # Create initial voxel grid from the original view
        voxel_grid = self.create_voxel_grid(original_depth, mask, resolution)
        
        # Integrate information from synthetic views
        for i, depth_path in enumerate(synthetic_views['depth_paths']):
            print(f"Integrating synthetic view {i+1}/{len(synthetic_views['depth_paths'])}...")
            
            # Load synthetic depth map
            synthetic_depth = self.load_depth_map(depth_path)
            
            # Create voxel grid for this view
            view_grid = self.create_voxel_grid(synthetic_depth, mask, resolution)
            
            # Combine with the main grid (using maximum values to ensure complete geometry)
            voxel_grid = np.maximum(voxel_grid, view_grid)
        
        # Apply 3D smoothing to clean up noise introduced by multiple views
        voxel_grid = ndimage.gaussian_filter(voxel_grid, sigma=0.8)
        
        return voxel_grid
    
    def load_depth_map(self, path):
        """
        Load a depth map and normalize it
        
        Args:
            path: Path to the depth map image
            
        Returns:
            Normalized depth map as numpy array
        """
        depth = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Normalize to [0, 1]
        if depth.max() > 1.0:
            depth = depth.astype(np.float32) / 255.0
        return depth
    
    def create_voxel_grid(self, depth_map, mask=None, resolution=128):
        """
        Create a 3D voxel grid from a depth map
        
        Args:
            depth_map: Depth map as a numpy array
            mask: Optional binary mask
            resolution: Voxel grid resolution
            
        Returns:
            3D voxel grid as numpy array
        """
        print(f"Creating voxel grid with resolution {resolution}...")
        
        # If no mask is provided, create a simple one
        if mask is None:
            mask = np.ones_like(depth_map)
            
        # Normalize depth map to [0, 1]
        if depth_map.max() > 1.0:
            depth_map = depth_map / 255.0
            
        # Apply the mask
        masked_depth = depth_map * (mask > 0)
        
        # Clean depth map
        kernel = np.ones((3,3), np.uint8)
        cleaned_depth = cv2.morphologyEx(masked_depth.astype(np.float32), cv2.MORPH_OPEN, kernel)
        
        # Resize to match voxel resolution
        depth_resized = cv2.resize(cleaned_depth, (resolution, resolution))
        
        # Create 3D grid
        voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        # Use a more sophisticated approach to create the 3D volume
        for x in range(resolution):
            for y in range(resolution):
                depth_val = depth_resized[y, x]
                
                if depth_val > 0.01:
                    z_val = int(depth_val * resolution)
                    z_range = max(2, int(resolution * 0.05))
                    
                    z_start = max(0, z_val - z_range)
                    z_end = min(resolution - 1, z_val + z_range)
                    
                    voxel_grid[y, x, z_start:z_end+1] = 1.0
        
        # Apply 3D smoothing
        voxel_grid = ndimage.gaussian_filter(voxel_grid, sigma=0.8)
        
        return voxel_grid
    
    def create_mesh_from_voxels(self, voxel_grid, threshold=0.3):
        """
        Create a 3D mesh from a voxel grid using Marching Cubes
        """
        print(f"Creating mesh from voxel grid using Marching Cubes (threshold={threshold})...")
        
        # Pad the voxel grid to avoid boundary issues
        padded_grid = np.pad(voxel_grid, 1, mode='constant', constant_values=0)
        
        # Apply marching cubes
        vertices, faces, normals, _ = measure.marching_cubes(
            padded_grid, 
            level=threshold,
            allow_degenerate=False
        )
        
        # Adjust vertices to account for padding
        vertices = vertices - 1
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
        
        # Apply Laplacian smoothing using the new method
        try:
            mesh = trimesh.graph.smooth_shade(mesh, iterations=3)
        except Exception as e:
            print(f"Warning: Smoothing failed - {e}")
        
        return mesh
    
    def apply_texture_to_mesh(self, mesh, image_path, mask_path=None):
        """
        Apply texture from the original image to the mesh
        
        Args:
            mesh: Trimesh object
            image_path: Path to the original image for texturing
            mask_path: Optional path to the mask image
            
        Returns:
            Textured mesh object
        """
        print(f"Applying texture from {image_path} to mesh...")
        
        # Load the original image for texturing
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to load texture image {image_path}. Using default coloring.")
            return self.apply_default_coloring(mesh)
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask if provided
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
        # Project texture onto mesh using improved UV mapping approach
        # First, establish the mapping from 3D to 2D
        vertices = mesh.vertices.copy()
        min_vals = np.min(vertices, axis=0)
        max_vals = np.max(vertices, axis=0)
        range_vals = max_vals - min_vals
        
        # Prevent division by zero
        range_vals[range_vals == 0] = 1
        
        # Normalize vertices
        normalized_vertices = (vertices - min_vals) / range_vals
        
        # Get image dimensions
        img_height, img_width = image_rgb.shape[:2]
        
        # Improved mapping: Use spherical coordinates for better texture wrapping
        # This works better for most object types
        
        # Center the vertices around origin for proper spherical mapping
        centered_verts = vertices - np.mean(vertices, axis=0)
        
        # Calculate radius, azimuth and elevation for spherical coordinates
        radius = np.sqrt(np.sum(centered_verts**2, axis=1))
        azimuth = np.arctan2(centered_verts[:, 0], centered_verts[:, 2])  # Using X and Z
        elevation = np.arcsin(centered_verts[:, 1] / (radius + 1e-10))  # Using Y
        
        # Normalize to [0, 1] range for texture mapping
        u = (azimuth / (2 * np.pi) + 0.5) % 1.0  # Azimuth to U coordinate
        v = (elevation / np.pi + 0.5)  # Elevation to V coordinate
        
        # Convert UV coordinates to image pixel coordinates
        img_coords_x = (u * img_width).astype(np.int32)
        img_coords_y = (v * img_height).astype(np.int32)
        
        # Clamp to image boundaries
        img_coords_x = np.clip(img_coords_x, 0, img_width - 1)
        img_coords_y = np.clip(img_coords_y, 0, img_height - 1)
        
        # Sample colors from image with enhanced color vibrancy
        vertex_colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        
        for i in range(len(vertices)):
            x, y = img_coords_x[i], img_coords_y[i]
            
            # Only apply texture where the mask is present (if mask is provided)
            if mask is None or mask[y, x] > 0:
                # Get color from the image and enhance it
                color = image_rgb[y, x].astype(np.float32)
                
                # Enhance color vibrancy (increase saturation)
                hsv = cv2.cvtColor(np.array([[color]]), cv2.COLOR_RGB2HSV)[0, 0]
                hsv[1] = min(hsv[1] * 1.3, 255)  # Increase saturation by 30%
                hsv[2] = min(hsv[2] * 1.1, 255)  # Increase value by 10%
                enhanced_color = cv2.cvtColor(np.array([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0]
                
                vertex_colors[i, :3] = enhanced_color
                vertex_colors[i, 3] = 255  # Full alpha channel
            else:
                # For points outside the mask, add a semi-transparent color
                vertex_colors[i, :3] = [200, 200, 200]  # Light gray
                vertex_colors[i, 3] = 128  # Semi-transparent
        
        # Create a new mesh with vertex colors
        textured_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            vertex_colors=vertex_colors
        )
        
        print("Texture mapping completed with enhanced spherical UV mapping and color vibrancy.")
        
        return textured_mesh
    
    def apply_default_coloring(self, mesh):
        """
        Apply default coloring to the mesh based on depth
        
        Args:
            mesh: Trimesh object
            
        Returns:
            Colored mesh object
        """
        # Get the vertices
        vertices = mesh.vertices
        
        # Determine min and max height (assuming Y is up)
        y_min = vertices[:, 1].min()
        y_max = vertices[:, 1].max()
        height_range = y_max - y_min
        
        if height_range == 0:
            height_range = 1  # Prevent division by zero
        
        # Normalize heights to [0, 1]
        normalized_heights = (vertices[:, 1] - y_min) / height_range
        
        # Create a gradient coloring using jet colormap
        # Map values from [0, 1] to colors
        colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        
        # Simple gradient from blue to red
        for i, h in enumerate(normalized_heights):
            if h < 0.25:
                # Blue to cyan
                b = 255
                g = int(255 * (h / 0.25))
                r = 0
            elif h < 0.5:
                # Cyan to green
                b = int(255 * (1 - (h - 0.25) / 0.25))
                g = 255
                r = 0
            elif h < 0.75:
                # Green to yellow
                b = 0
                g = 255
                r = int(255 * ((h - 0.5) / 0.25))
            else:
                # Yellow to red
                b = 0
                g = int(255 * (1 - (h - 0.75) / 0.25))
                r = 255
                
            colors[i] = [r, g, b, 255]
        
        # Create new mesh with vertex colors
        colored_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            vertex_colors=colors
        )
        
        return colored_mesh
    
    def process_image(self, img_path, segmentation="maskrcnn", depth_model="DPT_Large", 
                     voxel_resolution=128, apply_texture=True, generate_views=True, num_views=8):
        """
        Process a single image through the entire pipeline with multi-view synthesis
        
        Args:
            img_path: Path to the input image
            segmentation: Segmentation method ('maskrcnn' or 'yolo')
            depth_model: MiDaS model type
            voxel_resolution: Voxel grid resolution
            apply_texture: Whether to apply texture to the mesh
            generate_views: Whether to generate synthetic views
            num_views: Number of synthetic views to generate
            
        Returns:
            Dict with paths to all output files
        """
        img_name = Path(img_path).stem
        
        # Step 1: Segment objects
        print(f"\n--- Step 1: Object Segmentation ({segmentation}) ---")
        segmentation_results = self.segment_objects(img_path, method=segmentation)
        
        # Step 2: Generate depth map from masks
        print(f"\n--- Step 2: Depth Map Generation ({depth_model}) ---")
        depth_results = self.generate_depth_map(segmentation_results["masks"], model_type=depth_model)
        
        # Step 3 (New): Generate synthetic views if requested
        synthetic_views = None
        if generate_views:
            synthetic_views = self.generate_synthetic_views(
                img_path,
                depth_results["depth"],
                segmentation_results["masks"],
                num_views=num_views
            )
        
        # Step 4: Create voxel grid (now with multi-view support)
        print(f"\n--- Step 4: Voxel Grid Creation (resolution={voxel_resolution}) ---")
        if generate_views and synthetic_views:
            # Use multi-view approach for better 3D reconstruction
            voxel_grid = self.create_multi_view_voxel_grid(
                img_path,
                depth_results["depth"],
                segmentation_results["masks"],
                synthetic_views,
                resolution=voxel_resolution
            )
        else:
            # Fallback to single-view approach
            mask = cv2.imread(segmentation_results["masks"], cv2.IMREAD_GRAYSCALE) / 255.0
            voxel_grid = self.create_voxel_grid(depth_results["depth_data"], mask, resolution=voxel_resolution)
        
        # Save voxel grid
        voxel_path = os.path.join(self.voxels_dir, f"{img_name}_voxels.npy")
        np.save(voxel_path, voxel_grid)
        
        # Visualize voxel grid
        self._visualize_voxel_grid(voxel_grid, img_name)
        
        # Step 5: Create mesh
        print(f"\n--- Step 5: Mesh Creation ---")
        mesh = self.create_mesh_from_voxels(voxel_grid)
        
        # Step 6: Apply texture if requested
        if apply_texture:
            print(f"\n--- Step 6: Texture Mapping ---")
            try:
                textured_mesh = self.apply_texture_to_mesh(
                    mesh,
                    img_path,  # Use original image for texture
                    segmentation_results["masks"]  # Use mask to isolate the object
                )
                
                # Save textured mesh
                obj_path = os.path.join(self.meshes_dir, f"{img_name}_textured.obj")
                textured_mesh.export(
                    obj_path,
                    include_normals=True,
                    include_color=True,
                    write_texture=True
                )
                
                # Also save a PLY file which better preserves vertex colors
                ply_path = os.path.join(self.meshes_dir, f"{img_name}_textured.ply")
                textured_mesh.export(ply_path)
                
                # Create mesh preview with texture
                preview_path = os.path.join(self.meshes_dir, f"{img_name}_textured_preview.png")
                self._create_mesh_preview(textured_mesh, preview_path)
                
                print(f"Textured mesh saved to {obj_path} and {ply_path}")
                
                # Also save the untextured mesh for comparison
                untextured_obj_path = os.path.join(self.meshes_dir, f"{img_name}.obj")
                mesh.export(untextured_obj_path)
                
                # View the mesh with Open3D
                print("\nOpening 3D viewer...")
                self.view_mesh_with_open3d(obj_path)
                
            except Exception as e:
                print(f"Error applying texture: {e}")
                print("Saving untextured mesh only")
                obj_path = os.path.join(self.meshes_dir, f"{img_name}.obj")
                stl_path = os.path.join(self.meshes_dir, f"{img_name}.stl")
                mesh.export(obj_path)
                mesh.export(stl_path)
                
                # Create mesh preview without texture
                preview_path = os.path.join(self.meshes_dir, f"{img_name}_preview.png")
                self._create_mesh_preview(mesh, preview_path)
        else:
            # Save untextured mesh
            obj_path = os.path.join(self.meshes_dir, f"{img_name}.obj")
            mesh.export(obj_path)
            
            # View the mesh with Open3D
            self.view_mesh_with_open3d(obj_path)
            
            # Create mesh preview
            preview_path = os.path.join(self.meshes_dir, f"{img_name}_preview.png")
            self._create_mesh_preview(mesh, preview_path)
        
        print(f"\n--- Processing complete for {img_path} ---")
        print(f"All results saved in {self.output_dir} and subdirectories")
        
        return {
            "segmentation": segmentation_results,
            "depth": depth_results,
            "views": synthetic_views if generate_views else None,
            "voxel": voxel_path,
            "mesh": {"obj": obj_path, "preview": preview_path}
        }
    
    def process_all_images(self, segmentation="maskrcnn", depth_model="DPT_Large", 
                          voxel_resolution=128, apply_texture=True, generate_views=True, num_views=8):
        """
        Process all images in the input directory
        
        Args:
            segmentation: Segmentation method
            depth_model: MiDaS model type
            voxel_resolution: Voxel grid resolution
            apply_texture: Whether to apply texture to the meshes
            generate_views: Whether to generate synthetic views
            num_views: Number of synthetic views to generate
        """
        image_paths = []
        
        # Find all images in the input directory
        for ext in [".jpg", ".jpeg", ".png"]:
            image_paths.extend(list(Path(self.input_dir).glob(f"*{ext}")))
        
        if not image_paths:
            print(f"No images found in {self.input_dir}")
            return
        
        print(f"Found {len(image_paths)} images to process")
        
        results = {}
        for img_path in image_paths:
            print(f"\n=== Processing {img_path} ===")
            results[str(img_path)] = self.process_image(
                str(img_path),
                segmentation=segmentation,
                depth_model=depth_model,
                voxel_resolution=voxel_resolution,
                apply_texture=apply_texture,
                generate_views=generate_views,
                num_views=num_views
            )
        
        return results
    
    def _visualize_voxel_grid(self, voxel_grid, img_name):
        """
        Visualize a voxel grid with orthogonal slices
        """
        # Create a figure with 3 subplots for the 3 orthogonal slices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Take central slices from each dimension
        mid_x = voxel_grid.shape[0] // 2
        mid_y = voxel_grid.shape[1] // 2
        mid_z = voxel_grid.shape[2] // 2
        
        # Create visualizations for each slice
        axes[0].imshow(voxel_grid[mid_x, :, :], cmap='jet')
        axes[0].set_title(f'YZ Slice (X={mid_x})')
        
        axes[1].imshow(voxel_grid[:, mid_y, :], cmap='jet')
        axes[1].set_title(f'XZ Slice (Y={mid_y})')
        
        axes[2].imshow(voxel_grid[:, :, mid_z], cmap='jet')
        axes[2].set_title(f'XY Slice (Z={mid_z})')
        
        # Save the figure
        viz_path = os.path.join(self.voxels_dir, f"{img_name}_voxel_slices.png")
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
        print(f"Saved voxel visualization to {viz_path}")
    
    def _create_mesh_preview(self, mesh, output_path):
        """
        Create a preview image of a mesh
        """
        try:
            # Try to render with trimesh's built-in renderer
            scene = trimesh.Scene(mesh)
            png = scene.save_image(resolution=[1024, 768], visible=True)
            with open(output_path, 'wb') as f:
                f.write(png)
            print(f"Saved mesh preview: {output_path}")
        except Exception as e:
            print(f"Error with trimesh renderer: {e}")
            # Fall back to matplotlib
            try:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_trisurf(
                    mesh.vertices[:, 0], 
                    mesh.vertices[:, 1], 
                    mesh.vertices[:, 2], 
                    triangles=mesh.faces,
                    cmap='viridis', 
                    alpha=0.8
                )
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                print(f"Saved mesh preview using matplotlib: {output_path}")
            except Exception as e2:
                print(f"Failed to create mesh preview: {e2}")
    
    def evaluate_reconstruction(self, mesh_path, ground_truth_path=None):
        """
        Evaluate 3D reconstruction quality
        
        Args:
            mesh_path: Path to the reconstructed mesh
            ground_truth_path: Optional path to ground truth mesh for comparison
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n--- Evaluating Reconstruction Quality ---")
        
        # Load reconstructed mesh
        if isinstance(mesh_path, trimesh.Trimesh):
            reconstructed_mesh = mesh_path
        else:
            print(f"Loading reconstructed mesh from {mesh_path}")
            reconstructed_mesh = trimesh.load(mesh_path)
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Basic mesh quality metrics (self-evaluation)
        print("Computing mesh quality metrics...")
        mesh_metrics = EvaluationMetrics.mesh_self_consistency(reconstructed_mesh)
        metrics["mesh_quality"] = mesh_metrics
        
        # Create report with basic metrics
        report = f"""
        === RECONSTRUCTION QUALITY METRICS ===
        
        MESH INTEGRITY:
        - Watertight: {mesh_metrics['watertight']}
        - Connected Components: {mesh_metrics['connected_components']}
        - Surface Area: {mesh_metrics['surface_area']:.2f}
        - Volume: {mesh_metrics['volume']:.2f} (if watertight)
        
        MESH TOPOLOGY:
        - Manifold Edges Ratio: {mesh_metrics['manifold_edges_ratio']:.4f}
        - Mean Edge Length: {mesh_metrics['mean_edge_length']:.4f}
        - Edge Length Std Dev: {mesh_metrics['std_edge_length']:.4f}
        """
        
        # If ground truth is available, compute accuracy metrics
        if ground_truth_path is not None:
            print(f"Loading ground truth mesh from {ground_truth_path}")
            try:
                ground_truth_mesh = trimesh.load(ground_truth_path)
                
                # Sample points from both meshes for comparison
                print("Sampling points from meshes for comparison...")
                recon_points = reconstructed_mesh.sample(10000)
                gt_points = ground_truth_mesh.sample(10000)
                
                # Compute geometric metrics
                print("Computing Chamfer distance...")
                chamfer_dist = EvaluationMetrics.chamfer_distance(recon_points, gt_points)
                metrics["chamfer_distance"] = chamfer_dist
                
                print("Computing Hausdorff distance...")
                hausdorff_dist = EvaluationMetrics.hausdorff_distance(recon_points, gt_points)
                metrics["hausdorff_distance"] = hausdorff_dist
                
                print("Computing F-score...")
                f_score, precision, recall = EvaluationMetrics.f_score(recon_points, gt_points, threshold=0.01)
                metrics["f_score"] = f_score
                metrics["precision"] = precision
                metrics["recall"] = recall
                
                print("Computing normal consistency...")
                normal_cons = EvaluationMetrics.normal_consistency(reconstructed_mesh, ground_truth_mesh)
                metrics["normal_consistency"] = normal_cons
                
                # Add comparison metrics to report
                report += f"""
                COMPARISON TO GROUND TRUTH:
                - Chamfer Distance: {chamfer_dist:.6f} (lower is better)
                - Hausdorff Distance: {hausdorff_dist:.6f} (lower is better)
                - F-score (1cm): {f_score:.4f} (higher is better)
                - Precision: {precision:.4f}
                - Recall: {recall:.4f}
                - Normal Consistency: {normal_cons:.4f} (higher is better)
                """
            except Exception as e:
                print(f"Error comparing to ground truth: {e}")
                report += "\nCould not compare to ground truth due to an error."
        
        # Save report
        img_name = Path(mesh_path).stem if isinstance(mesh_path, str) else "reconstruction"
        report_path = os.path.join(self.eval_dir, f"{img_name}_evaluation.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        # Create a visual representation of metrics
        self._create_metrics_visualization(metrics, img_name)
        
        print(f"Evaluation complete. Report saved to {report_path}")
        return metrics
    
    def _create_metrics_visualization(self, metrics, img_name):
        """
        Create visual representation of evaluation metrics
        
        Args:
            metrics: Dictionary with evaluation metrics
            img_name: Base name for the output file
        """
        # Create figure for mesh quality metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract mesh quality metrics
        mesh_metrics = metrics.get("mesh_quality", {})
        
        # Create bar chart for key metrics
        metric_names = ["Manifold Edges", "Connected Components"]
        metric_values = [
            mesh_metrics.get("manifold_edges_ratio", 0),
            1.0 / max(1, mesh_metrics.get("connected_components", 1))  # Normalize: 1 component = 1.0, more = lower
        ]
        
        # Add accuracy metrics if available
        if "f_score" in metrics:
            metric_names.extend(["F-score", "Precision", "Recall", "Normal Consistency"])
            metric_values.extend([
                metrics.get("f_score", 0),
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("normal_consistency", 0)
            ])
        
        # Create bar chart
        bars = ax.bar(metric_names, metric_values, color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom')
        
        # Add titles and labels
        ax.set_title('Reconstruction Quality Metrics')
        ax.set_ylim(0, 1.1)  # Metrics are normalized to [0,1]
        ax.set_ylabel('Score (higher is better)')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x-labels for better fit
        plt.xticks(rotation=45, ha='right')
        
        # Save visualization
        plt.tight_layout()
        viz_path = os.path.join(self.eval_dir, f"{img_name}_metrics.png")
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Metrics visualization saved to {viz_path}")
    
    def evaluate_depth_estimation(self, predicted_depth_path, ground_truth_depth_path=None, mask_path=None):
        """
        Evaluate depth estimation quality
        
        Args:
            predicted_depth_path: Path to the predicted depth map
            ground_truth_depth_path: Optional path to ground truth depth map
            mask_path: Optional path to mask for evaluation
            
        Returns:
            Dictionary with depth evaluation metrics
        """
        # Load predicted depth map
        predicted_depth = cv2.imread(predicted_depth_path, cv2.IMREAD_GRAYSCALE)
        if predicted_depth is None:
            print(f"Error: Could not load predicted depth map from {predicted_depth_path}")
            return None
            
        # Normalize to [0,1] if needed
        if predicted_depth.max() > 1.0:
            predicted_depth = predicted_depth.astype(np.float32) / 255.0
        
        # Load mask if provided
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.max() > 1.0:
                mask = mask.astype(np.float32) / 255.0
        
        # If ground truth is available, compute accuracy metrics
        if ground_truth_depth_path and os.path.exists(ground_truth_depth_path):
            ground_truth_depth = cv2.imread(ground_truth_depth_path, cv2.IMREAD_GRAYSCALE)
            if ground_truth_depth is None:
                print(f"Error: Could not load ground truth depth from {ground_truth_depth_path}")
            else:
                # Normalize to [0,1] if needed
                if ground_truth_depth.max() > 1.0:
                    ground_truth_depth = ground_truth_depth.astype(np.float32) / 255.0
                
                # Compute depth accuracy metrics
                metrics = EvaluationMetrics.depth_accuracy(predicted_depth, ground_truth_depth, mask)
                
                # Create visualizations for the comparison
                self._visualize_depth_comparison(
                    predicted_depth, 
                    ground_truth_depth, 
                    Path(predicted_depth_path).stem,
                    mask
                )
                
                return metrics
        
        # If no ground truth, return self-evaluation metrics
        # For depth maps without ground truth, we can analyze smoothness, 
        # edge preservation, and distribution statistics
        
        # Compute gradient magnitude for smoothness assessment
        sobelx = cv2.Sobel(predicted_depth, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(predicted_depth, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Compute metrics
        metrics = {
            "mean_depth": float(np.mean(predicted_depth[predicted_depth > 0])),
            "std_depth": float(np.std(predicted_depth[predicted_depth > 0])),
            "median_depth": float(np.median(predicted_depth[predicted_depth > 0])),
            "gradient_mean": float(np.mean(gradient_magnitude)),
            "zero_depth_ratio": float(np.mean(predicted_depth == 0))
        }
        
        # Visualize depth map and analysis
        self._visualize_depth_analysis(predicted_depth, Path(predicted_depth_path).stem, mask)
        
        return metrics
    
    def _visualize_depth_comparison(self, predicted_depth, ground_truth_depth, img_name, mask=None):
        """
        Create visual comparison between predicted and ground truth depth maps
        
        Args:
            predicted_depth: Predicted depth map
            ground_truth_depth: Ground truth depth map
            img_name: Base name for the output file
            mask: Optional mask for focused evaluation
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Show predicted depth
        axes[0].imshow(predicted_depth, cmap='magma')
        axes[0].set_title('Predicted Depth')
        axes[0].axis('off')
        
        # Show ground truth depth
        axes[1].imshow(ground_truth_depth, cmap='magma')
        axes[1].set_title('Ground Truth Depth')
        axes[1].axis('off')
        
        # Show absolute error
        error = np.abs(predicted_depth - ground_truth_depth)
        if mask is not None:
            error = error * (mask > 0)
            
        im = axes[2].imshow(error, cmap='hot')
        axes[2].set_title('Absolute Error')
        axes[2].axis('off')
        
        # Add colorbar for error
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Save figure
        plt.tight_layout()
        viz_path = os.path.join(self.eval_dir, f"{img_name}_depth_comparison.png")
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Depth comparison visualization saved to {viz_path}")
    
    def _visualize_depth_analysis(self, depth_map, img_name, mask=None):
        """
        Create visual analysis of a depth map
        
        Args:
            depth_map: Depth map to analyze
            img_name: Base name for the output file
            mask: Optional mask for focused evaluation
        """
        # Apply mask if provided
        analysis_depth = depth_map.copy()
        if mask is not None:
            analysis_depth = analysis_depth * (mask > 0)
            nonzero_mask = (mask > 0)
        else:
            nonzero_mask = (analysis_depth > 0)
        
        # Create figure with multiple plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot depth map
        axes[0, 0].imshow(analysis_depth, cmap='magma')
        axes[0, 0].set_title('Depth Map')
        axes[0, 0].axis('off')
        
        # Plot gradient magnitude
        sobelx = cv2.Sobel(analysis_depth, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(analysis_depth, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        axes[0, 1].imshow(gradient, cmap='viridis')
        axes[0, 1].set_title('Gradient Magnitude')
        axes[0, 1].axis('off')
        
        # Plot depth histogram
        valid_depths = analysis_depth[nonzero_mask]
        if len(valid_depths) > 0:
            axes[1, 0].hist(valid_depths.flatten(), bins=50, color='skyblue')
            axes[1, 0].set_title('Depth Distribution')
            axes[1, 0].set_xlabel('Normalized Depth')
            axes[1, 0].set_ylabel('Frequency')
            
            # Calculate and display statistics
            mean_depth = np.mean(valid_depths)
            median_depth = np.median(valid_depths)
            std_depth = np.std(valid_depths)
            stats_text = f"Mean: {mean_depth:.4f}\nMedian: {median_depth:.4f}\nStd Dev: {std_depth:.4f}"
            axes[1, 0].text(0.95, 0.95, stats_text, transform=axes[1, 0].transAxes, 
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot depth cross-section (middle row)
        mid_row = depth_map.shape[0] // 2
        x = np.arange(depth_map.shape[1])
        axes[1, 1].plot(x, depth_map[mid_row, :], color='blue')
        axes[1, 1].set_title(f'Depth Cross-Section (Row {mid_row})')
        axes[1, 1].set_xlabel('Column')
        axes[1, 1].set_ylabel('Depth')
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        viz_path = os.path.join(self.eval_dir, f"{img_name}_depth_analysis.png")
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Depth analysis visualization saved to {viz_path}")

    def view_mesh_with_open3d(self, mesh_path):
        """
        Open and visualize the mesh using Open3D with improved error handling
        """
        try:
            # Load the mesh
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            
            # Ensure mesh has normals and colors
            mesh.compute_vertex_normals()
            if not mesh.has_vertex_colors():
                mesh.paint_uniform_color([0.7, 0.7, 0.7])
                
            # Create visualizer
            vis = o3d.visualization.Visualizer()
            
            # Initialize window
            vis_success = vis.create_window(
                window_name="3D Reconstruction Viewer",
                width=1024,
                height=768,
                visible=True
            )
            
            if not vis_success:
                raise RuntimeError("Failed to create Open3D window")
                
            # Add mesh to scene
            vis.add_geometry(mesh)
            
            # Configure render options
            render_option = vis.get_render_option()
            render_option.background_color = np.asarray([0.2, 0.2, 0.2])
            render_option.point_size = 1.0
            render_option.show_coordinate_frame = True
            render_option.mesh_show_wireframe = False
            render_option.mesh_show_back_face = True
            
            # Configure lighting
            render_option.light_on = True
            
            # Set better camera viewpoint
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
            
            # Update geometry and render
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
            
            # Run visualizer
            print("\nViewer controls:")
            print("- Left mouse: Rotate")
            print("- Middle mouse: Pan")
            print("- Right mouse: Zoom")
            print("- Q: Exit viewer")
            print("- H: Show help")
            
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            print(f"Error in Open3D visualization: {e}")
            print("Falling back to matplotlib visualization...")
            self._show_mesh_matplotlib(mesh_path)

    def _show_mesh_matplotlib(self, mesh_path):
        """Fallback visualization using matplotlib"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Load mesh using trimesh
        mesh = trimesh.load(mesh_path)
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mesh
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Plot vertices and faces
        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=faces,
            cmap='viridis',
            alpha=0.8
        )
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.title('3D Mesh Visualization')
        plt.show()

def main():
    """
    Main entry point for the 3D reconstruction pipeline
    """
    parser = argparse.ArgumentParser(description="3D Reconstruction Pipeline")
    parser.add_argument("--input", default="data/input", help="Input directory containing images")
    parser.add_argument("--output", default="data/output", help="Output directory for results")
    parser.add_argument("--segmentation", choices=["maskrcnn", "yolo"], default="maskrcnn", 
                       help="Segmentation method to use (maskrcnn or yolo)")
    parser.add_argument("--depth-model", choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"], 
                       default="DPT_Large", help="MiDaS depth model to use")
    parser.add_argument("--resolution", type=int, default=128, 
                       help="Voxel grid resolution (higher = more detail but slower)")
    parser.add_argument("--single-image", default=None, 
                       help="Process only a single image (provide path)")
    parser.add_argument("--no-texture", action="store_true",
                       help="Disable texture mapping (creates untextured meshes)")
    parser.add_argument("--no-views", action="store_true",
                       help="Disable synthetic view generation")
    parser.add_argument("--num-views", type=int, default=8,
                       help="Number of synthetic views to generate")
    
    # New evaluation arguments
    parser.add_argument("--evaluate", action="store_true",
                       help="Enable evaluation of reconstruction quality")
    parser.add_argument("--ground-truth", default=None,
                       help="Path to ground truth mesh for evaluation (optional)")
    parser.add_argument("--evaluate-mesh", default=None,
                       help="Path to an existing mesh to evaluate (skip reconstruction)")
    parser.add_argument("--evaluate-depth", default=None,
                       help="Path to a depth map to evaluate")
    parser.add_argument("--gt-depth", default=None,
                       help="Path to ground truth depth map for comparison (optional)")
    
    args = parser.parse_args()
    
    # Initialize the pipeline
    pipeline = Pipeline3D(input_dir=args.input, output_dir=args.output)
    
    # Handle evaluation-only mode
    if args.evaluate_mesh:
        if os.path.exists(args.evaluate_mesh):
            print(f"\n=== Evaluating Existing Mesh: {args.evaluate_mesh} ===")
            pipeline.evaluate_reconstruction(args.evaluate_mesh, args.ground_truth)
            return
        else:
            print(f"Error: Mesh file {args.evaluate_mesh} not found")
            return
            
    # Handle depth evaluation-only mode
    if args.evaluate_depth:
        if os.path.exists(args.evaluate_depth):
            print(f"\n=== Evaluating Depth Map: {args.evaluate_depth} ===")
            pipeline.evaluate_depth_estimation(args.evaluate_depth, args.gt_depth)
            return
        else:
            print(f"Error: Depth map {args.evaluate_depth} not found")
            return
    
    # Process images
    if args.single_image:
        if not os.path.exists(args.single_image):
            print(f"Error: Image {args.single_image} not found")
            return
        
        results = pipeline.process_image(
            args.single_image,
            segmentation=args.segmentation,
            depth_model=args.depth_model,
            voxel_resolution=args.resolution,
            apply_texture=not args.no_texture,
            generate_views=not args.no_views,
            num_views=args.num_views
        )
        
        # Evaluate if requested
        if args.evaluate:
            if 'mesh' in results and 'obj' in results['mesh']:
                mesh_path = results['mesh']['obj']
                print(f"\n=== Evaluating Reconstruction Quality for {args.single_image} ===")
                pipeline.evaluate_reconstruction(mesh_path, args.ground_truth)
                
                # Also evaluate depth map quality
                if 'depth' in results and 'depth' in results['depth']:
                    depth_path = results['depth']['depth']
                    print(f"\n=== Evaluating Depth Estimation Quality for {args.single_image} ===")
                    pipeline.evaluate_depth_estimation(depth_path, args.gt_depth, 
                                                     results['segmentation'].get('masks'))
    else:
        results = pipeline.process_all_images(
            segmentation=args.segmentation,
            depth_model=args.depth_model,
            voxel_resolution=args.resolution,
            apply_texture=not args.no_texture,
            generate_views=not args.no_views,
            num_views=args.num_views
        )
        
        # Evaluate all results if requested
        if args.evaluate and results:
            print("\n=== Evaluating All Reconstructions ===")
            for img_path, result in results.items():
                if 'mesh' in result and 'obj' in result['mesh']:
                    mesh_path = result['mesh']['obj']
                    print(f"\n--- Evaluating {Path(img_path).stem} ---")
                    pipeline.evaluate_reconstruction(mesh_path, args.ground_truth)
                    
                    # Also evaluate depth map quality
                    if 'depth' in result and 'depth' in result['depth']:
                        depth_path = result['depth']['depth']
                        pipeline.evaluate_depth_estimation(depth_path, args.gt_depth,
                                                         result['segmentation'].get('masks'))


if __name__ == "__main__":
    main()