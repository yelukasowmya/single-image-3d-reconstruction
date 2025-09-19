import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


class ViewSynthesizer:
    """
    Handles multi-view synthesis for 3D reconstruction
    """
    
    def __init__(self, output_dir="data/output/views", device=None):
        """
        Initialize the view synthesizer
        
        Args:
            output_dir: Directory to save synthesized views
            device: PyTorch device for processing
        """
        self.output_dir = output_dir
        self.device = device
    
    def generate_synthetic_views(self, image, depth_map, mask=None, num_views=8):
        """
        Generate synthetic views around the object
        
        Args:
            image: Input RGB image array
            depth_map: Depth map of the input image
            mask: Optional mask for the object
            num_views: Number of synthetic views to generate
            
        Returns:
            Dict with synthetic views and related data
        """
        print(f"Generating {num_views} synthetic views...")
        
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
        
        height, width = image.shape[:2]
        
        # Calculate camera intrinsics
        focal_length = max(height, width)
        cx, cy = width / 2, height / 2
        
        intrinsics = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Generate camera poses around the object
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        radius = 2.0  # Distance from camera to object
        
        views = []
        depths = []
        extrinsics = []
        
        for i, angle in enumerate(angles):
            # Calculate camera position based on angle
            camera_pos = np.array([
                radius * np.sin(angle),
                0.5,  # Slight elevation
                radius * np.cos(angle)
            ])
            
            # Look-at matrix construction
            z_axis = -camera_pos / np.linalg.norm(camera_pos)  # Forward
            temp = np.array([0, 1, 0])  # Temporary up vector
            x_axis = np.cross(temp, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)  # Right
            y_axis = np.cross(z_axis, x_axis)  # Up
            
            # Create rotation matrix
            rotation = np.column_stack([x_axis, y_axis, z_axis])
            
            # Create extrinsic matrix [R|t]
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = rotation
            extrinsic[:3, 3] = camera_pos
            
            # Generate synthetic view using 3D warping
            view, depth = self._generate_view_from_depth(
                image, depth_map, intrinsics, np.linalg.inv(extrinsic), width, height)
            
            # Apply post-processing to improve quality
            view = self._post_process_view(view)
            
            views.append(view)
            depths.append(depth)
            extrinsics.append(extrinsic)
            
            print(f"Created view {i+1}/{num_views}")
        
        return {
            "views": views,
            "depths": depths,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics
        }
    
    def _generate_view_from_depth(self, image, depth_map, intrinsics, extrinsic, width, height):
        """
        Generate a synthetic view from depth map using 3D warping
        
        Args:
            image: Input RGB image
            depth_map: Depth map of the input image
            intrinsics: Camera intrinsics matrix
            extrinsic: Target view extrinsic matrix
            width: Target view width
            height: Target view height
            
        Returns:
            Synthetic view RGB image and corresponding depth map
        """
        # Create pixel grid
        y, x = np.mgrid[0:height, 0:width]
        pixels = np.stack([x.flatten(), y.flatten(), np.ones_like(x.flatten())], axis=0)
        
        # Reproject depth points to 3D
        depth_values = depth_map.flatten()
        
        # Skip zero depth pixels
        valid_mask = depth_values > 0
        valid_pixels = pixels[:, valid_mask]
        valid_depths = depth_values[valid_mask]
        
        # Unproject to 3D points in the first camera coordinate system
        rays = np.linalg.inv(intrinsics) @ valid_pixels
        points_3d = rays * valid_depths[np.newaxis, :]
        
        # Convert to homogeneous coordinates
        points_3d_homogeneous = np.vstack([points_3d, np.ones_like(valid_depths)])
        
        # Transform to world coordinates and then to new view
        points_new_view = extrinsic @ points_3d_homogeneous
        
        # Project to 2D in the new view
        points_2d_homogeneous = intrinsics @ points_new_view[:3, :]
        points_2d = points_2d_homogeneous[:2, :] / (points_2d_homogeneous[2, :] + 1e-10)
        
        # Round to nearest pixel and check bounds
        points_2d_int = np.round(points_2d).astype(np.int32)
        
        # Create new view image and depth map
        new_view = np.zeros((height, width, 3), dtype=np.uint8)
        new_depth = np.zeros((height, width), dtype=np.float32)
        
        # Filter points within image bounds
        mask = ((points_2d_int[0, :] >= 0) & 
                (points_2d_int[0, :] < width) & 
                (points_2d_int[1, :] >= 0) & 
                (points_2d_int[1, :] < height))
        
        # Sort points by depth (nearest first for proper occlusion handling)
        z_values = points_new_view[2, :]
        sorted_indices = np.argsort(-z_values[mask])  # Descending order
        
        # Get pixel coordinates and corresponding values
        x_coords = points_2d_int[0, mask][sorted_indices]
        y_coords = points_2d_int[1, mask][sorted_indices]
        rgb_values = image[np.unravel_index(np.where(valid_mask)[0][mask][sorted_indices], (height, width))]
        depth_values = z_values[mask][sorted_indices]
        
        # Render view with z-buffer approach
        for x_coord, y_coord, rgb_value, depth_value in zip(x_coords, y_coords, rgb_values, depth_values):
            # Only update if closer than existing depth
            if new_depth[y_coord, x_coord] == 0 or depth_value < new_depth[y_coord, x_coord]:
                new_view[y_coord, x_coord] = rgb_value
                new_depth[y_coord, x_coord] = depth_value
        
        return new_view, new_depth
    
    def _post_process_view(self, view):
        """
        Apply post-processing to improve synthetic view quality
        
        Args:
            view: Synthetic view RGB image
            
        Returns:
            Post-processed RGB image
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
        
        # Find holes (black regions)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        
        # Only proceed if there are holes to fill
        if np.sum(mask == 0) > 0:
            # Dilate the mask to expand valid regions
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            
            # Find new pixels to fill (dilated - original mask)
            fill_mask = dilated & ~mask
            
            # For each channel, perform inpainting
            result = view.copy()
            
            # Simple inpainting by averaging nearby valid pixels
            for i in range(3):  # RGB channels
                valid_regions = view[:, :, i] * (mask > 0)
                
                # Use simple blur to fill small holes
                blurred = cv2.blur(valid_regions, (5, 5))
                
                # Only apply blurred values to holes
                result[:, :, i] = np.where(fill_mask > 0, blurred, result[:, :, i])
            
            # Apply bilateral filter to smooth while preserving edges
            result = cv2.bilateralFilter(result, 9, 75, 75)
            
            return result
    
    def _create_views_collage(self, views, img_name, view_dir):
        """
        Create a collage of all synthetic views
        
        Args:
            views: List of synthetic view RGB images
            img_name: Base name for output files
            view_dir: Directory to save collage
            
        Returns:
            Path to collage image
        """
        num_views = len(views)
        
        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(num_views)))
        rows, cols = grid_size, grid_size
        
        # Get image dimensions
        height, width = views[0].shape[:2]
        
        # Create collage
        collage = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
        
        for i, view in enumerate(views):
            if i >= rows * cols:
                break
                
            row, col = i // cols, i % cols
            y_start, y_end = row * height, (row + 1) * height
            x_start, x_end = col * width, (col + 1) * width
            
            collage[y_start:y_end, x_start:x_end] = view
        
        # Save collage
        collage_path = os.path.join(view_dir, f"{img_name}_views_collage.jpg")
        cv2.imwrite(collage_path, cv2.cvtColor(collage, cv2.COLOR_RGB2BGR))
        
        return collage_path
    
    def _create_depth_collage(self, depths, img_name, view_dir):
        """
        Create a collage of all depth maps
        
        Args:
            depths: List of depth maps
            img_name: Base name for output files
            view_dir: Directory to save collage
            
        Returns:
            Paths to grayscale and colored depth collages
        """
        num_depths = len(depths)
        
        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(num_depths)))
        rows, cols = grid_size, grid_size
        
        # Get image dimensions
        height, width = depths[0].shape[:2]
        
        # Create collages (grayscale and colored)
        collage = np.zeros((rows * height, cols * width), dtype=np.float32)
        collage_colored = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
        
        for i, depth in enumerate(depths):
            if i >= rows * cols:
                break
                
            row, col = i // cols, i % cols
            y_start, y_end = row * height, (row + 1) * height
            x_start, x_end = col * width, (col + 1) * width
            
            # Normalize depth map
            normalized_depth = depth / (np.max(depth) + 1e-10)
            
            # Add to grayscale collage
            collage[y_start:y_end, x_start:x_end] = normalized_depth
            
            # Add to colored collage
            colored_depth = plt.cm.plasma(normalized_depth)[:, :, :3]
            colored_depth = (colored_depth * 255).astype(np.uint8)
            collage_colored[y_start:y_end, x_start:x_end] = colored_depth
        
        # Save collages
        collage_path = os.path.join(view_dir, f"{img_name}_depth_collage.jpg")
        collage_colored_path = os.path.join(view_dir, f"{img_name}_depth_collage_colored.jpg")
        
        # Save grayscale depth collage
        plt.imsave(collage_path, collage, cmap='gray')
        
        # Save colored depth collage
        plt.imsave(collage_colored_path, collage_colored / 255.0)
        
        return collage_path, collage_colored_path
    
    def save_synthetic_views(self, synthetic_views_data, output_dir, img_name):
        """
        Save synthetic views to disk
        
        Args:
            synthetic_views_data: Dictionary with views, depths, etc.
            output_dir: Directory to save the views
            img_name: Base name for output files
            
        Returns:
            Dictionary with paths to saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        views = synthetic_views_data["views"]
        depths = synthetic_views_data["depths"]
        
        # Save individual views and depths
        view_paths = []
        depth_paths = []
        
        for i, (view, depth) in enumerate(zip(views, depths)):
            # Save synthetic view
            view_path = os.path.join(output_dir, f"{img_name}_view_{i:02d}.jpg")
            cv2.imwrite(view_path, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
            view_paths.append(view_path)
            
            # Save depth visualization
            depth_path = os.path.join(output_dir, f"{img_name}_view_{i:02d}_depth.jpg")
            plt.imsave(depth_path, depth, cmap='plasma')
            depth_paths.append(depth_path)
        
        # Create views collage
        collage_path = self._create_views_collage(views, img_name, output_dir)
        
        # Create depth collage
        depth_collage_path, depth_collage_colored_path = self._create_depth_collage(depths, img_name, output_dir)
        
        print(f"Saved {len(views)} synthetic views to {output_dir}")
        
        return {
            "view_paths": view_paths,
            "depth_paths": depth_paths,
            "collage_path": collage_path,
            "depth_collage_path": depth_collage_path,
            "depth_collage_colored_path": depth_collage_colored_path,
            "view_dir": output_dir
        }