import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure
import trimesh
from pathlib import Path


class VoxelMeshProcessor:
    """
    Handles voxel grid creation and mesh generation
    """
    
    def __init__(self, output_dir="data/output"):
        """
        Initialize the voxel and mesh processor
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = output_dir
        self.voxel_dir = os.path.join(output_dir, "voxels")
        self.mesh_dir = os.path.join(output_dir, "meshes")
        
        os.makedirs(self.voxel_dir, exist_ok=True)
        os.makedirs(self.mesh_dir, exist_ok=True)
    
    def create_voxel_grid(self, views, depths, intrinsics, extrinsics, img_name, resolution=128):
        """
        Create a voxel grid using multi-view depth maps
        
        Args:
            views: List of synthetic view images
            depths: List of corresponding depth maps
            intrinsics: Camera intrinsics matrix
            extrinsics: List of camera extrinsic matrices
            img_name: Base name for output files
            resolution: Voxel grid resolution
            
        Returns:
            Dict with paths to output files and voxel grid data
        """
        print(f"--- Creating Multi-View Voxel Grid ---")
        
        # Initialize voxel grid
        voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        # Voxel grid dimensions in world space
        grid_min = np.array([-1.0, -1.0, -1.0])
        grid_max = np.array([1.0, 1.0, 1.0])
        
        # Create meshgrid for voxel centers
        grid_range = np.linspace(-1.0, 1.0, resolution)
        yy, xx, zz = np.meshgrid(grid_range, grid_range, grid_range)
        
        # Reshape to Nx3 points
        grid_points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        
        # Process each view
        for i, (view, depth, extrinsic) in enumerate(zip(views, depths, extrinsics)):
            print(f"Integrating synthetic view {i+1}/{len(views)}...")
            
            # Create voxel grid from current view
            voxel_grid_view = self._create_voxel_grid_from_view(
                depth, intrinsics, extrinsic, resolution, grid_min, grid_max)
            
            # Combine with existing voxel grid (average)
            voxel_grid = np.maximum(voxel_grid, voxel_grid_view)
        
        # Save voxel grid
        voxel_path = os.path.join(self.voxel_dir, f"{img_name}_voxels.npy")
        np.save(voxel_path, voxel_grid)
        
        # Visualize voxel grid slices
        slices_path = self._visualize_voxel_slices(voxel_grid, img_name)
        
        return {
            "voxel_grid": voxel_grid,
            "voxel_path": voxel_path,
            "slices_path": slices_path
        }
    
    def _create_voxel_grid_from_view(self, depth, intrinsics, extrinsic, resolution, grid_min, grid_max):
        """
        Create a voxel grid from a single view
        
        Args:
            depth: Depth map
            intrinsics: Camera intrinsics matrix
            extrinsic: Camera extrinsic matrix
            resolution: Voxel grid resolution
            grid_min: Minimum coordinates of voxel grid
            grid_max: Maximum coordinates of voxel grid
            
        Returns:
            Voxel grid (numpy array)
        """
        print(f"Creating voxel grid with resolution {resolution}...")
        
        # Initialize voxel grid
        voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        # Create grid of 3D points
        grid_range = np.linspace(0.0, 1.0, resolution)
        grid_size = (grid_max - grid_min)
        
        # Project each voxel into camera view
        camera_matrix = intrinsics @ extrinsic[:3, :]
        
        # Generate all voxel centers
        xx, yy, zz = np.meshgrid(
            np.linspace(grid_min[0], grid_max[0], resolution),
            np.linspace(grid_min[1], grid_max[1], resolution),
            np.linspace(grid_min[2], grid_max[2], resolution)
        )
        voxel_centers = np.stack([xx.flatten(), yy.flatten(), zz.flatten(), np.ones_like(xx.flatten())], axis=0)
        
        # Project voxel centers to image space
        projected = camera_matrix @ voxel_centers
        projected = projected / (projected[2:3, :] + 1e-10)
        
        # Get pixel coordinates
        pix_x = np.round(projected[0, :]).astype(np.int32)
        pix_y = np.round(projected[1, :]).astype(np.int32)
        
        # Find pixels within image bounds
        height, width = depth.shape
        valid_idx = (pix_x >= 0) & (pix_x < width) & (pix_y >= 0) & (pix_y < height)
        
        # Get depth at each pixel
        valid_pix_x = pix_x[valid_idx]
        valid_pix_y = pix_y[valid_idx]
        
        # Get depth values from depth map
        depth_values = depth[valid_pix_y, valid_pix_x]
        
        # Compare projected depth with voxel depth
        voxel_depths = projected[2, valid_idx]
        
        # Occupancy determined by how close voxel depth is to observed depth
        occupancy = np.exp(-np.abs(voxel_depths - depth_values) * 10.0)
        
        # Update voxel grid
        flat_indices = np.where(valid_idx)[0]
        grid_indices = np.unravel_index(flat_indices, (resolution, resolution, resolution))
        
        # Assign occupancy values to voxel grid
        voxel_grid[grid_indices] = occupancy
        
        return voxel_grid
    
    def _visualize_voxel_slices(self, voxel_grid, img_name, num_slices=5):
        """
        Visualize voxel grid slices
        
        Args:
            voxel_grid: 3D voxel grid
            img_name: Base name for output files
            num_slices: Number of slices to visualize
            
        Returns:
            Path to visualization image
        """
        resolution = voxel_grid.shape[0]
        slice_indices = np.linspace(0, resolution-1, num_slices).astype(np.int32)
        
        fig, axs = plt.subplots(1, num_slices, figsize=(15, 3))
        
        for i, idx in enumerate(slice_indices):
            axs[i].imshow(voxel_grid[idx, :, :], cmap='jet')
            axs[i].set_title(f"Slice {idx}")
            axs[i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        slice_path = os.path.join(self.voxel_dir, f"{img_name}_voxel_slices.png")
        plt.savefig(slice_path, dpi=150)
        plt.close()
        
        return slice_path
    
    def create_mesh_from_voxels(self, voxel_grid, img_name, threshold=0.3):
        """
        Create mesh from voxel grid using Marching Cubes
        
        Args:
            voxel_grid: 3D voxel grid
            img_name: Base name for output files
            threshold: Threshold for isosurface extraction
            
        Returns:
            Dict with paths to output files and mesh object
        """
        print(f"Creating mesh from voxel grid using Marching Cubes (threshold={threshold})...")
        
        # Extract isosurface using Marching Cubes
        verts, faces, normals, values = measure.marching_cubes(voxel_grid, threshold)
        
        # Scale vertices to range [-1, 1]
        resolution = voxel_grid.shape[0]
        verts = verts / resolution * 2.0 - 1.0
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)
        
        # Apply smoothing
        mesh = mesh.smooth(method='laplacian', iterations=3)
        
        # Save mesh to OBJ file
        mesh_path = os.path.join(self.mesh_dir, f"{img_name}.obj")
        mesh.export(mesh_path)
        
        print(f"Mesh created with {len(verts)} vertices and {len(faces)} faces")
        print(f"Saved to {mesh_path}")
        
        return {
            "mesh": mesh,
            "mesh_path": mesh_path
        }
    
    def apply_texture_to_mesh(self, mesh, image_path, img_name, mask_path=None):
        """
        Apply texture from the original image to the mesh
        
        Args:
            mesh: Trimesh object
            image_path: Path to the original image for texturing
            img_name: Base name for output files
            mask_path: Optional path to the mask image
            
        Returns:
            Dict with paths to output files
        """
        print(f"Applying texture from {image_path} to mesh...")
        
        # Load the original image for texturing
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to load texture image {image_path}. Using default coloring.")
            return self._apply_default_coloring(mesh, img_name)
            
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
        
        # Save textured mesh
        obj_path = os.path.join(self.mesh_dir, f"{img_name}_textured.obj")
        ply_path = os.path.join(self.mesh_dir, f"{img_name}_textured.ply")
        
        textured_mesh.export(obj_path)
        textured_mesh.export(ply_path)
        
        # Create a preview image
        try:
            # Try using built-in renderer
            preview_path = os.path.join(self.mesh_dir, f"{img_name}_textured_preview.png")
            scene = trimesh.Scene([textured_mesh])
            png = scene.save_image(resolution=[640, 480])
            with open(preview_path, 'wb') as f:
                f.write(png)
            print(f"Saved mesh preview: {preview_path}")
        except Exception as e:
            # Fall back to matplotlib
            print(f"Error with trimesh renderer: {e}")
            preview_path = os.path.join(self.mesh_dir, f"{img_name}_textured_preview.png")
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the 3D mesh using matplotlib
            ax.plot_trisurf(
                textured_mesh.vertices[:, 0], 
                textured_mesh.vertices[:, 1], 
                textured_mesh.vertices[:, 2],
                triangles=textured_mesh.faces,
                cmap='viridis'
            )
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.tight_layout()
            plt.savefig(preview_path)
            plt.close()
            print(f"Saved mesh preview using matplotlib: {preview_path}")
        
        print(f"Textured mesh saved to {obj_path} and {ply_path}")
        
        return {
            "textured_mesh": textured_mesh,
            "obj_path": obj_path,
            "ply_path": ply_path,
            "preview_path": preview_path
        }
    
    def _apply_default_coloring(self, mesh, img_name):
        """
        Apply default coloring to the mesh (fallback)
        
        Args:
            mesh: Trimesh object
            img_name: Base name for output files
            
        Returns:
            Dict with paths to output files
        """
        # Create vertex colors using vertex normals
        normals = mesh.vertex_normals
        
        # Normalize normals to [0, 1] range for coloring
        normals = (normals + 1) / 2
        
        # Set alpha channel
        vertex_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
        vertex_colors[:, :3] = (normals * 255).astype(np.uint8)
        vertex_colors[:, 3] = 255  # Full opacity
        
        # Create a new mesh with vertex colors
        colored_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            vertex_colors=vertex_colors
        )
        
        # Save colored mesh
        obj_path = os.path.join(self.mesh_dir, f"{img_name}_colored.obj")
        ply_path = os.path.join(self.mesh_dir, f"{img_name}_colored.ply")
        
        colored_mesh.export(obj_path)
        colored_mesh.export(ply_path)
        
        print(f"Applied default coloring to mesh")
        print(f"Saved to {obj_path} and {ply_path}")
        
        return {
            "colored_mesh": colored_mesh,
            "obj_path": obj_path,
            "ply_path": ply_path
        }