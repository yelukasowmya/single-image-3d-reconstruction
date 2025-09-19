import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import cv2
from pathlib import Path
from scipy.spatial import cKDTree
from main import Pipeline3D, EvaluationMetrics

# Add explicit print for debugging
print("Starting evaluation script...")

def benchmark_pipeline(input_dir, output_dir, models=None, resolutions=None, segmentation="maskrcnn"):
    """
    Benchmark the pipeline with different parameters and create performance comparison
    
    Args:
        input_dir: Directory with test images
        output_dir: Directory to save results
        models: List of depth models to test
        resolutions: List of voxel resolutions to test
        segmentation: Segmentation method to use
    """
    print(f"Starting benchmark with input directory: {input_dir}")
    
    if models is None:
        models = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
    
    if resolutions is None:
        resolutions = [64, 128, 256]
    
    # Initialize results dictionaries
    timing_results = {}
    quality_results = {}
    
    # Find all images in the input directory
    image_paths = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_paths.extend(list(Path(input_dir).glob(f"*{ext}")))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images for benchmarking")
    
    # Create subfolder for benchmark results
    benchmark_dir = os.path.join(output_dir, "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Iterate over all model combinations
    for model in models:
        timing_results[model] = {}
        quality_results[model] = {}
        
        for res in resolutions:
            print(f"\n=== Testing Model: {model}, Resolution: {res} ===")
            
            model_key = f"{model}_{res}"
            timing_results[model][res] = {}
            quality_results[model][res] = {}
            
            # Process first image for timing analysis
            test_image = str(image_paths[0])
            
            # Initialize pipeline
            pipeline = Pipeline3D(input_dir=input_dir, 
                                  output_dir=os.path.join(benchmark_dir, model_key))
            
            # Time each major step
            # 1. Segmentation
            start_time = time.time()
            seg_results = pipeline.segment_objects(test_image, method=segmentation)
            seg_time = time.time() - start_time
            
            # 2. Depth Estimation
            start_time = time.time()
            depth_results = pipeline.generate_depth_map(seg_results["masks"], model_type=model)
            depth_time = time.time() - start_time
            
            # 3. Voxel Grid Creation
            start_time = time.time()
            mask = cv2.imread(seg_results["masks"], cv2.IMREAD_GRAYSCALE) / 255.0
            voxel_grid = pipeline.create_voxel_grid(depth_results["depth_data"], mask, resolution=res)
            voxel_time = time.time() - start_time
            
            # 4. Mesh Creation
            start_time = time.time()
            mesh = pipeline.create_mesh_from_voxels(voxel_grid)
            mesh_time = time.time() - start_time
            
            # 5. Texture Application
            start_time = time.time()
            textured_mesh = pipeline.apply_texture_to_mesh(mesh, test_image, seg_results["masks"])
            texture_time = time.time() - start_time
            
            # Total time
            total_time = seg_time + depth_time + voxel_time + mesh_time + texture_time
            
            # Store timing results
            timing_results[model][res] = {
                "segmentation": seg_time,
                "depth": depth_time,
                "voxel": voxel_time,
                "mesh": mesh_time,
                "texture": texture_time,
                "total": total_time
            }
            
            # Evaluate mesh quality
            quality_metrics = pipeline.evaluate_reconstruction(textured_mesh)
            quality_results[model][res] = quality_metrics
            
            # Save the mesh for visual comparison
            result_mesh_path = os.path.join(benchmark_dir, f"{model_key}_result.obj")
            textured_mesh.export(result_mesh_path)
            
            # Also generate a preview
            preview_path = os.path.join(benchmark_dir, f"{model_key}_preview.png")
            pipeline._create_mesh_preview(textured_mesh, preview_path)
    
    # Create summary visualizations
    visualize_benchmark_results(timing_results, quality_results, benchmark_dir)
    
    return timing_results, quality_results

def visualize_benchmark_results(timing_results, quality_results, output_dir):
    """
    Create visualizations of benchmark results
    
    Args:
        timing_results: Dictionary with timing measurements
        quality_results: Dictionary with quality metrics
        output_dir: Directory to save visualizations
    """
    # 1. Timing comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Prepare data for grouped bar chart
    models = list(timing_results.keys())
    resolutions = list(timing_results[models[0]].keys())
    
    # Position of bars on x-axis
    bar_width = 0.15
    index = np.arange(len(models))
    
    # Colors for different resolutions
    colors = ['skyblue', 'lightgreen', 'salmon', 'lightpurple', 'gold']
    
    # Plot total times for each model+resolution
    for i, res in enumerate(resolutions):
        total_times = [timing_results[model][res]['total'] for model in models]
        plt.bar(index + i * bar_width, total_times, bar_width,
                label=f'Resolution {res}', color=colors[i % len(colors)])
    
    plt.xlabel('Depth Model')
    plt.ylabel('Total Processing Time (seconds)')
    plt.title('Processing Time Comparison by Model and Resolution')
    plt.xticks(index + bar_width * (len(resolutions) - 1) / 2, models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save timing comparison
    timing_path = os.path.join(output_dir, "timing_comparison.png")
    plt.tight_layout()
    plt.savefig(timing_path)
    plt.close()
    
    # 2. Quality metrics comparison
    plt.figure(figsize=(14, 8))
    
    # For each model, plot a line showing how mesh quality varies with resolution
    for model in models:
        resolutions = sorted(list(quality_results[model].keys()))
        
        # Extract a quality metric (using manifold edges ratio as example)
        quality_values = [quality_results[model][res]['mesh_quality']['manifold_edges_ratio'] 
                         for res in resolutions]
        
        plt.plot(resolutions, quality_values, marker='o', label=model)
    
    plt.xlabel('Voxel Resolution')
    plt.ylabel('Manifold Edges Ratio')
    plt.title('Mesh Quality vs. Resolution by Model')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save quality comparison
    quality_path = os.path.join(output_dir, "quality_comparison.png")
    plt.tight_layout()
    plt.savefig(quality_path)
    plt.close()
    
    # 3. Pipeline step breakdown visualization
    plt.figure(figsize=(10, 8))
    
    # Choose a specific configuration to show breakdown
    selected_model = models[0]
    selected_res = resolutions[-1]  # Highest resolution
    
    # Get step times
    steps = ['segmentation', 'depth', 'voxel', 'mesh', 'texture']
    step_times = [timing_results[selected_model][selected_res][step] for step in steps]
    
    # Create pie chart
    plt.pie(step_times, labels=steps, autopct='%1.1f%%',
            startangle=90, shadow=True)
    plt.axis('equal')
    plt.title(f'Processing Time Breakdown\n{selected_model} at {selected_res}x{selected_res}x{selected_res}')
    
    # Save pie chart
    breakdown_path = os.path.join(output_dir, "time_breakdown.png")
    plt.tight_layout()
    plt.savefig(breakdown_path)
    plt.close()
    
    print(f"Benchmark visualizations saved to {output_dir}")

def compare_with_baselines(test_image, output_dir):
    """
    Compare our method with baseline methods and create comparison visuals
    
    Args:
        test_image: Path to test image
        output_dir: Directory to save results
    """
    # Create output directory
    baselines_dir = os.path.join(output_dir, "baselines")
    os.makedirs(baselines_dir, exist_ok=True)
    
    # Initialize our pipeline
    our_pipeline = Pipeline3D(output_dir=os.path.join(baselines_dir, "ours"))
    
    # Process with our method
    print("\n=== Processing with Our Method ===")
    our_results = our_pipeline.process_image(
        test_image,
        depth_model="DPT_Large",
        voxel_resolution=128,
        generate_views=True,
        num_views=8
    )
    
    # Create a simple baseline: basic shape from silhouette
    print("\n=== Processing with Baseline Method: Shape from Silhouette ===")
    baseline_pipeline = Pipeline3D(output_dir=os.path.join(baselines_dir, "baseline"))
    
    # For baseline, use segmentation but skip view synthesis
    baseline_seg = baseline_pipeline.segment_objects(test_image)
    baseline_depth = baseline_pipeline.generate_depth_map(baseline_seg["masks"], model_type="MiDaS_small")
    
    # Create a simplified voxel grid (depth projection without smoothing or view synthesis)
    mask = cv2.imread(baseline_seg["masks"], cv2.IMREAD_GRAYSCALE) / 255.0
    depth_map = baseline_pipeline.load_depth_map(baseline_depth["depth"])
    
    # Create a simple voxelization directly from depth
    simplified_grid = np.zeros((64, 64, 64), dtype=np.float32)
    depth_resized = cv2.resize(depth_map, (64, 64))
    
    for x in range(64):
        for y in range(64):
            if depth_resized[y, x] > 0.01:
                z_val = int(depth_resized[y, x] * 64)
                if 0 <= z_val < 64:
                    simplified_grid[y, x, z_val] = 1.0
    
    # Create mesh from simplified grid
    baseline_mesh = baseline_pipeline.create_mesh_from_voxels(simplified_grid, threshold=0.5)
    
    # Save baseline mesh
    baseline_mesh_path = os.path.join(baselines_dir, "baseline_result.obj")
    baseline_mesh.export(baseline_mesh_path)
    
    # Create comparison visualization
    compare_mesh_visualizations(
        our_results["mesh"]["obj"],
        baseline_mesh_path,
        test_image,
        os.path.join(baselines_dir, "comparison.png")
    )
    
    # Evaluate both methods
    our_metrics = our_pipeline.evaluate_reconstruction(our_results["mesh"]["obj"])
    baseline_metrics = baseline_pipeline.evaluate_reconstruction(baseline_mesh)
    
    # Create metrics comparison
    create_metrics_comparison(our_metrics, baseline_metrics, baselines_dir)
    
    return our_metrics, baseline_metrics

def compare_mesh_visualizations(our_mesh_path, baseline_mesh_path, input_image_path, output_path):
    """
    Create a visual comparison between our method and a baseline
    
    Args:
        our_mesh_path: Path to our reconstruction mesh
        baseline_mesh_path: Path to baseline mesh
        input_image_path: Path to input image
        output_path: Path to save comparison visualization
    """
    # Load meshes
    our_mesh = trimesh.load(our_mesh_path)
    baseline_mesh = trimesh.load(baseline_mesh_path)
    
    # Load input image
    input_image = cv2.imread(input_image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show input image
    axes[0].imshow(input_image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Render our mesh
    our_scene = trimesh.Scene(our_mesh)
    our_render = our_scene.save_image(resolution=[512, 512], visible=True)
    our_render = plt.imread(our_render, format='png')
    
    axes[1].imshow(our_render)
    axes[1].set_title('Our Method')
    axes[1].axis('off')
    
    # Render baseline mesh
    baseline_scene = trimesh.Scene(baseline_mesh)
    baseline_render = baseline_scene.save_image(resolution=[512, 512], visible=True)
    baseline_render = plt.imread(baseline_render, format='png')
    
    axes[2].imshow(baseline_render)
    axes[2].set_title('Baseline Method')
    axes[2].axis('off')
    
    # Save comparison
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Visual comparison saved to {output_path}")

def create_metrics_comparison(our_metrics, baseline_metrics, output_dir):
    """
    Create a visual comparison of evaluation metrics
    
    Args:
        our_metrics: Metrics for our method
        baseline_metrics: Metrics for baseline method
        output_dir: Directory to save visualization
    """
    # Extract key metrics
    metrics_to_plot = ['manifold_edges_ratio', 'connected_components', 'surface_area']
    
    our_values = [
        our_metrics['mesh_quality']['manifold_edges_ratio'],
        1.0 / max(1, our_metrics['mesh_quality']['connected_components']),  # Normalize: 1 component = 1.0
        min(1.0, our_metrics['mesh_quality']['surface_area'] / 10000)  # Normalize surface area
    ]
    
    baseline_values = [
        baseline_metrics['mesh_quality']['manifold_edges_ratio'],
        1.0 / max(1, baseline_metrics['mesh_quality']['connected_components']),
        min(1.0, baseline_metrics['mesh_quality']['surface_area'] / 10000)
    ]
    
    # Create bar chart comparison
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    plt.bar(x - width/2, our_values, width, label='Our Method')
    plt.bar(x + width/2, baseline_values, width, label='Baseline')
    
    plt.ylabel('Normalized Score')
    plt.title('Reconstruction Quality Comparison')
    plt.xticks(x, ['Manifold Edges', 'Component Coherence', 'Surface Detail'])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save comparison
    metrics_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.tight_layout()
    plt.savefig(metrics_path)
    plt.close()
    
    print(f"Metrics comparison saved to {metrics_path}")

def calculate_reconstruction_accuracy(mesh_path, output_dir="data/output/evaluation"):
    """
    Calculate detailed accuracy metrics for a 3D reconstruction
    
    Args:
        mesh_path: Path to the reconstructed mesh
        output_dir: Directory to save results
        
    Returns:
        Dictionary with accuracy metrics
    """
    print(f"\n===== CALCULATING RECONSTRUCTION ACCURACY =====")
    print(f"Mesh: {mesh_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reconstructed mesh
    mesh = trimesh.load(mesh_path)
    img_name = Path(mesh_path).stem
    
    # Calculate mesh statistics
    vertices_count = len(mesh.vertices)
    faces_count = len(mesh.faces)
    
    # Calculate self-consistency metrics
    quality_metrics = EvaluationMetrics.mesh_self_consistency(mesh)
    
    # Calculate geometric accuracy metrics
    # Sample points from the mesh surface
    points = mesh.sample(10000)
    
    # Get mesh volume for completeness score
    if quality_metrics["watertight"]:
        volume = mesh.volume
        density = vertices_count / volume  # Vertices per unit volume
    else:
        volume = 0
        density = 0
    
    # Calculate mesh regularity (standard deviation of edge lengths)
    edge_regularity = 1.0 - min(1.0, quality_metrics["std_edge_length"] / quality_metrics["mean_edge_length"])
    
    # Calculate mesh smoothness (based on dihedral angles)
    face_normals = mesh.face_normals
    mesh_smoothness = 0.0
    
    # Calculate dihedral angles between adjacent faces
    edges_unique = mesh.edges_unique
    edges_face_count = np.zeros(len(edges_unique))
    face_adjacency = mesh.face_adjacency
    
    if len(face_adjacency) > 0:
        face_pair_normals = face_normals[face_adjacency]
        # Dot product of adjacent face normals
        adjacent_normals_dot = np.sum(face_pair_normals[:, 0, :] * face_pair_normals[:, 1, :], axis=1)
        # Convert to angles in degrees
        adjacent_angles = np.degrees(np.arccos(np.clip(adjacent_normals_dot, -1.0, 1.0)))
        # Normalize smoothness (0 = very angular, 1 = very smooth)
        mesh_smoothness = 1.0 - np.mean(np.abs(adjacent_angles - 90) / 90)

    # Calculate mesh symmetry
    # For a simple approach, we compare left and right halves of the mesh
    center_x = np.mean(mesh.vertices[:, 0])
    left_vertices = mesh.vertices[mesh.vertices[:, 0] < center_x]
    right_vertices = mesh.vertices[mesh.vertices[:, 0] > center_x]
    
    # Reflect right vertices across the YZ plane
    if len(right_vertices) > 0:
        reflected_right = right_vertices.copy()
        reflected_right[:, 0] = 2 * center_x - reflected_right[:, 0]
        
        # If we have points on both sides, calculate symmetry
        if len(left_vertices) > 0:
            # Build KD-tree for left vertices
            if len(left_vertices) > 0 and len(reflected_right) > 0:
                left_tree = cKDTree(left_vertices)
                # Find distances from reflected right vertices to nearest left vertices
                distances, _ = left_tree.query(reflected_right)
                # Normalize by mesh bounding box diagonal
                bbox_diagonal = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
                symmetry_score = 1.0 - min(1.0, np.mean(distances) / (0.1 * bbox_diagonal))
            else:
                symmetry_score = 0.0
        else:
            symmetry_score = 0.0
    else:
        symmetry_score = 0.0
    
    # Compile all accuracy metrics
    accuracy_metrics = {
        "vertices_count": vertices_count,
        "faces_count": faces_count,
        "watertight": quality_metrics["watertight"],
        "manifold_ratio": quality_metrics["manifold_edges_ratio"],
        "connected_components": quality_metrics["connected_components"],
        "surface_area": quality_metrics["surface_area"],
        "volume": volume,
        "vertex_density": density,
        "edge_regularity": edge_regularity,
        "mesh_smoothness": mesh_smoothness,
        "symmetry_score": symmetry_score
    }
    
    # Create visual representation of accuracy metrics
    create_accuracy_visualization(accuracy_metrics, img_name, output_dir)
    
    # Create detailed accuracy report
    create_accuracy_report(accuracy_metrics, mesh_path, output_dir)
    
    print(f"Accuracy assessment complete. Results saved to {output_dir}")
    return accuracy_metrics

def create_accuracy_visualization(metrics, img_name, output_dir):
    """
    Create visual representation of accuracy metrics
    
    Args:
        metrics: Dictionary with accuracy metrics
        img_name: Base name for the output file
        output_dir: Directory to save visualization
    """
    # Create radar chart for key accuracy metrics
    plt.figure(figsize=(10, 8))
    
    # Define metrics to plot
    metric_names = [
        'Manifold Quality', 
        'Watertight', 
        'Structural Coherence',
        'Mesh Smoothness', 
        'Edge Regularity',
        'Symmetry'
    ]
    
    # Convert metrics to normalized values between 0 and 1
    metric_values = [
        metrics['manifold_ratio'],
        1.0 if metrics['watertight'] else 0.0,
        1.0 / max(1, metrics['connected_components']),
        metrics['mesh_smoothness'],
        metrics['edge_regularity'],
        metrics['symmetry_score']
    ]
    
    # Number of variables
    N = len(metric_names)
    
    # Create angle array for radar chart (in radians)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Make the plot circular by appending the first value to the end
    metric_values += [metric_values[0]]
    angles += [angles[0]]
    metric_names += [metric_names[0]]
    
    # Create the plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metric_names[:-1], size=12)
    
    # Draw the y-axis labels (0-100%)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot the data
    ax.plot(angles, metric_values, linewidth=2, linestyle='solid')
    
    # Fill the area
    ax.fill(angles, metric_values, alpha=0.25)
    
    # Add title
    plt.title("3D Reconstruction Accuracy Assessment", size=15, y=1.1)
    
    # Calculate overall accuracy score (average of all metrics)
    overall_score = round(100 * np.mean(metric_values[:-1]))  # Exclude the duplicate value
    
    # Add overall score in the center
    plt.text(0, 0, f"{overall_score}%", 
             horizontalalignment='center',
             verticalalignment='center', 
             size=24, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the figure
    accuracy_viz_path = os.path.join(output_dir, f"{img_name}_accuracy_radar.png")
    plt.tight_layout()
    plt.savefig(accuracy_viz_path)
    plt.close()
    
    # Create bar chart with detailed metrics
    plt.figure(figsize=(12, 6))
    
    # Define additional metrics for bar chart
    detailed_metrics = [
        ('Manifold Ratio', metrics['manifold_ratio']),
        ('Watertight', 1.0 if metrics['watertight'] else 0.0),
        ('Structural Coherence', 1.0 / max(1, metrics['connected_components'])),
        ('Mesh Smoothness', metrics['mesh_smoothness']),
        ('Edge Regularity', metrics['edge_regularity']),
        ('Symmetry Score', metrics['symmetry_score'])
    ]
    
    # Extract names and values
    names = [m[0] for m in detailed_metrics]
    values = [m[1] for m in detailed_metrics]
    
    # Create bars
    bars = plt.bar(names, values, color='royalblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Add labels and title
    plt.ylabel('Score (0-1)')
    plt.title('Detailed Reconstruction Accuracy Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    detail_path = os.path.join(output_dir, f"{img_name}_detailed_accuracy.png")
    plt.tight_layout()
    plt.savefig(detail_path)
    plt.close()

def create_accuracy_report(metrics, mesh_path, output_dir):
    """
    Create detailed text report with accuracy metrics
    
    Args:
        metrics: Dictionary with accuracy metrics
        mesh_path: Path to the original mesh file
        output_dir: Directory to save the report
    """
    img_name = Path(mesh_path).stem
    
    # Calculate overall accuracy score (average of normalized metrics)
    normalized_metrics = [
        metrics['manifold_ratio'],
        1.0 if metrics['watertight'] else 0.0,
        1.0 / max(1, metrics['connected_components']),
        metrics['mesh_smoothness'],
        metrics['edge_regularity'],
        metrics['symmetry_score']
    ]
    
    overall_score = np.mean(normalized_metrics) * 100
    
    # Create report text
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
===============================================
  COMPREHENSIVE 3D RECONSTRUCTION ACCURACY REPORT
===============================================

Date: {current_time}
Mesh: {mesh_path}

OVERALL ACCURACY SCORE: {overall_score:.1f}%

---------------------------------
MESH STATISTICS:
---------------------------------
- Vertices: {metrics['vertices_count']}
- Faces: {metrics['faces_count']}
- Surface Area: {metrics['surface_area']:.2f}
- Volume: {metrics['volume']:.2f} (if watertight)
- Vertex Density: {metrics['vertex_density']:.2f} vertices per unit volume

---------------------------------
TOPOLOGICAL QUALITY:
---------------------------------
- Watertight: {'Yes' if metrics['watertight'] else 'No'} (Score: {1.0 if metrics['watertight'] else 0.0:.2f})
- Manifold Ratio: {metrics['manifold_ratio']:.4f}
- Connected Components: {metrics['connected_components']} (Score: {1.0 / max(1, metrics['connected_components']):.2f})

---------------------------------
GEOMETRIC QUALITY:
---------------------------------
- Mesh Smoothness: {metrics['mesh_smoothness']:.4f}
- Edge Regularity: {metrics['edge_regularity']:.4f}
- Symmetry Score: {metrics['symmetry_score']:.4f}

---------------------------------
INTERPRETATION GUIDE:
---------------------------------
- Scores range from 0 (worst) to 1 (best)
- Watertight: Indicates if the mesh has no holes
- Manifold Ratio: Higher values indicate better mesh integrity
- Connected Components: Ideally 1 (single connected mesh)
- Mesh Smoothness: Higher values indicate smoother surface transitions
- Edge Regularity: Higher values indicate more uniform mesh density
- Symmetry Score: Higher values indicate better left-right symmetry

---------------------------------
RECOMMENDATIONS:
---------------------------------
"""
    
    # Add recommendations based on specific metrics
    if metrics['manifold_ratio'] < 0.8:
        report += "- Consider improving mesh manifold quality by adjusting the marching cubes threshold\n"
    
    if not metrics['watertight']:
        report += "- Mesh has holes; consider post-processing to make it watertight\n"
    
    if metrics['connected_components'] > 1:
        report += f"- Mesh has {metrics['connected_components']} separate parts; consider keeping only the largest component\n"
    
    if metrics['mesh_smoothness'] < 0.6:
        report += "- Apply additional mesh smoothing to improve surface quality\n"
    
    if metrics['edge_regularity'] < 0.5:
        report += "- Consider mesh regularization to improve edge length consistency\n"
    
    if metrics['symmetry_score'] < 0.5:
        report += "- Left-right symmetry could be improved through post-processing\n"
    
    if overall_score >= 80:
        report += "\nOVERALL: Excellent reconstruction quality suitable for most applications"
    elif overall_score >= 60:
        report += "\nOVERALL: Good reconstruction quality with minor issues"
    elif overall_score >= 40:
        report += "\nOVERALL: Fair reconstruction quality that may need improvements"
    else:
        report += "\nOVERALL: Poor reconstruction quality requiring significant improvements"
    
    # Save report
    report_path = os.path.join(output_dir, f"{img_name}_accuracy_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

def main():
    """
    Main entry point for the evaluation script
    """
    parser = argparse.ArgumentParser(description="3D Reconstruction Evaluation")
    parser.add_argument("--input", default="data/input", help="Input directory containing images")
    parser.add_argument("--output", default="data/output", help="Output directory for results")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark across different models and resolutions")
    parser.add_argument("--compare", default=None, help="Path to an image for baseline comparison")
    
    args = parser.parse_args()
    
    if args.benchmark:
        print("\n=== Running Performance Benchmark ===")
        benchmark_pipeline(args.input, args.output)
    
    if args.compare:
        if not os.path.exists(args.compare):
            print(f"Error: Image {args.compare} not found")
            return
            
        print("\n=== Running Baseline Comparison ===")
        compare_with_baselines(args.compare, args.output)
    
    if not args.benchmark and not args.compare:
        print("No evaluation actions specified. Use --benchmark or --compare.")
        print("Run with -h for more information.")

if __name__ == "__main__":
    main()