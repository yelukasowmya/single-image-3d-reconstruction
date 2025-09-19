import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import cv2
from pathlib import Path
from main import Pipeline3D, EvaluationMetrics

def run_simple_evaluation(image_path, output_dir="data/output"):
    """
    Run a simplified evaluation on a single image
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save results
    """
    print("\n===== SINGLE IMAGE 3D RECONSTRUCTION EVALUATION =====")
    print(f"Input image: {image_path}")
    
    # Create output directory
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Initialize pipeline
    print("\n[1/4] Initializing reconstruction pipeline...")
    pipeline = Pipeline3D(output_dir=output_dir)
    
    # Process image with our full method
    print("\n[2/4] Processing image with full pipeline...")
    full_results = pipeline.process_image(
        image_path,
        depth_model="DPT_Large",
        voxel_resolution=128,
        generate_views=True,
        num_views=8
    )
    
    # Process image with simplified method (no view synthesis)
    print("\n[3/4] Processing image with simplified pipeline (no view synthesis)...")
    simple_results = pipeline.process_image(
        image_path,
        depth_model="MiDaS_small",
        voxel_resolution=64,
        generate_views=False
    )
    
    # Evaluate and compare
    print("\n[4/4] Generating evaluation metrics and comparisons...")
    
    # Evaluate full method
    full_mesh_path = full_results["mesh"]["obj"]
    full_metrics = pipeline.evaluate_reconstruction(full_mesh_path)
    
    # Evaluate simplified method
    simple_mesh_path = simple_results["mesh"]["obj"]
    simple_metrics = pipeline.evaluate_reconstruction(simple_mesh_path)
    
    # Create comparison visualizations
    create_evaluation_visuals(
        image_path,
        full_results,
        simple_results,
        full_metrics,
        simple_metrics,
        eval_dir
    )
    
    print("\n===== EVALUATION COMPLETE =====")
    print(f"Results saved to {eval_dir}")
    return full_metrics, simple_metrics

def create_evaluation_visuals(image_path, full_results, simple_results, full_metrics, simple_metrics, output_dir):
    """
    Create evaluation visualizations comparing the full and simplified methods
    
    Args:
        image_path: Path to the input image
        full_results: Results from full pipeline
        simple_results: Results from simplified pipeline
        full_metrics: Metrics for full pipeline
        simple_metrics: Metrics for simplified pipeline
        output_dir: Directory to save visualizations
    """
    img_name = Path(image_path).stem
    
    # 1. Side-by-side visualization of reconstructions
    # Load input image
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show input image
    axes[0].imshow(input_image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Show full reconstruction preview
    full_preview = cv2.imread(full_results["mesh"]["preview"])
    full_preview = cv2.cvtColor(full_preview, cv2.COLOR_BGR2RGB)
    axes[1].imshow(full_preview)
    axes[1].set_title('Full Method\n(Multi-View, 128³ Resolution)')
    axes[1].axis('off')
    
    # Show simplified reconstruction preview
    simple_preview = cv2.imread(simple_results["mesh"]["preview"])
    simple_preview = cv2.cvtColor(simple_preview, cv2.COLOR_BGR2RGB)
    axes[2].imshow(simple_preview)
    axes[2].set_title('Simplified Method\n(Single-View, 64³ Resolution)')
    axes[2].axis('off')
    
    # Save comparison visualization
    comparison_path = os.path.join(output_dir, f"{img_name}_comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_path)
    plt.close()
    
    # 2. Metrics comparison chart
    plt.figure(figsize=(10, 6))
    
    # Extract key metrics
    metrics_to_plot = ['manifold_edges_ratio', 'connected_components', 'surface_area']
    metric_labels = ['Manifold Edges', 'Component Coherence', 'Surface Detail']
    
    full_values = [
        full_metrics['mesh_quality']['manifold_edges_ratio'],
        1.0 / max(1, full_metrics['mesh_quality']['connected_components']),  # Normalize: 1 component = 1.0
        min(1.0, full_metrics['mesh_quality']['surface_area'] / 10000)  # Normalize surface area
    ]
    
    simple_values = [
        simple_metrics['mesh_quality']['manifold_edges_ratio'],
        1.0 / max(1, simple_metrics['mesh_quality']['connected_components']),
        min(1.0, simple_metrics['mesh_quality']['surface_area'] / 10000)
    ]
    
    # Create bar chart comparison
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    plt.bar(x - width/2, full_values, width, label='Full Method', color='royalblue')
    plt.bar(x + width/2, simple_values, width, label='Simplified Method', color='lightcoral')
    
    # Add value labels on top of bars
    for i, v in enumerate(full_values):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    for i, v in enumerate(simple_values):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    
    plt.ylabel('Normalized Score')
    plt.title('Reconstruction Quality Comparison')
    plt.xticks(x, metric_labels)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save metrics comparison
    metrics_path = os.path.join(output_dir, f"{img_name}_metrics.png")
    plt.tight_layout()
    plt.savefig(metrics_path)
    plt.close()
    
    # 3. Create depth map comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show full method depth map
    full_depth = cv2.imread(full_results["depth"]["colored_depth"])
    full_depth = cv2.cvtColor(full_depth, cv2.COLOR_BGR2RGB)
    axes[0].imshow(full_depth)
    axes[0].set_title('Depth Map (DPT_Large)')
    axes[0].axis('off')
    
    # Show simplified method depth map
    simple_depth = cv2.imread(simple_results["depth"]["colored_depth"])
    simple_depth = cv2.cvtColor(simple_depth, cv2.COLOR_BGR2RGB)
    axes[1].imshow(simple_depth)
    axes[1].set_title('Depth Map (MiDaS_small)')
    axes[1].axis('off')
    
    # Save depth comparison
    depth_path = os.path.join(output_dir, f"{img_name}_depth_comparison.png")
    plt.tight_layout()
    plt.savefig(depth_path)
    plt.close()
    
    # 4. Create a text report with all metrics
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
    === 3D RECONSTRUCTION EVALUATION REPORT ===
    Input Image: {image_path}
    Date: {current_time}
    
    === FULL METHOD (Multi-View, 128³ Resolution) ===
    Mesh Quality:
    - Watertight: {full_metrics['mesh_quality']['watertight']}
    - Connected Components: {full_metrics['mesh_quality']['connected_components']}
    - Manifold Edges Ratio: {full_metrics['mesh_quality']['manifold_edges_ratio']:.4f}
    - Surface Area: {full_metrics['mesh_quality']['surface_area']:.2f}
    - Volume: {full_metrics['mesh_quality']['volume']:.2f} (if watertight)
    - Mean Edge Length: {full_metrics['mesh_quality']['mean_edge_length']:.4f}
    
    === SIMPLIFIED METHOD (Single-View, 64³ Resolution) ===
    Mesh Quality:
    - Watertight: {simple_metrics['mesh_quality']['watertight']}
    - Connected Components: {simple_metrics['mesh_quality']['connected_components']}
    - Manifold Edges Ratio: {simple_metrics['mesh_quality']['manifold_edges_ratio']:.4f}
    - Surface Area: {simple_metrics['mesh_quality']['surface_area']:.2f}
    - Volume: {simple_metrics['mesh_quality']['volume']:.2f} (if watertight)
    - Mean Edge Length: {simple_metrics['mesh_quality']['mean_edge_length']:.4f}
    
    === CONCLUSION ===
    The multi-view approach with higher resolution provides a {(full_metrics['mesh_quality']['manifold_edges_ratio'] / simple_metrics['mesh_quality']['manifold_edges_ratio'] - 1) * 100:.1f}% improvement
    in mesh quality (manifold edges) compared to the simplified approach.
    """
    
    # Save report
    report_path = os.path.join(output_dir, f"{img_name}_evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_evaluate.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found")
        sys.exit(1)
        
    run_simple_evaluation(image_path)