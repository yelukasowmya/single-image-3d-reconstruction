import os
import sys
import argparse
from pathlib import Path
from evaluate import calculate_reconstruction_accuracy

def main():
    """
    Calculate comprehensive accuracy metrics for 3D reconstruction mesh
    """
    parser = argparse.ArgumentParser(description="3D Reconstruction Accuracy Metrics")
    parser.add_argument("--mesh", required=True, help="Path to the reconstructed mesh file (.obj or .ply)")
    parser.add_argument("--output", default="data/output/evaluation", help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Check if mesh file exists
    if not os.path.exists(args.mesh):
        print(f"Error: Mesh file {args.mesh} not found")
        return 1
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n===== CALCULATING ACCURACY METRICS FOR {args.mesh} =====")
    
    # Calculate accuracy metrics
    metrics = calculate_reconstruction_accuracy(args.mesh, args.output)
    
    # Print summary to console
    img_name = Path(args.mesh).stem
    
    # Calculate overall accuracy score
    normalized_metrics = [
        metrics['manifold_ratio'],
        1.0 if metrics['watertight'] else 0.0,
        1.0 / max(1, metrics['connected_components']),
        metrics['mesh_smoothness'],
        metrics['edge_regularity'],
        metrics['symmetry_score']
    ]
    
    overall_score = sum(normalized_metrics) / len(normalized_metrics) * 100
    
    print(f"\n===== RECONSTRUCTION ACCURACY SUMMARY =====")
    print(f"Overall Accuracy Score: {overall_score:.1f}%")
    print(f"Mesh Statistics: {metrics['vertices_count']} vertices, {metrics['faces_count']} faces")
    print(f"Watertight: {'Yes' if metrics['watertight'] else 'No'}")
    print(f"Manifold Ratio: {metrics['manifold_ratio']:.4f}")
    print(f"Connected Components: {metrics['connected_components']}")
    print(f"Mesh Smoothness: {metrics['mesh_smoothness']:.4f}")
    print(f"Edge Regularity: {metrics['edge_regularity']:.4f}")
    print(f"Symmetry Score: {metrics['symmetry_score']:.4f}")
    
    print(f"\nDetailed report and visualizations saved to: {args.output}")
    print(f"  - {img_name}_accuracy_radar.png (Radar chart)")
    print(f"  - {img_name}_detailed_accuracy.png (Bar chart)")
    print(f"  - {img_name}_accuracy_report.txt (Detailed report)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())