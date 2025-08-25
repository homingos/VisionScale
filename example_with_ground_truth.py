#!/usr/bin/env python3
"""
Example: 3D Distance Measurement with Ground Truth
==================================================

This script demonstrates the new functionality for ground truth comparison
and organized output storage.

Usage:
    python example_with_ground_truth.py --image path/to/photo.jpg --ground_truth 2.5
"""

import argparse
import os
import json
import cv2
import numpy as np

def display_results_summary(output_folder):
    """Display a summary of the results from the output folder."""
    json_path = os.path.join(output_folder, "data.json")
    
    if not os.path.exists(json_path):
        print(f"❌ Results file not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*60)
    print("MEASUREMENT RESULTS SUMMARY")
    print("="*60)
    
    # Basic info
    print(f"📸 Image: {data['image_path']}")
    print(f"📐 Image size: {data['image_size']['width']} x {data['image_size']['height']} pixels")
    print(f"🔧 Mode: {data['mode']}")
    
    # Depth Pro prediction
    dp_pred = data['depth_pro_prediction']
    print(f"\n🔍 Depth Pro Prediction:")
    print(f"   • Distance: {dp_pred['distance_m']:.4f} meters")
    if dp_pred['error_m'] is not None:
        print(f"   • Error: {dp_pred['error_m']:.4f} meters ({dp_pred['error_percent']:.2f}%)")
    
    # GeoCalib prediction
    gc_pred = data['geocalib_prediction']
    if gc_pred['distance_m'] is not None:
        print(f"\n🔍 GeoCalib Prediction:")
        print(f"   • Distance: {gc_pred['distance_m']:.4f} meters")
        if gc_pred['error_m'] is not None:
            print(f"   • Error: {gc_pred['error_m']:.4f} meters ({gc_pred['error_percent']:.2f}%)")
    
    # Ground truth
    if data['ground_truth_m'] is not None:
        print(f"\n📏 Ground Truth: {data['ground_truth_m']:.4f} meters")
        
        # Compare predictions if both are available
        if gc_pred['distance_m'] is not None:
            diff = abs(dp_pred['distance_m'] - gc_pred['distance_m'])
            diff_percent = (diff / data['ground_truth_m']) * 100
            print(f"   • Prediction Difference: {diff:.4f} meters ({diff_percent:.2f}%)")
            
            # Show which is better
            if dp_pred['error_percent'] < gc_pred['error_percent']:
                print(f"   • Winner: Depth Pro (better by {gc_pred['error_percent'] - dp_pred['error_percent']:.2f} percentage points)")
            else:
                print(f"   • Winner: GeoCalib (better by {dp_pred['error_percent'] - gc_pred['error_percent']:.2f} percentage points)")
    
    # Points info
    points = data['measurement_points']
    print(f"\n📍 Measurement Points:")
    print(f"   • Point 1 (2D): ({points['point1_2d']['u']:.1f}, {points['point1_2d']['v']:.1f}) pixels")
    print(f"   • Point 2 (2D): ({points['point2_2d']['u']:.1f}, {points['point2_2d']['v']:.1f}) pixels")
    print(f"   • Point 1 depth: {points['point1_depth']:.3f} meters")
    print(f"   • Point 2 depth: {points['point2_depth']:.3f} meters")
    
    # Camera intrinsics
    intrinsics = data['camera_intrinsics']
    print(f"\n📐 Camera Intrinsics:")
    print(f"   • Depth Pro focal: {intrinsics['depth_pro_focal']:.2f} pixels")
    if intrinsics['geocalib_focal'] is not None:
        print(f"   • GeoCalib fx: {intrinsics['geocalib_focal']['fx']:.2f}, fy: {intrinsics['geocalib_focal']['fy']:.2f} pixels")
    
    # Output files
    files = data['output_files']
    print(f"\n💾 Output Files:")
    print(f"   • Depth map: {files['depth_map']}")
    print(f"   • Measurement visualization: {files['measurement_visualization']}")
    print(f"   • Data JSON: {files['data_json']}")
    
    # Check if files exist
    print(f"\n✅ File Status:")
    for key, path in files.items():
        if key != 'data_json':  # Skip the JSON file itself
            if os.path.exists(path):
                print(f"   • {key}: ✓ Found")
            else:
                print(f"   • {key}: ✗ Missing")


def analyze_multiple_measurements(image_folder, ground_truths):
    """Analyze multiple measurements and compare with ground truths."""
    print("\n" + "="*60)
    print("MULTIPLE MEASUREMENTS ANALYSIS")
    print("="*60)
    
    results_summary = []
    
    for image_file, gt_distance in ground_truths.items():
        image_path = os.path.join(image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
        
        # Run measurement
        print(f"\n📸 Processing: {image_file}")
        print(f"📏 Ground truth: {gt_distance} meters")
        
        # This would normally call the main script
        # For this example, we'll simulate the process
        print("   • Running measurement...")
        print("   • Click two points when prompted...")
        
        # Simulate output folder creation
        base_name = os.path.splitext(image_file)[0]
        output_folder = f"3d_{base_name}"
        
        print(f"   • Results saved to: {output_folder}")
        
        # Check if results exist
        json_path = os.path.join(output_folder, "data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            dp_pred = data['depth_pro_prediction']
            gc_pred = data['geocalib_prediction']
            
            if data['ground_truth_m'] is not None:
                results_summary.append({
                    'image': image_file,
                    'depth_pro_predicted': dp_pred['distance_m'],
                    'geocalib_predicted': gc_pred['distance_m'] if gc_pred['distance_m'] is not None else None,
                    'ground_truth': data['ground_truth_m'],
                    'depth_pro_error': dp_pred['error_m'],
                    'depth_pro_error_percent': dp_pred['error_percent'],
                    'geocalib_error': gc_pred['error_m'] if gc_pred['error_m'] is not None else None,
                    'geocalib_error_percent': gc_pred['error_percent'] if gc_pred['error_percent'] is not None else None
                })
                
                print(f"   • Depth Pro: {dp_pred['distance_m']:.4f} m")
                if gc_pred['distance_m'] is not None:
                    print(f"   • GeoCalib: {gc_pred['distance_m']:.4f} m")
                    print(f"   • DP Error: {dp_pred['error_m']:.4f} m ({dp_pred['error_percent']:.2f}%)")
                    print(f"   • GC Error: {gc_pred['error_m']:.4f} m ({gc_pred['error_percent']:.2f}%)")
                else:
                    print(f"   • DP Error: {dp_pred['error_m']:.4f} m ({dp_pred['error_percent']:.2f}%)")
                    print(f"   • GeoCalib: Not available")
    
    # Summary statistics
    if results_summary:
        print(f"\n📊 SUMMARY STATISTICS:")
        
        # Depth Pro statistics
        dp_errors = [r['depth_pro_error'] for r in results_summary]
        dp_error_percents = [r['depth_pro_error_percent'] for r in results_summary]
        
        print(f"\n🔍 Depth Pro Predictions:")
        print(f"   • Mean error: {np.mean(dp_errors):.4f} meters")
        print(f"   • Std error: {np.std(dp_errors):.4f} meters")
        print(f"   • Mean error %: {np.mean(dp_error_percents):.2f}%")
        print(f"   • Best result: {min(dp_error_percents):.2f}% error")
        print(f"   • Worst result: {max(dp_error_percents):.2f}% error")
        
        # GeoCalib statistics (only for results where GeoCalib was available)
        gc_results = [r for r in results_summary if r['geocalib_error'] is not None]
        if gc_results:
            gc_errors = [r['geocalib_error'] for r in gc_results]
            gc_error_percents = [r['geocalib_error_percent'] for r in gc_results]
            
            print(f"\n🎯 GeoCalib Predictions:")
            print(f"   • Available for {len(gc_results)}/{len(results_summary)} images")
            print(f"   • Mean error: {np.mean(gc_errors):.4f} meters")
            print(f"   • Std error: {np.std(gc_errors):.4f} meters")
            print(f"   • Mean error %: {np.mean(gc_error_percents):.2f}%")
            print(f"   • Best result: {min(gc_error_percents):.2f}% error")
            print(f"   • Worst result: {max(gc_error_percents):.2f}% error")
            
            # Comparison analysis
            print(f"\n📈 Comparison Analysis:")
            dp_better_count = sum(1 for r in gc_results if r['depth_pro_error_percent'] < r['geocalib_error_percent'])
            gc_better_count = len(gc_results) - dp_better_count
            
            print(f"   • Depth Pro better: {dp_better_count}/{len(gc_results)} cases")
            print(f"   • GeoCalib better: {gc_better_count}/{len(gc_results)} cases")
            
            if len(gc_results) > 0:
                improvements = [r['geocalib_error_percent'] - r['depth_pro_error_percent'] for r in gc_results]
                mean_improvement = np.mean(improvements)
                print(f"   • Mean improvement (DP vs GC): {mean_improvement:.2f} percentage points")
                
                if mean_improvement > 0:
                    print(f"   • Overall winner: Depth Pro")
                else:
                    print(f"   • Overall winner: GeoCalib")
        else:
            print(f"\n🎯 GeoCalib Predictions:")
            print(f"   • Not available for any images")


def main():
    parser = argparse.ArgumentParser(description="Example: 3D Distance Measurement with Ground Truth")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--ground_truth", type=float, help="Ground truth distance in meters")
    parser.add_argument("--mode", choices=["depthpro", "geocalib", "fused"], default="fused",
                        help="Measurement mode")
    parser.add_argument("--show_summary", action="store_true", help="Show results summary")
    parser.add_argument("--batch_folder", help="Folder containing multiple images for batch processing")
    
    args = parser.parse_args()
    
    if args.batch_folder:
        # Batch processing mode
        if not os.path.exists(args.batch_folder):
            print(f"❌ Batch folder not found: {args.batch_folder}")
            return
        
        # Example ground truths (you would provide these)
        ground_truths = {
            "image1.jpg": 2.5,
            "image2.jpg": 1.8,
            "image3.jpg": 3.2
        }
        
        analyze_multiple_measurements(args.batch_folder, ground_truths)
        
    else:
        # Single image mode
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            return
        
        print("="*60)
        print("3D DISTANCE MEASUREMENT WITH GROUND TRUTH")
        print("="*60)
        
        # Build command for the main script
        cmd_parts = [
            "python", "measure_3d_distance.py",
            "--image", args.image,
            "--mode", args.mode
        ]
        
        if args.ground_truth:
            cmd_parts.extend(["--ground_truth", str(args.ground_truth)])
        
        cmd = " ".join(cmd_parts)
        
        print(f"🚀 Running command:")
        print(f"   {cmd}")
        print(f"\n📋 Instructions:")
        print(f"   1. The script will process your image")
        print(f"   2. Click two points when prompted")
        print(f"   3. Enter ground truth distance when asked")
        print(f"   4. Results will be saved in '3d_[filename]' folder")
        
        if args.ground_truth:
            print(f"\n📏 Ground truth: {args.ground_truth} meters")
        
        # Ask user if they want to proceed
        response = input(f"\n❓ Run the measurement? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            import subprocess
            try:
                subprocess.run(cmd_parts, check=True)
                
                # Show summary if requested
                if args.show_summary:
                    base_name = os.path.splitext(os.path.basename(args.image))[0]
                    output_folder = f"3d_{base_name}"
                    display_results_summary(output_folder)
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ Error running measurement: {e}")
        else:
            print("❌ Measurement cancelled.")


if __name__ == "__main__":
    main()
