#!/usr/bin/env python3
"""
Batch 3D Distance Measurement Tool
=================================

This script processes multiple images and measures distances between specified points.
Useful for batch processing or when you have known point coordinates.
"""

import argparse
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from datetime import datetime

# Import the measurement system
try:
    from measure_3d_distance import DistanceMeasurer
    HAVE_MEASUREMENT_SYSTEM = True
except ImportError as e:
    HAVE_MEASUREMENT_SYSTEM = False
    print(f"❌ Could not import measurement system: {e}")


class BatchProcessor:
    """Batch processor for 3D distance measurements."""
    
    def __init__(self, mode: str = "fused", output_dir: str = "batch_results"):
        """
        Initialize batch processor.
        
        Args:
            mode: Measurement mode ("depthpro", "geocalib", "fused")
            output_dir: Directory to save results
        """
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.measurer = None
        if HAVE_MEASUREMENT_SYSTEM:
            try:
                self.measurer = DistanceMeasurer(mode=mode)
                print(f"✅ Batch processor initialized with mode: {mode}")
            except Exception as e:
                print(f"❌ Failed to initialize measurer: {e}")
    
    def process_single_image(self, image_path: str, point1: Tuple[float, float], 
                           point2: Tuple[float, float]) -> Dict[str, Any]:
        """
        Process a single image and measure distance.
        
        Args:
            image_path: Path to image file
            point1, point2: (u, v) coordinates in pixels
            
        Returns:
            Dictionary with measurement results
        """
        if not self.measurer:
            raise RuntimeError("Measurement system not available")
        
        try:
            result = self.measurer.measure_distance(image_path, point1, point2)
            if result:
                # Add metadata
                result['image_path'] = image_path
                result['point1_2d'] = point1
                result['point2_2d'] = point2
                result['mode'] = self.mode
                result['timestamp'] = datetime.now().isoformat()
                return result
            else:
                return None
        except Exception as e:
            print(f"❌ Failed to process {image_path}: {e}")
            return None
    
    def process_image_list(self, image_list: List[str], point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> List[Dict[str, Any]]:
        """
        Process a list of images with the same points.
        
        Args:
            image_list: List of image paths
            point1, point2: (u, v) coordinates in pixels
            
        Returns:
            List of measurement results
        """
        results = []
        
        print(f"🔧 Processing {len(image_list)} images...")
        print(f"   • Point 1: {point1}")
        print(f"   • Point 2: {point2}")
        print(f"   • Mode: {self.mode}")
        
        for i, image_path in enumerate(image_list):
            print(f"\n📸 [{i+1}/{len(image_list)}] Processing: {os.path.basename(image_path)}")
            
            if not os.path.exists(image_path):
                print(f"   ❌ File not found: {image_path}")
                continue
            
            start_time = time.time()
            result = self.process_single_image(image_path, point1, point2)
            end_time = time.time()
            
            if result:
                result['processing_time'] = end_time - start_time
                results.append(result)
                print(f"   ✅ Distance: {result['distance_m']:.4f} meters")
            else:
                print(f"   ❌ Measurement failed")
        
        return results
    
    def save_results_csv(self, results: List[Dict[str, Any]], filename: str = "batch_results.csv"):
        """Save results to CSV file."""
        if not results:
            print("⚠️  No results to save")
            return
        
        csv_path = self.output_dir / filename
        
        # Define CSV columns
        fieldnames = [
            'image_path', 'point1_2d', 'point2_2d', 'distance_m',
            'point1_3d_x', 'point1_3d_y', 'point1_3d_z',
            'point2_3d_x', 'point2_3d_y', 'point2_3d_z',
            'delta_3d_x', 'delta_3d_y', 'delta_3d_z',
            'mode', 'processing_time', 'timestamp'
        ]
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Flatten the result dictionary
                row = {
                    'image_path': result['image_path'],
                    'point1_2d': str(result['point1_2d']),
                    'point2_2d': str(result['point2_2d']),
                    'distance_m': result['distance_m'],
                    'point1_3d_x': result['point1_3d'].x,
                    'point1_3d_y': result['point1_3d'].y,
                    'point1_3d_z': result['point1_3d'].z,
                    'point2_3d_x': result['point2_3d'].x,
                    'point2_3d_y': result['point2_3d'].y,
                    'point2_3d_z': result['point2_3d'].z,
                    'delta_3d_x': result['delta_3d'].x,
                    'delta_3d_y': result['delta_3d'].y,
                    'delta_3d_z': result['delta_3d'].z,
                    'mode': result['mode'],
                    'processing_time': result['processing_time'],
                    'timestamp': result['timestamp']
                }
                writer.writerow(row)
        
        print(f"💾 Results saved to: {csv_path}")
    
    def save_results_json(self, results: List[Dict[str, Any]], filename: str = "batch_results.json"):
        """Save results to JSON file."""
        if not results:
            print("⚠️  No results to save")
            return
        
        json_path = self.output_dir / filename
        
        # Convert Point3D objects to dictionaries for JSON serialization
        json_results = []
        for result in results:
            json_result = result.copy()
            json_result['point1_3d'] = {
                'x': result['point1_3d'].x,
                'y': result['point1_3d'].y,
                'z': result['point1_3d'].z
            }
            json_result['point2_3d'] = {
                'x': result['point2_3d'].x,
                'y': result['point2_3d'].y,
                'z': result['point2_3d'].z
            }
            json_result['delta_3d'] = {
                'x': result['delta_3d'].x,
                'y': result['delta_3d'].y,
                'z': result['delta_3d'].z
            }
            json_results.append(json_result)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to: {json_path}")
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not results:
            return {}
        
        distances = [r['distance_m'] for r in results]
        processing_times = [r['processing_time'] for r in results]
        
        summary = {
            'total_images': len(results),
            'successful_measurements': len(results),
            'distance_stats': {
                'mean': sum(distances) / len(distances),
                'min': min(distances),
                'max': max(distances),
                'std': (sum((d - sum(distances)/len(distances))**2 for d in distances) / len(distances))**0.5
            },
            'processing_stats': {
                'mean_time': sum(processing_times) / len(processing_times),
                'total_time': sum(processing_times),
                'min_time': min(processing_times),
                'max_time': max(processing_times)
            },
            'mode': self.mode,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def save_summary(self, summary: Dict[str, Any], filename: str = "summary.json"):
        """Save summary to JSON file."""
        summary_path = self.output_dir / filename
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Summary saved to: {summary_path}")
        
        # Print summary
        print(f"\n📊 BATCH PROCESSING SUMMARY")
        print(f"=" * 40)
        print(f"   • Total images: {summary['total_images']}")
        print(f"   • Successful measurements: {summary['successful_measurements']}")
        print(f"   • Mode: {summary['mode']}")
        print(f"   • Distance range: {summary['distance_stats']['min']:.4f} - {summary['distance_stats']['max']:.4f} meters")
        print(f"   • Mean distance: {summary['distance_stats']['mean']:.4f} ± {summary['distance_stats']['std']:.4f} meters")
        print(f"   • Total processing time: {summary['processing_stats']['total_time']:.2f} seconds")
        print(f"   • Mean processing time: {summary['processing_stats']['mean_time']:.2f} seconds per image")


def find_images_in_directory(directory: str, extensions: List[str] = None) -> List[str]:
    """Find all image files in a directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    directory = Path(directory)
    image_files = []
    
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return [str(f) for f in sorted(image_files)]


def parse_point(point_str: str) -> Tuple[float, float]:
    """Parse point string like '100,200' to (100.0, 200.0)."""
    try:
        parts = point_str.split(',')
        if len(parts) != 2:
            raise ValueError("Point must have exactly 2 coordinates")
        return (float(parts[0]), float(parts[1]))
    except Exception as e:
        raise ValueError(f"Invalid point format '{point_str}': {e}")


def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description="Batch 3D Distance Measurement Tool")
    parser.add_argument("--input", required=True, 
                       help="Input directory or file with image paths")
    parser.add_argument("--point1", required=True, 
                       help="First point coordinates (u,v) e.g., '100,200'")
    parser.add_argument("--point2", required=True, 
                       help="Second point coordinates (u,v) e.g., '300,400'")
    parser.add_argument("--mode", choices=["depthpro", "geocalib", "fused"], 
                       default="fused", help="Measurement mode")
    parser.add_argument("--output", default="batch_results", 
                       help="Output directory for results")
    parser.add_argument("--format", choices=["csv", "json", "both"], 
                       default="both", help="Output format")
    
    args = parser.parse_args()
    
    # Parse points
    try:
        point1 = parse_point(args.point1)
        point2 = parse_point(args.point2)
    except ValueError as e:
        print(f"❌ Error parsing points: {e}")
        return
    
    # Find images
    if os.path.isdir(args.input):
        print(f"📁 Scanning directory: {args.input}")
        image_list = find_images_in_directory(args.input)
    elif os.path.isfile(args.input):
        # Assume it's a text file with image paths
        print(f"📄 Reading image list from: {args.input}")
        with open(args.input, 'r') as f:
            image_list = [line.strip() for line in f if line.strip()]
    else:
        print(f"❌ Input not found: {args.input}")
        return
    
    if not image_list:
        print("❌ No images found")
        return
    
    print(f"📸 Found {len(image_list)} images")
    
    # Initialize batch processor
    processor = BatchProcessor(mode=args.mode, output_dir=args.output)
    
    if not processor.measurer:
        print("❌ Measurement system not available")
        print("   Run: python setup.py")
        return
    
    # Process images
    results = processor.process_image_list(image_list, point1, point2)
    
    if not results:
        print("❌ No successful measurements")
        return
    
    # Save results
    if args.format in ["csv", "both"]:
        processor.save_results_csv(results)
    
    if args.format in ["json", "both"]:
        processor.save_results_json(results)
    
    # Generate and save summary
    summary = processor.generate_summary(results)
    processor.save_summary(summary)
    
    print(f"\n🎉 Batch processing complete!")
    print(f"   • Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
