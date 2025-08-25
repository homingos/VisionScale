#!/usr/bin/env python3
"""
Example: Using GeoCalib for Camera Intrinsics Estimation
=======================================================

This script demonstrates how to use GeoCalib to estimate camera intrinsics
from a single image, which can then be used for 3D distance measurements.

Usage:
    python example_geocalib_usage.py --image path/to/photo.jpg
"""

import argparse
import cv2
import numpy as np
from PIL import Image
import torch
import sys
import os

# Add the geocalib path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'geocalib'))

try:
    from geocalib import GeoCalib
    HAVE_GEOCALIB = True
    print("[✓] GeoCalib imported successfully")
except Exception as e:
    HAVE_GEOCALIB = False
    print(f"[✗] GeoCalib not available: {e}")
    print("Please install GeoCalib first:")
    print("cd 3d_distances/src/geocalib && pip install -e .")
    sys.exit(1)


def estimate_intrinsics(image_path):
    """Estimate camera intrinsics using GeoCalib."""
    if not HAVE_GEOCALIB:
        print("❌ GeoCalib not available")
        return None
    
    print(f"[GeoCalib] Processing {image_path}...")
    
    # Load image
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    
    # Initialize GeoCalib
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GeoCalib] Using device: {device}")
    
    gc = GeoCalib().to(device)
    
    # Load image tensor
    t = gc.load_image(pil_image)
    
    # Run calibration
    print("[GeoCalib] Running calibration...")
    res = gc.calibrate(t.to(device))
    
    # Extract results
    camera = res["camera"]
    gravity = res["gravity"]
    
    # Get intrinsics
    fx = float(camera.f[0, 0].item())
    fy = float(camera.f[0, 1].item())
    cx = float(camera.c[0, 0].item())
    cy = float(camera.c[0, 1].item())
    
    # Get uncertainties
    focal_uncertainty = float(res.get("focal_uncertainty", [0.0])[0].item()) if "focal_uncertainty" in res else 0.0
    
    # Get gravity direction (roll, pitch)
    roll, pitch = gravity.rp.unbind(-1)
    roll_deg = float(roll.item() * 180 / np.pi)
    pitch_deg = float(pitch.item() * 180 / np.pi)
    
    # Get field of view
    vfov_deg = float(camera.vfov.item() * 180 / np.pi)
    hfov_deg = float(camera.hfov.item() * 180 / np.pi)
    
    # Get image size
    width, height = camera.size.unbind(-1)
    width = int(width.item())
    height = int(height.item())
    
    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'focal_uncertainty': focal_uncertainty,
        'roll_deg': roll_deg,
        'pitch_deg': pitch_deg,
        'vfov_deg': vfov_deg,
        'hfov_deg': hfov_deg,
        'width': width,
        'height': height,
        'camera': camera,
        'gravity': gravity,
        'full_results': res
    }


def print_results(results):
    """Print the calibration results in a formatted way."""
    if results is None:
        return
    
    print("\n" + "="*60)
    print("GEOcalib Camera Calibration Results")
    print("="*60)
    
    print(f"\n📐 Image Size: {results['width']} x {results['height']} pixels")
    
    print(f"\n🔧 Camera Intrinsics:")
    print(f"   • Focal length (fx): {results['fx']:.2f} pixels")
    print(f"   • Focal length (fy): {results['fy']:.2f} pixels")
    print(f"   • Principal point (cx): {results['cx']:.2f} pixels")
    print(f"   • Principal point (cy): {results['cy']:.2f} pixels")
    print(f"   • Focal uncertainty: ±{results['focal_uncertainty']:.2f} pixels")
    
    print(f"\n📐 Field of View:")
    print(f"   • Vertical FoV: {results['vfov_deg']:.2f}°")
    print(f"   • Horizontal FoV: {results['hfov_deg']:.2f}°")
    
    print(f"\n🌍 Gravity Direction:")
    print(f"   • Roll: {results['roll_deg']:.2f}°")
    print(f"   • Pitch: {results['pitch_deg']:.2f}°")
    
    print(f"\n📊 Quality Metrics:")
    # Check if confidence maps are available
    if 'up_confidence' in results['full_results']:
        up_conf = results['full_results']['up_confidence']
        lat_conf = results['full_results']['latitude_confidence']
        print(f"   • Up confidence: {up_conf.mean():.3f} (avg)")
        print(f"   • Latitude confidence: {lat_conf.mean():.3f} (avg)")


def save_results(results, output_file):
    """Save results to a text file."""
    if results is None:
        return
    
    with open(output_file, 'w') as f:
        f.write("GeoCalib Camera Calibration Results\n")
        f.write("="*40 + "\n\n")
        
        f.write(f"Image Size: {results['width']} x {results['height']} pixels\n\n")
        
        f.write("Camera Intrinsics:\n")
        f.write(f"  fx: {results['fx']:.2f} pixels\n")
        f.write(f"  fy: {results['fy']:.2f} pixels\n")
        f.write(f"  cx: {results['cx']:.2f} pixels\n")
        f.write(f"  cy: {results['cy']:.2f} pixels\n")
        f.write(f"  focal_uncertainty: {results['focal_uncertainty']:.2f} pixels\n\n")
        
        f.write("Field of View:\n")
        f.write(f"  vertical_fov: {results['vfov_deg']:.2f} degrees\n")
        f.write(f"  horizontal_fov: {results['hfov_deg']:.2f} degrees\n\n")
        
        f.write("Gravity Direction:\n")
        f.write(f"  roll: {results['roll_deg']:.2f} degrees\n")
        f.write(f"  pitch: {results['pitch_deg']:.2f} degrees\n")
    
    print(f"💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Estimate camera intrinsics using GeoCalib")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Output file for results (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        return
    
    # Estimate intrinsics
    results = estimate_intrinsics(args.image)
    
    if results is None:
        print("❌ Failed to estimate intrinsics")
        return
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output)
    else:
        # Save with default name
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        default_output = f"geocalib_results_{base_name}.txt"
        save_results(results, default_output)


if __name__ == "__main__":
    main()
