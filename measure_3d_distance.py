#!/usr/bin/env python3
"""
3D Distance Measurement Tool
============================

This script measures 3D distances between two points in an image using:
- Depth Pro: Monocular depth estimation in meters
- GeoCalib: Camera intrinsics estimation (optional)
- Interactive point selection
- Ground truth comparison

Usage:
    python measure_3d_distance.py --image path/to/photo.jpg --mode fused
"""

import argparse
import math
import numpy as np
import cv2
from PIL import Image
import os
import sys
import json
from typing import Tuple, Optional, Dict, Any
import time

# ---- Depth Pro (required) ----
try:
    import depth_pro
    HAVE_DEPTHPRO = True
    print("[✓] Depth Pro imported successfully")
except Exception as e:
    HAVE_DEPTHPRO = False
    print(f"[✗] Depth Pro not available: {e}")

# ---- GeoCalib (optional) ----
try:
    from geocalib import GeoCalib
    import torch
    HAVE_GEOCALIB = True
    print("[✓] GeoCalib imported successfully")
    print(f"[✓] GeoCalib version available: {GeoCalib.__name__}")
    print(f"[✓] PyTorch available: {torch.__version__}")
    print(f"[✓] CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    HAVE_GEOCALIB = False
    print(f"[✗] GeoCalib not available: {e}")
    print(f"[✗] Please install GeoCalib: cd src/geocalib && pip install -e .")


def bilinear_sample(img, x, y):
    """Sample float image (H,W) at subpixel (x,y). Returns float or np.nan."""
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return np.nan
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    v00, v10 = img[y0, x0], img[y0, x1]
    v01, v11 = img[y1, x0], img[y1, x1]
    return float(((1 - dy) * ((1 - dx) * v00 + dx * v10) +
                  dy * ((1 - dx) * v01 + dx * v11)))


def robust_depth_sample(depth, x, y):
    """Bilinear first; if NaN/invalid, fall back to median 3x3."""
    z = bilinear_sample(depth, x, y)
    if np.isfinite(z) and z > 0:
        return z
    xi, yi = int(round(x)), int(round(y))
    h, w = depth.shape
    x0, x1 = max(0, xi - 1), min(w, xi + 2)
    y0, y1 = max(0, yi - 1), min(h, yi + 2)
    patch = depth[y0:y1, x0:x1].reshape(-1)
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return np.nan
    return float(np.median(patch))


def backproject(u, v, Z, fx, fy, cx, cy):
    """Back-project 2D image point to 3D camera coordinates."""
    # Camera frame: X right, Y down (image convention), Z forward (meters)
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return np.array([X, Y, Z], float)


def run_depthpro(image_path, f_px_override=None):
    """Run Depth Pro inference to get depth map and focal length."""
    assert HAVE_DEPTHPRO, "Depth Pro not installed. `pip install -e .` inside ml-depth-pro."
    
    print(f"[Depth Pro] Processing {image_path}...")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    
    # Load image
    image, _, f_px_exif = depth_pro.load_rgb(image_path)
    image_t = transform(image)
    
    # Run inference
    pred = model.infer(image_t, f_px=f_px_override if f_px_override is not None else f_px_exif)
    
    depth_m = pred["depth"]                  # (H,W), meters
    f_px_pred = float(pred["focallength_px"])# pixels
    
    print(f"[Depth Pro] Depth range: [{depth_m.min():.3f}, {depth_m.max():.3f}] meters")
    print(f"[Depth Pro] Focal length: {f_px_pred:.2f} pixels")
    
    return depth_m, f_px_pred


def run_geocalib(image_bgr):
    """Run GeoCalib on BGR image to get camera intrinsics."""
    try:
        import torch
        from geocalib import GeoCalib
        import tempfile
        import os
        
        # Save the BGR image to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Convert BGR to RGB and save
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            cv2.imwrite(tmp_file.name, image_rgb)
            temp_path = tmp_file.name
        
        try:
            # Initialize GeoCalib model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = GeoCalib().to(device)
            
            # Load image using GeoCalib's loader (expects file path)
            image_tensor = model.load_image(temp_path).to(device)
            
            # Run calibration
            result = model.calibrate(image_tensor)
            
            # Extract camera intrinsics
            camera = result["camera"]
            
            # Get focal lengths (fx, fy) - handle different matrix structures
            try:
                # Try the standard 2x2 focal matrix
                if camera.f.shape == (2, 2):
                    fx = camera.f[0, 0].item()  # focal length in x direction
                    fy = camera.f[1, 1].item()  # focal length in y direction
                elif camera.f.shape == (1, 2):
                    # Single focal length for both x and y
                    fx = fy = camera.f[0, 0].item()
                elif camera.f.shape == (1, 1):
                    # Single focal length
                    fx = fy = camera.f[0, 0].item()
                else:
                    # Fallback: use the first element as focal length
                    fx = fy = camera.f.flatten()[0].item()
                    print(f"[GeoCalib] Warning: Unexpected focal matrix shape {camera.f.shape}, using first element")
            except Exception as e:
                print(f"[GeoCalib] Error extracting focal lengths: {e}")
                print(f"[GeoCalib] Focal matrix shape: {camera.f.shape}")
                print(f"[GeoCalib] Focal matrix: {camera.f}")
                return None, None, None, None, None, None
            
            # Get principal point (cx, cy) - GeoCalib assumes center of image
            H, W = image_bgr.shape[:2]
            cx = W / 2.0
            cy = H / 2.0
            
            # Get gravity direction if available
            gravity = result.get("gravity", None)
            
            # Calculate focal uncertainty (not directly provided by GeoCalib)
            focal_uncertainty = None
            
            print(f"[GeoCalib] Camera intrinsics extracted:")
            print(f"   • fx: {fx:.2f} pixels")
            print(f"   • fy: {fy:.2f} pixels")
            print(f"   • cx: {cx:.2f} pixels (image center)")
            print(f"   • cy: {cy:.2f} pixels (image center)")
            if gravity is not None:
                print(f"   • Gravity direction: {gravity}")
                print(f"   • Gravity type: {type(gravity)}")
                print(f"   • Gravity dir: {dir(gravity)}")
                if hasattr(gravity, 'data'):
                    print(f"   • Gravity data: {gravity.data}")
                if hasattr(gravity, 'shape'):
                    print(f"   • Gravity shape: {gravity.shape}")
            
            return fx, fy, cx, cy, focal_uncertainty, gravity
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ GeoCalib error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None


def create_depth_visualization(depth_m, output_path=None):
    """Create a depth visualization for debugging."""
    # Convert tensor to numpy if needed
    if hasattr(depth_m, 'cpu'):
        d = depth_m.cpu().numpy()
    else:
        d = depth_m.copy()
    
    d[~np.isfinite(d)] = 0
    
    # Clip to reasonable range
    d = np.clip(d, np.percentile(d, 1), np.percentile(d, 99))
    d = (d - d.min()) / max(1e-9, d.max() - d.min())
    
    # Create visualization
    d_vis = (255 * (1.0 - d)).astype(np.uint8)
    d_vis = cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO)
    
    if output_path:
        cv2.imwrite(output_path, d_vis)
        print(f"[Visualization] Depth map saved to {output_path}")
    
    return d_vis


def create_measurement_visualization(img, clicks, dist_m_dp, dist_m_gc=None, ground_truth_m=None, output_path=None):
    """Create a visualization with the measurement line and distances from both methods."""
    vis = img.copy()
    
    # Convert float coordinates to integers for OpenCV drawing functions
    clicks_int = [(int(x), int(y)) for x, y in clicks]
    
    # Draw the line between points
    cv2.line(vis, clicks_int[0], clicks_int[1], (0, 255, 0), 3)
    
    # Draw points
    for i, (x, y) in enumerate(clicks_int):
        cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(vis, (x, y), 8, (0, 0, 0), 2)
        cv2.putText(vis, str(i+1), (x+12, y-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Calculate text position (midpoint of the line)
    mid_x = int((clicks[0][0] + clicks[1][0]) / 2)
    mid_y = int((clicks[0][1] + clicks[1][1]) / 2)
    
    # Draw Depth Pro prediction in blue
    dp_text = f"Depth Pro: {dist_m_dp:.4f} m"
    cv2.putText(vis, dp_text, (mid_x-120, mid_y-40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Draw GeoCalib prediction in yellow if available
    if dist_m_gc is not None:
        gc_text = f"GeoCalib: {dist_m_gc:.4f} m"
        cv2.putText(vis, gc_text, (mid_x-120, mid_y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw ground truth in red if provided
    if ground_truth_m is not None:
        gt_text = f"Ground Truth: {ground_truth_m:.4f} m"
        cv2.putText(vis, gt_text, (mid_x-120, mid_y+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate and display errors
        error_dp = abs(dist_m_dp - ground_truth_m)
        error_percent_dp = (error_dp / ground_truth_m) * 100
        error_dp_text = f"DP Error: {error_dp:.4f} m ({error_percent_dp:.2f}%)"
        cv2.putText(vis, error_dp_text, (mid_x-120, mid_y+40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        if dist_m_gc is not None:
            error_gc = abs(dist_m_gc - ground_truth_m)
            error_percent_gc = (error_gc / ground_truth_m) * 100
            error_gc_text = f"GC Error: {error_gc:.4f} m ({error_percent_gc:.2f}%)"
            cv2.putText(vis, error_gc_text, (mid_x-120, mid_y+70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show which method is better
            if error_percent_dp < error_percent_gc:
                better_text = "Depth Pro is better"
                better_color = (255, 0, 0)  # Blue
            else:
                better_text = "GeoCalib is better"
                better_color = (0, 255, 255)  # Yellow
            cv2.putText(vis, better_text, (mid_x-120, mid_y+100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, better_color, 2)
    
    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"[Visualization] Measurement visualization saved to {output_path}")
    
    return vis


def get_ground_truth_input():
    """Get ground truth distance from user input."""
    while True:
        try:
            gt_input = input("\n📏 Enter ground truth distance in meters (or press Enter to skip): ").strip()
            if not gt_input:
                return None
            gt_distance = float(gt_input)
            if gt_distance <= 0:
                print("❌ Distance must be positive. Please try again.")
                continue
            return gt_distance
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")
            continue


def save_results_to_json(results_data, output_path):
    """Save all results data to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"💾 Results data saved to: {output_path}")


def main():
    """Main function for 3D distance measurement."""
    parser = argparse.ArgumentParser(description="Click two points to measure 3D distance using Depth Pro (and optionally GeoCalib).")
    parser.add_argument("--image", required=True, help="Path to image (jpg/png).")
    parser.add_argument("--mode", choices=["depthpro", "geocalib", "fused"], default="depthpro",
                        help="Which intrinsics to use for back-projection.")
    parser.add_argument("--show_depth", action="store_true", help="Display depth visualization for reference.")
    parser.add_argument("--save_depth", help="Save depth visualization to file.")
    parser.add_argument("--point1", nargs=2, type=float, help="First point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point2", nargs=2, type=float, help="Second point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--ground_truth", type=float, help="Ground truth distance in meters")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        return
    
    # Load original image for display / clicking
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ Could not load image: {args.image}")
        return
    
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    
    print(f"\n📐 Image: {W}x{H} pixels")
    print(f"📐 Principal point: ({cx:.1f}, {cy:.1f})")
    
    # (Optional) GeoCalib intrinsics - always try to run GeoCalib for comparison
    fx_g = fy_g = cx_g = cy_g = None
    focal_uncertainty = None
    gravity = None
    if HAVE_GEOCALIB:
        try:
            fx_g, fy_g, cx_g, cy_g, focal_uncertainty, gravity = run_geocalib(img)
            if fx_g is not None and fy_g is not None:
                print(f"[GeoCalib] Successfully obtained intrinsics: fx={fx_g:.2f}, fy={fy_g:.2f}")
            else:
                print(f"[GeoCalib] Failed to obtain valid intrinsics")
                fx_g = fy_g = cx_g = cy_g = None
        except Exception as e:
            print(f"❌ GeoCalib failed: {e}")
            fx_g = fy_g = cx_g = cy_g = None
            focal_uncertainty = None
            gravity = None
    else:
        print("❌ GeoCalib not available; install it for dual prediction comparison.")
    
    # Depth Pro inference (metric depth + focal)
    if not HAVE_DEPTHPRO:
        print("❌ Depth Pro not available. Please install it first.")
        return
    
    # For fused mode, condition Depth Pro with GeoCalib focal if available
    f_pass_to_dp = None
    if args.mode == "fused" and fx_g and fy_g:
        f_pass_to_dp = 0.5 * (fx_g + fy_g)  # condition DP on GeoCalib focal
        print(f"[Fusion] Conditioning Depth Pro with GeoCalib focal: {f_pass_to_dp:.2f}")
    
    try:
        depth_m, f_dp = run_depthpro(args.image, f_px_override=f_pass_to_dp)
    except Exception as e:
        print(f"❌ Depth Pro failed: {e}")
        return
    
    # Ensure depth size == display size
    if depth_m.shape != (H, W):
        print(f"[Resize] Depth map {depth_m.shape} -> ({H}, {W})")
        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Select intrinsics for back-projection based on mode
    if args.mode == "depthpro":
        # Use Depth Pro intrinsics for the main calculation
        fx_used = fy_used = f_dp
        cx_used = cx
        cy_used = cy
        intr_src = "DepthPro"
    elif args.mode == "geocalib":
        if fx_g is None:
            print("❌ GeoCalib intrinsics not available.")
            return
        # Use GeoCalib intrinsics for the main calculation
        fx_used = fx_g
        fy_used = fy_g
        cx_used = cx_g if cx_g is not None else cx
        cy_used = cy_g if cy_g is not None else cy
        intr_src = "GeoCalib"
    else:  # fused
        if fx_g is None:
            # Fall back to Depth Pro only
            fx_used = fy_used = f_dp
            cx_used = cx
            cy_used = cy
            intr_src = "DepthPro (GeoCalib failed)"
        else:
            # Use geometric mean for fused mode
            f_geo = 0.5 * (fx_g + fy_g)
            fx_used = fy_used = math.sqrt(max(1e-6, f_dp) * max(1e-6, f_geo))
            cx_used = cx_g if cx_g is not None else cx
            cy_used = cy_g if cy_g is not None else cy
            intr_src = "Fused(DP×GC)"
    
    print(f"\n📐 Using intrinsics [{intr_src}]:")
    print(f"   • fx={fx_used:.2f}, fy={fy_used:.2f} pixels")
    print(f"   • cx={cx_used:.2f}, cy={cy_used:.2f} pixels")
    if focal_uncertainty is not None:
        print(f"   • Focal uncertainty: {focal_uncertainty:.2f} pixels")
    print(f"   • Depth range: [{depth_m.min():.3f}, {depth_m.max():.3f}] meters")
    
    # Get points (interactive or command line)
    if args.point1 and args.point2:
        # Non-interactive mode
        clicks = [args.point1, args.point2]
        print(f"\n📏 Non-interactive mode:")
        print(f"   • Point 1: ({clicks[0][0]:.1f}, {clicks[0][1]:.1f})")
        print(f"   • Point 2: ({clicks[1][0]:.1f}, {clicks[1][1]:.1f})")
    else:
        # Interactive clicks
        clicks = []
        vis = img.copy()
        cv2.namedWindow("Click two points (ESC to finish)", cv2.WINDOW_NORMAL)
        
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
                clicks.append((float(x), float(y)))
                # Draw point
                cv2.circle(vis, (x, y), 6, (0, 255, 0), 2)
                # Draw point number
                cv2.putText(vis, str(len(clicks)), (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Draw line between points if we have 2
                if len(clicks) == 2:
                    cv2.line(vis, (int(clicks[0][0]), int(clicks[0][1])), 
                            (int(clicks[1][0]), int(clicks[1][1])), (255, 0, 0), 2)
                cv2.imshow("Click two points (ESC to finish)", vis)
        
        cv2.setMouseCallback("Click two points (ESC to finish)", on_mouse)
        cv2.imshow("Click two points (ESC to finish)", vis)
        
        print(f"\n🖱️  Click two points on the image to measure distance...")
        print(f"   • Press ESC to finish")
        print(f"   • Points will be numbered 1 and 2")
        
        while True:
            key = cv2.waitKey(20)
            if key == 27 or len(clicks) == 2:  # ESC or done
                break
        
        cv2.destroyAllWindows()
        
        if len(clicks) != 2:
            print("❌ Need exactly two points. Exiting.")
            return
    
    # Measure distance
    (u1, v1), (u2, v2) = clicks
    
    print(f"\n🔍 Sampling depths...")
    Z1 = robust_depth_sample(depth_m, u1, v1)
    Z2 = robust_depth_sample(depth_m, u2, v2)
    
    print(f"   • Point 1 depth: {Z1:.3f} meters")
    print(f"   • Point 2 depth: {Z2:.3f} meters")
    
    if not (np.isfinite(Z1) and Z1 > 0 and np.isfinite(Z2) and Z2 > 0):
        print("❌ Invalid depth at selected points.")
        print("   • Try different points")
        print("   • Avoid featureless areas (sky, glass, shadows)")
        print("   • Check depth visualization with --show_depth")
        return
    
    print(f"\n📐 Back-projecting points to 3D...")
    print(f"   • Point 1 (2D): ({u1:.1f}, {v1:.1f}) pixels, depth: {Z1:.3f} m")
    print(f"   • Point 2 (2D): ({u2:.1f}, {v2:.1f}) pixels, depth: {Z2:.3f} m")
    
    # Back-project to 3D using Depth Pro intrinsics (always calculate this)
    print(f"\n🔍 Depth Pro back-projection:")
    print(f"   • Using focal: {f_dp:.2f} pixels")
    print(f"   • Using principal point: ({cx:.2f}, {cy:.2f})")
    
    P1_dp = backproject(u1, v1, Z1, f_dp, f_dp, cx, cy)
    P2_dp = backproject(u2, v2, Z2, f_dp, f_dp, cx, cy)
    dXYZ_dp = P2_dp - P1_dp
    dist_m_dp = float(np.linalg.norm(dXYZ_dp))
    
    print(f"   • Point 1 (3D): ({P1_dp[0]:.4f}, {P1_dp[1]:.4f}, {P1_dp[2]:.4f}) meters")
    print(f"   • Point 2 (3D): ({P2_dp[0]:.4f}, {P2_dp[1]:.4f}, {P2_dp[2]:.4f}) meters")
    print(f"   • Distance: {dist_m_dp:.4f} meters")
    
    # Back-project to 3D using GeoCalib intrinsics (if available)
    dist_m_gc = None
    dXYZ_gc = None
    P1_gc = None
    P2_gc = None
    
    if fx_g is not None and fy_g is not None:
        # Use GeoCalib principal points if available, otherwise use image center
        cx_gc = cx_g if cx_g is not None else cx
        cy_gc = cy_g if cy_g is not None else cy
        
        print(f"\n🔍 GeoCalib back-projection:")
        print(f"   • Using focal: fx={fx_g:.2f}, fy={fy_g:.2f} pixels")
        print(f"   • Using principal point: ({cx_gc:.2f}, {cy_gc:.2f})")
        
        P1_gc = backproject(u1, v1, Z1, fx_g, fy_g, cx_gc, cy_gc)
        P2_gc = backproject(u2, v2, Z2, fx_g, fy_g, cx_gc, cy_gc)
        dXYZ_gc = P2_gc - P1_gc
        dist_m_gc = float(np.linalg.norm(dXYZ_gc))
        
        print(f"   • Point 1 (3D): ({P1_gc[0]:.4f}, {P1_gc[1]:.4f}, {P1_gc[2]:.4f}) meters")
        print(f"   • Point 2 (3D): ({P2_gc[0]:.4f}, {P2_gc[1]:.4f}, {P2_gc[2]:.4f}) meters")
        print(f"   • Distance: {dist_m_gc:.4f} meters")
    else:
        print(f"\n❌ GeoCalib prediction skipped - intrinsics not available")
    
    # Get ground truth distance
    ground_truth_m = args.ground_truth
    if ground_truth_m is None:
        ground_truth_m = get_ground_truth_input()
    
    # Create output folder
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    output_folder = f"3d_{base_name}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create visualizations
    depth_vis_path = os.path.join(output_folder, "depth_map.png")
    depth_vis = create_depth_visualization(depth_m, depth_vis_path)
    
    measurement_vis_path = os.path.join(output_folder, "measurement_visualization.png")
    measurement_vis = create_measurement_visualization(img, clicks, dist_m_dp, dist_m_gc, ground_truth_m, measurement_vis_path)
    
    # Optional depth visualization display
    if args.show_depth:
        cv2.imshow("Depth (scaled, TURBO)", depth_vis)
        cv2.waitKey(1)
    
    # Print results
    print(f"\n📏 3D Distance Measurement Results:")
    print(f"   • Point 1 (2D): ({u1:.1f}, {v1:.1f}) pixels")
    print(f"   • Point 2 (2D): ({u2:.1f}, {v2:.1f}) pixels")
    print(f"   • Point 1 depth: {Z1:.3f} meters")
    print(f"   • Point 2 depth: {Z2:.3f} meters")
    
    print(f"\n🔍 Depth Pro Prediction:")
    print(f"   • Point 1 (3D): ({P1_dp[0]:.4f}, {P1_dp[1]:.4f}, {P1_dp[2]:.4f}) meters")
    print(f"   • Point 2 (3D): ({P2_dp[0]:.4f}, {P2_dp[1]:.4f}, {P2_dp[2]:.4f}) meters")
    print(f"   • Delta (3D): ({dXYZ_dp[0]:.4f}, {dXYZ_dp[1]:.4f}, {dXYZ_dp[2]:.4f}) meters")
    print(f"   • Distance: {dist_m_dp:.4f} meters")
    
    if dist_m_gc is not None:
        print(f"\n🔍 GeoCalib Prediction:")
        print(f"   • Point 1 (3D): ({P1_gc[0]:.4f}, {P1_gc[1]:.4f}, {P1_gc[2]:.4f}) meters")
        print(f"   • Point 2 (3D): ({P2_gc[0]:.4f}, {P2_gc[1]:.4f}, {P2_gc[2]:.4f}) meters")
        print(f"   • Delta (3D): ({dXYZ_gc[0]:.4f}, {dXYZ_gc[1]:.4f}, {dXYZ_gc[2]:.4f}) meters")
        print(f"   • Distance: {dist_m_gc:.4f} meters")
    
    if ground_truth_m is not None:
        print(f"\n📊 Error Analysis:")
        print(f"   • Ground Truth: {ground_truth_m:.4f} meters")
        
        # Calculate errors for Depth Pro prediction
        error_dp = abs(dist_m_dp - ground_truth_m)
        error_percent_dp = (error_dp / ground_truth_m) * 100
        print(f"   • Depth Pro Error: {error_dp:.4f} meters ({error_percent_dp:.2f}%)")
        
        # Calculate errors for GeoCalib prediction (if available)
        if dist_m_gc is not None:
            error_gc = abs(dist_m_gc - ground_truth_m)
            error_percent_gc = (error_gc / ground_truth_m) * 100
            print(f"   • GeoCalib Error: {error_gc:.4f} meters ({error_percent_gc:.2f}%)")
            
            # Compare predictions
            diff = abs(dist_m_dp - dist_m_gc)
            diff_percent = (diff / ground_truth_m) * 100
            print(f"   • Prediction Difference: {diff:.4f} meters ({diff_percent:.2f}%)")
    
    # Additional info
    print(f"\n📊 Additional Information:")
    print(f"   • Depth Pro focal: {f_dp:.2f} pixels")
    if fx_g is not None:
        print(f"   • GeoCalib fx: {fx_g:.2f}, fy: {fy_g:.2f} pixels")
        print(f"   • GeoCalib cx: {cx_g:.2f}, cy: {cy_g:.2f} pixels")
        if focal_uncertainty is not None:
            print(f"   • GeoCalib focal uncertainty: {focal_uncertainty:.2f} pixels")
    print(f"   • Used principal point: cx={cx:.2f}, cy={cy:.2f} pixels")
    
    # Prepare data for JSON storage
    results_data = {
        "image_path": args.image,
        "image_size": {"width": W, "height": H},
        "mode": args.mode,
        "camera_intrinsics": {
            "depth_pro_focal": f_dp,
            "geocalib_focal": {"fx": fx_g, "fy": fy_g} if fx_g is not None else None,
            "geocalib_principal_point": {"cx": cx_g, "cy": cy_g} if cx_g is not None else None,
            "focal_uncertainty": focal_uncertainty,
            "used_principal_point": {"cx": cx, "cy": cy}
        },
        "measurement_points": {
            "point1_2d": {"u": u1, "v": v1},
            "point2_2d": {"u": u2, "v": v2},
            "point1_depth": Z1,
            "point2_depth": Z2
        },
        "depth_pro_prediction": {
            "point1_3d": {"x": P1_dp[0], "y": P1_dp[1], "z": P1_dp[2]},
            "point2_3d": {"x": P2_dp[0], "y": P2_dp[1], "z": P2_dp[2]},
            "delta_3d": {"x": dXYZ_dp[0], "y": dXYZ_dp[1], "z": dXYZ_dp[2]},
            "distance_m": dist_m_dp,
            "error_m": abs(dist_m_dp - ground_truth_m) if ground_truth_m is not None else None,
            "error_percent": (abs(dist_m_dp - ground_truth_m) / ground_truth_m * 100) if ground_truth_m is not None else None
        },
        "geocalib_prediction": {
            "available": dist_m_gc is not None,
            "focal_length": {
                "fx": fx_g,
                "fy": fy_g
            },
            "principal_point": {
                "cx": cx_g,
                "cy": cy_g
            },
            "focal_uncertainty": focal_uncertainty,
            "gravity_direction": gravity.data.tolist() if gravity is not None and hasattr(gravity, 'data') else None,
            "points_3d": {
                "P1": P1_gc.tolist() if P1_gc is not None else None,
                "P2": P2_gc.tolist() if P2_gc is not None else None
            },
            "delta_xyz": dXYZ_gc.tolist() if dXYZ_gc is not None else None,
            "distance_meters": dist_m_gc,
            "error_absolute": abs(dist_m_gc - ground_truth_m) if dist_m_gc is not None and ground_truth_m is not None else None,
            "error_percentage": (abs(dist_m_gc - ground_truth_m) / ground_truth_m * 100) if dist_m_gc is not None and ground_truth_m is not None else None
        },
        "ground_truth_m": ground_truth_m,
        "depth_map_info": {
            "depth_range_m": {"min": float(depth_m.min()), "max": float(depth_m.max())},
            "depth_map_shape": depth_m.shape
        },
        "output_files": {
            "depth_map": depth_vis_path,
            "measurement_visualization": measurement_vis_path,
            "data_json": os.path.join(output_folder, "data.json")
        }
    }
    
    # Save results to JSON
    json_path = os.path.join(output_folder, "data.json")
    save_results_to_json(results_data, json_path)
    
    print(f"\n💾 All results saved to folder: {output_folder}")
    print(f"   • Depth map: {depth_vis_path}")
    print(f"   • Measurement visualization: {measurement_vis_path}")
    print(f"   • Data: {json_path}")


if __name__ == "__main__":
    main()
