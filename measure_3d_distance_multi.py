#!/usr/bin/env python3
"""
3D Distance Measurement Tool (Multi-Point)
==========================================

This script measures 3D distances between multiple points in an image using:
- Depth Pro: Monocular depth estimation in meters
- GeoCalib: Camera intrinsics estimation (optional)
- Interactive point selection
- Ground truth comparison

Calculates distances between all point combinations (avoiding duplicates).

Usage:
    python measure_3d_distance_multi.py --image path/to/photo.jpg --mode fused
"""

import argparse
import math
import numpy as np
import cv2
from PIL import Image
import os
import sys
import json
from typing import Tuple, Optional, Dict, Any, List
import time
from itertools import combinations

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


def create_measurement_visualization(img, clicks, point_pairs_dp, point_pairs_gc=None, ground_truth_m=None, output_path=None):
    """Create a visualization with all measurement lines and distances from both methods."""
    vis = img.copy()
    
    # Convert float coordinates to integers for OpenCV drawing functions
    clicks_int = [(int(x), int(y)) for x, y in clicks]
    
    # Draw all lines between points with different colors for each method
    for i, j in combinations(range(len(clicks)), 2):
        # Draw Depth Pro lines in blue
        cv2.line(vis, clicks_int[i], clicks_int[j], (255, 0, 0), 2)
        
        # Draw GeoCalib lines in yellow if available
        if point_pairs_gc:
            cv2.line(vis, clicks_int[i], clicks_int[j], (0, 255, 255), 1)
    
    # Draw points
    for i, (x, y) in enumerate(clicks_int):
        cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(vis, (x, y), 8, (0, 0, 0), 2)
        cv2.putText(vis, str(i+1), (x+12, y-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Add distance labels on the image for Depth Pro
    for idx, (i, j, P1, P2, dXYZ, dist) in enumerate(point_pairs_dp):
        # Calculate midpoint for text placement
        mid_x = int((clicks[i-1][0] + clicks[j-1][0]) / 2)
        mid_y = int((clicks[i-1][1] + clicks[j-1][1]) / 2)
        
        # Draw Depth Pro distance label in blue
        dp_text = f"DP {i}-{j}: {dist:.3f}m"
        cv2.putText(vis, dp_text, (mid_x-50, mid_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Add distance labels for GeoCalib if available
    if point_pairs_gc:
        for idx, (i, j, P1, P2, dXYZ, dist) in enumerate(point_pairs_gc):
            # Calculate midpoint for text placement
            mid_x = int((clicks[i-1][0] + clicks[j-1][0]) / 2)
            mid_y = int((clicks[i-1][1] + clicks[j-1][1]) / 2)
            
            # Draw GeoCalib distance label in yellow
            gc_text = f"GC {i}-{j}: {dist:.3f}m"
            cv2.putText(vis, gc_text, (mid_x-50, mid_y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Add summary text at the top
    summary_y = 30
    cv2.putText(vis, f"Total points: {len(clicks)}", (10, summary_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Total distances: {len(point_pairs_dp)}", (10, summary_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add legend
    legend_y = summary_y + 60
    cv2.putText(vis, "DP: Depth Pro (Blue)", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if point_pairs_gc:
        cv2.putText(vis, "GC: GeoCalib (Yellow)", (10, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"[Visualization] Measurement visualization saved to {output_path}")
    
    return vis





def save_results_to_json(results_data, output_path):
    """Save all results data to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"💾 Results data saved to: {output_path}")


def main():
    """Main function for 3D distance measurement."""
    parser = argparse.ArgumentParser(description="Click multiple points to measure 3D distances using Depth Pro (and optionally GeoCalib).")
    parser.add_argument("--image", required=True, help="Path to image (jpg/png).")
    parser.add_argument("--mode", choices=["depthpro", "geocalib", "fused"], default="depthpro",
                        help="Which intrinsics to use for back-projection.")
    parser.add_argument("--show_depth", action="store_true", help="Display depth visualization for reference.")
    parser.add_argument("--save_depth", help="Save depth visualization to file.")
    parser.add_argument("--point1", nargs=2, type=float, help="First point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point2", nargs=2, type=float, help="Second point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point3", nargs=2, type=float, help="Third point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point4", nargs=2, type=float, help="Fourth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point5", nargs=2, type=float, help="Fifth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point6", nargs=2, type=float, help="Sixth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point7", nargs=2, type=float, help="Seventh point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point8", nargs=2, type=float, help="Eighth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point9", nargs=2, type=float, help="Ninth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point10", nargs=2, type=float, help="Tenth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point11", nargs=2, type=float, help="Eleventh point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point12", nargs=2, type=float, help="Twelfth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point13", nargs=2, type=float, help="Thirteenth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point14", nargs=2, type=float, help="Fourteenth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point15", nargs=2, type=float, help="Fifteenth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point16", nargs=2, type=float, help="Sixteenth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point17", nargs=2, type=float, help="Seventeenth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point18", nargs=2, type=float, help="Eighteenth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point19", nargs=2, type=float, help="Nineteenth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--point20", nargs=2, type=float, help="Twentieth point (u, v) in pixels (non-interactive mode)")
    parser.add_argument("--ground_truth", type=float, help="Ground truth distance in meters (optional)")

    
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
        # Non-interactive mode - support up to 20 points
        clicks = [args.point1, args.point2]
        
        # Add additional points if provided (up to 20 total)
        for i in range(3, 21):
            point_attr = f'point{i}'
            if hasattr(args, point_attr) and getattr(args, point_attr):
                clicks.append(getattr(args, point_attr))
        
        print(f"\n📏 Non-interactive mode:")
        print(f"   • Total points provided: {len(clicks)}")
        for i, point in enumerate(clicks):
            print(f"   • Point {i+1}: ({point[0]:.1f}, {point[1]:.1f})")
    else:
        # Interactive clicks
        clicks = []
        vis = img.copy()
        cv2.namedWindow("Click multiple points (ESC to finish)", cv2.WINDOW_NORMAL)
        
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((float(x), float(y)))
                # Draw point
                cv2.circle(vis, (x, y), 6, (0, 255, 0), 2)
                # Draw point number
                cv2.putText(vis, str(len(clicks)), (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Draw lines between all points
                for i in range(len(clicks)):
                    for j in range(i+1, len(clicks)):
                        cv2.line(vis, (int(clicks[i][0]), int(clicks[i][1])), 
                                (int(clicks[j][0]), int(clicks[j][1])), (255, 0, 0), 1)
                cv2.imshow("Click multiple points (ESC to finish)", vis)
        
        cv2.setMouseCallback("Click multiple points (ESC to finish)", on_mouse)
        cv2.imshow("Click multiple points (ESC to finish)", vis)
        
        print(f"\n🖱️  Click multiple points on the image to measure distances...")
        print(f"   • Press ESC to finish")
        print(f"   • Points will be numbered sequentially")
        print(f"   • All distances between points will be calculated")
        
        while True:
            key = cv2.waitKey(20)
            if key == 27:  # ESC to finish
                break
        
        cv2.destroyAllWindows()
        
        if len(clicks) < 2:
            print("❌ Need at least two points. Exiting.")
            return
    
    # Measure distances between all point combinations
    print(f"\n🔍 Sampling depths for {len(clicks)} points...")
    
    # Sample depths for all points
    depths = []
    for i, (u, v) in enumerate(clicks):
        Z = robust_depth_sample(depth_m, u, v)
        depths.append(Z)
        print(f"   • Point {i+1} depth: {Z:.3f} meters")
    
    # Check if all depths are valid
    invalid_depths = [i for i, z in enumerate(depths) if not (np.isfinite(z) and z > 0)]
    if invalid_depths:
        print("❌ Invalid depth at selected points:")
        for i in invalid_depths:
            print(f"   • Point {i+1}: {depths[i]:.3f} meters")
        print("   • Try different points")
        print("   • Avoid featureless areas (sky, glass, shadows)")
        print("   • Check depth visualization with --show_depth")
        return
    
    print(f"\n📐 Back-projecting {len(clicks)} points to 3D...")
    for i, ((u, v), Z) in enumerate(zip(clicks, depths)):
        print(f"   • Point {i+1} (2D): ({u:.1f}, {v:.1f}) pixels, depth: {Z:.3f} m")
    
    # Back-project all points to 3D using Depth Pro intrinsics
    print(f"\n🔍 Depth Pro back-projection:")
    print(f"   • Using focal: {f_dp:.2f} pixels")
    print(f"   • Using principal point: ({cx:.2f}, {cy:.2f})")
    
    points_3d_dp = []
    for i, ((u, v), Z) in enumerate(zip(clicks, depths)):
        P = backproject(u, v, Z, f_dp, f_dp, cx, cy)
        points_3d_dp.append(P)
        print(f"   • Point {i+1} (3D): ({P[0]:.4f}, {P[1]:.4f}, {P[2]:.4f}) meters")
    
    # Calculate distances between all combinations
    distances_dp = []
    point_pairs_dp = []
    for i, j in combinations(range(len(clicks)), 2):
        P1, P2 = points_3d_dp[i], points_3d_dp[j]
        dXYZ = P2 - P1
        dist = float(np.linalg.norm(dXYZ))
        distances_dp.append(dist)
        point_pairs_dp.append((i+1, j+1, P1, P2, dXYZ, dist))
        print(f"   • Distance {i+1}-{j+1}: {dist:.4f} meters")
    
    # Back-project all points to 3D using GeoCalib intrinsics (if available)
    points_3d_gc = []
    distances_gc = []
    point_pairs_gc = []
    
    if fx_g is not None and fy_g is not None:
        # Use GeoCalib principal points if available, otherwise use image center
        cx_gc = cx_g if cx_g is not None else cx
        cy_gc = cy_g if cy_g is not None else cy
        
        print(f"\n🔍 GeoCalib back-projection:")
        print(f"   • Using focal: fx={fx_g:.2f}, fy={fy_g:.2f} pixels")
        print(f"   • Using principal point: ({cx_gc:.2f}, {cy_gc:.2f})")
        
        for i, ((u, v), Z) in enumerate(zip(clicks, depths)):
            P = backproject(u, v, Z, fx_g, fy_g, cx_gc, cy_gc)
            points_3d_gc.append(P)
            print(f"   • Point {i+1} (3D): ({P[0]:.4f}, {P[1]:.4f}, {P[2]:.4f}) meters")
        
        # Calculate distances between all combinations for GeoCalib
        for i, j in combinations(range(len(clicks)), 2):
            P1, P2 = points_3d_gc[i], points_3d_gc[j]
            dXYZ = P2 - P1
            dist = float(np.linalg.norm(dXYZ))
            distances_gc.append(dist)
            point_pairs_gc.append((i+1, j+1, P1, P2, dXYZ, dist))
            print(f"   • Distance {i+1}-{j+1}: {dist:.4f} meters")
    else:
        print(f"\n❌ GeoCalib prediction skipped - intrinsics not available")
    
    # Handle ground truth input
    ground_truth_m = args.ground_truth
    if ground_truth_m is None:
        # For multi-point mode, we don't have a single ground truth
        # Just set to None for now
        ground_truth_m = None
    

    
    # Create output folder
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    output_folder = f"3d_multi_{base_name}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create visualizations
    depth_vis_path = os.path.join(output_folder, "depth_map.png")
    depth_vis = create_depth_visualization(depth_m, depth_vis_path)
    
    measurement_vis_path = os.path.join(output_folder, "measurement_visualization.png")
    measurement_vis = create_measurement_visualization(img, clicks, point_pairs_dp, point_pairs_gc, ground_truth_m, measurement_vis_path)
    
    # Save input image with points drawn on it
    input_with_points_path = os.path.join(output_folder, "input_with_points.png")
    input_with_points = img.copy()
    clicks_int = [(int(x), int(y)) for x, y in clicks]
    
    # Draw points on input image
    for i, (x, y) in enumerate(clicks_int):
        cv2.circle(input_with_points, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(input_with_points, (x, y), 8, (0, 0, 0), 2)
        cv2.putText(input_with_points, str(i+1), (x+12, y-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imwrite(input_with_points_path, input_with_points)
    print(f"[Visualization] Input image with points saved to {input_with_points_path}")
    
    # Optional depth visualization display
    if args.show_depth:
        cv2.imshow("Depth (scaled, TURBO)", depth_vis)
        cv2.waitKey(1)
    
    # Print results
    print(f"\n📏 3D Distance Measurement Results:")
    print(f"   • Total points: {len(clicks)}")
    print(f"   • Total distances calculated: {len(point_pairs_dp)}")
    
    for i, ((u, v), Z) in enumerate(zip(clicks, depths)):
        print(f"   • Point {i+1} (2D): ({u:.1f}, {v:.1f}) pixels, depth: {Z:.3f} m")
    
    print(f"\n🔍 Depth Pro Predictions:")
    for i, (point_i, point_j, P1, P2, dXYZ, dist) in enumerate(point_pairs_dp):
        print(f"   • Distance {point_i}-{point_j}: {dist:.4f} meters")
        print(f"     - Point {point_i} (3D): ({P1[0]:.4f}, {P1[1]:.4f}, {P1[2]:.4f}) meters")
        print(f"     - Point {point_j} (3D): ({P2[0]:.4f}, {P2[1]:.4f}, {P2[2]:.4f}) meters")
        print(f"     - Delta (3D): ({dXYZ[0]:.4f}, {dXYZ[1]:.4f}, {dXYZ[2]:.4f}) meters")
    
    if point_pairs_gc:
        print(f"\n🔍 GeoCalib Predictions:")
        for i, (point_i, point_j, P1, P2, dXYZ, dist) in enumerate(point_pairs_gc):
            print(f"   • Distance {point_i}-{point_j}: {dist:.4f} meters")
            print(f"     - Point {point_i} (3D): ({P1[0]:.4f}, {P1[1]:.4f}, {P1[2]:.4f}) meters")
            print(f"     - Point {point_j} (3D): ({P2[0]:.4f}, {P2[1]:.4f}, {P2[2]:.4f}) meters")
            print(f"     - Delta (3D): ({dXYZ[0]:.4f}, {dXYZ[1]:.4f}, {dXYZ[2]:.4f}) meters")
    
    # Summary statistics
    if distances_dp:
        print(f"\n📊 Depth Pro Summary:")
        print(f"   • Min distance: {min(distances_dp):.4f} meters")
        print(f"   • Max distance: {max(distances_dp):.4f} meters")
        print(f"   • Mean distance: {np.mean(distances_dp):.4f} meters")
        print(f"   • Std distance: {np.std(distances_dp):.4f} meters")
    
    if distances_gc:
        print(f"\n📊 GeoCalib Summary:")
        print(f"   • Min distance: {min(distances_gc):.4f} meters")
        print(f"   • Max distance: {max(distances_gc):.4f} meters")
        print(f"   • Mean distance: {np.mean(distances_gc):.4f} meters")
        print(f"   • Std distance: {np.std(distances_gc):.4f} meters")
    

    
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
            "total_points": len(clicks),
            "points_2d": [{"point_id": i+1, "u": u, "v": v, "depth_m": Z} 
                         for i, ((u, v), Z) in enumerate(zip(clicks, depths))],
            "points_3d_dp": [{"point_id": i+1, "x": P[0], "y": P[1], "z": P[2]} 
                            for i, P in enumerate(points_3d_dp)],
            "points_3d_gc": [{"point_id": i+1, "x": P[0], "y": P[1], "z": P[2]} 
                            for i, P in enumerate(points_3d_gc)] if points_3d_gc else None
        },
        "depth_pro_predictions": {
            "total_distances": len(point_pairs_dp),
            "distances": [
                {
                    "point_pair": f"{i}-{j}",
                    "point1_id": i,
                    "point2_id": j,
                    "point1_3d": {"x": P1[0], "y": P1[1], "z": P1[2]},
                    "point2_3d": {"x": P2[0], "y": P2[1], "z": P2[2]},
                    "delta_3d": {"x": dXYZ[0], "y": dXYZ[1], "z": dXYZ[2]},
                    "distance_m": dist
                }
                for i, j, P1, P2, dXYZ, dist in point_pairs_dp
            ],
            "summary": {
                "min_distance": min(distances_dp) if distances_dp else None,
                "max_distance": max(distances_dp) if distances_dp else None,
                "mean_distance": np.mean(distances_dp) if distances_dp else None,
                "std_distance": np.std(distances_dp) if distances_dp else None
            }
        },
        "geocalib_predictions": {
            "available": len(point_pairs_gc) > 0,
            "total_distances": len(point_pairs_gc),
            "focal_length": {
                "fx": fx_g,
                "fy": fy_g
            } if fx_g is not None else None,
            "principal_point": {
                "cx": cx_g,
                "cy": cy_g
            } if cx_g is not None else None,
            "focal_uncertainty": focal_uncertainty,
            "gravity_direction": gravity.data.tolist() if gravity is not None and hasattr(gravity, 'data') else None,
            "distances": [
                {
                    "point_pair": f"{i}-{j}",
                    "point1_id": i,
                    "point2_id": j,
                    "point1_3d": {"x": P1[0], "y": P1[1], "z": P1[2]},
                    "point2_3d": {"x": P2[0], "y": P2[1], "z": P2[2]},
                    "delta_3d": {"x": dXYZ[0], "y": dXYZ[1], "z": dXYZ[2]},
                    "distance_m": dist
                }
                for i, j, P1, P2, dXYZ, dist in point_pairs_gc
            ],
            "summary": {
                "min_distance": min(distances_gc) if distances_gc else None,
                "max_distance": max(distances_gc) if distances_gc else None,
                "mean_distance": np.mean(distances_gc) if distances_gc else None,
                "std_distance": np.std(distances_gc) if distances_gc else None
            }
        } if point_pairs_gc else None,
        "depth_map_info": {
            "depth_range_m": {"min": float(depth_m.min()), "max": float(depth_m.max())},
            "depth_map_shape": depth_m.shape
        },
        "output_files": {
            "depth_map": depth_vis_path,
            "measurement_visualization": measurement_vis_path,
            "input_with_points": input_with_points_path,
            "data_json": os.path.join(output_folder, "data.json")
        }
    }
    
    # Save results to JSON
    json_path = os.path.join(output_folder, "data.json")
    save_results_to_json(results_data, json_path)
    
    print(f"\n💾 All results saved to folder: {output_folder}")
    print(f"   • Input image with points: {input_with_points_path}")
    print(f"   • Depth map: {depth_vis_path}")
    print(f"   • Measurement visualization: {measurement_vis_path}")
    print(f"   • Data: {json_path}")


if __name__ == "__main__":
    main()
