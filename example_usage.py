#!/usr/bin/env python3
"""
Example Usage of 3D Distance Measurement System
==============================================

This script demonstrates how to use the 3D distance measurement tool.
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if all dependencies are available."""
    print("🔍 Checking dependencies...")
    
    # Check basic imports
    try:
        import numpy as np
        import cv2
        print("✅ Basic dependencies (NumPy, OpenCV) available")
    except ImportError as e:
        print(f"❌ Basic dependencies missing: {e}")
        return False
    
    # Check Depth Pro
    try:
        import depth_pro
        print("✅ Depth Pro available")
    except ImportError:
        print("❌ Depth Pro not available")
        print("   Run: python setup.py")
        return False
    
    # Check GeoCalib
    try:
        from geocalib import GeoCalib
        print("✅ GeoCalib available")
    except ImportError:
        print("⚠️  GeoCalib not available (optional)")
    
    return True

def find_test_images():
    """Find test images in the current directory."""
    print("\n📸 Looking for test images...")
    
    # Common test image names
    test_names = [
        "test_image.jpg", "test_image.png",
        "sample.jpg", "sample.png",
        "example.jpg", "example.png",
        "room.jpg", "room.png",
        "indoor.jpg", "indoor.png",
        "outdoor.jpg", "outdoor.png"
    ]
    
    found_images = []
    for name in test_names:
        if os.path.exists(name):
            found_images.append(name)
            print(f"   ✅ Found: {name}")
    
    if not found_images:
        print("   ❌ No test images found")
        print("   📝 Place a test image in the current directory")
        print("   📝 Supported formats: jpg, jpeg, png, bmp, tiff")
    
    return found_images

def run_interactive_example(image_path):
    """Run interactive example with user clicks."""
    print(f"\n🎬 Running interactive example with {image_path}")
    print("   • A window will open showing the image")
    print("   • Click two points to measure distance")
    print("   • Press ESC to finish")
    
    try:
        from measure_3d_distance import main
        
        # Set up command line arguments
        original_argv = sys.argv
        sys.argv = [
            "measure_3d_distance.py",
            "--image", image_path,
            "--mode", "fused",
            "--show_depth"
        ]
        
        # Run the measurement
        main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print("✅ Interactive example completed")
        
    except Exception as e:
        print(f"❌ Interactive example failed: {e}")
        print("   This might be due to missing dependencies")

def run_non_interactive_example(image_path):
    """Run non-interactive example with predefined points."""
    print(f"\n🔧 Running non-interactive example with {image_path}")
    print("   • Using predefined points (100,100) and (300,300)")
    
    try:
        from measure_3d_distance import main
        
        # Set up command line arguments
        original_argv = sys.argv
        sys.argv = [
            "measure_3d_distance.py",
            "--image", image_path,
            "--mode", "depthpro",
            "--point1", "100", "100",
            "--point2", "300", "300"
        ]
        
        # Run the measurement
        main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print("✅ Non-interactive example completed")
        
    except Exception as e:
        print(f"❌ Non-interactive example failed: {e}")

def run_batch_example():
    """Run batch processing example if multiple images are available."""
    print(f"\n📦 Running batch processing example...")
    
    # Find all images in current directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    images = []
    
    for ext in image_extensions:
        images.extend(Path('.').glob(f"*{ext}"))
        images.extend(Path('.').glob(f"*{ext.upper()}"))
    
    images = [str(img) for img in images]
    
    if len(images) < 2:
        print("   ⚠️  Need at least 2 images for batch processing")
        return
    
    print(f"   📸 Found {len(images)} images for batch processing")
    
    try:
        from batch_measure import main as batch_main
        
        # Set up command line arguments
        original_argv = sys.argv
        sys.argv = [
            "batch_measure.py",
            "--input", ".",
            "--point1", "100,100",
            "--point2", "300,300",
            "--mode", "depthpro",
            "--output", "example_batch_results"
        ]
        
        # Run batch processing
        batch_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print("✅ Batch processing example completed")
        
    except Exception as e:
        print(f"❌ Batch processing example failed: {e}")

def show_usage_instructions():
    """Show usage instructions."""
    print(f"\n📖 USAGE INSTRUCTIONS")
    print(f"=" * 50)
    print(f"")
    print(f"🔧 Basic Usage:")
    print(f"   • Interactive mode: python measure_3d_distance.py --image your_image.jpg")
    print(f"   • Non-interactive: python measure_3d_distance.py --image your_image.jpg --point1 100 100 --point2 200 200")
    print(f"")
    print(f"🎛️  Modes:")
    print(f"   • --mode depthpro    (Depth Pro only, fastest)")
    print(f"   • --mode geocalib    (GeoCalib only, most accurate)")
    print(f"   • --mode fused       (Combined, best of both)")
    print(f"")
    print(f"📊 Options:")
    print(f"   • --show_depth       (Display depth visualization)")
    print(f"   • --save_depth file  (Save depth map to file)")
    print(f"")
    print(f"📦 Batch Processing:")
    print(f"   • python batch_measure.py --input directory --point1 100,100 --point2 200,200")
    print(f"")
    print(f"🧪 Testing:")
    print(f"   • python test_system.py")
    print(f"   • python setup.py")

def main():
    """Main example function."""
    print("🚀 3D Distance Measurement System - Example Usage")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies not satisfied")
        print("   Run: python setup.py")
        return
    
    # Find test images
    test_images = find_test_images()
    
    if not test_images:
        show_usage_instructions()
        return
    
    # Run examples
    print(f"\n🎯 Running examples...")
    
    # Example 1: Interactive with first image
    if test_images:
        run_interactive_example(test_images[0])
    
    # Example 2: Non-interactive with first image
    if test_images:
        run_non_interactive_example(test_images[0])
    
    # Example 3: Batch processing if multiple images
    if len(test_images) >= 2:
        run_batch_example()
    
    # Show usage instructions
    show_usage_instructions()
    
    print(f"\n🎉 Examples completed!")
    print(f"   • Check the output files for results")
    print(f"   • Try different modes and options")
    print(f"   • Use your own images for testing")

if __name__ == "__main__":
    main()
