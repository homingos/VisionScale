#!/usr/bin/env python3
"""
Setup script for 3D Distance Measurement System
===============================================

This script sets up the environment for 3D distance measurement using:
- Depth Pro: Monocular depth estimation
- GeoCalib: Camera intrinsics estimation
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import urllib.request
import zipfile

def run_command(command, description, cwd=None):
    """Run a shell command and handle errors."""
    print(f"\n🔧 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, cwd=cwd)
        print(f"   ✅ Success")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False

def download_file(url, filename, description):
    """Download a file with progress indication."""
    print(f"\n📥 {description}")
    print(f"   URL: {url}")
    print(f"   File: {filename}")
    
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"   ✅ Download complete")
        return True
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  GPU not available, will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")
        return False

def install_requirements():
    """Install Python requirements."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        return run_command(
            f"pip3 install -r {requirements_file}",
            "Installing Python requirements"
        )
    else:
        print("⚠️  requirements.txt not found")
        return True

def setup_depth_pro():
    """Set up Depth Pro."""
    print("\n📦 Setting up Depth Pro...")
    
    # Check if already installed
    try:
        import depth_pro
        print("✅ Depth Pro already installed")
        return True
    except ImportError:
        pass
    
    # Create ml-depth-pro directory if it doesn't exist
    ml_depth_pro_dir = Path(__file__).parent / "ml-depth-pro"
    ml_depth_pro_dir.mkdir(exist_ok=True)
    
    # Check if repository already exists
    if (ml_depth_pro_dir / ".git").exists():
        print("✅ Depth Pro repository already exists")
    else:
        # Clone repository
        if not run_command(
            "git clone https://github.com/apple/ml-depth-pro.git",
            "Cloning Depth Pro repository",
            cwd=Path(__file__).parent
        ):
            return False
    
    # Install Depth Pro
    if not run_command(
        "pip3 install -e .",
        "Installing Depth Pro",
        cwd=ml_depth_pro_dir
    ):
        return False
    
    # Create checkpoints directory
    checkpoints_dir = ml_depth_pro_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Download pretrained models
    model_url = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
    model_path = checkpoints_dir / "depth_pro.pt"
    
    if not model_path.exists():
        if not download_file(model_url, str(model_path), "Downloading Depth Pro model"):
            return False
    else:
        print("✅ Depth Pro model already exists")
    
    print("✅ Depth Pro setup complete")
    return True

def setup_geocalib():
    """Set up GeoCalib."""
    print("\n📦 Setting up GeoCalib...")
    
    # Check if already installed
    try:
        from geocalib import GeoCalib
        print("✅ GeoCalib already installed")
        return True
    except ImportError:
        pass
    
    # Create src directory if it doesn't exist
    src_dir = Path(__file__).parent / "src"
    src_dir.mkdir(exist_ok=True)
    
    # Check if GeoCalib repository already exists
    geocalib_dir = src_dir / "geocalib"
    if geocalib_dir.exists():
        print("✅ GeoCalib repository already exists")
    else:
        # Clone GeoCalib repository
        if not run_command(
            "git clone https://github.com/cvg/GeoCalib.git geocalib",
            "Cloning GeoCalib repository",
            cwd=src_dir
        ):
            return False
    
    # Install GeoCalib
    if not run_command(
        "pip3 install -e .",
        "Installing GeoCalib",
        cwd=geocalib_dir
    ):
        return False
    
    print("✅ GeoCalib setup complete")
    return True

def setup_additional_models():
    """Set up additional models and checkpoints."""
    print("\n📦 Setting up additional models...")
    
    # Create checkpoints directory
    checkpoints_dir = Path(__file__).parent / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Download additional models if needed
    models_to_download = {
        "depth_pro.pt": "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt",
    }
    
    for model_name, model_url in models_to_download.items():
        model_path = checkpoints_dir / model_name
        if not model_path.exists():
            if not download_file(model_url, str(model_path), f"Downloading {model_name}"):
                print(f"⚠️  Failed to download {model_name}, continuing...")
        else:
            print(f"✅ {model_name} already exists")
    
    print("✅ Additional models setup complete")
    return True

def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    # Test imports
    try:
        import depth_pro
        print("✅ Depth Pro import successful")
    except ImportError as e:
        print(f"❌ Depth Pro import failed: {e}")
        return False
    
    try:
        from geocalib import GeoCalib
        print("✅ GeoCalib import successful")
    except ImportError as e:
        print(f"❌ GeoCalib import failed: {e}")
        return False
    
    # Test basic functionality
    try:
        import cv2
        import numpy as np
        import torch
        print("✅ OpenCV, NumPy, and PyTorch import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test GeoCalib functionality
    try:
        from geocalib import GeoCalib
        import torch
        
        # Create a simple test
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = GeoCalib().to(device)
        print("✅ GeoCalib model creation successful")
    except Exception as e:
        print(f"❌ GeoCalib model creation failed: {e}")
        return False
    
    print("✅ All tests passed!")
    return True

def create_example_script():
    """Create an example usage script."""
    example_script = """#!/usr/bin/env python3
\"\"\"
Example usage of 3D Distance Measurement Tool
============================================

This script demonstrates how to use the 3D distance measurement tool.
\"\"\"

import os
from measure_3d_distance import main
import sys

def run_example():
    \"\"\"Run an example measurement.\"\"\"
    # Check if we have a test image
    test_images = [
        "test_image.jpg",
        "sample.jpg", 
        "example.png"
    ]
    
    for img in test_images:
        if os.path.exists(img):
            print(f"📸 Found test image: {img}")
            print(f"🔧 Running measurement...")
            
            # Set up command line arguments
            sys.argv = [
                "measure_3d_distance.py",
                "--image", img,
                "--mode", "fused",
                "--show_depth"
            ]
            
            # Run measurement
            main()
            return
    
    print("❌ No test image found.")
    print("   Please place a test image (jpg/png) in the current directory")
    print("   or run: python measure_3d_distance.py --image your_image.jpg")

if __name__ == "__main__":
    run_example()
"""
    
    with open("example_usage.py", "w") as f:
        f.write(example_script)
    
    print("✅ Created example_usage.py")

def create_readme():
    """Create a comprehensive README file."""
    readme_content = """# 3D Distance Measurement System

This system measures 3D distances between two points in images using:
- **Depth Pro**: Monocular depth estimation in meters
- **GeoCalib**: Camera intrinsics estimation for improved accuracy

## Features

- Interactive point selection
- Dual prediction system (Depth Pro + GeoCalib)
- Ground truth comparison
- Organized output with JSON data
- Depth map visualization
- Multiple camera models support

## Installation

Run the setup script to install all dependencies:

```bash
python3 setup.py
```

This will:
1. Install Python requirements
2. Download and install Depth Pro
3. Download and install GeoCalib
4. Download pretrained models
5. Test the installation

## Usage

### Basic Usage

```bash
python3 measure_3d_distance.py --image your_image.jpg
```

### Advanced Usage

```bash
# With depth visualization
python3 measure_3d_distance.py --image your_image.jpg --show_depth

# Different modes
python3 measure_3d_distance.py --image your_image.jpg --mode depthpro
python3 measure_3d_distance.py --image your_image.jpg --mode geocalib
python3 measure_3d_distance.py --image your_image.jpg --mode fused

# With ground truth
python3 measure_3d_distance.py --image your_image.jpg --ground_truth 0.5

# Pre-selected points
python3 measure_3d_distance.py --image your_image.jpg --point1 100 200 --point2 300 400
```

### Modes

- **depthpro**: Uses only Depth Pro for focal length estimation
- **geocalib**: Uses only GeoCalib for camera intrinsics
- **fused**: Combines both methods for robust estimation (default)

## Output

The system creates a folder named `3d_[filename]` containing:
- `data.json`: Comprehensive measurement data
- `depth_map.png`: Depth map visualization
- `measurement_visualization.png`: Measurement visualization

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- PIL (Pillow)

## Troubleshooting

1. **Import errors**: Run `python3 setup.py` to reinstall dependencies
2. **Model download issues**: Check internet connection and try again
3. **GPU issues**: The system will automatically fall back to CPU

## Examples

See `example_usage.py` for a complete example.
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created README.md")

def main():
    """Main setup function."""
    print("🚀 3D Distance Measurement System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check GPU
    check_gpu()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Setup Depth Pro
    if not setup_depth_pro():
        print("❌ Failed to setup Depth Pro")
        return False
    
    # Setup GeoCalib
    if not setup_geocalib():
        print("❌ Failed to setup GeoCalib")
        return False
    
    # Setup additional models
    if not setup_additional_models():
        print("❌ Failed to setup additional models")
        return False
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        return False
    
    # Create example script and README
    create_example_script()
    create_readme()
    
    print("\n🎉 Setup complete!")
    print("\n📖 Usage:")
    print("   • Interactive mode: python3 measure_3d_distance.py --image your_image.jpg")
    print("   • With depth visualization: python3 measure_3d_distance.py --image your_image.jpg --show_depth")
    print("   • Different modes: --mode depthpro, --mode geocalib, --mode fused")
    print("   • Example: python3 example_usage.py")
    print("\n📚 Documentation:")
    print("   • README.md: Complete usage guide")
    print("   • example_usage.py: Example script")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
