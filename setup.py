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
    
    print("\n🎉 Setup complete!")
    print("\n📖 Usage:")
    print("   • Interactive mode: python3 measure_3d_distance.py --image your_image.jpg")
    print("   • With depth visualization: python3 measure_3d_distance.py --image your_image.jpg --show_depth")
    print("   • Different modes: --mode depthpro, --mode geocalib, --mode fused")
    print("   • Multi-point: python3 measure_3d_distance_multi.py --image your_image.jpg")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
