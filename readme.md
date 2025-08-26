# 3D Distance Measurement System

A comprehensive computer vision system for measuring 3D distances from single images using state-of-the-art deep learning models.

## 🎯 Overview

This system combines two cutting-edge technologies:
- **Depth Pro**: Apple's monocular depth estimation model for metric depth prediction
- **GeoCalib**: Single-image camera calibration for improved accuracy

The system provides both single-point and multi-point measurement capabilities with interactive and non-interactive modes.

## 🏗️ System Architecture

### Core Components

1. **Depth Pro Pipeline**: Monocular depth estimation in meters
2. **GeoCalib Pipeline**: Camera intrinsics estimation (focal length, principal point)
3. **3D Back-projection**: Converts 2D image points to 3D world coordinates
4. **Distance Calculation**: Euclidean distance between 3D points
5. **Visualization**: Depth maps and measurement overlays

### Pipeline Flow

```
Input Image → Depth Pro → Depth Map → Point Selection → 3D Back-projection → Distance Calculation
     ↓
GeoCalib → Camera Intrinsics → Improved Accuracy
```

## 📁 Project Structure

```
3d_distances/
├── measure_3d_distance.py          # Single-point measurement pipeline
├── measure_3d_distance_multi.py    # Multi-point measurement pipeline (up to 20 points)
├── setup.py                        # Automated setup script
├── requirements.txt                # Python dependencies
├── src/geocalib/                   # GeoCalib source code
├── ml-depth-pro/                   # Depth Pro source code
├── checkpoints/                    # Model checkpoints
└── 3d_multi_image/                 # Output folder for multi-point results
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone <repository-url>
cd VisionScale

# Run automated setup
python3 setup.py
```

The setup script will:
- ✅ Install Python dependencies
- ✅ Download and install Depth Pro
- ✅ Download and install GeoCalib
- ✅ Download pretrained models
- ✅ Verify GPU availability

### 2. Basic Usage

#### Single-Point Measurement
```bash
# Interactive mode (click two points)
python3 measure_3d_distance.py --image your_image.jpg --mode fused

# Non-interactive mode (predefined points)
python3 measure_3d_distance.py --image your_image.jpg --point1 100 200 --point2 300 400
```

#### Multi-Point Measurement
```bash
# Interactive mode (click multiple points)
python3 measure_3d_distance_multi.py --image your_image.jpg --mode fused

# Non-interactive mode (up to 20 points)
python3 measure_3d_distance_multi.py --image your_image.jpg \
  --point1 100 200 --point2 300 400 --point3 500 600 \
  --point4 700 800 --point5 900 1000
```

## 🔧 Detailed Pipeline Explanation

### 1. Single-Point Pipeline (`measure_3d_distance.py`)

**Purpose**: Measure distance between exactly two points in an image.

**Workflow**:
1. **Image Loading**: Load and validate input image
2. **Depth Estimation**: Run Depth Pro to get depth map in meters
3. **Camera Calibration**: Run GeoCalib to estimate camera intrinsics
4. **Point Selection**: Interactive clicking or command-line input
5. **Depth Sampling**: Robust depth sampling with bilinear interpolation
6. **3D Back-projection**: Convert 2D points to 3D using camera intrinsics
7. **Distance Calculation**: Euclidean distance between 3D points
8. **Output Generation**: JSON data, visualizations, and analysis

**Modes**:
- `depthpro`: Uses only Depth Pro intrinsics
- `geocalib`: Uses only GeoCalib intrinsics
- `fused`: Combines both methods (geometric mean)

**Output**:
- `3d_[filename]/` folder containing:
  - `data.json`: Complete measurement data
  - `depth_map.png`: Depth visualization
  - `measurement_visualization.png`: Measurement overlay

### 2. Multi-Point Pipeline (`measure_3d_distance_multi.py`)

**Purpose**: Measure distances between multiple points (up to 20) and calculate all combinations.

**Workflow**:
1. **Image Loading**: Load and validate input image
2. **Depth Estimation**: Run Depth Pro to get depth map in meters
3. **Camera Calibration**: Run GeoCalib to estimate camera intrinsics
4. **Point Selection**: Interactive clicking (unlimited) or command-line (up to 20)
5. **Depth Sampling**: Sample depths for all points
6. **3D Back-projection**: Convert all 2D points to 3D
7. **Combination Calculation**: Calculate distances between all point pairs
8. **Output Generation**: Comprehensive results and visualizations

**Features**:
- **Unlimited Points**: Interactive mode supports unlimited points
- **Command-line Support**: Up to 20 points via command-line arguments
- **All Combinations**: Calculates distances between every point pair
- **Duplicate Prevention**: Only calculates each unique pair once (1-2, not 2-1)
- **Statistical Analysis**: Min, max, mean, standard deviation of distances

**Output**:
- `3d_multi_image/` folder containing:
  - `input_with_points.png`: Original image with numbered points
  - `depth_map.png`: Depth visualization
  - `measurement_visualization.png`: All measurements with color coding
  - `data.json`: Complete data with all point combinations

## 🎨 Visualization Features

### Single-Point Visualization
- Points numbered 1 and 2
- Distance line between points
- Depth Pro and GeoCalib predictions
- Ground truth comparison (if provided)
- Error analysis

### Multi-Point Visualization
- All points numbered sequentially
- All connections between points
- Color-coded measurements:
  - **Blue**: Depth Pro predictions
  - **Yellow**: GeoCalib predictions
- Distance labels on each connection
- Summary statistics overlay

## 📊 Output Data Structure

### JSON Output Format
```json
{
  "image_path": "path/to/image.jpg",
  "image_size": {"width": 1920, "height": 1080},
  "mode": "fused",
  "camera_intrinsics": {
    "depth_pro_focal": 1234.56,
    "geocalib_focal": {"fx": 1234.56, "fy": 1234.56},
    "used_principal_point": {"cx": 960, "cy": 540}
  },
  "measurement_points": {
    "total_points": 4,
    "points_2d": [...],
    "points_3d_dp": [...],
    "points_3d_gc": [...]
  },
  "depth_pro_predictions": {
    "total_distances": 6,
    "distances": [...],
    "summary": {
      "min_distance": 1.234,
      "max_distance": 5.678,
      "mean_distance": 3.456,
      "std_distance": 1.789
    }
  },
  "geocalib_predictions": {...},
  "depth_map_info": {...},
  "output_files": {...}
}
```

## ⚙️ Advanced Usage

### Command Line Options

#### Single-Point Pipeline
```bash
python3 measure_3d_distance.py \
  --image photo.jpg \
  --mode fused \
  --show_depth \
  --point1 100 200 \
  --point2 300 400 \
  --ground_truth 0.5
```

#### Multi-Point Pipeline
```bash
python3 measure_3d_distance_multi.py \
  --image photo.jpg \
  --mode fused \
  --point1 100 200 \
  --point2 300 400 \
  --point3 500 600 \
  --point4 700 800 \
  --point5 900 1000
```

### Parameters
- `--image`: Input image path (required)
- `--mode`: Measurement mode (`depthpro`, `geocalib`, `fused`)
- `--show_depth`: Display depth visualization
- `--point1` to `--point20`: Predefined point coordinates (u, v)
- `--ground_truth`: Known distance for comparison (single-point only)

## 🔬 Technical Details

### Depth Pro Integration
- **Model**: Apple's Depth Pro for monocular depth estimation
- **Output**: Metric depth map in meters
- **Focal Length**: Estimated from image EXIF or model prediction
- **Accuracy**: State-of-the-art depth estimation

### GeoCalib Integration
- **Model**: Single-image camera calibration
- **Output**: Camera intrinsics (fx, fy, cx, cy)
- **Features**: Gravity direction estimation
- **Accuracy**: Improved distance measurement accuracy

### 3D Back-projection
```python
# Camera frame: X right, Y down, Z forward (meters)
X = (u - cx) / fx * Z
Y = (v - cy) / fy * Z
Z = depth_at_point
```

### Robust Depth Sampling
- **Bilinear Interpolation**: Subpixel depth sampling
- **Median Fallback**: 3x3 median filter for invalid depths
- **Validation**: Checks for finite, positive depth values

## 🛠️ System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for models and dependencies

### Software
- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher (for GPU acceleration)
- **OS**: Linux, macOS, or Windows

### Dependencies
- PyTorch >= 1.9.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- Pillow >= 8.3.0
- Matplotlib >= 3.4.0
- SciPy >= 1.7.0

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall dependencies
   python3 setup.py
   ```

2. **GPU Issues**
   - System automatically falls back to CPU
   - Check CUDA installation: `nvidia-smi`

3. **Model Download Failures**
   - Check internet connection
   - Verify disk space
   - Retry setup script

4. **Depth Sampling Issues**
   - Avoid featureless areas (sky, glass, shadows)
   - Try different points
   - Use `--show_depth` to visualize depth map

### Performance Optimization

1. **GPU Usage**: Ensure CUDA is properly installed
2. **Memory**: Close other applications during processing
3. **Batch Processing**: Use non-interactive mode for multiple images

## 📈 Performance Metrics

### Accuracy
- **Depth Pro**: State-of-the-art monocular depth estimation
- **GeoCalib**: Improved camera calibration accuracy
- **Fused Mode**: Best overall performance

### Speed
- **GPU**: ~2-5 seconds per image
- **CPU**: ~10-30 seconds per image
- **Multi-point**: Scales with number of point combinations

## 🔮 Future Enhancements

- **Batch Processing**: Process multiple images simultaneously
- **Real-time Processing**: Video stream support
- **Additional Models**: Support for other depth estimation models
- **Web Interface**: Browser-based point selection
- **API Integration**: REST API for remote processing

## 📚 References

### Research Papers
- Depth Pro: [Apple's Depth Pro Paper]
- GeoCalib: [ECCV 2024 Paper]

### Model Repositories
- Depth Pro: https://github.com/apple/ml-depth-pro
- GeoCalib: https://github.com/cvg/GeoCalib



## 🙏 Acknowledgements

### Research Teams
- **Apple Research**: For developing and open-sourcing Depth Pro, a state-of-the-art monocular depth estimation model that provides metric depth predictions from single images.
- **Computer Vision Group (CVG)**: For developing GeoCalib, an innovative single-image camera calibration system that estimates camera intrinsics and gravity direction from a single image.

### Model Contributors
- **Depth Pro Team**: Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, Marc Pollefeys
- **GeoCalib Team**: Computer Vision Group at ETH Zurich

### Open Source Libraries
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization library
- **SciPy**: Scientific computing

### Development Tools
- **Git**: Version control
- **Python**: Programming language
- **CUDA**: GPU acceleration

### Community
- **Open Source Community**: For providing the foundation libraries and tools that make this project possible
- **Research Community**: For advancing the state-of-the-art in computer vision and depth estimation

### Special Thanks
- **Apple**: For making Depth Pro available to the research community
- **ETH Zurich**: For developing and open-sourcing GeoCalib
- **All Contributors**: For their valuable feedback and contributions to this project

---

**Note**: This system combines cutting-edge research from multiple institutions to provide accurate 3D distance measurements from single images. The integration of Depth Pro and GeoCalib represents a novel approach to monocular 3D measurement.
