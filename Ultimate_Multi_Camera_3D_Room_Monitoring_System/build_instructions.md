# Build Instructions

## Prerequisites

### System Requirements
- Ubuntu 20.04+ or Windows 10+ with WSL2
- 8GB+ RAM recommended
- CUDA-capable GPU (optional but recommended)
- Multiple USB cameras or video files

### Required Software
- CMake 3.20+
- C++17 compatible compiler (GCC 9+ or MSVC 2019+)
- Python 3.8+
- Git

## Step-by-Step Build Process

### 1. Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopencv-dev \
    libeigen3-dev \
    libsqlite3-dev \
    libyaml-cpp-dev \
    python3 \
    python3-pip \
    wget \
    unzip
```

#### Windows (with vcpkg):
```powershell
# Install vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install packages
.\vcpkg install opencv eigen3 sqlite3 yaml-cpp
```

### 2. Install ONNX Runtime

#### Linux:
```bash
# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp -r onnxruntime-linux-x64-1.16.0/* /usr/local/
```

#### Windows:
```powershell
# Download and extract ONNX Runtime
Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-win-x64-1.16.0.zip" -OutFile "onnxruntime.zip"
Expand-Archive -Path "onnxruntime.zip" -DestinationPath "C:\onnxruntime"
```

### 3. Install ORB-SLAM3 (Optional)

```bash
# Clone ORB-SLAM3
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3

# Install dependencies
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libatlas-base-dev libeigen3-dev
sudo apt-get install libsuitesparse-dev

# Build
chmod +x build.sh
./build.sh

# Set environment variables
export ORB_SLAM3_ROOT=/path/to/ORB_SLAM3
```

### 4. Install Python Dependencies

```bash
pip3 install opencv-python numpy pyyaml open3d
```

### 5. Build the Project

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT=/usr/local \
    -DORB_SLAM3_ROOT=/path/to/ORB_SLAM3

# Build
make -j$(nproc)  # Linux
# or
cmake --build . --config Release  # Windows
```

### 6. Download Models

```bash
# Make script executable
chmod +x download_models.sh

# Download models
./download_models.sh
```

## Configuration

### 1. Camera Calibration

```bash
# Run calibration script
python3 scripts/calibrate.py --cameras 2

# This will create config/calibration.yaml
```

### 2. Update Configuration Files

Edit the following files for your setup:

- `config/cameras.yaml`: Camera parameters and sources
- `config/tracking.yaml`: Detection and tracking parameters
- `config/stereo.yaml`: ORB-SLAM3 configuration

### 3. Test the Build

```bash
# Test monitor application
./build/monitor/room_monitor

# Test mapper application (if ORB-SLAM3 is available)
./build/mapper/room_mapper

# Run Python tests
python3 scripts/test_tracker.py
```

## Troubleshooting

### Common Build Issues

1. **ONNX Runtime not found**:
   ```bash
   export ONNXRUNTIME_ROOT=/usr/local
   # or set in CMakeLists.txt
   ```

2. **OpenCV not found**:
   ```bash
   sudo apt-get install libopencv-dev
   # or use vcpkg on Windows
   ```

3. **ORB-SLAM3 not found**:
   ```bash
   # Either install ORB-SLAM3 or disable mapper
   cmake .. -DORB_SLAM3_ROOT=""
   ```

4. **CUDA not available**:
   ```bash
   # Install CUDA toolkit
   sudo apt-get install nvidia-cuda-toolkit
   ```

### Performance Optimization

1. **Enable GPU acceleration**:
   - Install CUDA toolkit
   - Set `use_gpu=true` in ONNXRuntime constructors

2. **Optimize build**:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
   ```

3. **Reduce model size**:
   - Use quantized models (INT8)
   - Use smaller model variants

## Docker Build (Alternative)

```dockerfile
FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libopencv-dev libeigen3-dev \
    libsqlite3-dev libyaml-cpp-dev \
    python3 python3-pip

# Install ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.0.tgz \
    && cp -r onnxruntime-linux-x64-1.16.0/* /usr/local/

# Build project
COPY . /app
WORKDIR /app
RUN mkdir build && cd build && cmake .. && make -j4

CMD ["./build/monitor/room_monitor"]
```

## Verification

After building, verify the installation:

```bash
# Check executables
ls -la build/monitor/room_monitor
ls -la build/mapper/room_mapper

# Check models
ls -la models/

# Test with sample data
./build/monitor/room_monitor --help
```

## Next Steps

1. Configure your cameras in `config/cameras.yaml`
2. Run camera calibration
3. Start with mapping phase (optional)
4. Begin monitoring phase
5. Set up alerts and notifications
