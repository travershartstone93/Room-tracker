# Ultimate Multi-Camera 3D Room Monitoring System

A comprehensive real-time 3D object tracking and monitoring system using multiple cameras, computer vision, and SLAM technology.

## Features

- **Multi-Camera Fusion**: Combines multiple camera views for accurate 3D positioning
- **Real-Time Object Detection**: YOLOv8-based detection with pose estimation
- **3D Depth Estimation**: MiDaS and Depth-Anything models for accurate depth
- **Object Tracking**: Kalman filter-based tracking with occlusion handling
- **Event Detection**: Detects object movement, picking up, and state changes
- **SLAM Integration**: ORB-SLAM3 for mapping and localization
- **Database Logging**: SQLite database for event and object persistence

## System Architecture

### Components

1. **Common Library** (`common/`)
   - Shared types and data structures
   - ONNX Runtime wrapper
   - Geometry utilities
   - Configuration management

2. **Monitor Application** (`monitor/`)
   - Real-time object detection and tracking
   - Multi-camera fusion
   - Event detection and logging
   - Database integration

3. **Mapper Application** (`mapper/`)
   - SLAM-based room mapping
   - Object detection for mapping
   - Point cloud export
   - Known object database creation

4. **Scripts** (`scripts/`)
   - Camera calibration utilities
   - Visualization tools
   - Testing framework

## Dependencies

### Required Libraries
- OpenCV 4.5+
- Eigen3 3.3+
- SQLite3
- yaml-cpp
- ONNX Runtime
- ORB-SLAM3 (optional, for mapping)

### Python Dependencies
- opencv-python
- numpy
- pyyaml
- open3d

## Installation

### 1. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake git
sudo apt-get install libopencv-dev libeigen3-dev libsqlite3-dev libyaml-cpp-dev

# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp -r onnxruntime-linux-x64-1.16.0/* /usr/local/

# Install ORB-SLAM3 (optional)
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3
chmod +x build.sh
./build.sh
```

### 2. Build the Project

```bash
mkdir build
cd build
cmake ..
make -j4
```

### 3. Download Models

```bash
chmod +x download_models.sh
./download_models.sh
```

## Configuration

### Camera Setup

1. **Calibrate Cameras**:
```bash
python scripts/calibrate.py --cameras 2
```

2. **Configure Camera Parameters**:
Edit `config/cameras.yaml` with your camera intrinsics and extrinsics.

3. **Configure Tracking Parameters**:
Edit `config/tracking.yaml` to adjust detection thresholds and behavior.

## Usage

### 1. Mapping Phase (Optional)

Create a 3D map of your room and identify known objects:

```bash
./build/mapper/room_mapper
```

This will:
- Create a 3D point cloud map (`room_map.ply`)
- Identify and catalog objects in the room
- Save known objects to database

### 2. Monitoring Phase

Start real-time monitoring:

```bash
./build/monitor/room_monitor
```

The system will:
- Detect and track objects in real-time
- Log events to database
- Print alerts to console

### 3. Visualization

View the 3D map:
```bash
python scripts/visualize_map.py room_map.ply
```

## Configuration Files

### `config/cameras.yaml`
Camera intrinsics, extrinsics, and source configuration.

### `config/tracking.yaml`
Detection thresholds, tracking parameters, and alert settings.

### `config/stereo.yaml`
ORB-SLAM3 configuration for stereo camera setup.

## Database Schema

The system uses SQLite with the following tables:
- `known_objects`: Objects identified during mapping
- `objects`: Current tracked objects
- `events`: Object movement and state change events
- `people`: Person tracking data
- `camera_status`: Camera health monitoring

## Event Types

- `appeared`: New object detected
- `disappeared`: Object lost from tracking
- `moved`: Object moved beyond threshold
- `picked_up`: Object picked up by person
- `put_down`: Object placed down

## Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera permissions and device paths
2. **Poor depth estimation**: Ensure good lighting and texture
3. **Tracking failures**: Adjust detection confidence thresholds
4. **SLAM not working**: Verify stereo camera calibration

### Performance Optimization

- Use GPU acceleration for ONNX models
- Adjust detection frequency for performance
- Optimize database queries for large datasets
- Use appropriate camera resolutions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ORB-SLAM3 for SLAM functionality
- Ultralytics for YOLOv8 models
- OpenCV community for computer vision tools
- ONNX Runtime for model inference
