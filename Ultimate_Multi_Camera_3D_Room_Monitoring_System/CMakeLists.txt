cmake_minimum_required(VERSION 3.20)
project(RoomMonitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find packages
find_package(OpenCV 4.5 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "/usr/local")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# ORB-SLAM3 (optional, for mapping only)
find_package(ORB_SLAM3 QUIET)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/common
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Subdirectories
add_subdirectory(common)
add_subdirectory(monitor)

if(ORB_SLAM3_FOUND)
    add_subdirectory(mapper)
    message(STATUS "ORB-SLAM3 found - Building mapper")
else()
    message(WARNING "ORB-SLAM3 not found - Skipping mapper (mapping phase only)")
endif()

# Install models
add_custom_target(download_models
    COMMAND ${CMAKE_SOURCE_DIR}/download_models.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Downloading ONNX models..."
)
