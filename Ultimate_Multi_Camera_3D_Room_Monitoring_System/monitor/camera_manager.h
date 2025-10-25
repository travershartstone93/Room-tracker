#pragma once
#include "types.h"
#include <opencv2/opencv.hpp>
#include <mutex>

class CameraManager {
public:
    CameraManager(const CameraCalibration& config);
    cv::Mat get_frame();
    Eigen::Vector3f deproject_to_3d(const cv::Rect& bbox, const cv::Mat& depth_map);
    Eigen::Vector3f deproject(float u, float v, float d);
    int id;

private:
    cv::VideoCapture cap;
    cv::Mat frame;
    std::mutex mutex;
    CameraCalibration config;
};
