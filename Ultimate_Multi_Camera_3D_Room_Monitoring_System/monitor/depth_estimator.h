#pragma once
#include <opencv2/opencv.hpp>
#include "onnx_runtime.h"

class DepthEstimator {
public:
    DepthEstimator(const std::string& model_path);
    cv::Mat estimate(const cv::Mat& frame);

private:
    ONNXRuntime session;
};
