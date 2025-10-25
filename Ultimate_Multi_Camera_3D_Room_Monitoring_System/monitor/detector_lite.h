#pragma once
#include "types.h"
#include <vector>
#include "onnx_runtime.h"

class DetectorLite {
public:
    DetectorLite(const std::string& detection_model_path, const std::string& pose_model_path);
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    ONNXRuntime detection_session;
    ONNXRuntime pose_session;

    void nms(std::vector<Detection>& dets, float threshold);
    std::string get_class_name(int id);
    std::vector<Detection> run_detection(const cv::Mat& frame);
    std::vector<Detection> run_pose(const cv::Mat& frame);
};
