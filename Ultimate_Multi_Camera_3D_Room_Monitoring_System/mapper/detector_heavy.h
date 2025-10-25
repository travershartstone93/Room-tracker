#pragma once
#include "types.h"
#include <vector>
#include "onnx_runtime.h"

class DetectorHeavy {
public:
    DetectorHeavy(const std::string& model_path);
    std::vector<Object> detect(const cv::Mat& frame);

private:
    ONNXRuntime session;
    void nms(std::vector<Object>& dets, float threshold);
    std::string get_class_name(int id);
};
