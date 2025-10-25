#include "depth_estimator.h"

DepthEstimator::DepthEstimator(const std::string& model_path) : session(model_path, false) {
}

cv::Mat DepthEstimator::estimate(const cv::Mat& frame) {
    // Preprocess: resize to model input size, e.g. for MiDaS small 1x3x256x256
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(256, 256));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    std::vector<float> input(256 * 256 * 3);
    if (resized.isContinuous()) {
        input.assign((float*)resized.datastart, (float*)resized.dataend);
    } else {
        for (int i = 0; i < resized.rows; i++) {
            memcpy(&input[i * 256 * 3], resized.ptr(i), 256 * 3 * sizeof(float));
        }
    }
    std::vector<int64_t> input_shape = {1, 3, 256, 256};
    std::vector<int64_t> output_shape = {1, 1, 256, 256}; // For MiDaS
    auto output = session.run(input, input_shape, output_shape);

    // Create depth map
    cv::Mat depth(256, 256, CV_32F);
    memcpy(depth.data, output.data(), output.size() * sizeof(float));

    // Resize to original
    cv::resize(depth, depth, frame.size());

    return depth;
}
