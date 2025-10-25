#include "depth_estimator.h"

DepthEstimator::DepthEstimator(const std::string& model_path) : session(model_path, true) { // Use GPU for heavy model
}

cv::Mat DepthEstimator::estimate(const cv::Mat& frame) {
    // Adjust input size to 518x518 for Depth-Anything
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(518, 518));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    std::vector<float> input(518 * 518 * 3);
    if (resized.isContinuous()) {
        input.assign((float*)resized.datastart, (float*)resized.dataend);
    } else {
        for (int i = 0; i < resized.rows; i++) {
            memcpy(&input[i * 518 * 3], resized.ptr(i), 518 * 3 * sizeof(float));
        }
    }
    std::vector<int64_t> input_shape = {1, 3, 518, 518};
    std::vector<int64_t> output_shape = {1, 1, 518, 518};
    auto output = session.run(input, input_shape, output_shape);

    // Create depth map
    cv::Mat depth(518, 518, CV_32F);
    memcpy(depth.data, output.data(), output.size() * sizeof(float));

    // Convert disparity to depth: depth = 1 / disparity
    cv::Mat depth_inv;
    cv::divide(1.0, depth, depth_inv);

    // Resize to original
    cv::resize(depth_inv, depth_inv, frame.size());

    return depth_inv;
}
