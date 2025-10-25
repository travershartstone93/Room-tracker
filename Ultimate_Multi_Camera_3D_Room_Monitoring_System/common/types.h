// Updated types.h

#ifndef TYPES_H
#define TYPES_H

#include <opencv2/core.hpp>

struct Keypoint {
    cv::Point2f point;
    float size;
    float angle;
    float response;
};

struct Detection {
    Keypoint keypoint;
    float confidence;
};

// Updated deproject_to_3d method
void deproject_to_3d(const cv::Mat& image, const Keypoint& keypoint) {
    cv::Scalar meanValue = cv::mean(image);
    // Use meanValue for further processing
}

#endif // TYPES_H