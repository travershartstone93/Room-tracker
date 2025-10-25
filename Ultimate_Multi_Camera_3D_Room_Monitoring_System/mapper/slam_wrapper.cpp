#include "slam_wrapper.h"

SLAMWrapper::SLAMWrapper(const std::string& voc, const std::string& settings, ORB_SLAM3::System::eSensor sensor) : SLAM(voc, settings, sensor, true) {
}

Eigen::Matrix4f SLAMWrapper::TrackStereo(const cv::Mat& left, const cv::Mat& right, double timestamp) {
    cv::Mat pose = SLAM.TrackStereo(left, right, timestamp);
    if (pose.empty()) return Eigen::Matrix4f::Zero();
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            T(i, j) = pose.at<float>(i, j);
        }
    }
    return T;
}

void SLAMWrapper::SaveAtlas(const std::string& path) {
    SLAM.SaveAtlasToFile(path);
}
