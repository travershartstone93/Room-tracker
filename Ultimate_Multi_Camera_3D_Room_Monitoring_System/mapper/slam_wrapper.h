#pragma once
#include "ORB_SLAM3/include/System.h"
#include <Eigen/Dense>

class SLAMWrapper {
public:
    SLAMWrapper(const std::string& voc, const std::string& settings, ORB_SLAM3::System::eSensor sensor);
    Eigen::Matrix4f TrackStereo(const cv::Mat& left, const cv::Mat& right, double timestamp);
    void SaveAtlas(const std::string& path);
    ORB_SLAM3::System& get_system() { return SLAM; }

private:
    ORB_SLAM3::System SLAM;
};
