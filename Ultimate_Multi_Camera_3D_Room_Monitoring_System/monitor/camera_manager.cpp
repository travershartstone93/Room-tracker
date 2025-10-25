#include "camera_manager.h"

CameraManager::CameraManager(const CameraCalibration& config) : config(config), id(config.id) {
    cap = cv::VideoCapture(config.source);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, config.resolution.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.resolution.height);
    cap.set(cv::CAP_PROP_FPS, config.fps);
}

cv::Mat CameraManager::get_frame() {
    std::lock_guard<std::mutex> lock(mutex);
    cap >> frame;
    if (!config.distortion.empty()) {
        cv::Mat undist;
        cv::undistort(frame, undist, config.K, config.distortion);
        frame = undist;
    }
    return frame.clone();
}

Eigen::Vector3f CameraManager::deproject(float u, float v, float d) {
    return config.deproject(u, v, d);
}

Eigen::Vector3f CameraManager::deproject_to_3d(const cv::Rect& bbox, const cv::Mat& depth_map) {
    return config.deproject_to_3d(bbox, depth_map);
}
