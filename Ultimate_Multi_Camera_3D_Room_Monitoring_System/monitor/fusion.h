#pragma once
#include "types.h"
#include <vector>
#include <map>
#include <mutex>

class MultiCameraFusion {
public:
    MultiCameraFusion(const std::vector<CameraCalibration>& cameras, const TrackingConfig& config);
    void add_detections(int camera_id, const std::vector<Detection>& detections);
    std::vector<Object> get_fused_objects();
    std::vector<Person> get_people();

private:
    std::vector<CameraCalibration> cameras;
    TrackingConfig config;
    std::map<int, std::vector<Detection>> current_detections;
    std::mutex mutex;

    std::vector<Object> fuse_objects(const std::map<int, std::vector<Detection>>& dets);
    std::vector<Person> fuse_people(const std::map<int, std::vector<Detection>>& dets);
};
