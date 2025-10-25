#pragma once

#include "types.h"

#include <string>
#include <vector>
#include <map>
#include <set>

struct TrackingConfig {
    float detection_confidence = 0.5f;
    float depth_confidence = 0.6f;
    float nms_threshold = 0.45f;
    int max_frames_lost = 30;
    float position_threshold = 0.15f;
    float velocity_smoothing = 0.7f;
    float grace_period = 5.0f;
    float held_distance = 0.4f;
    int occlusion_timeout = 30;
    int transient_period = 3600;
    int min_cameras_to_confirm = 2;
    float triangulation_max_error = 0.5f;
    float depth_inconsistency = 1.0f;
    float max_association_distance = 1.0f; // Added
    std::map<std::string, float> velocity_thresholds;
    std::set<std::string> deformable_classes;
    std::vector<std::string> whitelist;
    std::vector<std::string> blacklist;
};

struct Config {
    std::vector<CameraCalibration> cameras;
    TrackingConfig tracking;
    Config(const std::string& cameras_yaml, const std::string& tracking_yaml);
};
