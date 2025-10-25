#include "config.h"
#include <yaml-cpp/yaml.h>

Config::Config(const std::string& cameras_yaml, const std::string& tracking_yaml) {
    if (!cameras_yaml.empty()) {
        YAML::Node cam_node = YAML::LoadFile(cameras_yaml);
        for (const auto& cam : cam_node["cameras"]) {
            CameraCalibration c;
            c.id = cam["id"].as<int>();
            c.source = cam["source"].as<std::string>();
            c.resolution = cv::Size(cam["resolution"][0].as<int>(), cam["resolution"][1].as<int>());
            c.fps = cam["fps"].as<int>();
            c.K = cv::Mat(3,3, CV_32F, 0.0f);
            c.K.at<float>(0,0) = cam["intrinsics"]["fx"].as<float>();
            c.K.at<float>(1,1) = cam["intrinsics"]["fy"].as<float>();
            c.K.at<float>(0,2) = cam["intrinsics"]["cx"].as<float>();
            c.K.at<float>(1,2) = cam["intrinsics"]["cy"].as<float>();
            c.K.at<float>(2,2) = 1.0f;
            auto dist = cam["intrinsics"]["distortion"].as<std::vector<float>>();
            c.distortion = cv::Mat(dist);
            auto trans = cam["extrinsics"]["translation"].as<std::vector<float>>();
            auto rot = cam["extrinsics"]["rotation"].as<std::vector<float>>();
            Eigen::Quaternionf q(rot[0], rot[1], rot[2], rot[3]);
            c.T_world_cam.block<3,3>(0,0) = q.toRotationMatrix();
            c.T_world_cam(0,3) = trans[0];
            c.T_world_cam(1,3) = trans[1];
            c.T_world_cam(2,3) = trans[2];
            cameras.push_back(c);
        }
    }

    if (!tracking_yaml.empty()) {
        YAML::Node track_node = YAML::LoadFile(tracking_yaml);
        auto tracking_node = track_node["tracking"];
        tracking.detection_confidence = tracking_node["detection_confidence"].as<float>(tracking.detection_confidence);
        tracking.depth_confidence = tracking_node["depth_confidence"].as<float>(tracking.depth_confidence);
        tracking.nms_threshold = tracking_node["nms_threshold"].as<float>(tracking.nms_threshold);
        tracking.max_frames_lost = tracking_node["max_frames_lost"].as<int>(tracking.max_frames_lost);
        tracking.position_threshold = tracking_node["position_threshold"].as<float>(tracking.position_threshold);
        tracking.velocity_smoothing = tracking_node["velocity_smoothing"].as<float>(tracking.velocity_smoothing);
        tracking.grace_period = tracking_node["grace_period"].as<float>(tracking.grace_period);
        tracking.held_distance = tracking_node["held_distance"].as<float>(tracking.held_distance);
        tracking.occlusion_timeout = tracking_node["occlusion_timeout"].as<int>(tracking.occlusion_timeout);
        tracking.transient_period = tracking_node["transient_period"].as<int>(tracking.transient_period);
        tracking.min_cameras_to_confirm = tracking_node["min_cameras_to_confirm"].as<int>(tracking.min_cameras_to_confirm);
        tracking.triangulation_max_error = tracking_node["triangulation_max_error"].as<float>(tracking.triangulation_max_error);
        tracking.depth_inconsistency = tracking_node["depth_inconsistency"].as<float>(tracking.depth_inconsistency);
        for (const auto& kv : tracking_node["velocity_alerts"]) {
            tracking.velocity_thresholds[kv.first.as<std::string>()] = kv.second.as<float>();
        }
        for (const auto& cls : tracking_node["deformable_classes"]) {
            tracking.deformable_classes.insert(cls.as<std::string>());
        }
        tracking.whitelist = tracking_node["whitelist"].as<std::vector<std::string>>(tracking.whitelist);
        tracking.blacklist = tracking_node["blacklist"].as<std::vector<std::string>>(tracking.blacklist);
    }
}
