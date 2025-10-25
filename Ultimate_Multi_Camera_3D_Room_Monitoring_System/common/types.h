#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <chrono>

using Timestamp = std::chrono::time_point<std::chrono::system_clock>;

struct Detection {
    int class_id;
    std::string class_name;
    cv::Rect bbox;
    float confidence;
    Eigen::Vector3f pos_local;      // Camera-relative 3D position
    float depth_confidence;
    int camera_id;
    Timestamp timestamp;
    std::vector<Keypoint> keypoints; // Optional for persons
};

struct Keypoint {
    std::string name;
    Eigen::Vector2f pos_2d;
    Eigen::Vector3f pos_3d;
    float confidence;
};

struct Person {
    int id;
    int tracking_id;
    Eigen::Vector3f pos;
    std::vector<Keypoint> keypoints;
    float hand_velocity;
    Timestamp last_seen;
};

struct Object {
    int id;
    std::string global_id;          // UUID for persistence
    int class_id;
    std::string class_name;
    
    Eigen::Vector3f pos;            // Global 3D position
    Eigen::Vector3f velocity;       // m/s
    cv::Rect bbox;
    
    float confidence;
    float depth_confidence;
    float uncertainty;              // Kalman covariance
    
    std::string state;              // "static", "held", "occluded", etc.
    int parent_id;                  // Person holding this object (-1 if none)
    int support_surface_id;         // Object this is resting on (-1 if none)
    
    int seen_count;
    int frames_lost;
    Timestamp first_seen;
    Timestamp last_seen;
    
    std::vector<int> camera_ids;    // Cameras that see this object
};

struct CameraCalibration {
    int id;
    std::string source;
    cv::Size resolution;
    int fps;
    
    // Intrinsics
    cv::Mat K;                      // 3x3 camera matrix
    cv::Mat distortion;             // Distortion coefficients
    
    // Extrinsics (global pose)
    Eigen::Matrix4f T_world_cam;    // 4x4 transformation matrix

    Eigen::Vector3f deproject(float u, float v, float d) const {
        Eigen::Vector3f p;
        p.x() = d * (u - K.at<float>(0,2)) / K.at<float>(0,0);
        p.y() = d * (v - K.at<float>(1,2)) / K.at<float>(1,1);
        p.z() = d;
        return p;
    }

    Eigen::Vector3f deproject_to_3d(const cv::Rect& bbox, const cv::Mat& depth_map) const {
        cv::Mat roi = depth_map(bbox);
        float d = cv::mean(roi)[0];
        float u = bbox.x + bbox.width / 2.0f;
        float v = bbox.y + bbox.height / 2.0f;
        return deproject(u, v, d);
    }
};

struct Event {
    int object_id;
    std::string event_type;         // "moved", "picked_up", etc.
    Eigen::Vector3f old_pos;
    Eigen::Vector3f new_pos;
    float distance_moved;
    int attributed_to;              // Person ID (-1 if unknown)
    float confidence;
    int camera_id;
    Timestamp timestamp;
};
