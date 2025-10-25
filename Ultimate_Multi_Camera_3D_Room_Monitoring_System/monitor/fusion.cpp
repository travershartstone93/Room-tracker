#include "fusion.h"
#include "geometry.h" // for average

MultiCameraFusion::MultiCameraFusion(const std::vector<CameraCalibration>& cameras, const TrackingConfig& config) : cameras(cameras), config(config) {}

void MultiCameraFusion::add_detections(int camera_id, const std::vector<Detection>& detections) {
    std::lock_guard<std::mutex> lock(mutex);

    current_detections[camera_id] = detections;

    // Compute global positions
    for (auto& det : current_detections[camera_id]) {
        Eigen::Vector4f pos_local_hom = Eigen::Vector4f(det.pos_local.x(), det.pos_local.y(), det.pos_local.z(), 1.0f);
        Eigen::Vector4f pos_global_hom = cameras[camera_id].T_world_cam * pos_local_hom;
        det.pos_local = pos_global_hom.head<3>(); // Overwrite pos_local with global for simplicity (rename if needed)
        for (auto& kp : det.keypoints) {
            Eigen::Vector4f kp_local_hom = Eigen::Vector4f(kp.pos_3d.x(), kp.pos_3d.y(), kp.pos_3d.z(), 1.0f);
            Eigen::Vector4f kp_global_hom = cameras[camera_id].T_world_cam * kp_local_hom;
            kp.pos_3d = kp_global_hom.head<3>();
        }
    }
}

std::vector<Object> MultiCameraFusion::get_fused_objects() {
    std::lock_guard<std::mutex> lock(mutex);
    auto fused = fuse_objects(current_detections);
    current_detections.clear();
    return fused;
}

std::vector<Person> MultiCameraFusion::get_people() {
    std::lock_guard<std::mutex> lock(mutex);
    auto fused = fuse_people(current_detections);
    return fused;
}

std::vector<Object> MultiCameraFusion::fuse_objects(const std::map<int, std::vector<Detection>>& dets) {
    std::map<std::string, std::vector<Detection>> class_to_dets;
    for (const auto& kv : dets) {
        for (const auto& det : kv.second) {
            if (det.class_name != "person") {
                class_to_dets[det.class_name].push_back(det);
            }
        }
    }

    std::vector<Object> fused;
    for (const auto& kv : class_to_dets) {
        const auto& class_dets = kv.second;
        std::vector<std::vector<Detection>> clusters;
        for (const auto& det : class_dets) {
            bool found = false;
            for (auto& cluster : clusters) {
                std::vector<Eigen::Vector3f> positions;
                for (const auto& d : cluster) positions.push_back(d.pos_local);
                Eigen::Vector3f mean = average(positions);
                if ((mean - det.pos_local).norm() < config.triangulation_max_error) {
                    cluster.push_back(det);
                    found = true;
                    break;
                }
            }
            if (!found) {
                clusters.push_back({det});
            }
        }

        for (const auto& cluster : clusters) {
            if (cluster.size() < config.min_cameras_to_confirm) continue;
            Object obj;
            obj.class_name = kv.first;
            obj.class_id = cluster[0].class_id;
            std::vector<Eigen::Vector3f> positions;
            for (const auto& d : cluster) positions.push_back(d.pos_local);
            obj.pos = average(positions);
            float sum_conf = 0.0f;
            for (const auto& d : cluster) sum_conf += d.confidence;
            obj.confidence = sum_conf / cluster.size();
            float sum_depth = 0.0f;
            for (const auto& d : cluster) sum_depth += d.depth_confidence;
            obj.depth_confidence = sum_depth / cluster.size();
            obj.bbox = cluster[0].bbox; // Take first
            obj.state = "confirmed";
            for (const auto& d : cluster) obj.camera_ids.push_back(d.camera_id);
            fused.push_back(obj);
        }
    }
    return fused;
}

std::vector<Person> MultiCameraFusion::fuse_people(const std::map<int, std::vector<Detection>>& dets) {
    std::vector<Detection> person_dets;
    for (const auto& kv : dets) {
        for (const auto& det : kv.second) {
            if (det.class_name == "person") person_dets.push_back(det);
        }
    }

    std::vector<std::vector<Detection>> clusters;
    for (const auto& det : person_dets) {
        bool found = false;
        for (auto& cluster : clusters) {
            std::vector<Eigen::Vector3f> positions;
            for (const auto& d : cluster) positions.push_back(d.pos_local);
            Eigen::Vector3f mean = average(positions);
            if ((mean - det.pos_local).norm() < config.triangulation_max_error) {
                cluster.push_back(det);
                found = true;
                break;
            }
        }
        if (!found) {
            clusters.push_back({det});
        }
    }

    std::vector<Person> fused;
    for (const auto& cluster : clusters) {
        if (cluster.size() < config.min_cameras_to_confirm) continue;
        Person p;
        std::vector<Eigen::Vector3f> positions;
        for (const auto& d : cluster) positions.push_back(d.pos_local);
        p.pos = average(positions);
        p.hand_velocity = 0.0f; // Calculated in tracker
        p.last_seen = std::chrono::system_clock::now();
        // Fuse keypoints
        std::map<std::string, std::vector<std::pair<Eigen::Vector3f, float>>> kp_data;
        for (const auto& det : cluster) {
            for (const auto& kp : det.keypoints) {
                kp_data[kp.name].push_back({kp.pos_3d, kp.confidence});
            }
        }
        for (const auto& kv : kp_data) {
            float sum_conf = 0.0f;
            Eigen::Vector3f sum_pos = Eigen::Vector3f::Zero();
            for (const auto& pair : kv.second) {
                sum_pos += pair.first * pair.second;
                sum_conf += pair.second;
            }
            Keypoint kp;
            kp.name = kv.first;
            kp.pos_3d = sum_pos / sum_conf;
            kp.confidence = sum_conf / kv.second.size();
            p.keypoints.push_back(kp);
        }
        fused.push_back(p);
    }
    return fused;
}
