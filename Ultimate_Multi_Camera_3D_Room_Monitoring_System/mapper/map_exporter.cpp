#include "map_exporter.h"
#include "database.h"
#include "geometry.h"
#include <fstream>

void MapExporter::export_point_cloud(const SLAMWrapper& slam, const std::string& ply_path) {
    ORB_SLAM3::Atlas* atlas = slam.get_system().GetAtlas(); // FIXED
    ORB_SLAM3::Map* map = atlas->GetCurrentMap();
    auto spMapPoints = map->GetAllMapPoints();
    std::ofstream f(ply_path);
    f << "ply\nformat ascii 1.0\nelement vertex " << spMapPoints.size() << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    for (auto pMP : spMapPoints) {
        auto pos = pMP->GetWorldPos();
        f << pos(0) << " " << pos(1) << " " << pos(2) << "\n";
    }
}

void MapExporter::export_objects(const std::vector<Object>& all_detections, const std::string& db_path) {
    Database db(db_path);
    db.initialize();

    std::map<std::string, std::vector<Eigen::Vector3f>> class_to_pos;
    for (const auto& det : all_detections) {
        if (det.confidence < 0.6f) continue;
        class_to_pos[det.class_name].push_back(det.pos_local); // global
    }

    for (const auto& kv : class_to_pos) {
        const std::string& cls = kv.first;
        const auto& positions = kv.second;
        std::vector<std::vector<Eigen::Vector3f>> clusters;
        for (const auto& pos : positions) {
            bool found = false;
            for (auto& cluster : clusters) {
                Eigen::Vector3f mean = average(cluster);
                if ((mean - pos).norm() < 0.2f) {
                    cluster.push_back(pos);
                    found = true;
                    break;
                }
            }
            if (!found) {
                clusters.push_back({pos});
            }
        }

        for (const auto& cluster : clusters) {
            Eigen::Vector3f mean = average(cluster);
            Object obj;
            obj.global_id = generate_uuid();
            obj.class_name = cls;
            obj.pos = mean;
            obj.state = "static";
            obj.bbox = cv::Rect(); // Average if needed
            db.add_known_object(obj);
        }
    }
}
