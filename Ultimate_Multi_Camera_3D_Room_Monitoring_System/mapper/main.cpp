#include <iostream>
#include <opencv2/opencv.hpp>
#include "slam_wrapper.h"
#include "detector_heavy.h"
#include "depth_estimator.h"
#include "map_exporter.h"
#include "config.h"

int main() {
    Config config("config/cameras.yaml", "");

    DetectorHeavy detector("models/yolov8x.onnx"); // Assume similar to DetectorLite but for heavy
    DepthEstimator depth_estimator("models/depth_anything_v2.onnx"); // Heavy depth

    SLAMWrapper slam("Vocabulary/ORBvoc.txt", "config/stereo.yaml", ORB_SLAM3::System::STEREO);

    cv::VideoCapture cap0(config.cameras[0].source);
    cap0.set(cv::CAP_PROP_FRAME_WIDTH, config.cameras[0].resolution.width);
    cap0.set(cv::CAP_PROP_FRAME_HEIGHT, config.cameras[0].resolution.height);
    cap0.set(cv::CAP_PROP_FPS, config.cameras[0].fps);

    cv::VideoCapture cap1(config.cameras[1].source);
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, config.cameras[1].resolution.width);
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, config.cameras[1].resolution.height);
    cap1.set(cv::CAP_PROP_FPS, config.cameras[1].fps);

    auto start = std::chrono::system_clock::now();

    std::vector<Object> all_detections;

    while (true) {
        cv::Mat left, right;
        cap0 >> left;
        cap1 >> right;
        if (left.empty() || right.empty()) break;

        double t = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();

        Eigen::Matrix4f Tcw = slam.TrackStereo(left, right, t);

        if (Tcw.isZero()) continue; // Not initialized

        // Detect on left frame every 10 frames
        static int count = 0;
        if (++count % 10 == 0) {
            auto detections = detector.detect(left);
            cv::Mat depth_map = depth_estimator.estimate(left);
            for (auto& det : detections) {
                det.pos_local = config.cameras[0].deproject_to_3d(det.bbox, depth_map);
                Eigen::Vector4f pos_hom = Eigen::Vector4f(det.pos_local.x(), det.pos_local.y(), det.pos_local.z(), 1.0f);
                Eigen::Vector4f pos_global = Tcw.inverse() * pos_hom;
                det.pos_local = pos_global.head<3>(); // Reuse for global
            }
            all_detections.insert(all_detections.end(), detections.begin(), detections.end());
        }

        if (std::chrono::system_clock::now() - start > std::chrono::seconds(60)) break;
    }

    slam.SaveAtlas("atlas.bin");

    MapExporter exporter;
    exporter.export_point_cloud(slam, "room_map.ply");
    exporter.export_objects(all_detections, "known_objects.db");

    return 0;
}
