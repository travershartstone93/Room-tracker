#include <iostream>
#include <thread>
#include <chrono>
#include "camera_manager.h"
#include "detector_lite.h"
#include "depth_estimator.h"
#include "tracker.h"
#include "fusion.h"
#include "database.h"
#include "config.h"
#include <iomanip> // for time
#include <ctime>

int main(int argc, char** argv) {
    try {
        // Load configuration
        Config config("config/cameras.yaml", "config/tracking.yaml");

        // Initialize database
        Database db("room_monitor.db");
        db.initialize();

        // Load known objects from mapping phase
        auto known_objects = db.load_known_objects();

        // Initialize components
        std::vector<CameraManager> cameras;
        for (const auto& cam_config : config.cameras) {
            cameras.emplace_back(cam_config);
        }

        DetectorLite detector("models/yolov8n-int8.onnx", "models/yolov8n-pose-int8.onnx");
        DepthEstimator depth_estimator("models/midas_small.onnx");
        MultiCameraFusion fusion(config.cameras, config.tracking);
        ObjectTracker tracker(config.tracking, known_objects);

        // Start camera threads
        std::vector<std::thread> camera_threads;
        for (auto& cam : cameras) {
            camera_threads.emplace_back([&cam, &detector, &depth_estimator, &fusion]() {
                while (true) {
                    cv::Mat frame = cam.get_frame();
                    if (frame.empty()) continue;

                    // Detect objects + poses
                    auto detections = detector.detect(frame);

                    // Estimate depth
                    cv::Mat depth_map = depth_estimator.estimate(frame);

                    // Convert to 3D
                    for (auto& det : detections) {
                        det.pos_local = cam.deproject_to_3d(det.bbox, depth_map);
                        det.depth_confidence = cv::mean(depth_map(det.bbox))[0]; // Average depth as confidence proxy
                        if (!det.keypoints.empty()) {
                            for (auto& kp : det.keypoints) {
                                int x = static_cast<int>(kp.pos_2d.x);
                                int y = static_cast<int>(kp.pos_2d.y);
                                float d = depth_map.at<float>(y, x);
                                kp.pos_3d = cam.deproject(kp.pos_2d.x, kp.pos_2d.y, d);
                            }
                        }
                    }

                    // Send to fusion
                    fusion.add_detections(cam.id, detections);
                }
            });
        }

        // Main fusion loop
        while (true) {
            auto fused_objects = fusion.get_fused_objects();
            auto people = fusion.get_people();

            // Track objects
            tracker.update(fused_objects, people);

            // Get events
            auto events = tracker.get_events();

            // Log to database
            for (const auto& event : events) {
                db.log_event(event);

                // Print to console
                auto time = std::chrono::system_clock::to_time_t(event.timestamp);
                std::cout << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "] [" << event.event_type << "] "
                          << "Object #" << event.object_id 
                          << " " << event.event_type << " " << event.distance_moved << "m";
                if (event.attributed_to >= 0) {
                    std::cout << " (by Person #" << event.attributed_to << ")";
                }
                std::cout << std::endl;
            }

            // Update object states in database
            for (const auto& obj : tracker.get_tracked_objects()) {
                db.update_object(obj);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 20 FPS
        }

        // Join threads
        for (auto& thread : camera_threads) {
            thread.join();
        }
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
