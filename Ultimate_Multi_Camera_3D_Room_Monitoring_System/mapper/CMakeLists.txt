add_executable(room_mapper main.cpp slam_wrapper.cpp detector_heavy.cpp depth_estimator.cpp map_exporter.cpp)
target_link_libraries(room_mapper common ORB_SLAM3::ORB_SLAM3 ${OpenCV_LIBS} yaml-cpp onnxruntime)
