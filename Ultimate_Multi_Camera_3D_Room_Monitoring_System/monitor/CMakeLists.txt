add_executable(room_monitor main.cpp camera_manager.cpp detector_lite.cpp depth_estimator.cpp tracker.cpp fusion.cpp database.cpp)

target_link_libraries(room_monitor common ${OpenCV_LIBS} ${SQLITE3_LIBRARIES} yaml-cpp onnxruntime)
