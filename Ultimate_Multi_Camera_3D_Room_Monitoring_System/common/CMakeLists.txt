add_library(common STATIC
    onnx_runtime.cpp
    geometry.cpp
    config.cpp
)

target_link_libraries(common ${OpenCV_LIBS} Eigen3::Eigen yaml-cpp onnxruntime)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
