#include "detector_lite.h"
#include "geometry.h" // for bbox_iou
#include <algorithm>

const std::vector<std::string> coco_classes = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

std::vector<std::string> coco_keypoints = {"nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"};

DetectorLite::DetectorLite(const std::string& detection_model_path, const std::string& pose_model_path) : detection_session(detection_model_path, false), pose_session(pose_model_path, false) {
    // Verify model dimensions
    auto input_shape = detection_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape[2] != 640 || input_shape[3] != 640) {
        throw std::runtime_error("Detection model must have 640x640 input");
    }
}

std::vector<Detection> DetectorLite::detect(const cv::Mat& frame) {
    auto object_dets = run_detection(frame);

    auto pose_dets = run_pose(frame);

    // Match pose to person detections
    for (auto& pose : pose_dets) {
        bool matched = false;
        for (auto& obj : object_dets) {
            if (obj.class_name == "person" && bbox_iou(obj.bbox, pose.bbox) > 0.7f) {
                obj.keypoints = pose.keypoints;
                matched = true;
                break;
            }
        }
        if (!matched) {
            object_dets.push_back(pose); // Add if no match
        }
    }

    return object_dets;
}

std::vector<Detection> DetectorLite::run_detection(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(640, 640));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    std::vector<float> input(640 * 640 * 3);
    if (resized.isContinuous()) {
        input.assign((float*)resized.datastart, (float*)resized.dataend);
    } else {
        for (int i = 0; i < resized.rows; i++) {
            memcpy(&input[i * 640 * 3], resized.ptr(i), 640 * 3 * sizeof(float));
        }
    }
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    std::vector<int64_t> output_shape = {1, 8400, 84}; // FIXED
    auto output = detection_session.run(input, input_shape, output_shape);

    int num_proposals = 8400;
    int dims = 84; // 4 bbox + 80 class
    std::vector<Detection> dets;
    for (int i = 0; i < num_proposals; i++) {
        float* ptr = &output[i * dims];
        float max_class = 0.0f;
        int class_id = -1;
        for (int j = 0; j < 80; j++) {
            float cls_conf = ptr[4 + j];
            if (cls_conf > max_class) {
                max_class = cls_conf;
                class_id = j;
            }
        }
        if (max_class < 0.5f) continue;
        float cx = ptr[0];
        float cy = ptr[1];
        float w = ptr[2];
        float h = ptr[3];
        float x = (cx - w / 2) * (frame.cols / 640.0f);
        float y = (cy - h / 2) * (frame.rows / 640.0f);
        float ww = w * (frame.cols / 640.0f);
        float hh = h * (frame.rows / 640.0f);
        Detection det;
        det.bbox = cv::Rect(static_cast<int>(x), static_cast<int>(y), static_cast<int>(ww), static_cast<int>(hh));
        det.class_id = class_id;
        det.class_name = get_class_name(class_id);
        det.confidence = max_class;
        det.timestamp = std::chrono::system_clock::now();
        dets.push_back(det);
    }
    nms(dets, 0.45f);
    return dets;
}

std::vector<Detection> DetectorLite::run_pose(const cv::Mat& frame) {
    // Similar, output_shape = {1, 8400, 56} for pose (4 + 1 + 51)
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(640, 640));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    std::vector<float> input(640 * 640 * 3);
    if (resized.isContinuous()) {
        input.assign((float*)resized.datastart, (float*)resized.dataend);
    } else {
        for (int i = 0; i < resized.rows; i++) {
            memcpy(&input[i * 640 * 3], resized.ptr(i), 640 * 3 * sizeof(float));
        }
    }
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    std::vector<int64_t> output_shape = {1, 8400, 56};
    auto output = pose_session.run(input, input_shape, output_shape);

    int num_proposals = 8400;
    int dims = 56; // 4 bbox + 52 keypoints (17*3)
    std::vector<Detection> dets;
    for (int i = 0; i < num_proposals; i++) {
        float* ptr = &output[i * dims];
        float conf = ptr[4];
        if (conf < 0.5f) continue;
        float cx = ptr[0];
        float cy = ptr[1];
        float w = ptr[2];
        float h = ptr[3];
        float x = (cx - w / 2) * (frame.cols / 640.0f);
        float y = (cy - h / 2) * (frame.rows / 640.0f);
        float ww = w * (frame.cols / 640.0f);
        float hh = h * (frame.rows / 640.0f);
        Detection det;
        det.bbox = cv::Rect(static_cast<int>(x), static_cast<int>(y), static_cast<int>(ww), static_cast<int>(hh));
        det.class_id = 0;
        det.class_name = "person";
        det.confidence = conf;
        det.timestamp = std::chrono::system_clock::now();
        // Parse keypoints
        for (int k = 0; k < 17; k++) {
            Keypoint kp;
            kp.name = coco_keypoints[k];
            kp.pos_2d.x = ptr[5 + k*3] * (frame.cols / 640.0f);
            kp.pos_2d.y = ptr[5 + k*3 + 1] * (frame.rows / 640.0f);
            kp.confidence = ptr[5 + k*3 + 2];
            if (kp.confidence > 0.3f) det.keypoints.push_back(kp);
        }
        dets.push_back(det);
    }
    nms(dets, 0.45f);
    return dets;
}

void DetectorLite::nms(std::vector<Detection>& dets, float threshold) {
    if (dets.empty()) return;
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    std::vector<bool> keep(dets.size(), true);
    for (size_t i = 0; i < dets.size(); i++) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (bbox_iou(dets[i].bbox, dets[j].bbox) > threshold) {
                keep[j] = false;
            }
        }
    }
    std::vector<Detection> filtered;
    for (size_t i = 0; i < dets.size(); i++) {
        if (keep[i]) filtered.push_back(dets[i]);
    }
    dets = filtered;
}

std::string DetectorLite::get_class_name(int id) {
    if (id < 0 || id >= coco_classes.size()) return "unknown";
    return coco_classes[id];
}
