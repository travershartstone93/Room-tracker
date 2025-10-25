#include "detector_heavy.h"
#include "geometry.h"

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

DetectorHeavy::DetectorHeavy(const std::string& model_path) : session(model_path, true) { // Use GPU
}

std::vector<Object> DetectorHeavy::detect(const cv::Mat& frame) {
    // Similar to run_detection, but for x model, output_shape = {1, 8400, 84} same logic, but larger model for accuracy
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
    std::vector<int64_t> output_shape = {1, 8400, 84};
    auto output = session.run(input, input_shape, output_shape);

    int num_proposals = 8400;
    int dims = 84; // 4 bbox + 80 class
    std::vector<Object> dets;
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
        if (max_class < 0.6f) continue; // Higher threshold for heavy model
        float cx = ptr[0];
        float cy = ptr[1];
        float w = ptr[2];
        float h = ptr[3];
        float x = (cx - w / 2) * (frame.cols / 640.0f);
        float y = (cy - h / 2) * (frame.rows / 640.0f);
        float ww = w * (frame.cols / 640.0f);
        float hh = h * (frame.rows / 640.0f);
        Object det;
        det.bbox = cv::Rect(static_cast<int>(x), static_cast<int>(y), static_cast<int>(ww), static_cast<int>(hh));
        det.class_id = class_id;
        det.class_name = get_class_name(class_id);
        det.confidence = max_class;
        det.last_seen = std::chrono::system_clock::now();
        dets.push_back(det);
    }
    nms(dets, 0.45f);
    return dets;
}

void DetectorHeavy::nms(std::vector<Object>& dets, float threshold) {
    if (dets.empty()) return;
    std::sort(dets.begin(), dets.end(), [](const Object& a, const Object& b) { return a.confidence > b.confidence; });
    std::vector<bool> keep(dets.size(), true);
    for (size_t i = 0; i < dets.size(); i++) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (bbox_iou(dets[i].bbox, dets[j].bbox) > threshold) {
                keep[j] = false;
            }
        }
    }
    std::vector<Object> filtered;
    for (size_t i = 0; i < dets.size(); i++) {
        if (keep[i]) filtered.push_back(dets[i]);
    }
    dets = filtered;
}

std::string DetectorHeavy::get_class_name(int id) {
    if (id < 0 || id >= coco_classes.size()) return "unknown";
    return coco_classes[id];
}
