#include "geometry.h"

#include <random>
#include <sstream>

std::string generate_uuid() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    int i;
    ss << std::hex;
    for (i = 0; i < 8; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 4; i++) {
        ss << dis(gen);
    }
    ss << "-4";
    for (i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    ss << dis2(gen);
    for (i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 12; i++) {
        ss << dis(gen);
    }
    return ss.str();
}

float bbox_iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    float area_inter = inter.area();
    float area_union = a.area() + b.area() - area_inter;
    return area_inter / area_union;
}

Eigen::Vector3f average(const std::vector<Eigen::Vector3f>& points) {
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    for (const auto& p : points) sum += p;
    return sum / static_cast<float>(points.size());
}
