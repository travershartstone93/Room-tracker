#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

std::string generate_uuid();

float bbox_iou(const cv::Rect& a, const cv::Rect& b);

Eigen::Vector3f average(const std::vector<Eigen::Vector3f>& points);
