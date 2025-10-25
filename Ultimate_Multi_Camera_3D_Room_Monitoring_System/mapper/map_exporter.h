#pragma once
#include "types.h"
#include "slam_wrapper.h"
#include <string>

class MapExporter {
public:
    void export_point_cloud(const SLAMWrapper& slam, const std::string& ply_path);
    void export_objects(const std::vector<Object>& all_detections, const std::string& db_path);
};
