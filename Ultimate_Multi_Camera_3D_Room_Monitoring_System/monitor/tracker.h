#pragma once
#include "types.h"
#include <vector>
#include <map>
#include <mutex>

class KalmanFilter {
public:
    KalmanFilter();
    void predict(float dt);
    void update(const Eigen::Vector3f& measurement);
    Eigen::Vector3f get_position() const { return state.head<3>(); }
    Eigen::Vector3f get_velocity() const { return state.tail<3>(); }
    float get_uncertainty() const { return covariance.trace(); }
    
private:
    Eigen::VectorXf state;           // [x, y, z, vx, vy, vz]
    Eigen::MatrixXf covariance;      // 6x6
    Eigen::MatrixXf process_noise;
    Eigen::MatrixXf measurement_noise;
};

class ObjectTracker {
public:
    ObjectTracker(const TrackingConfig& config, 
                  const std::vector<Object>& known_objects);
    
    void update(const std::vector<Object>& detections,
                const std::vector<Person>& people);
    
    std::vector<Event> get_events();
    std::vector<Object> get_tracked_objects() const;
    
private:
    // Edge case handlers
    bool is_object_held(const Person& person, const Object& object);
    int find_likely_mover(const Object& obj, const std::vector<Person>& people);
    void resolve_overlaps(std::vector<Object>& objects);
    bool match_deformable(const Object& obj1, const Object& obj2);
    void handle_occlusion(Object& obj);
    void classify_permanence(Object& obj);
    bool should_alert(const Object& obj);
    
    // Tracking logic
    void associate_detections(const std::vector<Object>& detections);
    void predict_all();
    void create_new_tracks(const std::vector<Object>& unmatched);
    void prune_lost_tracks();
    
    std::map<int, KalmanFilter> kalman_filters;
    std::map<int, Object> tracked_objects;
    std::vector<Event> pending_events;
    std::vector<Object> known_objects;
    TrackingConfig config;
    int next_id = 0;
};
