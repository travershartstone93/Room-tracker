#include "tracker.h"
#include "geometry.h"
#include <algorithm>
#include <limits>

KalmanFilter::KalmanFilter() {
    state = Eigen::VectorXf::Zero(6);
    covariance = Eigen::MatrixXf::Identity(6,6) * 0.1f;
    process_noise = Eigen::MatrixXf::Identity(6,6) * 0.01f;
    measurement_noise = Eigen::MatrixXf::Identity(3,3) * 0.1f;
}

void KalmanFilter::predict(float dt) {
    Eigen::MatrixXf F(6,6);
    F.setIdentity();
    F(0,3) = dt;
    F(1,4) = dt;
    F(2,5) = dt;
    state = F * state;
    covariance = F * covariance * F.transpose() + process_noise;
}

void KalmanFilter::update(const Eigen::Vector3f& measurement) {
    Eigen::MatrixXf H(3,6);
    H.setZero();
    H.block<3,3>(0,0) = Eigen::Matrix3f::Identity();
    Eigen::Vector3f innovation = measurement - H * state;
    Eigen::MatrixXf S = H * covariance * H.transpose() + measurement_noise;
    Eigen::MatrixXf K = covariance * H.transpose() * S.inverse();
    state = state + K * innovation;
    covariance = (Eigen::MatrixXf::Identity(6,6) - K * H) * covariance;
}

ObjectTracker::ObjectTracker(const TrackingConfig& config,
                             const std::vector<Object>& known_objects)
    : config(config), known_objects(known_objects) {}

void ObjectTracker::update(const std::vector<Object>& detections,
                           const std::vector<Person>& people) {
    // 1. Predict all tracked objects
    predict_all();
    
    // 2. Handle overlapping objects (depth layering)
    auto detections_copy = detections;
    resolve_overlaps(detections_copy);
    
    // 3. Associate detections with existing tracks
    associate_detections(detections_copy);
    
    // 4. Check for objects held by people
    for (auto& [id, obj] : tracked_objects) {
        for (const auto& person : people) {
            if (is_object_held(person, obj)) {
                if (obj.state != "held") {
                    // State transition: picked up
                    Event event{
                        .object_id = id,
                        .event_type = "picked_up",
                        .old_pos = obj.pos,
                        .new_pos = obj.pos,
                        .distance_moved = 0.0f,
                        .attributed_to = person.id,
                        .confidence = 0.9f,
                        .timestamp = std::chrono::system_clock::now()
                    };
                    pending_events.push_back(event);
                }
                obj.state = "held";
                obj.parent_id = person.id;
                break;
            } else if (obj.state == "held" && obj.parent_id == person.id) {
                // State transition: put down
                Event event{
                    .object_id = id,
                    .event_type = "put_down",
                    .old_pos = obj.pos,
                    .new_pos = obj.pos,
                    .distance_moved = 0.0f,
                    .attributed_to = person.id,
                    .confidence = 0.8f,
                    .timestamp = std::chrono::system_clock::now()
                };
                pending_events.push_back(event);
                obj.state = "static";
                obj.parent_id = -1;
            }
        }
    }
    
    // 5. Check for moved objects and generate events
    for (auto& [id, obj] : tracked_objects) {
        if (obj.frames_lost == 0 && obj.velocity.norm() > config.position_threshold) {
            float vel = obj.velocity.norm();
            auto threshold_it = config.velocity_thresholds.find(obj.class_name);
            float threshold = (threshold_it != config.velocity_thresholds.end()) 
                              ? threshold_it->second : 1.0f;
            
            if (vel > threshold && should_alert(obj) ) {
                Event event{
                    .object_id = id,
                    .event_type = "moved",
                    .old_pos = obj.pos - obj.velocity * 0.05f, // 50ms ago
                    .new_pos = obj.pos,
                    .distance_moved = vel * 0.05f,
                    .attributed_to = find_likely_mover(obj, people),
                    .confidence = obj.confidence,
                    .timestamp = obj.last_seen
                };
                pending_events.push_back(event);
            }
        }
    }
    
    // 6. Handle lost tracks (occlusions, missing objects)
    prune_lost_tracks();
    
    // 7. Classify permanence of new objects
    for (auto& [id, obj] : tracked_objects) {
        classify_permanence(obj);
    }
}

void ObjectTracker::predict_all() {
    for (auto& [id, filter] : kalman_filters) {
        filter.predict(0.05f); // 50ms timestep
        tracked_objects[id].pos = filter.get_position();
        tracked_objects[id].velocity = filter.get_velocity();
        tracked_objects[id].uncertainty = filter.get_uncertainty();
    }
}

void ObjectTracker::associate_detections(const std::vector<Object>& detections) {
    std::vector<bool> matched_detections(detections.size(), false);
    std::vector<bool> matched_tracks; // Changed to avoid size issue
    // Note: matched_tracks not used, removed for simplicity

    for (auto& [track_id, tracked_obj] : tracked_objects) {
        float best_dist = std::numeric_limits<float>::max();
        int best_det_idx = -1;
        
        for (size_t i = 0; i < detections.size(); i++) {
            if (matched_detections[i]) continue;
            
            const auto& det = detections[i];
            
            // Class must match (unless deformable)
            bool class_match = (det.class_name == tracked_obj.class_name) ||
                               match_deformable(tracked_obj, det);
            if (!class_match) continue;
            
            // Position distance (Euclidean)
            float dist = (det.pos - tracked_obj.pos).norm();
            
            // Consider uncertainty
            float uncertainty_factor = 1.0f + tracked_obj.uncertainty;
            dist /= uncertainty_factor;
            
            if (dist < best_dist && dist < config.max_association_distance) {
                best_dist = dist;
                best_det_idx = i;
            }
        }
        
        if (best_det_idx >= 0) {
            // Update Kalman filter
            kalman_filters[track_id].update(detections[best_det_idx].pos);
            
            // Update object state
            tracked_obj.pos = kalman_filters[track_id].get_position();
            tracked_obj.velocity = kalman_filters[track_id].get_velocity();
            tracked_obj.confidence = detections[best_det_idx].confidence;
            tracked_obj.depth_confidence = detections[best_det_idx].depth_confidence;
            tracked_obj.bbox = detections[best_det_idx].bbox;
            tracked_obj.last_seen = detections[best_det_idx].last_seen;
            tracked_obj.frames_lost = 0;
            tracked_obj.seen_count++;
            
            matched_detections[best_det_idx] = true;
        } else {
            tracked_obj.frames_lost++;
            handle_occlusion(tracked_obj);
        }
    }
    
    // Create new tracks for unmatched detections
    std::vector<Object> unmatched;
    for (size_t i = 0; i < detections.size(); i++) {
        if (!matched_detections[i]) {
            unmatched.push_back(detections[i]);
        }
    }
    create_new_tracks(unmatched);
}

void ObjectTracker::create_new_tracks(const std::vector<Object>& unmatched) {
    for (const auto& det : unmatched) {
        Object new_obj = det;
        new_obj.id = next_id++;
        new_obj.global_id = generate_uuid();
        new_obj.state = "transient";
        new_obj.first_seen = det.last_seen;
        new_obj.seen_count = 1;
        new_obj.frames_lost = 0;
        new_obj.parent_id = -1;
        new_obj.support_surface_id = -1;
        
        tracked_objects[new_obj.id] = new_obj;
        
        // Initialize Kalman filter
        KalmanFilter filter;
        kalman_filters[new_obj.id] = filter;
        kalman_filters[new_obj.id].update(det.pos);
        
        // Log new object event
        Event event{
            .object_id = new_obj.id,
            .event_type = "appeared",
            .old_pos = Eigen::Vector3f::Zero(),
            .new_pos = det.pos,
            .distance_moved = 0.0f,
            .attributed_to = -1,
            .confidence = det.confidence,
            .timestamp = det.last_seen
        };
        pending_events.push_back(event);
    }
}

void ObjectTracker::prune_lost_tracks() {
    std::vector<int> to_remove;
    
    for (auto& [id, obj] : tracked_objects) {
        if (obj.frames_lost > config.max_frames_lost) {
            // Object is missing
            Event event{
                .object_id = id,
                .event_type = "disappeared",
                .old_pos = obj.pos,
                .new_pos = obj.pos,
                .distance_moved = 0.0f,
                .attributed_to = -1,
                .confidence = 0.7f,
                .timestamp = std::chrono::system_clock::now()
            };
            pending_events.push_back(event);
            to_remove.push_back(id);
        }
    }
    
    for (int id : to_remove) {
        tracked_objects.erase(id);
        kalman_filters.erase(id);
    }
}

bool ObjectTracker::is_object_held(const Person& person, const Object& object) {
    // Check distance to hand keypoints
    for (const auto& kp : person.keypoints) {
        if (kp.name == "left_wrist" || kp.name == "right_wrist") {
            float dist = (kp.pos_3d - object.pos).norm();
            if (dist < config.held_distance && kp.confidence > 0.5) {
                return true;
            }
        }
    }
    return false;
}

int ObjectTracker::find_likely_mover(const Object& obj, 
                                     const std::vector<Person>& people) {
    int closest_person_id = -1;
    float min_dist = std::numeric_limits<float>::max();
    
    for (const auto& person : people) {
        float dist = (person.pos - obj.pos).norm();
        if (dist < min_dist && dist < 1.5f) { // 1.5m threshold
            min_dist = dist;
            closest_person_id = person.id;
        }
    }
    
    // Verify person was moving their hands
    if (closest_person_id >= 0) {
        for (const auto& person : people) {
            if (person.id == closest_person_id && person.hand_velocity > 0.5f) {
                return person.id;
            }
        }
    }
    
    return -1; // Unknown mover
}

void ObjectTracker::resolve_overlaps(std::vector<Object>& objects) {
    for (size_t i = 0; i < objects.size(); i++) {
        for (size_t j = i + 1; j < objects.size(); j++) {
            // Calculate IoU
            cv::Rect intersection = objects[i].bbox & objects[j].bbox;
            cv::Rect union_rect = objects[i].bbox | objects[j].bbox;
            float iou = static_cast<float>(intersection.area()) / union_rect.area();
            
            if (iou > 0.7) {
                float depth_diff = std::abs(objects[i].pos.z() - objects[j].pos.z());
                
                if (depth_diff < 0.15f) { // Within 15cm depth
                    // Determine stacking order
                    if (objects[i].pos.z() < objects[j].pos.z()) {
                        objects[i].support_surface_id = objects[j].id;
                    } else {
                        objects[j].support_surface_id = objects[i].id;
                    }
                }
            }
        }
    }
}

bool ObjectTracker::match_deformable(const Object& obj1, const Object& obj2) {
    if (config.deformable_classes.find(obj1.class_name) == 
        config.deformable_classes.end()) {
        return false;
    }
    
    float centroid_dist = (obj1.pos - obj2.pos).norm();
    return (centroid_dist < 0.5f && obj1.class_name == obj2.class_name);
}

void ObjectTracker::handle_occlusion(Object& obj) {
    if (obj.frames_lost > 5 && obj.frames_lost < config.max_frames_lost) {
        obj.state = "occluded";
    } else if (obj.frames_lost >= config.max_frames_lost) {
        obj.state = "missing";
    }
}

void ObjectTracker::classify_permanence(Object& obj) {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        now - obj.first_seen).count();
    
    if (obj.seen_count < 10) {
        obj.state = "transient";
    } else if (duration < config.transient_period) {
        obj.state = "temporary";
    } else if (obj.seen_count > 50 || duration > config.transient_period) {
        if (obj.state != "permanent") {
            obj.state = "permanent";
            // Optionally add to known_objects
        }
    }
}

bool ObjectTracker::should_alert(const Object& obj) {
    // Don't alert for transient objects
    if (obj.state == "transient") return false;
    
    // Don't alert if held by person
    if (obj.state == "held") return false;
    
    // Don't alert if low confidence
    if (obj.confidence < 0.6 || obj.depth_confidence < 0.5) return false;
    
    return true;
}

std::vector<Event> ObjectTracker::get_events() {
    auto events = pending_events;
    pending_events.clear();
    return events;
}

std::vector<Object> ObjectTracker::get_tracked_objects() const {
    std::vector<Object> objects;
    for (const auto& [id, obj] : tracked_objects) {
        objects.push_back(obj);
    }
    return objects;
}
