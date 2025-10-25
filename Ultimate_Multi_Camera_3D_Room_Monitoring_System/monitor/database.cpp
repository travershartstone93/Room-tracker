#include "database.h"
#include <iostream>
#include <cstring>

Database::Database(const std::string& db_path) : db_path(db_path), db(nullptr) {
    int rc = sqlite3_open(db_path.c_str(), &db);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to open database: " + 
                                 std::string(sqlite3_errmsg(db)));
    }
}

Database::~Database() {
    if (stmt_insert_event) sqlite3_finalize(stmt_insert_event);
    if (stmt_update_object) sqlite3_finalize(stmt_update_object);
    if (stmt_insert_person) sqlite3_finalize(stmt_insert_person);
    if (stmt_insert_known_object) sqlite3_finalize(stmt_insert_known_object); // FIXED
    if (db) sqlite3_close(db);
}

void Database::initialize() {
    // Create tables
    const char* schema = R"(
        CREATE TABLE IF NOT EXISTS known_objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            global_id TEXT UNIQUE,
            class TEXT NOT NULL,
            x REAL, y REAL, z REAL,
            bbox_w REAL, bbox_h REAL,
            mapped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            global_id TEXT,
            camera_id INTEGER,
            class TEXT NOT NULL,
            x REAL, y REAL, z REAL,
            vx REAL, vy REAL, vz REAL,
            confidence REAL,
            depth_confidence REAL,
            state TEXT,
            parent_id INTEGER,
            support_surface_id INTEGER,
            uncertainty REAL,
            first_seen TIMESTAMP,
            last_seen TIMESTAMP,
            seen_count INTEGER DEFAULT 1
        );
        
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_id INTEGER,
            event_type TEXT,
            old_x REAL, old_y REAL, old_z REAL,
            new_x REAL, new_y REAL, new_z REAL,
            distance_moved REAL,
            attributed_to INTEGER,
            confidence REAL,
            camera_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracking_id INTEGER,
            x REAL, y REAL, z REAL,
            keypoints TEXT,
            hand_velocity REAL,
            last_seen TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS camera_status (
            camera_id INTEGER PRIMARY KEY,
            is_online BOOLEAN,
            fps REAL,
            last_frame_timestamp TIMESTAMP,
            frames_since_detection INTEGER
        );
        
        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_objects_global_id ON objects(global_id);
    )";
    
    execute(schema);
    prepare_statements();
}

void Database::execute(const std::string& sql) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::string error = "SQL error: " + std::string(err_msg);
        sqlite3_free(err_msg);
        throw std::runtime_error(error);
    }
}

void Database::prepare_statements() {
    const char* sql_event = R"(
        INSERT INTO events (object_id, event_type, old_x, old_y, old_z,
                            new_x, new_y, new_z, distance_moved,
                            attributed_to, confidence, camera_id, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";
    sqlite3_prepare_v2(db, sql_event, -1, &stmt_insert_event, nullptr);
    
    const char* sql_object = R"(
        INSERT OR REPLACE INTO objects 
        (id, global_id, camera_id, class, x, y, z, vx, vy, vz,
         confidence, depth_confidence, state, parent_id, support_surface_id,
         uncertainty, first_seen, last_seen, seen_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";
    sqlite3_prepare_v2(db, sql_object, -1, &stmt_update_object, nullptr);
    
    const char* sql_known = R"(
        INSERT INTO known_objects (global_id, class, x, y, z, bbox_w, bbox_h)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    )";
    sqlite3_prepare_v2(db, sql_known, -1, &stmt_insert_known_object, nullptr);
}

void Database::log_event(const Event& event) {
    sqlite3_reset(stmt_insert_event);
    
    sqlite3_bind_int(stmt_insert_event, 1, event.object_id);
    sqlite3_bind_text(stmt_insert_event, 2, event.event_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt_insert_event, 3, event.old_pos.x());
    sqlite3_bind_double(stmt_insert_event, 4, event.old_pos.y());
    sqlite3_bind_double(stmt_insert_event, 5, event.old_pos.z());
    sqlite3_bind_double(stmt_insert_event, 6, event.new_pos.x());
    sqlite3_bind_double(stmt_insert_event, 7, event.new_pos.y());
    sqlite3_bind_double(stmt_insert_event, 8, event.new_pos.z());
    sqlite3_bind_double(stmt_insert_event, 9, event.distance_moved);
    sqlite3_bind_int(stmt_insert_event, 10, event.attributed_to);
    sqlite3_bind_double(stmt_insert_event, 11, event.confidence);
    sqlite3_bind_int(stmt_insert_event, 12, event.camera_id);
    
    auto timestamp = std::chrono::system_clock::to_time_t(event.timestamp);
    sqlite3_bind_int64(stmt_insert_event, 13, timestamp);
    
    if (sqlite3_step(stmt_insert_event) != SQLITE_DONE) {
        std::cerr << "Failed to insert event: " << sqlite3_errmsg(db) << std::endl;
    }
}

void Database::update_object(const Object& object) {
    sqlite3_reset(stmt_update_object);
    
    sqlite3_bind_int(stmt_update_object, 1, object.id);
    sqlite3_bind_text(stmt_update_object, 2, object.global_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt_update_object, 3, object.camera_ids.empty() ? -1 : object.camera_ids[0]);
    sqlite3_bind_text(stmt_update_object, 4, object.class_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt_update_object, 5, object.pos.x());
    sqlite3_bind_double(stmt_update_object, 6, object.pos.y());
    sqlite3_bind_double(stmt_update_object, 7, object.pos.z());
    sqlite3_bind_double(stmt_update_object, 8, object.velocity.x());
    sqlite3_bind_double(stmt_update_object, 9, object.velocity.y());
    sqlite3_bind_double(stmt_update_object, 10, object.velocity.z());
    sqlite3_bind_double(stmt_update_object, 11, object.confidence);
    sqlite3_bind_double(stmt_update_object, 12, object.depth_confidence);
    sqlite3_bind_text(stmt_update_object, 13, object.state.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt_update_object, 14, object.parent_id);
    sqlite3_bind_int(stmt_update_object, 15, object.support_surface_id);
    sqlite3_bind_double(stmt_update_object, 16, object.uncertainty);
    
    auto first_seen = std::chrono::system_clock::to_time_t(object.first_seen);
    auto last_seen = std::chrono::system_clock::to_time_t(object.last_seen);
    sqlite3_bind_int64(stmt_update_object, 17, first_seen);
    sqlite3_bind_int64(stmt_update_object, 18, last_seen);
    sqlite3_bind_int(stmt_update_object, 19, object.seen_count);
    
    if (sqlite3_step(stmt_update_object) != SQLITE_DONE) {
        std::cerr << "Failed to update object: " << sqlite3_errmsg(db) << std::endl;
    }
}

void Database::add_known_object(const Object& obj) {
    sqlite3_reset(stmt_insert_known_object);
    
    sqlite3_bind_text(stmt_insert_known_object, 1, obj.global_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_insert_known_object, 2, obj.class_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt_insert_known_object, 3, obj.pos.x());
    sqlite3_bind_double(stmt_insert_known_object, 4, obj.pos.y());
    sqlite3_bind_double(stmt_insert_known_object, 5, obj.pos.z());
    sqlite3_bind_double(stmt_insert_known_object, 6, obj.bbox.width);
    sqlite3_bind_double(stmt_insert_known_object, 7, obj.bbox.height);
    
    if (sqlite3_step(stmt_insert_known_object) != SQLITE_DONE) {
        std::cerr << "Failed to insert known object: " << sqlite3_errmsg(db) << std::endl;
    }
}

std::vector<Object> Database::load_known_objects() {
    std::vector<Object> objects;
    
    const char* sql = "SELECT global_id, class, x, y, z FROM known_objects";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Object obj;
        obj.global_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        obj.class_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        obj.pos.x() = sqlite3_column_double(stmt, 2);
        obj.pos.y() = sqlite3_column_double(stmt, 3);
        obj.pos.z() = sqlite3_column_double(stmt, 4);
        obj.state = "known";
        objects.push_back(obj);
    }
    
    sqlite3_finalize(stmt);
    return objects;
}
