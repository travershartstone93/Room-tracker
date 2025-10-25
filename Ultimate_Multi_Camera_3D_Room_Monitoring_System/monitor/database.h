#pragma once
#include "types.h"
#include <string>
#include <vector>
#include <sqlite3.h>

class Database {
public:
    explicit Database(const std::string& db_path);
    ~Database();
    
    void initialize();
    void log_event(const Event& event);
    void update_object(const Object& object);
    void update_person(const Person& person);
    void update_camera_status(int camera_id, bool online, float fps);
    
    std::vector<Object> load_known_objects();
    void add_known_object(const Object& obj); // Added
    std::vector<Event> get_recent_events(int limit = 100);
    
private:
    sqlite3* db;
    std::string db_path;
    
    void execute(const std::string& sql);
    void prepare_statements();
    
    sqlite3_stmt* stmt_insert_event;
    sqlite3_stmt* stmt_update_object;
    sqlite3_stmt* stmt_insert_person;
    sqlite3_stmt* stmt_insert_known_object; // Added
};
