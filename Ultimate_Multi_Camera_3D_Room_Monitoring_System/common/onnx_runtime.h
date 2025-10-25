#pragma once

#include <vector>
#include <string>

class ONNXRuntime {
public:
    ONNXRuntime(const std::string& model_path, bool use_gpu = false);
    ~ONNXRuntime();

    std::vector<float> run(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape, const std::vector<int64_t>& output_shape);

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name;
    std::string output_name;
};
