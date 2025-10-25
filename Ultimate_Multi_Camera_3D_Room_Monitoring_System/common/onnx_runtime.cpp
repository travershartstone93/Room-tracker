#include "onnx_runtime.h"
#include <onnxruntime_cxx_api.h>

ONNXRuntime::ONNXRuntime(const std::string& model_path, bool use_gpu) : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime") {
    Ort::SessionOptions options;
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options{};
        options.AppendExecutionProvider_CUDA(cuda_options);
    }
    
    session = Ort::Session(env, model_path.c_str(), options);
    
    // FIXED name retrieval:
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    input_name = std::string(input_name_ptr.get());
    
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    output_name = std::string(output_name_ptr.get());
}

ONNXRuntime::~ONNXRuntime() {
}

std::vector<float> ONNXRuntime::run(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape, const std::vector<int64_t>& output_shape) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor(
        memory_info, 
        const_cast<float*>(input_data.data()), 
        input_data.size(), 
        input_shape.data(), 
        input_shape.size()
    );
    
    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};
    
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, 
        &input_tensor, 
        1, 
        output_names, 
        1
    );
    
    float* float_array = output_tensors[0].GetTensorMutableData<float>();
    size_t total_elements = 1;
    for (auto dim : output_shape) total_elements *= dim;
    
    return std::vector<float>(float_array, float_array + total_elements);
}
