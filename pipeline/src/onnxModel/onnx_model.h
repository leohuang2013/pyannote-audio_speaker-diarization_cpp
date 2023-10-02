#ifndef DIARIZATION_ONNX_MODEL_H_
#define DIARIZATION_ONNX_MODEL_H_

#include <string>
#include <vector>

#include "cpu_provider_factory.h"  // NOLINT
#include "onnxruntime_cxx_api.h"  // NOLINT

class OnnxModel 
{
    public:
        static void InitEngineThreads(int num_threads = 1);
        OnnxModel(const std::string& model_path);

    protected:
        static Ort::Env env_;
        static Ort::SessionOptions session_options_;

        std::shared_ptr<Ort::Session> session_ = nullptr;
        Ort::MemoryInfo memory_info_ =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

        std::vector<const char*> input_node_names_;
        std::vector<const char*> output_node_names_;
};

#endif  // DIARIZATION_ONNX_MODEL_H_
