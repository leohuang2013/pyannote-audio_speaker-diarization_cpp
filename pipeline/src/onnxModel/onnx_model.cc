// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <sstream>
#include <iostream>
#include <unordered_map>

#include "onnxModel/onnx_model.h"

Ort::Env OnnxModel::env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
//Ort::Env OnnxModel::env_ = Ort::Env(ORT_LOGGING_LEVEL_INFO, "");
//Ort::Env OnnxModel::env_ = Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "");
Ort::SessionOptions OnnxModel::session_options_ = Ort::SessionOptions();

void OnnxModel::InitEngineThreads(int num_threads) {
  session_options_.SetIntraOpNumThreads(num_threads);
  //session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

static std::wstring ToWString(const std::string& str) {
  unsigned len = str.size() * 2;
  setlocale(LC_CTYPE, "");
  wchar_t* p = new wchar_t[len];
  mbstowcs(p, str.c_str(), len);
  std::wstring wstr(p);
  delete[] p;
  return wstr;
}

OnnxModel::OnnxModel(const std::string& model_path) 
{
    InitEngineThreads(1);

    /*
     * To avoid following error
     * [E:onnxruntime:, execution_providers.h:29 Add] Provider CUDAExecutionProvider has already been registered.
     *   terminate called after throwing an instance of 'Ort::Exception'
     *     what():  Provider CUDAExecutionProvider has already been registered.
     *   Aborted (core dumped)
     *   
     * Need set CPU provider, otherwise got following warning,
     * VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers
     * which may or may not have an negative impact on performance
     * */
    static bool init = false;
    if( !init )
    {
        // C++ way, but cannot find AppendExecutionProvider_CPU
        //OrtCUDAProviderOptions cuda_options;
        //cuda_options.device_id = 0;
        //session_options_.AppendExecutionProvider_CUDA(cuda_options);
        //session_options_.AppendExecutionProvider_CPU( 0 );

        // C API way
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0);
        std::vector<std::string> providers = Ort::GetAvailableProviders();
        status = OrtSessionOptionsAppendExecutionProvider_CPU( session_options_, 0 );

        init = true;
    }

#ifdef _MSC_VER
    session_ = std::make_shared<Ort::Session>(env_, ToWString(model_path).c_str(),
            session_options_);
#else
    session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
            session_options_);
#endif
    Ort::AllocatorWithDefaultOptions allocator;
    // Input info
    int num_nodes = session_->GetInputCount();
    input_node_names_.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        //input_node_names_[i] = session_->GetInputName(i, allocator); // onnx verion 1.20
        auto name_ptr = session_->GetInputNameAllocated(i, allocator); // onnx version 1.41
        auto ptr = name_ptr.get();
        char* tmp = new char[strlen( ptr ) + 1]; // TODO: release this memory
        strcpy( tmp, ptr );
        input_node_names_[i] = tmp; // name_ptr.get();
        std::cout << "Input names[" << i << "]: " << input_node_names_[i] <<std::endl;
    }
    // Output info
    num_nodes = session_->GetOutputCount();
    output_node_names_.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        //output_node_names_[i] = session_->GetOutputName(i, allocator);
        auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
        auto ptr = name_ptr.get();
        char* tmp = new char[strlen( ptr ) + 1]; // TODO: release this memory
        strcpy( tmp, ptr );
        output_node_names_[i] = tmp;
        std::cout << "Output names[" << i << "]: " << output_node_names_[i] <<std::endl;
    }
}
