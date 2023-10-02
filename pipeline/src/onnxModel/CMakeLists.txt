add_library(onnxModel STATIC onnx_model.cc )
target_link_libraries(onnxModel PUBLIC onnxruntime)
