cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(speaker_diarization VERSION 0.1)

option(CXX11_ABI "whether to use CXX11_ABI libtorch" OFF)
option(GPU "whether to build with GPU" OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base
  "fc_base-${CMAKE_CXX_COMPILER_ID}"
  REALPATH BASE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
set(FETCHCONTENT_BASE_DIR ${fc_base})
option(BUILD_TESTING "whether to build unit test" OFF)

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fPIC")
else()
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_compile_options(/W0 /wd4150 /wd4244 /wd4267)
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}, src)

include(samplerate) # Note: must include libsamplerate before glog
include(onnx)
include(torch)

add_subdirectory(src/frontend)
add_subdirectory(src/onnxModel)
add_subdirectory(src/clustering)

add_executable(speakerDiarizer src/speakerDiarizer.cpp)
#target_link_libraries(speakerDiarizer PUBLIC torch frontend onnxModel clustering gflags )
target_link_libraries(speakerDiarizer PUBLIC torch frontend onnxModel clustering )

