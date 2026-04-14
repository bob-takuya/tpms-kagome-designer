cmake_minimum_required(VERSION 3.20)
project(wgf_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen is header-only.
set(EIGEN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_compile_options(-O3 -Wall -Wno-sign-compare -DEIGEN_NO_DEBUG)

# Native (host) build — used for development / smoke tests.
if(NOT EMSCRIPTEN)
    add_executable(smoke_native src/smoke_native.cpp)
endif()

# Emscripten build — WASM module with a C API.
if(EMSCRIPTEN)
    add_executable(wgf_wasm src/wgf_main.cpp)
    set_target_properties(wgf_wasm PROPERTIES OUTPUT_NAME "wgf")
    target_link_options(wgf_wasm PRIVATE
        "-sMODULARIZE=1"
        "-sEXPORT_ES6=1"
        "-sEXPORT_NAME=createWgfModule"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sINITIAL_MEMORY=67108864"
        "-sMAXIMUM_MEMORY=2147483648"
        "-sENVIRONMENT=web,worker"
        "-sFILESYSTEM=0"
        "-sEXPORTED_FUNCTIONS=['_malloc','_free','_wgf_run','_wgf_result_num_families','_wgf_result_num_segments','_wgf_result_segment_family','_wgf_result_segment_face','_wgf_result_segment_points','_wgf_result_free','_wgf_result_free_all','_wgf_alg1_curl_initial','_wgf_alg1_curl_final','_wgf_alg1_iters','_wgf_num_singularities']"
        "-sEXPORTED_RUNTIME_METHODS=['HEAPF64','HEAPF32','HEAP32','HEAPU8','ccall','cwrap']"
        "-sASSERTIONS=1"
        "-O3"
    )
    set_target_properties(wgf_wasm PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../public/wasm"
    )
endif()
