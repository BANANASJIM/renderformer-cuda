cmake_minimum_required(VERSION 3.24)
project(renderformer_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(ZLIB REQUIRED)

# third-party header-only libs
add_library(third_party INTERFACE)
target_include_directories(third_party INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# cuDNN frontend (header-only)
add_library(cudnn_frontend INTERFACE)
target_include_directories(cudnn_frontend INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn-frontend/include)

# main library
file(GLOB_RECURSE CUDA_SOURCES src/kernels/*.cu src/layers/*.cu src/pipeline/*.cu)
file(GLOB_RECURSE CPP_SOURCES src/core/*.cpp src/io/*.cpp src/model/*.cpp src/pipeline/*.cpp src/layers/*.cpp)

add_library(renderformer_lib STATIC ${CUDA_SOURCES} ${CPP_SOURCES})
target_include_directories(renderformer_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(renderformer_lib PUBLIC
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cudart
    cudnn
    ${HDF5_C_LIBRARIES}
    ZLIB::ZLIB
    third_party
    cudnn_frontend
)
target_include_directories(renderformer_lib PUBLIC ${HDF5_INCLUDE_DIRS})
target_compile_options(renderformer_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# main executable
add_executable(renderformer src/main.cpp)
target_link_libraries(renderformer PRIVATE renderformer_lib)

# interactive viewer (optional — requires GLFW + OpenGL)
find_package(glfw3 QUIET)
find_package(OpenGL QUIET)
if(glfw3_FOUND AND OpenGL_FOUND)
    add_executable(renderformer_app src/app/viewer.cpp)
    target_link_libraries(renderformer_app PRIVATE renderformer_lib glfw OpenGL::GL)
    message(STATUS "Building interactive viewer (renderformer_app)")
else()
    message(STATUS "Skipping viewer — install glfw3 and OpenGL for renderformer_app")
endif()

# tests
add_executable(test_encoder test/test_encoder.cu)
target_link_libraries(test_encoder PRIVATE renderformer_lib)
target_compile_options(test_encoder PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_executable(test_compare test/test_compare.cu)
target_link_libraries(test_compare PRIVATE renderformer_lib)
target_compile_options(test_compare PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_executable(test_enc_layer test/test_enc_layer.cu)
target_link_libraries(test_enc_layer PRIVATE renderformer_lib)
target_compile_options(test_enc_layer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
