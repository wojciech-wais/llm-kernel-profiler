cmake_minimum_required(VERSION 3.18)
project(llm_kernel_lab_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 QUIET)

# Library sources
add_library(llm_kernel_lab_cuda STATIC
    src/benchmark.cpp
    src/timing.cu
)

target_include_directories(llm_kernel_lab_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(llm_kernel_lab_cuda PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(llm_kernel_lab_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Python bindings (optional, requires pybind11)
if(pybind11_FOUND)
    pybind11_add_module(llm_kernel_lab_cuda_py
        bindings/pybind_module.cpp
    )

    target_link_libraries(llm_kernel_lab_cuda_py PRIVATE
        llm_kernel_lab_cuda
    )

    set_target_properties(llm_kernel_lab_cuda_py PROPERTIES
        OUTPUT_NAME "llm_kernel_lab_cuda"
    )
endif()
