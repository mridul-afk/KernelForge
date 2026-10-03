cmake_minimum_required(VERSION 3.25)

project(KernelForge
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 3050 = Compute Capability 8.6
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cuda_test
  tests/cuda_test.cu
)

set_target_properties(cuda_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
