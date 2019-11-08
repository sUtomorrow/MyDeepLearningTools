cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 11)

project(pytorch_extension)

set(PYTORCH_INCLUDE /home/lty/.conda/envs/pytorch10/lib/python3.6/site-packages/torch/lib/include/)

include_directories(${PYTORCH_INCLUDE} /home/lty/.conda/envs/pytorch10/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include/)

add_executable(pytorch_extension_nms nms/cpu/nms_cpu.cpp nms/cpu/nms_cpu.h)

add_executable(pytorch_extension_test_relu test_relu/test_relu.cpp test_relu/cpu/test_relu_cpu.h test_relu/cpu/test_relu_cpu.cpp test_relu/cuda/test_relu_cuda.h test_relu/cuda/test_relu_cuda.cu)

add_executable(pytorch_extension pytorch_extension_main.cpp)

