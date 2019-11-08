//
// Created by lty on 10/11/19.
//
#include "cpu/test_relu_cpu.h"

//#define WITH_CUDA None

#ifdef WITH_CUDA
#include "cuda/test_relu_cuda.h"
#endif

at::Tensor test_relu_forward(at::Tensor& input){
    at::Tensor result;
    if (input.type().is_cuda()){
        #ifdef WITH_CUDA
            if (input.numel() == 0)
                return at::empty({0}, input.options().dtype(at::kLong).device(at::kCPU));
            result = forward_cuda(input);
            return result;
        #else
            AT_ERROR("test_relu_forward Not compiled with GPU support");
        #endif
    }
    result = forward_cpu(input);
    return result;
}

at::Tensor test_relu_backward(at::Tensor& grad, const at::Tensor& input){
    at::Tensor result;
    if (grad.type().is_cuda()){
        #ifdef WITH_CUDA
            result = backward_cuda(grad, input);
            return result;
        #else
            AT_ERROR("test_relu_backward Not compiled with GPU support");
        #endif
    }
    result = backward_cpu(grad, input);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &test_relu_forward, "test relu forward");
    m.def("backward", &test_relu_backward, "test relu backward");
}