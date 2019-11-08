//
// Created by lty on 10/11/19.
//
#include "test_relu_cpu.h"

template <typename scalar_t>
at::Tensor forward_kernel(const at::Tensor& input){
    AT_ASSERTM(!input.type().is_cuda(), "input must be a CPU tensor");

    if (input.numel() == 0) {
        return at::empty({0}, input.options().dtype(at::kLong).device(at::kCPU));
    }else{
        at::Tensor result = at::zeros_like(input);
        // TODO: for(int i = 0, i < )
        return input;
    }
}

template <typename scalar_t>
at::Tensor backward_kernel(const at::Tensor& grad, const at::Tensor& input){
    return grad;
}

at::Tensor forward_cpu(const at::Tensor& input){
    at::Tensor result;
    AT_DISPATCH_ALL_TYPES(input.type(), "test_relu_forward_cpu", [&] {
        result = forward_kernel<scalar_t>(input);
    });
    return result;
}

at::Tensor backward_cpu(const at::Tensor& grad, const at::Tensor& input){
    at::Tensor result;
    AT_DISPATCH_ALL_TYPES(grad.type(), "test_relu_backward_cpu", [&] {
        result = backward_kernel<scalar_t>(grad, input);
    });
    return result;
}

