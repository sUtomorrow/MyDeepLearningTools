//
// Created by lty on 10/11/19.
//

#include "test_relu_cuda.h"

template <typename scalar_t>
at::Tensor forward_kernel(const at::Tensor& input){
    return input;
}

template <typename scalar_t>
at::Tensor backward_kernel(const at::Tensor& grad, const at::Tensor& input){
    return grad;
}

at::Tensor forward_cuda(const at::Tensor& input){
    at::Tensor result;
    AT_DISPATCH_ALL_TYPES(input.type(), "test_relu_forward_cuda", [&] {
        result = forward_kernel<scalar_t>(input);
    });
    return result;
}

at::Tensor backward_cuda(const at::Tensor& grad, const at::Tensor& input){
    at::Tensor result;
    AT_DISPATCH_ALL_TYPES(grad.type(), "test_relu_backward_cuda", [&] {
        result = backward_kernel<scalar_t>(grad, input);
    });
    return result;
}

