//
// Created by lty on 10/11/19.
//

//

#pragma once

#include <torch/extension.h>

#ifndef PYTORCH_EXTENSION_TEST_RELU_CUDA_H
#define PYTORCH_EXTENSION_TEST_RELU_CUDA_H

at::Tensor forward_cuda(const at::Tensor& input);

at::Tensor backward_cuda(const at::Tensor& grad, const at::Tensor& input);

#endif //PYTORCH_EXTENSION_TEST_RELU_CUDA_H

