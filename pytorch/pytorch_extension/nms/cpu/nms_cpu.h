//
// Created by lty on 10/9/19.
//

#pragma once
#include <torch/extension.h>


#ifndef PYTORCH_EXTENSION_NMS_CPU_H
#define PYTORCH_EXTENSION_NMS_CPU_H

at::Tensor nms_cpu(const at::Tensor& dets,
                   const at::Tensor& scores,
                   const float threshold);

#endif //PYTORCH_EXTENSION_NMS_CPU_H
