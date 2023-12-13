#pragma once

#include "extension.h"

#define CHECK_CUDA(x) \
    AT_ASSERTM(x.device().is_cuda(), #x "must be CUDA Tensor")
