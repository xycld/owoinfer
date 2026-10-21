// This is purely so that it works with torch 2.1. For torch 2.2+ we can include ATen/cuda/PhiloxUtils.cuh

#pragma once
#include "mi_torch_shim.h"  // MossInfer: was <ATen/cuda/detail/UnpackRaw.cuh> (at::cuda::philox::unpack)
