// Test-only hooks of kernels/attention_q8 (not part of the engine interface).
#pragma once

#include <cuda_runtime.h>

namespace mi::attention_q8 {

// Timing: when set, attention_q8 records `e` on its stream between the quantization pre-pass and the attention
// kernel. Process-wide; nullptr turns it off.
void set_prepass_event_for_testing(cudaEvent_t e);

}  // namespace mi::attention_q8
