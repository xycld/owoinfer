// moss-server's protocol spellings of the C API enums, with their C values. They are the pipeline's own names
// (sampler_name(), scheduler_name(), noise_layout_name(), noise_source_name(), tier_name(), attention_name()), but the
// server links only the C API, so it keeps its own tables; tests/unit/test_server_json.cpp checks them against the
// pipeline's names.
#pragma once

#include <string_view>

#include "mossinfer/mossinfer.h"

namespace mi::server {

template <class E>
struct Name {
  std::string_view name;
  E value;
};

inline constexpr Name<mi_sampler> kSamplerNames[] = {
    {"euler", MI_SAMPLER_EULER},
    {"ab2", MI_SAMPLER_AB2},
    {"euler_ancestral", MI_SAMPLER_EULER_ANCESTRAL},
    {"heun", MI_SAMPLER_HEUN},
    {"dpmpp_2m", MI_SAMPLER_DPMPP_2M},
    {"dpmpp_2m_sde", MI_SAMPLER_DPMPP_2M_SDE},
    {"er_sde", MI_SAMPLER_ER_SDE},
    {"res_multistep", MI_SAMPLER_RES_MULTISTEP},
    {"dataspace", MI_SAMPLER_DATASPACE},
};

inline constexpr Name<mi_scheduler> kSchedulerNames[] = {
    {"moss", MI_SCHEDULER_MOSS},
    {"simple", MI_SCHEDULER_SIMPLE},
    {"normal", MI_SCHEDULER_NORMAL},
    {"karras", MI_SCHEDULER_KARRAS},
    {"exponential", MI_SCHEDULER_EXPONENTIAL},
    {"beta", MI_SCHEDULER_BETA},
    {"sgm_uniform", MI_SCHEDULER_SGM_UNIFORM},
};

inline constexpr Name<mi_noise_layout> kNoiseLayoutNames[] = {
    {"official", MI_NOISE_LAYOUT_OFFICIAL},
    {"batch_stream", MI_NOISE_LAYOUT_BATCH_STREAM},
};

inline constexpr Name<mi_noise_source> kNoiseSourceNames[] = {
    {"philox", MI_NOISE_SOURCE_PHILOX},
    {"philox_device", MI_NOISE_SOURCE_PHILOX_DEVICE},
    {"brownian_tree", MI_NOISE_SOURCE_BROWNIAN_TREE},
};

inline constexpr Name<mi_tier> kTierNames[] = {
    {"auto", MI_TIER_AUTO},
    {"reference", MI_TIER_REFERENCE},
    {"high", MI_TIER_HIGH},
    {"turbo", MI_TIER_TURBO},
};

// resize_mode_name()'s spellings.
inline constexpr Name<mi_resize_mode> kResizeNames[] = {
    {"crop", MI_RESIZE_CROP},
    {"fit", MI_RESIZE_FIT},
    {"stretch", MI_RESIZE_STRETCH},
};

// The server's own spellings (the pipeline has no mask encodings: it takes 8-bit masks).
inline constexpr Name<mi_mask_encoding> kMaskEncodingNames[] = {
    {"u8", MI_MASK_U8},
    {"binary01", MI_MASK_BINARY01},
};

// "default" is the server's spelling of an unset attention (the tier's choice); the others are attention_name()'s.
inline constexpr Name<mi_attention> kAttentionNames[] = {
    {"default", MI_ATTENTION_DEFAULT},
    {"exact", MI_ATTENTION_EXACT},
    {"q8_fp16", MI_ATTENTION_Q8_FP16},
    {"q8_fp16_f32acc", MI_ATTENTION_Q8_FP16_F32ACC},
    {"q8_fp8", MI_ATTENTION_Q8_FP8},
    {"q8_fp8_f32acc", MI_ATTENTION_Q8_FP8_F32ACC},
};

// accel::step_cache_name()'s spellings.
inline constexpr Name<mi_step_cache> kStepCacheNames[] = {
    {"off", MI_STEP_CACHE_OFF},
    {"fbcache", MI_STEP_CACHE_FBCACHE},
};

}  // namespace mi::server
