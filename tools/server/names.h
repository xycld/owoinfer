// moss-server's protocol spellings of the C API enums, with their C values. They are the pipeline's own names
// (sampler_name(), scheduler_name(), noise_layout_name(), noise_source_name(), tier_name(), attention_name(),
// planner::mode_name(), moss_dit::block_path_name()), but the server links only the C API, so it keeps its own tables;
// tests/unit/test_server_json.cpp checks them against the pipeline's names. The server validates every enum parameter
// against these tables and lists them in hello's "values" (kValueTables).
#pragma once

#include <array>
#include <cstddef>
#include <span>
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

inline constexpr Name<mi_mode> kModeNames[] = {
    {"min_memory", MI_MODE_MIN_MEMORY},
    {"balanced", MI_MODE_BALANCED},
    {"max_speed", MI_MODE_MAX_SPEED},
};

// "default" is the server's spelling of an unset block path (the engine's default); the others are block_path_name()'s.
inline constexpr Name<mi_block_path> kBlockPathNames[] = {
    {"default", MI_BLOCK_PATH_DEFAULT},
    {"op_by_op", MI_BLOCK_PATH_OP_BY_OP},
    {"fused", MI_BLOCK_PATH_FUSED},
    {"fused_graphs", MI_BLOCK_PATH_FUSED_GRAPHS},
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

// The server's own spellings of mi_direct_io (the pipeline has an optional bool: unset, false, true).
inline constexpr Name<mi_direct_io> kDitDirectIoNames[] = {
    {"auto", MI_DIRECT_IO_AUTO},
    {"buffered", MI_DIRECT_IO_BUFFERED},
    {"direct", MI_DIRECT_IO_DIRECT},
};

// planner::te_host_pin_name()'s spellings.
inline constexpr Name<mi_te_host_pin> kTeHostPinNames[] = {
    {"auto", MI_TE_HOST_PIN_AUTO},
    {"on", MI_TE_HOST_PIN_ON},
    {"off", MI_TE_HOST_PIN_OFF},
};

// accel::step_cache_name()'s spellings.
inline constexpr Name<mi_step_cache> kStepCacheNames[] = {
    {"off", MI_STEP_CACHE_OFF},
    {"fbcache", MI_STEP_CACHE_FBCACHE},
};

// A table's names, for hello's "values".
template <class E, std::size_t N>
constexpr std::array<std::string_view, N> names_of(const Name<E> (&table)[N]) {
  std::array<std::string_view, N> out{};
  for (std::size_t i = 0; i < N; ++i) out[i] = table[i].name;
  return out;
}

namespace detail {
inline constexpr auto kModes = names_of(kModeNames);
inline constexpr auto kBlockPaths = names_of(kBlockPathNames);
inline constexpr auto kTiers = names_of(kTierNames);
inline constexpr auto kAttentions = names_of(kAttentionNames);
inline constexpr auto kDitDirectIos = names_of(kDitDirectIoNames);
inline constexpr auto kTeHostPins = names_of(kTeHostPinNames);
inline constexpr auto kSamplers = names_of(kSamplerNames);
inline constexpr auto kSchedulers = names_of(kSchedulerNames);
inline constexpr auto kNoiseLayouts = names_of(kNoiseLayoutNames);
inline constexpr auto kNoiseSources = names_of(kNoiseSourceNames);
inline constexpr auto kResizes = names_of(kResizeNames);
inline constexpr auto kMaskEncodings = names_of(kMaskEncodingNames);
inline constexpr auto kStepCaches = names_of(kStepCacheNames);
}  // namespace detail

// The parameter each table validates, for hello's "values": load's (mode .. te_host_pin), then generate's and
// preload's (sampler .. step_cache).
struct ValueTable {
  std::string_view param;
  std::span<const std::string_view> names;
};
inline constexpr ValueTable kValueTables[] = {
    {"mode", detail::kModes},
    {"block_path", detail::kBlockPaths},
    {"tier", detail::kTiers},
    {"attention", detail::kAttentions},
    {"dit_direct_io", detail::kDitDirectIos},
    {"te_host_pin", detail::kTeHostPins},
    {"sampler", detail::kSamplers},
    {"scheduler", detail::kSchedulers},
    {"noise_layout", detail::kNoiseLayouts},
    {"noise_source", detail::kNoiseSources},
    {"resize", detail::kResizes},
    {"mask_encoding", detail::kMaskEncodings},
    {"step_cache", detail::kStepCaches},
};

}  // namespace mi::server
