#include "models/lora/lora_merge_device.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/lora/lora_merge.h"

namespace mi::lora {

namespace {

[[noreturn]] void fail(const std::string& msg) { throw LoraError("mi::lora: " + msg); }

constexpr std::size_t kAlign = 256;
constexpr std::int64_t kRowTile = 32;  // rows of the kernel's CTA tile: chunks are multiples of it

std::size_t align_up(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// The adapters merge_bf16 adds to `module` (merge_bf16's prepare(): merged ones with a nonzero coefficient), with
// their rank offsets; ok false when one is not low rank or carries a bias delta.
struct Plan {
  std::vector<const Adapter*> adapters;
  std::vector<int> bounds{0};
  std::vector<float> coef;
  std::int64_t rank = 0, k = 0;
  bool ok = true;
};

Plan plan_of(const LoraStack& stack, std::string_view module) {
  Plan p;
  for (const Adapter* a : stack.adapters_of(module)) {
    if (a->side_branch) continue;
    const float c = stack.coef(*a);
    if (!std::isfinite(c)) fail("coefficient strength * alpha / rank of '" + a->source + "' is not finite");
    if (c == 0.0f) continue;  // strength 0: the weight stays bit-identical
    if (a->kind != AdapterKind::kLowRank || a->bias_diff) p.ok = false;
    p.adapters.push_back(a);
    p.coef.push_back(c);
    p.rank += a->rank;
    p.bounds.push_back(static_cast<int>(p.rank));
    p.k = a->k;
  }
  return p;
}

// Scratch layout: [bad flag | bounds, coef | down [R, k] | up [chunk, R] | (host weight) w [chunk, k]].
std::size_t fixed_bytes(const Plan& p) {
  return align_up(sizeof(int)) + align_up(p.bounds.size() * sizeof(int) + p.coef.size() * sizeof(float)) +
         align_up(static_cast<std::size_t>(p.rank * p.k) * sizeof(float));
}
std::size_t chunk_bytes(const Plan& p, std::int64_t rows, bool host_weight) {
  return align_up(static_cast<std::size_t>(rows * p.rank) * sizeof(float)) +
         (host_weight ? align_up(static_cast<std::size_t>(rows * p.k) * sizeof(std::uint16_t)) : 0);
}

// Rows per chunk for `scratch_bytes` of scratch: the most that fit (whole row tiles, or all n rows), counting the
// chunk regions' real alignment padding; 0 when not even one row tile (or all n rows below one tile) fits.
std::int64_t chunk_rows(const Plan& p, std::int64_t n, bool host_weight, std::size_t scratch_bytes) {
  const std::size_t fixed = fixed_bytes(p);
  if (scratch_bytes < fixed) return 0;
  const std::size_t per_row = static_cast<std::size_t>(p.rank) * sizeof(float) + (host_weight ? static_cast<std::size_t>(p.k) * 2 : 0);
  if (per_row == 0) return n;  // nothing to merge
  std::int64_t chunk = std::min(n, static_cast<std::int64_t>((scratch_bytes - fixed) / per_row));
  if (chunk < n) chunk = chunk / kRowTile * kRowTile;
  while (chunk > 0 && fixed + chunk_bytes(p, chunk, host_weight) > scratch_bytes) {
    chunk = chunk % kRowTile == 0 ? chunk - kRowTile : chunk / kRowTile * kRowTile;
  }
  return chunk;
}

std::int64_t rows_of(const LoraStack& stack, std::string_view module) {
  const TargetModule* t = stack.model().find(module);
  if (t == nullptr) fail("no module '" + std::string(module) + "' in the model");
  return t->n;
}

}  // namespace

bool device_mergeable(const LoraStack& stack, std::string_view module) { return plan_of(stack, module).ok; }

std::size_t device_merge_min_scratch(const LoraStack& stack, std::string_view module, bool host_weight) {
  const Plan p = plan_of(stack, module);
  return fixed_bytes(p) + chunk_bytes(p, std::min(kRowTile, rows_of(stack, module)), host_weight);
}

std::int64_t device_merge_chunk_rows(const LoraStack& stack, std::string_view module, bool host_weight,
                                     std::size_t scratch_bytes) {
  return chunk_rows(plan_of(stack, module), rows_of(stack, module), host_weight, scratch_bytes);
}

std::size_t device_merge_full_scratch(const LoraStack& stack, std::string_view module, bool host_weight) {
  const Plan p = plan_of(stack, module);
  return fixed_bytes(p) + chunk_bytes(p, rows_of(stack, module), host_weight);
}

int merge_bf16_device(const LoraStack& stack, std::string_view module, std::uint16_t* w, std::int64_t n, std::int64_t k,
                      bool host_weight, void* scratch, std::size_t scratch_bytes, cudaStream_t stream) {
  const TargetModule* t = stack.model().find(module);
  if (t == nullptr) fail("no module '" + std::string(module) + "' in the model");
  if (t->n != n || t->k != k) {
    fail("module '" + std::string(module) + "' is [" + std::to_string(t->n) + ", " + std::to_string(t->k) + "], got [" +
         std::to_string(n) + ", " + std::to_string(k) + "]");
  }
  const Plan p = plan_of(stack, module);
  if (!p.ok) fail("'" + std::string(module) + "' has adapters the device merge does not take (not low rank, or a bias delta)");
  if (p.adapters.empty()) return 0;
  if (w == nullptr || scratch == nullptr || reinterpret_cast<std::uintptr_t>(scratch) % kAlign != 0) {
    fail("merge_bf16_device: null weight or unaligned scratch for '" + std::string(module) + "'");
  }
  const std::size_t fixed = fixed_bytes(p);
  const std::int64_t chunk = chunk_rows(p, n, host_weight, scratch_bytes);
  if (chunk <= 0) {
    fail("merge_bf16_device: " + std::to_string(scratch_bytes) + " bytes of scratch are too few for '" +
         std::string(module) + "' (" + std::to_string(device_merge_min_scratch(stack, module, host_weight)) + " needed)");
  }
  auto* const base = static_cast<std::byte*>(scratch);
  int* const bad = reinterpret_cast<int*>(base);
  auto* const params = base + align_up(sizeof(int));
  int* const bounds = reinterpret_cast<int*>(params);
  float* const coef = reinterpret_cast<float*>(params + p.bounds.size() * sizeof(int));
  float* const down = reinterpret_cast<float*>(params + align_up(p.bounds.size() * sizeof(int) + p.coef.size() * sizeof(float)));
  float* const up = reinterpret_cast<float*>(base + fixed);
  auto* const wchunk = reinterpret_cast<std::uint16_t*>(base + fixed + align_up(static_cast<std::size_t>(chunk * p.rank) * sizeof(float)));

  // Pageable host-to-device copies return once the host buffer has been staged, so the host vectors may be reused.
  std::vector<std::byte> head(p.bounds.size() * sizeof(int) + p.coef.size() * sizeof(float));
  std::memcpy(head.data(), p.bounds.data(), p.bounds.size() * sizeof(int));
  std::memcpy(head.data() + p.bounds.size() * sizeof(int), p.coef.data(), p.coef.size() * sizeof(float));
  // On an exception once work is queued (a CUDA error, a bad factor read between chunks), wait for that work before
  // the caller frees or reuses the scratch and the weight.
  struct SyncOnThrow {
    cudaStream_t stream;
    int pending = std::uncaught_exceptions();
    ~SyncOnThrow() {
      if (std::uncaught_exceptions() > pending) static_cast<void>(cudaStreamSynchronize(stream));
    }
  } sync_on_throw{stream};
  MI_CUDA_CHECK(cudaMemsetAsync(bad, 0, sizeof(int), stream));
  MI_CUDA_CHECK(cudaMemcpyAsync(params, head.data(), head.size(), cudaMemcpyHostToDevice, stream));
  {
    std::vector<float> d(static_cast<std::size_t>(p.rank * k));
    for (std::size_t i = 0; i < p.adapters.size(); ++i) {
      const std::vector<float> f = stack.read(p.adapters[i]->parts[0]);  // down [rank, k]
      std::copy(f.begin(), f.end(), d.begin() + static_cast<std::ptrdiff_t>(p.bounds[i]) * k);
    }
    MI_CUDA_CHECK(cudaMemcpyAsync(down, d.data(), d.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  }
  std::vector<float> u(static_cast<std::size_t>(chunk * p.rank));
  for (std::int64_t r0 = 0; r0 < n; r0 += chunk) {
    const std::int64_t rows = std::min(chunk, n - r0);
    for (std::size_t i = 0; i < p.adapters.size(); ++i) {
      const Adapter& a = *p.adapters[i];
      const std::vector<float> f = stack.read(a.parts[1], a.row0 + r0, rows);  // up rows [rows, rank]
      for (std::int64_t r = 0; r < rows; ++r) {
        std::copy_n(f.begin() + r * a.rank, a.rank, u.begin() + r * p.rank + p.bounds[i]);
      }
    }
    MI_CUDA_CHECK(cudaMemcpyAsync(up, u.data(), static_cast<std::size_t>(rows * p.rank) * sizeof(float),
                                  cudaMemcpyHostToDevice, stream));
    std::uint16_t* const target = host_weight ? wchunk : w + r0 * k;
    const std::size_t wbytes = static_cast<std::size_t>(rows * k) * sizeof(std::uint16_t);
    if (host_weight) MI_CUDA_CHECK(cudaMemcpyAsync(wchunk, w + r0 * k, wbytes, cudaMemcpyHostToDevice, stream));
    kernels::lora::merge_low_rank_bf16(reinterpret_cast<__nv_bfloat16*>(target), k, rows, k, up, p.rank, down, k, bounds,
                                       coef, static_cast<int>(p.adapters.size()), bad, stream);
    if (host_weight) MI_CUDA_CHECK(cudaMemcpyAsync(w + r0 * k, wchunk, wbytes, cudaMemcpyDeviceToHost, stream));
  }
  int flag = 0;
  MI_CUDA_CHECK(cudaMemcpyAsync(&flag, bad, sizeof(int), cudaMemcpyDeviceToHost, stream));
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (flag != 0) fail("merged weight of '" + std::string(module) + "' is not finite");
  return static_cast<int>(p.adapters.size());
}

}  // namespace mi::lora
