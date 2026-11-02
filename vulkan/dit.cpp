#include "dit.h"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

#include "safetensors.h"

namespace mi::vk {
namespace {

constexpr std::uint32_t kDim = kDiTDim, kFfn = kDiTFfnDim, kHead = kDiTHeadDim, kQkv = 3 * kDiTDim;
constexpr VkDeviceSize kBf16 = 2;

// The RoPE band frequencies 1 / 10000^(i / 16), as the reference builds them in fp32 on the CPU (kernels/basic/rope.h).
constexpr std::uint32_t kBandBits[16] = {0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887, 0x3dcccccd, 0x3d6655c2,
                                         0x3d0186e3, 0x3c91ad39, 0x3c23d70a, 0x3bb8449c, 0x3b4f3e38, 0x3ae91528,
                                         0x3a83126f, 0x3a136a16, 0x39a5cb60, 0x393a7753};

std::uint16_t to_bf16(float v) {
  const auto u = std::bit_cast<std::uint32_t>(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
  return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float from_bf16(std::uint16_t b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("vulkan dit: " + what); }

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

VkDeviceAddress at(const Buffer& b, VkDeviceSize byte_offset) { return b.address() + byte_offset; }

}  // namespace

std::vector<std::uint16_t> rope_table(std::uint32_t height, std::uint32_t width, std::uint32_t text_length) {
  const std::uint64_t hw = std::uint64_t{height} * width, n = hw + text_length;
  const float h_mid = static_cast<float>((static_cast<double>(height) - 1.0) / 2.0);
  const float w_mid = static_cast<float>((static_cast<double>(width) - 1.0) / 2.0);
  std::vector<std::uint16_t> out(n * kHead);
  for (std::uint64_t r = 0; r < n; ++r) {
    for (std::uint32_t p = 0; p < 64; ++p) {
      const std::uint32_t axis = p >> 4, band = p & 15;
      float coord = 0.0f;
      if (r < hw) {
        if (axis == 1) coord = static_cast<float>(text_length);
        if (axis == 2) coord = static_cast<float>(r / width) - h_mid;
        if (axis == 3) coord = static_cast<float>(r % width) - w_mid;
      } else if (axis == 1) {
        coord = static_cast<float>(r - hw);
      }
      const float phase = coord * std::bit_cast<float>(kBandBits[band]);
      // sinf/cosf to fp32 as torch's CUDA kernel; libm's are correctly rounded or within an ulp, and bf16 hides that.
      out[r * kHead + p] = to_bf16(std::sin(phase));
      out[r * kHead + 64 + p] = to_bf16(std::cos(phase));
    }
  }
  return out;
}

Dit::Dit(Device& device, const BlockKernels& kernels, const std::filesystem::path& transformer_dir, const DitOptions& options)
    : device_(device), kernels_(kernels) {
  max_gflop_ = options.submit_gflop > 0 ? options.submit_gflop : kernels.path() == BlockPath::kCoop ? 200.0 : 20.0;
  const st::Index index(transformer_dir);
  std::vector<std::byte> host;
  // Places tensors in one new buffer, each at a 256-byte aligned offset, and uploads them (or, for a host buffer, reads
  // them straight into its mapping); returns their offsets.
  const auto place = [&](const std::vector<std::string>& names, const std::vector<std::vector<std::int64_t>>& shapes,
                         Buffer::Kind kind) {
    std::vector<VkDeviceSize> offsets;
    VkDeviceSize total = 0;
    for (std::size_t i = 0; i < names.size(); ++i) {
      const st::TensorInfo& t = index.at(names[i]);
      if (t.dtype != "BF16") fail(names[i] + " is " + t.dtype + ", expected BF16");
      if (t.shape != shapes[i]) fail(names[i] + " has an unexpected shape");
      offsets.push_back(total);
      total += (t.bytes + 255) / 256 * 256;
    }
    owned_.push_back(std::make_unique<Buffer>(device_, total, kind));
    Buffer& buf = *owned_.back();
    for (std::size_t i = 0; i < names.size(); ++i) {
      const std::uint64_t bytes = index.at(names[i]).bytes;
      if (kind == Buffer::Kind::kStaging) {
        index.read_into(names[i], static_cast<std::byte*>(buf.mapped()) + offsets[i], bytes);
      } else {
        host.resize(bytes);
        index.read_into(names[i], host.data(), host.size());
        buf.upload(host.data(), host.size(), offsets[i]);
      }
    }
    if (kind == Buffer::Kind::kDevice) weight_bytes_ += total;
    else host_weight_bytes_ += total;
    return offsets;
  };
  const auto load = [&](const std::vector<std::string>& names, const std::vector<std::vector<std::int64_t>>& shapes) {
    const std::vector<VkDeviceSize> offsets = place(names, shapes, Buffer::Kind::kDevice);
    std::vector<VkDeviceAddress> addresses;
    for (const VkDeviceSize o : offsets) addresses.push_back(owned_.back()->address() + o);
    return addresses;
  };
  const std::int64_t d = kDim, f = kFfn;

  auto a = load({"video_embedder.weight", "video_embedder.bias", "time_embedding.0.weight", "time_embedding.0.bias",
                 "time_embedding.2.weight", "time_embedding.2.bias", "shared_block_modulation.weight",
                 "shared_block_modulation.bias", "video_head_modulation.weight", "video_head_modulation.bias",
                 "final_norm_video.weight", "final_linear_video.weight", "text_embedder.proj.weight",
                 "text_embedder.proj.bias"},
                {{d, kDiTLatentChannels}, {d}, {d, kDiTFreqDim}, {d}, {d, d}, {d}, {6 * d, d}, {6 * d}, {2 * d, d}, {2 * d},
                 {d}, {kDiTLatentChannels, d}, {d, kDiTTextDim}, {d}});
  g_ = {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]};

  for (int b = 0; b < 2; ++b) {
    const std::string p = "text_embedder.blocks." + std::to_string(b) + ".";
    auto t = load({p + "norm1.weight", p + "norm2.weight", p + "q_norm.weight", p + "k_norm.weight", p + "qkv.weight",
                   p + "qkv.bias", p + "proj.weight", p + "proj.bias", p + "ffn_up.weight", p + "ffn_up.bias",
                   p + "ffn_gate.weight", p + "ffn_gate.bias", p + "ffn_down.weight", p + "ffn_down.bias"},
                  {{d}, {d}, {kHead}, {kHead}, {3 * d, d}, {3 * d}, {d, d}, {d}, {f, d}, {f}, {f, d}, {f}, {d, f}, {d}});
    text_blocks_.push_back({t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13]});
  }

  // Blocks: the first `resident` in device memory, the rest in host memory, each copied into one device slot before it
  // runs. q, k and v sit back to back (256-aligned sizes of 18 MiB each), so they form the [9216, 3072] QKV weight.
  const std::vector<std::vector<std::int64_t>> block_shapes = {{d}, {d}, {kHead}, {kHead}, {d, d}, {d, d}, {d, d}, {d, d},
                                                                {2 * f, d}, {d, f}};
  const auto block_names = [](int l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    return std::vector<std::string>{p + "norm1.weight", p + "norm2.weight", p + "q_norm.weight", p + "k_norm.weight",
                                    p + "proj.q.weight", p + "proj.k.weight", p + "proj.v.weight", p + "proj.o.weight",
                                    p + "proj.ffn_up_gate.weight", p + "proj.ffn_down.weight"};
  };
  const auto block_weights = [&](VkDeviceAddress base, const std::vector<VkDeviceSize>& o) {
    if (o[5] != o[4] + kDim * kDim * kBf16 || o[6] != o[5] + kDim * kDim * kBf16) fail("q, k and v are not contiguous");
    return BlockWeights{base + o[0], base + o[1], base + o[2], base + o[3], base + o[4], base + o[7], base + o[8], base + o[9]};
  };
  VkDeviceSize block_bytes = 0;
  for (const std::string& n : block_names(0)) block_bytes += (index.at(n).bytes + 255) / 256 * 256;
  int resident = kDiTLayers;
  if (options.device_budget > 0) {
    const std::uint64_t free = options.device_budget > weight_bytes_ ? options.device_budget - weight_bytes_ : 0;
    // A streamed block needs the slot too.
    resident = free >= kDiTLayers * block_bytes ? kDiTLayers
                                                : static_cast<int>(free > block_bytes ? free / block_bytes - 1 : 0);
  }
  for (int l = 0; l < kDiTLayers; ++l) {
    const bool on_device = l < resident;
    const std::vector<VkDeviceSize> o = place(block_names(l), block_shapes, on_device ? Buffer::Kind::kDevice : Buffer::Kind::kStaging);
    if (on_device) {
      blocks_.push_back(block_weights(owned_.back()->address(), o));
      host_blocks_.push_back(nullptr);
    } else {
      if (!slot_) {
        slot_ = std::make_unique<Buffer>(device_, block_bytes);
        weight_bytes_ += block_bytes;
      }
      blocks_.push_back(block_weights(slot_->address(), o));
      host_blocks_.push_back(owned_.back().get());
    }
  }
  resident_blocks_ = resident;
}

Dit::~Dit() = default;

void Dit::run_block_op(int op, const BlockWeights& w, const BlockBuffers& b) {
  // Multiply-adds per token, x2 for FLOP; attention's per query token over its sequence's keys.
  static constexpr double kFlopPerToken[kBlockOps] = {0, 2.0 * kDim * kQkv, 0, 0, 2.0 * kDim * kDim, 0,
                                                      2.0 * kDim * 2 * kFfn, 2.0 * kFfn * kDim};
  const double limit = max_gflop_ * 1e9;
  if (op == 3) {
    // Attention splits along each sequence's queries: every query row reads all of its sequence's keys.
    for (std::size_t s = 0; s + 1 < b.cu_seqlens.size(); ++s) {
      const auto start = static_cast<std::uint32_t>(b.cu_seqlens[s]), len = static_cast<std::uint32_t>(b.cu_seqlens[s + 1]) - start;
      const double per_query = 4.0 * len * kDim;
      std::uint32_t chunk = len;
      if (limit > 0 && per_query * len > limit) chunk = std::max<std::uint32_t>(64, static_cast<std::uint32_t>(limit / per_query) / 64 * 64);
      for (std::uint32_t q0 = 0; q0 < len; q0 += chunk) {
        const std::uint32_t n = std::min(chunk, len - q0);
        const VkDeviceAddress base = b.qkv + VkDeviceAddress{start} * kQkv * kBf16;
        const VkDeviceAddress q = base + VkDeviceAddress{q0} * kQkv * kBf16;
        const AttentionArgs args{q, base + kDim * kBf16, base + 2 * kDim * kBf16, b.attn + (VkDeviceAddress{start} + q0) * kDim * kBf16,
                                 kQkv, kQkv, kQkv, kDim, kHead, kHead, kHead, kHead, {0, static_cast<std::int32_t>(n)},
                                 {0, static_cast<std::int32_t>(len)}, kDiTHeads,
                                 static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHead)))};
        device_.run([&](VkCommandBuffer cb) { kernels_.record_attention(cb, args); });
      }
    }
    return;
  }
  // Row-local ops split along the tokens, in multiples of the GEMM tiles' 128 rows.
  std::uint32_t chunk = b.tokens;
  if (limit > 0 && kFlopPerToken[op] * b.tokens > limit) {
    chunk = std::max<std::uint32_t>(128, static_cast<std::uint32_t>(limit / kFlopPerToken[op]) / 128 * 128);
  }
  for (std::uint32_t t0 = 0; t0 < b.tokens; t0 += chunk) {
    BlockBuffers part = b;
    const VkDeviceAddress t = t0;
    part.x += t * kDim * kBf16;
    part.y += t * kDim * kBf16;
    part.qkv += t * kQkv * kBf16;
    part.attn += t * kDim * kBf16;
    part.ffn += t * kFfn * kBf16;
    part.rope += t * kHead * kBf16;
    part.tokens = std::min(chunk, b.tokens - t0);
    device_.run([&](VkCommandBuffer cb) { record_block_op(kernels_, cb, op, w, part); });
  }
}

std::uint64_t Dit::activation_bytes(std::uint64_t tokens, std::uint64_t hw) {
  // forward(): x, y and attn [tokens, 3072]; qkv [tokens, 9216]; ffn [tokens, 8192]; rope [tokens, 128]; the video tokens
  // and the head's projection [hw, 128]; the modulation and time scratch. encode_text() reuses them for a prompt no longer
  // than the joint sequence, with its own up, gate [S, 8192] and embeddings [S, 4096].
  const std::uint64_t per_token = (3 * kDim + kQkv + kFfn + kHead) * kBf16;
  return tokens * per_token + 2 * hw * kDiTLatentChannels * kBf16 + 2 * tokens * kFfn * kBf16 +
         tokens * kDiTTextDim * kBf16 + (8 * kDim + 2 * kDim + kDiTFreqDim) * kBf16;
}

Buffer& Dit::scratch(std::unique_ptr<Buffer>& slot, VkDeviceSize bytes) {
  if (!slot || slot->bytes() < bytes) {
    slot.reset();
    slot = std::make_unique<Buffer>(device_, bytes);
  }
  return *slot;
}

std::unique_ptr<Buffer> Dit::encode_text(const std::vector<std::uint16_t>& embeds, std::uint32_t length) {
  if (length == 0 || embeds.size() != std::size_t{length} * kDiTTextDim) fail("encode_text: embeds must be [length, 4096]");
  const VkDeviceSize s = length;
  Buffer& e = scratch(embeds_, s * kDiTTextDim * kBf16);
  e.upload(embeds.data(), embeds.size() * kBf16);
  Buffer& y = scratch(y_, s * kDim * kBf16);
  Buffer& qkv = scratch(qkv_, s * kQkv * kBf16);
  Buffer& attn = scratch(attn_, s * kDim * kBf16);
  Buffer& up = scratch(up_, s * kFfn * kBf16);
  Buffer& gate = scratch(gate_, s * kFfn * kBf16);
  Buffer& ffn = scratch(ffn_, s * kFfn * kBf16);
  Buffer& rope = scratch(rope_, s * kHead * kBf16);
  const std::vector<std::uint16_t> table = rope_table(0, 0, length);
  rope.upload(table.data(), table.size() * kBf16);
  auto out = std::make_unique<Buffer>(device_, s * kDim * kBf16);
  const VkDeviceAddress x = out->address();
  const VkDeviceAddress ya = y.address(), qa = qkv.address(), aa = attn.address(), ua = up.address(), ga = gate.address(),
                        fa = ffn.address(), ea = e.address();
  // Each linear in token ranges of at most max_gflop_ (see run_block_op); norms and elementwise ops run whole.
  const auto rows = [&](double flop_per_token, const std::function<void(VkCommandBuffer, VkDeviceAddress, std::uint32_t)>& op) {
    const double limit = max_gflop_ * 1e9;
    std::uint32_t chunk = length;
    if (limit > 0 && flop_per_token * length > limit) {
      chunk = std::max<std::uint32_t>(128, static_cast<std::uint32_t>(limit / flop_per_token) / 128 * 128);
    }
    for (std::uint32_t t0 = 0; t0 < length; t0 += chunk) {
      device_.run([&](VkCommandBuffer cb) { op(cb, t0, std::min(chunk, length - t0)); });
    }
  };
  const auto off = [](VkDeviceAddress base, std::uint64_t t, std::uint64_t width) { return base + t * width * kBf16; };

  rows(2.0 * kDiTTextDim * kDim, [&](VkCommandBuffer cb, VkDeviceAddress t, std::uint32_t n) {
    kernels_.record_gemm_bias(cb, off(x, t, kDim), kDim, off(ea, t, kDiTTextDim), kDiTTextDim, g_.text_w, kDiTTextDim, g_.text_b,
                              n, kDim, kDiTTextDim);
  });
  for (const TextWeights& b : text_blocks_) {
    device_.run([&](VkCommandBuffer cb) { kernels_.record_norm(cb, x, kDim, b.norm1, ya, kDim, length, kDim); });
    rows(2.0 * kDim * kQkv, [&](VkCommandBuffer cb, VkDeviceAddress t, std::uint32_t n) {
      kernels_.record_gemm_bias(cb, off(qa, t, kQkv), kQkv, off(ya, t, kDim), kDim, b.qkv_w, kDim, b.qkv_b, n, kQkv, kDim);
    });
    device_.run([&](VkCommandBuffer cb) {
      kernels_.record_qk_norm_rope(cb, qa, kQkv, length, kDiTHeads, b.q_norm, b.k_norm, rope.address(), kHead);
    });
    run_block_op(3, BlockWeights{}, BlockBuffers{0, 0, qa, aa, 0, 0, 0, {0, static_cast<std::int32_t>(length)}, length});
    rows(2.0 * kDim * kDim, [&](VkCommandBuffer cb, VkDeviceAddress t, std::uint32_t n) {
      kernels_.record_gemm_bias_residual(cb, off(x, t, kDim), kDim, off(aa, t, kDim), kDim, b.proj_w, kDim, b.proj_b, n, kDim,
                                         kDim);
    });
    device_.run([&](VkCommandBuffer cb) { kernels_.record_norm(cb, x, kDim, b.norm2, ya, kDim, length, kDim); });
    rows(4.0 * kDim * kFfn, [&](VkCommandBuffer cb, VkDeviceAddress t, std::uint32_t n) {
      kernels_.record_gemm_bias(cb, off(ua, t, kFfn), kFfn, off(ya, t, kDim), kDim, b.up_w, kDim, b.up_b, n, kFfn, kDim);
      kernels_.record_gemm_bias(cb, off(ga, t, kFfn), kFfn, off(ya, t, kDim), kDim, b.gate_w, kDim, b.gate_b, n, kFfn, kDim);
    });
    device_.run([&](VkCommandBuffer cb) { kernels_.record_situ(cb, ua, ga, fa, length * kFfn); });
    rows(2.0 * kFfn * kDim, [&](VkCommandBuffer cb, VkDeviceAddress t, std::uint32_t n) {
      kernels_.record_gemm_bias_residual(cb, off(x, t, kDim), kDim, off(fa, t, kFfn), kFfn, b.down_w, kFfn, b.down_b, n, kDim,
                                         kFfn);
    });
  }
  return out;
}

void Dit::time_modulation(float timestep) {
  // P:232-241: a float64 sinusoid rounded to fp32 then bf16 (the .float() and back are exact), then the bf16 MLP.
  constexpr int kHalf = kDiTFreqDim / 2;
  std::vector<std::uint16_t> emb(kDiTFreqDim);
  for (int i = 0; i < kHalf; ++i) {
    const double freq = std::pow(10000.0, -static_cast<double>(i) / kHalf);
    const double phase = static_cast<double>(timestep) * freq;
    emb[static_cast<std::size_t>(i)] = to_bf16(static_cast<float>(std::cos(phase)));
    emb[static_cast<std::size_t>(kHalf + i)] = to_bf16(static_cast<float>(std::sin(phase)));
  }
  Buffer& t_in = scratch(t_in_, kDiTFreqDim * kBf16);
  t_in.upload(emb.data(), emb.size() * kBf16);
  Buffer& hidden = scratch(hidden_, kDim * kBf16);
  Buffer& time = scratch(time_, kDim * kBf16);
  Buffer& mod = scratch(mod_, 6 * kDim * kBf16);
  Buffer& head = scratch(head_, 2 * kDim * kBf16);
  device_.run([&](VkCommandBuffer cb) {
    kernels_.record_gemm_bias(cb, hidden.address(), kDim, t_in.address(), kDiTFreqDim, g_.time0_w, kDiTFreqDim, g_.time0_b,
                              1, kDim, kDiTFreqDim);
    kernels_.record_silu(cb, hidden.address(), hidden.address(), kDim);
    kernels_.record_gemm_bias(cb, time.address(), kDim, hidden.address(), kDim, g_.time2_w, kDim, g_.time2_b, 1, kDim, kDim);
    kernels_.record_gemm_bias(cb, mod.address(), 6 * kDim, time.address(), kDim, g_.mod_w, kDim, g_.mod_b, 1, 6 * kDim, kDim);
    kernels_.record_gemm_bias(cb, head.address(), 2 * kDim, time.address(), kDim, g_.head_w, kDim, g_.head_b, 1, 2 * kDim,
                              kDim);
  });
}

std::vector<std::vector<float>> Dit::forward(const std::vector<float>& latents, std::uint32_t height, std::uint32_t width,
                                             float timestep, const std::vector<Row>& rows) {
  const std::uint64_t hw = std::uint64_t{height} * width;
  if (rows.empty() || hw == 0 || latents.size() != hw * kDiTLatentChannels) fail("forward: latents must be [128, h, w]");
  std::vector<std::uint64_t> off{0};
  for (const Row& r : rows) {
    if (r.text == nullptr || r.length == 0 || r.text->bytes() < std::uint64_t{r.length} * kDim * kBf16) {
      fail("forward: a row has no refined text");
    }
    off.push_back(off.back() + hw + r.length);
  }
  const std::uint64_t total = off.back();
  op_ms_.assign(kBlockOps + 1, 0.0);
  const double t_start = now_ms();
  double copy_ms = 0.0;

  Buffer& x = scratch(x_, total * kDim * kBf16);
  scratch(y_, total * kDim * kBf16);
  scratch(qkv_, total * kQkv * kBf16);
  scratch(attn_, total * kDim * kBf16);
  scratch(ffn_, total * kFfn * kBf16);
  Buffer& rope = scratch(rope_, total * kHead * kBf16);
  Buffer& tokens = scratch(tokens_, hw * kDiTLatentChannels * kBf16);
  Buffer& proj = scratch(proj_, hw * kDiTLatentChannels * kBf16);

  time_modulation(timestep);

  // The video tokens [hw, 128] in bf16 (the reference's .to(bf16) of the transposed latents), the same for every row.
  std::vector<std::uint16_t> host_tokens(hw * kDiTLatentChannels);
  for (std::uint64_t p = 0; p < hw; ++p) {
    for (std::uint32_t c = 0; c < kDiTLatentChannels; ++c) host_tokens[p * kDiTLatentChannels + c] = to_bf16(latents[c * hw + p]);
  }
  tokens.upload(host_tokens.data(), host_tokens.size() * kBf16);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const std::vector<std::uint16_t> table = rope_table(height, width, rows[r].length);
    rope.upload(table.data(), table.size() * kBf16, off[r] * kHead * kBf16);
  }
  device_.run([&](VkCommandBuffer cb) {
    for (std::size_t r = 0; r < rows.size(); ++r) {
      kernels_.record_gemm_bias(cb, at(x, off[r] * kDim * kBf16), kDim, tokens.address(), kDiTLatentChannels, g_.video_w,
                                kDiTLatentChannels, g_.video_b, static_cast<std::uint32_t>(hw), kDim, kDiTLatentChannels);
      record_copy(cb, *rows[r].text, 0, x, (off[r] + hw) * kDim * kBf16, std::uint64_t{rows[r].length} * kDim * kBf16);
    }
  });

  for (int l = 0; l < kDiTLayers; ++l) {
    // Each maximal run of consecutive rows that executes block l is one varlen batch over its contiguous tokens.
    for (std::size_t r0 = 0; r0 < rows.size();) {
      if (rows[r0].skip.test(static_cast<std::size_t>(l))) {
        ++r0;
        continue;
      }
      std::size_t r1 = r0;
      std::vector<std::int32_t> cu{0};
      while (r1 < rows.size() && !rows[r1].skip.test(static_cast<std::size_t>(l))) {
        cu.push_back(static_cast<std::int32_t>(off[r1 + 1] - off[r0]));
        ++r1;
      }
      const BlockBuffers b{at(x, off[r0] * kDim * kBf16), y_->address(), qkv_->address(), attn_->address(), ffn_->address(),
                           at(rope, off[r0] * kHead * kBf16), mod_->address(), cu,
                           static_cast<std::uint32_t>(off[r1] - off[r0])};
      if (const Buffer* host = host_blocks_[static_cast<std::size_t>(l)]; host != nullptr && streamed_layer_ != l) {
        const double t0 = now_ms();
        device_.run([&](VkCommandBuffer cb) { record_copy(cb, *host, 0, *slot_, 0, host->bytes()); });
        streamed_layer_ = l;
        copy_ms += now_ms() - t0;
      }
      for (int op = 0; op < kBlockOps; ++op) {
        const double t0 = now_ms();
        run_block_op(op, blocks_[static_cast<std::size_t>(l)], b);
        op_ms_[static_cast<std::size_t>(op)] += now_ms() - t0;
      }
      r0 = r1;
    }
  }

  // Head (P:256-259): final norm with the head modulation (shift = head[0], scale = head[1]), linear, transpose.
  std::vector<std::vector<float>> out(rows.size(), std::vector<float>(hw * kDiTLatentChannels));
  std::vector<std::uint16_t> host_proj(hw * kDiTLatentChannels);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    device_.run([&](VkCommandBuffer cb) {
      kernels_.record_norm_modulate(cb, at(x, off[r] * kDim * kBf16), kDim, g_.final_norm, at(*head_, kDim * kBf16),
                                    head_->address(), y_->address(), kDim, static_cast<std::uint32_t>(hw), kDim);
      kernels_.record_gemm(cb, proj.address(), kDiTLatentChannels, y_->address(), kDim, g_.final_w, kDim,
                           static_cast<std::uint32_t>(hw), kDiTLatentChannels, kDim);
    });
    proj.download(host_proj.data(), host_proj.size() * kBf16);
    for (std::uint64_t p = 0; p < hw; ++p) {
      for (std::uint32_t c = 0; c < kDiTLatentChannels; ++c) out[r][c * hw + p] = from_bf16(host_proj[p * kDiTLatentChannels + c]);
    }
  }
  double blocks = 0.0;
  for (int op = 0; op < kBlockOps; ++op) blocks += op_ms_[static_cast<std::size_t>(op)];
  op_ms_[kBlockOps] = now_ms() - t_start - blocks;
  last_copy_ms_ = copy_ms;
  return out;
}

}  // namespace mi::vk
