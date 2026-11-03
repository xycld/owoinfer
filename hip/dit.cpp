#include "dit.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "dit_kernels.h"
#include "safetensors.h"

namespace mi::hip {
namespace {

using bf16 = std::uint16_t;
constexpr std::int64_t kDim = kDiTDim, kFfn = kDiTFfnDim, kHead = kDiTHeadDim, kQkv = 3 * kDiTDim;
constexpr std::size_t kBf16 = 2;
constexpr std::size_t kAlign = 256;

// The RoPE band frequencies 1 / 10000^(i / 16), as the reference builds them in fp32 on the CPU (kernels/basic/rope.h).
constexpr std::uint32_t kBandBits[16] = {0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887, 0x3dcccccd, 0x3d6655c2,
                                         0x3d0186e3, 0x3c91ad39, 0x3c23d70a, 0x3bb8449c, 0x3b4f3e38, 0x3ae91528,
                                         0x3a83126f, 0x3a136a16, 0x39a5cb60, 0x393a7753};

bf16 to_bf16(float v) {
  const auto u = std::bit_cast<std::uint32_t>(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<bf16>((u >> 16) | 0x0040u);
  return static_cast<bf16>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float from_bf16(bf16 b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("hip dit: " + what); }

std::size_t aligned(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

// A [rows, cols] row-major bf16 matrix into hip_block::pack_fragments' layout: 16 x 16 fragments contiguous, row groups
// of 16 rows, column groups of 16 columns. rows and cols multiples of 16.
void pack_host(bf16* to, const bf16* from, std::int64_t rows, std::int64_t cols) {
  const std::int64_t groups = cols / 16;
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t g = 0; g < groups; ++g) std::memcpy(to + ((r / 16 * groups + g) * 256 + r % 16 * 16), from + r * cols + g * 16, 32);
  }
}

bf16* at(const DeviceBuffer& b, std::size_t byte_offset) {
  return reinterpret_cast<bf16*>(static_cast<char*>(b.data()) + byte_offset);
}

// One block's tensors: the norms, then the GEMM weights packed, q, k and v back to back as the [9216, 3072] QKV weight.
struct BlockLayout {
  std::size_t norm1, norm2, q_norm, k_norm, qkv, o, up_gate, down, bytes;
};
constexpr BlockLayout block_layout() {
  BlockLayout l{};
  std::size_t o = 0;
  const auto place = [&](std::size_t bytes) {
    const std::size_t at_ = o;
    o += (bytes + kAlign - 1) / kAlign * kAlign;
    return at_;
  };
  l.norm1 = place(kDim * kBf16);
  l.norm2 = place(kDim * kBf16);
  l.q_norm = place(kHead * kBf16);
  l.k_norm = place(kHead * kBf16);
  l.qkv = place(kQkv * kDim * kBf16);
  l.o = place(kDim * kDim * kBf16);
  l.up_gate = place(2 * kFfn * kDim * kBf16);
  l.down = place(kDim * kFfn * kBf16);
  l.bytes = o;
  return l;
}
constexpr BlockLayout kBlock = block_layout();

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
      // sinf and cosf to fp32 as torch's CUDA kernel; the host's are correctly rounded or within an ulp, and bf16 hides that.
      out[r * kHead + p] = to_bf16(std::sin(phase));
      out[r * kHead + 64 + p] = to_bf16(std::cos(phase));
    }
  }
  return out;
}

Dit::Dit(Device& device, const std::filesystem::path& transformer_dir, const DitOptions& options)
    : device_(device), options_(options) {
  device_.make_current();
  const st::Index index(transformer_dir);
  const auto check_tensor = [&](const std::string& name, const std::vector<std::int64_t>& shape) {
    const st::TensorInfo& t = index.at(name);
    if (t.dtype != "BF16") fail(name + " is " + t.dtype + ", expected BF16");
    if (t.shape != shape) fail(name + " has an unexpected shape");
    return t.bytes;
  };
  // Tensors into one new device buffer, each at a 256-byte aligned offset; returns their device pointers.
  std::vector<std::uint8_t> host;
  const auto load = [&](const std::vector<std::string>& names, const std::vector<std::vector<std::int64_t>>& shapes) {
    std::vector<std::size_t> offsets;
    std::size_t total = 0;
    for (std::size_t i = 0; i < names.size(); ++i) {
      offsets.push_back(total);
      total += aligned(check_tensor(names[i], shapes[i]));
    }
    owned_.emplace_back(total);
    DeviceBuffer& buf = owned_.back();
    for (std::size_t i = 0; i < names.size(); ++i) {
      host.resize(index.at(names[i]).bytes);
      index.read_into(names[i], host.data(), host.size());
      buf.upload(host.data(), host.size(), offsets[i]);
    }
    weight_bytes_ += total;
    std::vector<const bf16*> out;
    for (const std::size_t o : offsets) out.push_back(at(buf, o));
    return out;
  };
  const std::int64_t d = kDim, f = kFfn;

  const auto a = load({"video_embedder.weight", "video_embedder.bias", "time_embedding.0.weight", "time_embedding.0.bias",
                       "time_embedding.2.weight", "time_embedding.2.bias", "shared_block_modulation.weight",
                       "shared_block_modulation.bias", "video_head_modulation.weight", "video_head_modulation.bias",
                       "final_norm_video.weight", "final_linear_video.weight", "text_embedder.proj.weight",
                       "text_embedder.proj.bias"},
                      {{d, kDiTLatentChannels}, {d}, {d, kDiTFreqDim}, {d}, {d, d}, {d}, {6 * d, d}, {6 * d}, {2 * d, d}, {2 * d},
                       {d}, {kDiTLatentChannels, d}, {d, kDiTTextDim}, {d}});
  g_ = {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]};

  // The refiner's ffn_up and ffn_gate (and their biases) back to back form the [hidden | gate] weight of one GEMM, as the
  // engine fuses them.
  for (int b = 0; b < 2; ++b) {
    const std::string p = "text_embedder.blocks." + std::to_string(b) + ".";
    const auto t = load({p + "norm1.weight", p + "norm2.weight", p + "q_norm.weight", p + "k_norm.weight", p + "qkv.weight",
                         p + "qkv.bias", p + "proj.weight", p + "proj.bias", p + "ffn_up.weight", p + "ffn_gate.weight",
                         p + "ffn_up.bias", p + "ffn_gate.bias", p + "ffn_down.weight", p + "ffn_down.bias"},
                        {{d}, {d}, {kHead}, {kHead}, {3 * d, d}, {3 * d}, {d, d}, {d}, {f, d}, {f, d}, {f}, {f}, {d, f}, {d}});
    if (t[9] != t[8] + f * d || t[11] != t[10] + f) fail("the refiner's ffn_up and ffn_gate are not contiguous");
    text_blocks_.push_back({t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[10], t[12], t[13]});
  }

  // The blocks: the first `resident` on the device, the rest in pinned host memory, all packed on the host.
  const auto block_names = [](int l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    return std::vector<std::string>{p + "norm1.weight", p + "norm2.weight", p + "q_norm.weight", p + "k_norm.weight",
                                    p + "proj.q.weight", p + "proj.k.weight", p + "proj.v.weight", p + "proj.o.weight",
                                    p + "proj.ffn_up_gate.weight", p + "proj.ffn_down.weight"};
  };
  const std::vector<std::vector<std::int64_t>> block_shapes = {{d}, {d}, {kHead}, {kHead}, {d, d}, {d, d}, {d, d}, {d, d},
                                                               {2 * f, d}, {d, f}};
  int resident = kDiTLayers;
  if (options.device_budget > 0) {
    const std::uint64_t free = options.device_budget > weight_bytes_ ? options.device_budget - weight_bytes_ : 0;
    // Streamed blocks need two slots: the next one copies while the current one runs.
    resident = free >= kDiTLayers * kBlock.bytes ? kDiTLayers
                                                 : static_cast<int>(free > 2 * kBlock.bytes ? free / kBlock.bytes - 2 : 0);
  }
  HostBuffer staging(resident > 0 ? kBlock.bytes : 0);
  std::vector<bf16> row_major;
  for (int l = 0; l < kDiTLayers; ++l) {
    const std::vector<std::string> names = block_names(l);
    for (std::size_t i = 0; i < names.size(); ++i) check_tensor(names[i], block_shapes[i]);
    const bool on_device = l < resident;
    HostBuffer streamed;
    if (!on_device) streamed = HostBuffer(kBlock.bytes);
    char* dst = static_cast<char*>(on_device ? staging.data() : streamed.data());
    std::memset(dst, 0, kBlock.bytes);
    const std::size_t plain[4] = {kBlock.norm1, kBlock.norm2, kBlock.q_norm, kBlock.k_norm};
    for (std::size_t i = 0; i < 4; ++i) index.read_into(names[i], dst + plain[i], index.at(names[i]).bytes);
    // q, k and v are packed one after the other: each holds whole row groups, so that is the [9216, 3072] weight's packing.
    const std::size_t packed_at[6] = {kBlock.qkv, kBlock.qkv + d * d * kBf16, kBlock.qkv + 2 * d * d * kBf16, kBlock.o,
                                      kBlock.up_gate, kBlock.down};
    for (std::size_t i = 4; i < names.size(); ++i) {
      const std::vector<std::int64_t>& s = block_shapes[i];
      row_major.resize(static_cast<std::size_t>(s[0] * s[1]));
      index.read_into(names[i], row_major.data(), row_major.size() * kBf16);
      pack_host(reinterpret_cast<bf16*>(dst + packed_at[i - 4]), row_major.data(), s[0], s[1]);
    }
    const bf16* base = nullptr;
    if (on_device) {
      owned_.emplace_back(kBlock.bytes);
      owned_.back().upload(staging.data(), kBlock.bytes);
      weight_bytes_ += kBlock.bytes;
      base = at(owned_.back(), 0);
      slot_of_.push_back(-1);
    } else {
      const int s = (l - resident) % 2;
      if (slots_[s].bytes() == 0) {
        slots_[s] = DeviceBuffer(kBlock.bytes);
        slot_free_[s] = std::make_unique<Event>();
        copy_done_[s] = std::make_unique<Event>();
        weight_bytes_ += kBlock.bytes;
      }
      base = at(slots_[s], 0);
      slot_of_.push_back(s);
      host_weight_bytes_ += kBlock.bytes;
    }
    host_blocks_.push_back(std::move(streamed));
    const auto p = [&](std::size_t o) { return reinterpret_cast<const bf16*>(reinterpret_cast<const char*>(base) + o); };
    hip_block::BlockWeights w{p(kBlock.norm1), p(kBlock.norm2), p(kBlock.q_norm), p(kBlock.k_norm), nullptr, nullptr, nullptr,
                              nullptr};
    w.qkv_packed = p(kBlock.qkv);
    w.o_packed = p(kBlock.o);
    w.up_gate_packed = p(kBlock.up_gate);
    w.down_packed = p(kBlock.down);
    blocks_.push_back(w);
  }
  resident_blocks_ = resident;
}

Dit::~Dit() {
  // Copies in flight read host buffers this destructor frees.
  try {
    copy_stream_.synchronize();
    stream_.synchronize();
  } catch (const std::exception&) {
  }
}

std::uint64_t Dit::activation_bytes(std::uint64_t tokens, std::uint64_t rows, std::uint64_t hw) {
  // x, y, attn [T, 3072]; qkv [T, 9216]; ffn [T, 8192]; rope [T, 128]; the packed GEMM input [T rounded up to 96, 8192]
  // and the packed attention input [T + 32 rows, 9216]; the video tokens [hw, 128] and each row's projection; the time
  // and modulation vectors.
  const std::uint64_t pack_rows = (tokens + hip_block::kPackRowsMultiple - 1) / hip_block::kPackRowsMultiple * hip_block::kPackRowsMultiple;
  return (tokens * (3 * kDim + kQkv + kFfn + kHead) + pack_rows * kFfn + (tokens + 32 * rows) * kQkv +
          hw * kDiTLatentChannels * (1 + rows) + 10 * kDim + kDiTFreqDim) *
         kBf16;
}

DeviceBuffer& Dit::scratch(DeviceBuffer& b, std::size_t bytes) {
  if (b.bytes() < bytes) {
    b = DeviceBuffer();
    b = DeviceBuffer(bytes);
  }
  return b;
}

void Dit::mark(int what) {
  if (!options_.time_ops) return;
  if (events_used_ == events_.size()) events_.push_back(std::make_unique<Event>(true));
  event_what_.resize(events_.size());
  event_what_[events_used_] = what;
  events_[events_used_++]->record(stream_.get());
}

DeviceBuffer Dit::encode_text(const std::vector<std::uint16_t>& embeds, std::uint32_t length) {
  if (length == 0 || embeds.size() != std::size_t{length} * kDiTTextDim) fail("encode_text: embeds must be [length, 4096]");
  device_.make_current();
  stream_.synchronize();  // the scratch below may still be in use by earlier work
  const std::size_t s = length;
  const hipStream_t st = stream_.get();
  DeviceBuffer& e = scratch(embeds_, s * kDiTTextDim * kBf16);
  e.upload(embeds.data(), embeds.size() * kBf16);
  DeviceBuffer& y = scratch(y_, s * kDim * kBf16);
  DeviceBuffer& qkv = scratch(qkv_, s * kQkv * kBf16);
  DeviceBuffer& attn = scratch(attn_, s * kDim * kBf16);
  DeviceBuffer& up_gate = scratch(up_gate_, s * 2 * kFfn * kBf16);
  DeviceBuffer& ffn = scratch(ffn_, s * kFfn * kBf16);
  DeviceBuffer& rope = scratch(rope_, s * kHead * kBf16);
  const std::vector<std::uint16_t> table = rope_table(0, 0, length);
  rope.upload(table.data(), table.size() * kBf16);
  DeviceBuffer& cu = scratch(cu_, 2 * sizeof(std::int32_t));
  const std::int32_t cu_host[2] = {0, static_cast<std::int32_t>(length)};
  cu.upload(cu_host, sizeof(cu_host));
  DeviceBuffer out(s * kDim * kBf16);
  const int m = static_cast<int>(length);
  bf16* x = out.as<bf16>();

  hip_block::gemm_bias(x, kDim, e.as<bf16>(), kDiTTextDim, g_.text_w, kDiTTextDim, g_.text_b, m, kDim, kDiTTextDim, st);  // P:120
  for (const TextBlock& b : text_blocks_) {  // _TextBlock.forward, P:104-110
    hip_block::rms_norm(x, kDim, b.norm1, y.as<bf16>(), kDim, m, kDim, st);
    hip_block::gemm_bias(qkv.as<bf16>(), kQkv, y.as<bf16>(), kDim, b.qkv_w, kDim, b.qkv_b, m, kQkv, kDim, st);
    hip_block::qk_norm_rope(qkv.as<bf16>(), kQkv, m, kDiTHeads, b.q_norm, b.k_norm, rope.as<bf16>(), kHead, st);
    hip_block::attention({qkv.as<bf16>(), qkv.as<bf16>() + kDim, qkv.as<bf16>() + 2 * kDim, attn.as<bf16>(), kQkv, kQkv, kQkv,
                          kDim, kHead, kHead, kHead, kHead, cu.as<std::int32_t>(), cu.as<std::int32_t>(), 1, m, kDiTHeads,
                          static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHead)))},
                         st);
    hip_block::gemm_bias_residual(x, kDim, attn.as<bf16>(), kDim, b.proj_w, kDim, b.proj_b, m, kDim, kDim, st);
    hip_block::rms_norm(x, kDim, b.norm2, y.as<bf16>(), kDim, m, kDim, st);
    hip_block::gemm_bias(up_gate.as<bf16>(), 2 * kFfn, y.as<bf16>(), kDim, b.up_gate_w, kDim, b.up_gate_b, m, 2 * kFfn, kDim, st);
    hip_block::situ_glu(up_gate.as<bf16>(), 2 * kFfn, ffn.as<bf16>(), kFfn, m, kFfn, st);
    hip_block::gemm_bias_residual(x, kDim, ffn.as<bf16>(), kFfn, b.down_w, kFfn, b.down_b, m, kDim, kFfn, st);
  }
  stream_.synchronize();
  return out;
}

void Dit::time_modulation(float timestep) {
  // P:232-241: a float64 sinusoid rounded to fp32 then bf16 (the .float() and back are exact), then the bf16 MLP.
  constexpr int kHalf = kDiTFreqDim / 2;
  std::vector<bf16> emb(kDiTFreqDim);
  for (int i = 0; i < kHalf; ++i) {
    const double freq = std::pow(10000.0, -static_cast<double>(i) / kHalf);
    const double phase = static_cast<double>(timestep) * freq;
    emb[static_cast<std::size_t>(i)] = to_bf16(static_cast<float>(std::cos(phase)));
    emb[static_cast<std::size_t>(kHalf + i)] = to_bf16(static_cast<float>(std::sin(phase)));
  }
  DeviceBuffer& t_in = scratch(t_in_, kDiTFreqDim * kBf16);
  t_in.upload(emb.data(), emb.size() * kBf16);
  DeviceBuffer& hidden = scratch(hidden_, kDim * kBf16);
  DeviceBuffer& time = scratch(time_, kDim * kBf16);
  DeviceBuffer& mod = scratch(mod_, 6 * kDim * kBf16);
  DeviceBuffer& head = scratch(head_, 2 * kDim * kBf16);
  const hipStream_t st = stream_.get();
  hip_block::gemm_bias(hidden.as<bf16>(), kDim, t_in.as<bf16>(), kDiTFreqDim, g_.time0_w, kDiTFreqDim, g_.time0_b, 1, kDim,
                       kDiTFreqDim, st);
  hip_block::silu(hidden.as<bf16>(), hidden.as<bf16>(), kDim, st);
  hip_block::gemm_bias(time.as<bf16>(), kDim, hidden.as<bf16>(), kDim, g_.time2_w, kDim, g_.time2_b, 1, kDim, kDim, st);
  hip_block::gemm_bias(mod.as<bf16>(), 6 * kDim, time.as<bf16>(), kDim, g_.mod_w, kDim, g_.mod_b, 1, 6 * kDim, kDim, st);
  hip_block::gemm_bias(head.as<bf16>(), 2 * kDim, time.as<bf16>(), kDim, g_.head_w, kDim, g_.head_b, 1, 2 * kDim, kDim, st);
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
  device_.make_current();
  stream_.synchronize();  // uploads below go to buffers earlier work may still read
  const hipStream_t st = stream_.get();
  events_used_ = 0;
  op_ms_.clear();

  DeviceBuffer& x = scratch(x_, total * kDim * kBf16);
  scratch(y_, total * kDim * kBf16);
  scratch(qkv_, total * kQkv * kBf16);
  scratch(attn_, total * kDim * kBf16);
  scratch(ffn_, total * kFfn * kBf16);
  const std::uint64_t pack_rows =
      (total + hip_block::kPackRowsMultiple - 1) / hip_block::kPackRowsMultiple * hip_block::kPackRowsMultiple;
  scratch(pack_, pack_rows * kFfn * kBf16);
  scratch(pack_attn_, (total + 32 * rows.size()) * kQkv * kBf16);
  DeviceBuffer& rope = scratch(rope_, total * kHead * kBf16);
  DeviceBuffer& tokens = scratch(tokens_, hw * kDiTLatentChannels * kBf16);
  DeviceBuffer& proj = scratch(proj_, rows.size() * hw * kDiTLatentChannels * kBf16);

  // The video tokens [hw, 128] in bf16 (the reference's .to(bf16) of the transposed latents), the same for every row.
  std::vector<bf16> host_tokens(hw * kDiTLatentChannels);
  for (std::uint64_t p = 0; p < hw; ++p) {
    for (std::uint32_t c = 0; c < kDiTLatentChannels; ++c) host_tokens[p * kDiTLatentChannels + c] = to_bf16(latents[c * hw + p]);
  }
  tokens.upload(host_tokens.data(), host_tokens.size() * kBf16);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const std::vector<bf16> table = rope_table(height, width, rows[r].length);
    rope.upload(table.data(), table.size() * kBf16, off[r] * kHead * kBf16);
  }

  // Each block's runs: the maximal runs of consecutive rows that execute it, each one varlen batch over its contiguous
  // tokens. Their sequence offsets go to the device once.
  struct Run {
    std::size_t r0, r1, cu_at;
  };
  std::vector<std::vector<Run>> runs(kDiTLayers);
  std::vector<std::int32_t> cu_host;
  for (int l = 0; l < kDiTLayers; ++l) {
    for (std::size_t r0 = 0; r0 < rows.size();) {
      if (rows[r0].skip.test(static_cast<std::size_t>(l))) {
        ++r0;
        continue;
      }
      std::size_t r1 = r0;
      while (r1 < rows.size() && !rows[r1].skip.test(static_cast<std::size_t>(l))) ++r1;
      const std::size_t at_ = cu_host.size();
      for (std::size_t r = r0; r <= r1; ++r) cu_host.push_back(static_cast<std::int32_t>(off[r] - off[r0]));
      runs[static_cast<std::size_t>(l)].push_back({r0, r1, at_});
      r0 = r1;
    }
  }
  DeviceBuffer& cu = scratch(cu_, cu_host.size() * sizeof(std::int32_t));
  cu.upload(cu_host.data(), cu_host.size() * sizeof(std::int32_t));

  mark(-1);
  time_modulation(timestep);
  hip_block::gemm_bias(at(x, 0), kDim, tokens.as<bf16>(), kDiTLatentChannels, g_.video_w, kDiTLatentChannels, g_.video_b,
                       static_cast<int>(hw), kDim, kDiTLatentChannels, st);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    if (r > 0) copy_async(at(x, off[r] * kDim * kBf16), at(x, 0), hw * kDim * kBf16, st);
    copy_async(at(x, (off[r] + hw) * kDim * kBf16), rows[r].text->data(), std::uint64_t{rows[r].length} * kDim * kBf16, st);
  }
  mark(-1);

  for (int l = 0; l < kDiTLayers; ++l) {
    const int s = slot_of_[static_cast<std::size_t>(l)];
    if (s >= 0) {
      // The copy waits until the slot's previous block has run; this block's ops wait for the copy. The host queues
      // everything ahead, so the next streamed block's copy (the other slot) runs while this block computes.
      slot_free_[s]->wait(copy_stream_.get());
      const HostBuffer& src = host_blocks_[static_cast<std::size_t>(l)];
      copy_async(slots_[s].data(), src.data(), src.bytes(), copy_stream_.get());
      copy_done_[s]->record(copy_stream_.get());
      copy_done_[s]->wait(st);
      mark(hip_block::kBlockOps);
    }
    for (const Run& run : runs[static_cast<std::size_t>(l)]) {
      hip_block::BlockBuffers b{};
      b.x = at(x, off[run.r0] * kDim * kBf16);
      b.y = y_.as<bf16>();
      b.qkv = qkv_.as<bf16>();
      b.attn = attn_.as<bf16>();
      b.ffn = ffn_.as<bf16>();
      b.rope = at(rope, off[run.r0] * kHead * kBf16);
      b.mod = mod_.as<bf16>();
      b.cu_seqlens = cu.as<std::int32_t>() + run.cu_at;
      b.batch = static_cast<int>(run.r1 - run.r0);
      b.max_seqlen = 0;
      for (std::size_t r = run.r0; r < run.r1; ++r) b.max_seqlen = std::max(b.max_seqlen, static_cast<int>(off[r + 1] - off[r]));
      b.tokens = static_cast<int>(off[run.r1] - off[run.r0]);
      b.pack = pack_.as<bf16>();
      b.pack_attn = pack_attn_.as<bf16>();
      for (int op = 0; op < hip_block::kBlockOps; ++op) {
        hip_block::run_block_op(op, blocks_[static_cast<std::size_t>(l)], b, st);
        mark(op);
      }
    }
    if (s >= 0) slot_free_[s]->record(st);
  }

  // Head (P:256-259): final norm with the head modulation (shift = head[0], scale = head[1]), linear, transpose.
  for (std::size_t r = 0; r < rows.size(); ++r) {
    hip_block::rms_norm_modulate(at(x, off[r] * kDim * kBf16), kDim, g_.final_norm, at(head_, kDim * kBf16), head_.as<bf16>(),
                                 y_.as<bf16>(), kDim, static_cast<std::int64_t>(hw), kDim, st);
    hip_block::gemm(at(proj, r * hw * kDiTLatentChannels * kBf16), kDiTLatentChannels, y_.as<bf16>(), kDim, g_.final_w, kDim,
                    static_cast<int>(hw), kDiTLatentChannels, kDim, st);
  }
  mark(-1);
  stream_.synchronize();
  std::vector<bf16> host_proj(rows.size() * hw * kDiTLatentChannels);
  proj.download(host_proj.data(), host_proj.size() * kBf16);
  std::vector<std::vector<float>> out(rows.size(), std::vector<float>(hw * kDiTLatentChannels));
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const bf16* pr = host_proj.data() + r * hw * kDiTLatentChannels;
    for (std::uint64_t p = 0; p < hw; ++p) {
      for (std::uint32_t c = 0; c < kDiTLatentChannels; ++c) out[r][c * hw + p] = from_bf16(pr[p * kDiTLatentChannels + c]);
    }
  }
  if (options_.time_ops) {
    // Per op kind, then the waits for streamed copies, then the rest (time embedding, embedders, head) as the last entry.
    op_ms_.assign(hip_block::kBlockOps + 2, 0.0);
    for (std::size_t i = 1; i < events_used_; ++i) {
      const int what = event_what_[i];
      const double ms = static_cast<double>(events_[i]->elapsed_ms(*events_[i - 1]));
      op_ms_[static_cast<std::size_t>(what < 0 ? hip_block::kBlockOps + 1 : what)] += ms;
    }
  }
  return out;
}

}  // namespace mi::hip
