#include "models/flux2_vae/vae_file.h"

#include <algorithm>
#include <set>
#include <string_view>

#include "util/path.h"
#include "weights/safetensors.h"

namespace mi::flux2_vae {
namespace {

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

// "<digits>." at the front of `rest`: parses it into n and advances.
bool take_index(std::string_view& rest, int& n) { return weights::take_layer_index(rest, n); }

[[noreturn]] void index_fail(const std::filesystem::path& file, const std::string& msg) {
  throw weights::IndexError(util::path_utf8(file) + ": " + msg);
}

}  // namespace

std::string vae_file_tensor_name(const std::string& key) {
  std::string_view k = key;
  if (starts_with(k, "encoder.") || starts_with(k, "quant_conv.") || k == "bn.num_batches_tracked") return "";
  if (starts_with(k, "decoder.post_quant_conv.")) return std::string(k.substr(8));
  if (!starts_with(k, "decoder.")) return key;  // post_quant_conv.*, bn.*
  std::string_view rest = k.substr(8);
  if (starts_with(rest, "mid.")) {
    rest.remove_prefix(4);
    if (starts_with(rest, "block_1.")) return "decoder.mid_block.resnets.0." + std::string(rest.substr(8));
    if (starts_with(rest, "block_2.")) return "decoder.mid_block.resnets.1." + std::string(rest.substr(8));
    if (starts_with(rest, "attn_1.")) {
      rest.remove_prefix(7);
      const std::string a = "decoder.mid_block.attentions.0.";
      for (const auto& [from, to] : {std::pair<std::string_view, std::string_view>{"norm.", "group_norm."}, {"q.", "to_q."},
                                     {"k.", "to_k."}, {"v.", "to_v."}, {"proj_out.", "to_out.0."}})
        if (starts_with(rest, from)) return a + std::string(to) + std::string(rest.substr(from.size()));
    }
    return key;  // unknown: rejected by index_vae_file
  }
  if (starts_with(rest, "up.")) {
    rest.remove_prefix(3);
    int i = 0, j = 0;
    if (!take_index(rest, i) || i > 3) return key;
    const std::string block = "decoder.up_blocks." + std::to_string(3 - i) + ".";
    if (starts_with(rest, "upsample.conv.")) return block + "upsamplers.0.conv." + std::string(rest.substr(14));
    if (starts_with(rest, "block.")) {
      rest.remove_prefix(6);
      if (!take_index(rest, j)) return key;
      std::string r(rest);
      if (starts_with(r, "nin_shortcut.")) r = "conv_shortcut." + r.substr(13);
      return block + "resnets." + std::to_string(j) + "." + r;
    }
    return key;
  }
  if (starts_with(rest, "norm_out.")) return "decoder.conv_norm_out." + std::string(rest.substr(9));
  return key;  // diffusers names and conv_in / conv_out (same in both layouts)
}

weights::ModelIndex index_vae_file(const std::filesystem::path& file) {
  const weights::SafetensorsFile st(file);
  // Optional common prefix.
  std::string prefix;
  for (const char* p : {"vae.", "first_stage_model."}) {
    const bool all = !st.tensors().empty() && std::all_of(st.tensors().begin(), st.tensors().end(),
                                                         [&](const weights::TensorInfo& t) { return starts_with(t.name, p); });
    if (all) prefix = p;
  }

  weights::ModelIndex index;
  index.bundle_dir = file.parent_path();
  // Not a cached index: the file's identity is only recorded for messages.
  index.files.push_back({util::path_utf8(file.filename()), st.file_size(), 0});
  weights::WeightGroup g;
  g.component = weights::Component::kVae;
  g.name = "decoder";
  weights::ExcludedSummary excluded;
  excluded.component = weights::Component::kVae;
  excluded.prefix = "encoder.";
  std::set<std::string> seen;
  for (const weights::TensorInfo& t : st.tensors()) {  // file-offset order, as upload_group reads best
    const std::string key = t.name.substr(prefix.size());
    const std::string name = vae_file_tensor_name(key);
    if (name.empty()) {
      ++excluded.tensors;
      excluded.bytes += t.nbytes;
      continue;
    }
    const bool known = starts_with(name, "decoder.") || starts_with(name, "post_quant_conv.") || starts_with(name, "bn.running_");
    if (!known || starts_with(name, "decoder.mid.") || starts_with(name, "decoder.up.") || starts_with(name, "decoder.norm_out."))
      index_fail(file, "tensor '" + t.name + "' is not part of a FLUX.2 VAE decoder");
    if (t.dtype != weights::DType::kF32) index_fail(file, "tensor '" + t.name + "' is " + weights::dtype_name(t.dtype) + ", expected F32");
    if (!seen.insert(name).second) index_fail(file, "two tensors map to '" + name + "'");
    weights::IndexedTensor it;
    it.name = name;
    it.dtype = t.dtype;
    it.shape = t.shape;
    // BFL attention projections are 1x1 convs: the same bytes as a [C, C] linear weight.
    if (it.shape.size() == 4 && it.shape[2] == 1 && it.shape[3] == 1 && starts_with(name, "decoder.mid_block.attentions."))
      it.shape.resize(2);
    it.file = 0;
    it.offset = t.offset;
    it.nbytes = t.nbytes;
    g.total_bytes += t.nbytes;
    g.total_params += t.numel();
    g.tensors.push_back(std::move(it));
  }
  for (const char* need : {"decoder.conv_in.weight", "decoder.conv_out.weight", "post_quant_conv.weight", "bn.running_mean",
                           "bn.running_var"})
    if (!seen.count(need)) index_fail(file, std::string("no FLUX.2 VAE decoder: '") + need + "' missing");
  index.groups.push_back(std::move(g));
  if (excluded.tensors) index.excluded.push_back(excluded);
  return index;
}

}  // namespace mi::flux2_vae
