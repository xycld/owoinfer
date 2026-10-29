#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <sstream>

#include "test_util.h"
#include "util/path.h"
#include "util/xxh64.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"

namespace {

using namespace mi::weights;
namespace fs = std::filesystem;
using mi::testing::TempDir;
using mi::testing::TensorSpec;
using mi::testing::write_safetensors;

void write_text(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
}

// Tiny bundle mirroring the real layout: 3 TE layers (linear, linear, full) plus vision/lm_head,
// 2 DiT blocks, a decoder-and-encoder VAE. TE is sharded, the others are single files.
struct Bundle {
  explicit Bundle(const std::string& tag) : dir(tag) {}
  TempDir dir;
  fs::path root() const { return dir.path(); }

  void build(const std::string& te_types = R"(["linear_attention","linear_attention","full_attention"])",
             bool drop_vae_decoder = false, bool extra_te_tensor = false) {
    write_text(root() / "text_encoder/config.json", R"({"text_config":{"layer_types":)" + te_types + "}}");
    std::vector<TensorSpec> a = {{"language_model.embed_tokens.weight", DType::kBF16, {8, 4}},
                                 {"language_model.layers.0.input_layernorm.weight", DType::kBF16, {4}},
                                 {"language_model.layers.0.linear_attn.A_log", DType::kBF16, {2}},
                                 {"language_model.layers.1.linear_attn.in_proj_a.weight", DType::kBF16, {2, 4}},
                                 {"visual.blocks.0.attn.qkv.weight", DType::kBF16, {6, 2}}};
    std::vector<TensorSpec> b = {{"language_model.layers.2.self_attn.q_proj.weight", DType::kBF16, {4, 4}},
                                 {"language_model.layers.2.self_attn.q_norm.weight", DType::kBF16, {4}},
                                 {"language_model.norm.weight", DType::kBF16, {4}},
                                 {"lm_head.weight", DType::kBF16, {8, 4}}};
    if (extra_te_tensor) b.push_back({"mystery.weight", DType::kBF16, {1}});
    write_safetensors(root() / "text_encoder/model-00001-of-00002.safetensors", a);
    write_safetensors(root() / "text_encoder/model-00002-of-00002.safetensors", b);
    std::string wm;
    for (const auto& t : a) wm += "\"" + t.name + "\":\"model-00001-of-00002.safetensors\",";
    for (const auto& t : b) wm += "\"" + t.name + "\":\"model-00002-of-00002.safetensors\",";
    wm.pop_back();
    write_text(root() / "text_encoder/model.safetensors.index.json", "{\"metadata\":{},\"weight_map\":{" + wm + "}}");

    write_text(root() / "transformer/config.json", R"({"num_layers":2})");
    write_safetensors(root() / "transformer/diffusion_pytorch_model.safetensors",
                      {{"layers.0.proj.q.weight", DType::kBF16, {4, 4}},
                       {"layers.0.norm1.weight", DType::kBF16, {4}},
                       {"layers.1.proj.q.weight", DType::kBF16, {4, 4}},
                       {"layers.1.norm1.weight", DType::kBF16, {4}},
                       {"text_embedder.proj.weight", DType::kBF16, {4, 4}},
                       {"text_embedder.blocks.0.qkv.weight", DType::kBF16, {12, 4}},
                       {"video_embedder.weight", DType::kBF16, {4, 2}},
                       {"time_embedding.0.bias", DType::kBF16, {4}},
                       {"rope.bands", DType::kF32, {2}}});

    write_text(root() / "vae/config.json", "{}");
    std::vector<TensorSpec> v = {{"encoder.conv_in.weight", DType::kF32, {2, 2}},
                                 {"quant_conv.weight", DType::kF32, {2}},
                                 {"post_quant_conv.weight", DType::kF32, {3}},
                                 {"bn.running_mean", DType::kF32, {2}},
                                 {"bn.num_batches_tracked", DType::kI64, {}}};
    if (!drop_vae_decoder) v.push_back({"decoder.conv_in.weight", DType::kF32, {5}});
    write_safetensors(root() / "vae/diffusion_pytorch_model.safetensors", v);
  }
};

TEST(ModelIndex, ClassifiesSyntheticBundle) {
  Bundle b("mi_ok");
  b.build();
  ModelIndex idx = build_index(b.root());

  auto te = idx.groups_of(Component::kTextEncoder);
  ASSERT_EQ(te.size(), 5u);  // embed_tokens, 3 layers, final_norm
  EXPECT_EQ(te[0]->name, "embed_tokens");
  EXPECT_EQ(te[4]->name, "final_norm");
  EXPECT_EQ(idx.find_group(Component::kTextEncoder, "layers.0")->layer_type, "linear_attention");
  EXPECT_EQ(idx.find_group(Component::kTextEncoder, "layers.2")->layer_type, "full_attention");
  EXPECT_EQ(idx.find_group(Component::kTextEncoder, "layers.0")->tensors.size(), 2u);
  EXPECT_EQ(idx.find_group(Component::kTextEncoder, "layers.2")->total_params, 16 + 4);
  EXPECT_EQ(idx.find_group(Component::kTextEncoder, "embed_tokens")->total_bytes, 8u * 4 * 2);

  // Layer 2 lives in the second shard, layer 1 in the first: file ids differ.
  EXPECT_NE(idx.find_group(Component::kTextEncoder, "layers.1")->tensors[0].file,
            idx.find_group(Component::kTextEncoder, "layers.2")->tensors[0].file);

  auto dit = idx.groups_of(Component::kTransformer);
  ASSERT_EQ(dit.size(), 4u);  // globals, text_embedder, block.0, block.1
  EXPECT_EQ(idx.find_group(Component::kTransformer, "block.1")->tensors.size(), 2u);
  EXPECT_EQ(idx.find_group(Component::kTransformer, "text_embedder")->tensors.size(), 2u);
  EXPECT_EQ(idx.find_group(Component::kTransformer, "globals")->tensors.size(), 3u);  // incl. rope.bands

  auto vae = idx.groups_of(Component::kVae);
  ASSERT_EQ(vae.size(), 1u);
  EXPECT_EQ(vae[0]->tensors.size(), 4u);  // decoder, post_quant_conv, bn x2
  for (const auto& t : vae[0]->tensors) EXPECT_TRUE(t.name.rfind("encoder.", 0) != 0 && t.name != "quant_conv.weight") << t.name;

  std::uint64_t excluded = 0;
  for (const auto& e : idx.excluded) excluded += e.tensors;
  EXPECT_EQ(excluded, 4u);  // visual, lm_head, encoder., quant_conv.
}

TEST(ModelIndex, RoundTripsThroughMossidxWithChecksums) {
  Bundle b("mi_rt");
  b.build();
  ModelIndex built = build_index(b.root(), {.checksum = true});
  for (const auto& g : built.groups) ASSERT_TRUE(g.checksum.has_value()) << g.name;
  write_index(built, b.root() / "moss.mossidx");
  ModelIndex loaded = load_index(b.root() / "moss.mossidx");

  ASSERT_EQ(loaded.groups.size(), built.groups.size());
  ASSERT_EQ(loaded.files.size(), built.files.size());
  for (std::size_t i = 0; i < built.groups.size(); ++i) {
    const auto& x = built.groups[i];
    const auto& y = loaded.groups[i];
    EXPECT_EQ(x.name, y.name);
    EXPECT_EQ(x.layer_type, y.layer_type);
    EXPECT_EQ(x.total_bytes, y.total_bytes);
    EXPECT_EQ(x.total_params, y.total_params);
    EXPECT_EQ(x.checksum, y.checksum);
    ASSERT_EQ(x.tensors.size(), y.tensors.size());
    for (std::size_t t = 0; t < x.tensors.size(); ++t) {
      EXPECT_EQ(x.tensors[t].name, y.tensors[t].name);
      EXPECT_EQ(x.tensors[t].dtype, y.tensors[t].dtype);
      EXPECT_EQ(x.tensors[t].shape, y.tensors[t].shape);
      EXPECT_EQ(x.tensors[t].file, y.tensors[t].file);
      EXPECT_EQ(x.tensors[t].offset, y.tensors[t].offset);
      EXPECT_EQ(x.tensors[t].nbytes, y.tensors[t].nbytes);
    }
    // Stored checksums still describe the bytes on disk.
    EXPECT_EQ(checksum_group(loaded, y), *y.checksum) << y.name;
  }
  EXPECT_EQ(loaded.excluded.size(), built.excluded.size());
}

TEST(ModelIndex, DetectsStaleFilesAndBadVersions) {
  Bundle b("mi_stale");
  b.build();
  write_index(build_index(b.root()), b.root() / "moss.mossidx");
  EXPECT_NO_THROW(load_index(b.root() / "moss.mossidx"));

  // Version mismatch.
  std::ifstream in(b.root() / "moss.mossidx");
  std::stringstream ss;
  ss << in.rdbuf();
  in.close();
  std::string text = ss.str();
  std::string v2 = text;
  v2.replace(v2.find("\"version\": 1"), 12, "\"version\": 9");
  write_text(b.root() / "future.mossidx", v2);
  try {
    load_index(b.root() / "future.mossidx");
    FAIL() << "expected IndexError";
  } catch (const StaleIndexError&) {
    FAIL() << "version mismatch is not staleness";
  } catch (const IndexError& e) {
    EXPECT_NE(std::string(e.what()).find("version 9"), std::string::npos);
  }

  // Corrupt totals.
  std::string bad = text;
  auto pos = bad.find("\"bytes\":", bad.find("\"groups\""));
  bad.replace(pos, 8, "\"bytes\": 1");
  write_text(b.root() / "bad.mossidx", bad);
  EXPECT_THROW(load_index(b.root() / "bad.mossidx"), IndexError);
  write_text(b.root() / "junk.mossidx", "{\"format\":\"nope\"}");
  EXPECT_THROW(load_index(b.root() / "junk.mossidx"), IndexError);

  // mtime change -> stale.
  auto shard = b.root() / "vae/diffusion_pytorch_model.safetensors";
  fs::last_write_time(shard, fs::last_write_time(shard) + std::chrono::hours(1));
  EXPECT_THROW(load_index(b.root() / "moss.mossidx"), StaleIndexError);
  // size change -> stale (restore mtime first so only size differs).
  fs::last_write_time(shard, fs::last_write_time(shard) - std::chrono::hours(1));
  EXPECT_NO_THROW(load_index(b.root() / "moss.mossidx"));
  // A file gone: its component's weights are missing, which no rebuild mends.
  fs::remove(shard);
  try {
    load_index(b.root() / "moss.mossidx");
    FAIL() << "expected IndexError";
  } catch (const StaleIndexError& e) {
    FAIL() << "a missing file is not staleness: " << e.what();
  } catch (const IndexError& e) {
    EXPECT_NE(std::string(e.what()).find("missing weights for vae (vae/diffusion_pytorch_model.safetensors): add them or "
                                         "pass its component file"),
              std::string::npos)
        << e.what();
    EXPECT_EQ(std::string(e.what()).find("rebuild"), std::string::npos) << e.what();
  }
}

TEST(ModelIndex, RejectsUnclassifiedAndInconsistentBundles) {
  {
    Bundle b("mi_unk");
    b.build(R"(["linear_attention","linear_attention","full_attention"])", false, /*extra_te_tensor=*/true);
    try {
      build_index(b.root());
      FAIL() << "expected IndexError";
    } catch (const IndexError& e) {
      EXPECT_NE(std::string(e.what()).find("unclassified"), std::string::npos) << e.what();
      EXPECT_NE(std::string(e.what()).find("mystery.weight"), std::string::npos) << e.what();
    }
  }
  {
    Bundle b("mi_wrongtype");
    b.build(R"(["linear_attention","full_attention","full_attention"])");
    try {
      build_index(b.root());
      FAIL() << "expected IndexError";
    } catch (const IndexError& e) {
      EXPECT_NE(std::string(e.what()).find("layers.1"), std::string::npos) << e.what();
    }
  }
  {
    Bundle b("mi_missinglayer");
    b.build(R"(["linear_attention","linear_attention","full_attention","full_attention"])");  // config has 4 layers
    EXPECT_THROW(build_index(b.root()), IndexError);
  }
  {
    Bundle b("mi_extralayer");
    b.build(R"(["linear_attention","linear_attention"])");  // checkpoint has layer 2
    EXPECT_THROW(build_index(b.root()), IndexError);
  }
  {
    Bundle b("mi_novae");
    b.build(R"(["linear_attention","linear_attention","full_attention"])", /*drop_vae_decoder=*/false);
    write_safetensors(b.root() / "vae/diffusion_pytorch_model.safetensors", {{"stranger.weight", DType::kF32, {1}}});
    EXPECT_THROW(build_index(b.root()), IndexError);
  }
  {
    Bundle b("mi_missingcomp");
    b.build();
    fs::remove_all(b.root() / "vae");
    EXPECT_THROW(build_index(b.root()), IndexError);
  }
  {
    Bundle b("mi_orphanshard");
    b.build();
    write_safetensors(b.root() / "text_encoder/model-00003-of-00002.safetensors", {{"x", DType::kU8, {1}}});
    EXPECT_THROW(build_index(b.root()), IndexError);
  }
}

std::string read_text(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Recomputes group totals so a mutated index fails on the property under test, not on its totals.
void retotal(ModelIndex& idx) {
  for (auto& g : idx.groups) {
    g.total_bytes = 0;
    g.total_params = 0;
    for (const auto& t : g.tensors) {
      g.total_bytes += t.nbytes;
      g.total_params += t.numel() < 0 ? 0 : t.numel();
    }
  }
}

WeightGroup& group(ModelIndex& idx, Component c, const std::string& name) {
  for (auto& g : idx.groups) {
    if (g.component == c && g.name == name) return g;
  }
  throw std::runtime_error("no group " + name);
}

// The .mossidx ships inside a downloaded bundle; tampered ones must fail with IndexError naming the
// problem, before any file outside the bundle is touched or any size is trusted.
TEST(ModelIndex, LoadRejectsTamperedIndexes) {
  Bundle b("mi_evil");
  b.build();
  const ModelIndex good = build_index(b.root());
  write_index(good, b.root() / "good.mossidx");
  ASSERT_NO_THROW(load_index(b.root() / "good.mossidx"));

  int checked = 0;
  // Structurally invalid content is an IndexError; a well-formed index that disagrees with the bundle
  // (the cross-check against a fresh build_index) is a StaleIndexError, whose remedy is to rebuild.
  auto run = [&](bool stale, const char* what, const std::string& message, const std::function<void(ModelIndex&)>& mutate) {
    SCOPED_TRACE(what);
    ModelIndex evil = good;
    mutate(evil);
    write_index(evil, b.root() / "evil.mossidx");
    try {
      load_index(b.root() / "evil.mossidx");
      ADD_FAILURE() << "expected an error";
    } catch (const StaleIndexError& e) {
      EXPECT_TRUE(stale) << "tampering is not staleness: " << e.what();
      EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << e.what();
    } catch (const IndexError& e) {
      EXPECT_FALSE(stale) << "expected StaleIndexError: " << e.what();
      EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << e.what();
    }
    ++checked;
  };
  auto expect_rejected = [&](const char* what, const std::string& message, const std::function<void(ModelIndex&)>& mutate) {
    run(false, what, message, mutate);
  };
  auto expect_stale = [&](const char* what, const std::string& message, const std::function<void(ModelIndex&)>& mutate) {
    run(true, what, message, mutate);
  };
  const std::string path_msg = "is not a plain bundle-relative path";

  expect_rejected("overflowing shape", "element count overflow", [](ModelIndex& m) {
    group(m, Component::kTransformer, "block.0").tensors[0].shape = {4611686018427387904, 4};
  });
  expect_rejected("int64 overflow hidden by a later zero dim", "element count overflow", [](ModelIndex& m) {
    group(m, Component::kTransformer, "block.0").tensors[0].shape = {4611686018427387904, 2, 0};
  });
  expect_rejected("negative dim", "negative dimension", [](ModelIndex& m) {
    group(m, Component::kTransformer, "block.0").tensors[0].shape = {-4, -4};
  });
  expect_rejected("rank above the cap", "rank 9", [](ModelIndex& m) {
    auto& t = group(m, Component::kTransformer, "block.0").tensors[0];
    t.shape = {1, 1, 1, 1, 1, 1, 1, 4, 4};
  });
  expect_rejected("../ path", path_msg, [](ModelIndex& m) { m.files[0].path = "../outside/model.safetensors"; });
  expect_rejected("nested .. path", path_msg, [](ModelIndex& m) { m.files[0].path = "text_encoder/../../x.safetensors"; });
  expect_rejected("absolute path", path_msg, [](ModelIndex& m) { m.files[0].path = "/etc/passwd"; });
  expect_rejected("drive path", path_msg, [](ModelIndex& m) { m.files[0].path = "C:/Windows/win.ini"; });
  expect_rejected("drive-relative path", path_msg, [](ModelIndex& m) { m.files[0].path = "C:win.ini"; });
  expect_rejected("UNC path", path_msg, [](ModelIndex& m) { m.files[0].path = "\\\\server\\share\\x"; });
  expect_rejected("backslash traversal", path_msg, [](ModelIndex& m) { m.files[0].path = "text_encoder\\..\\..\\x"; });
  expect_rejected("empty component", path_msg, [](ModelIndex& m) { m.files[0].path = "text_encoder//x"; });
  expect_rejected("empty path", path_msg, [](ModelIndex& m) { m.files[0].path = ""; });
  expect_rejected("overlapping ranges in one group", "overlaps", [](ModelIndex& m) {
    auto& ts = group(m, Component::kTransformer, "block.0").tensors;
    ASSERT_EQ(ts[0].file, ts[1].file);
    ts[1].offset = ts[0].offset + 2;
  });
  expect_rejected("tensor listed in two groups", "overlaps", [](ModelIndex& m) {
    auto& from = group(m, Component::kTransformer, "block.0");
    group(m, Component::kTransformer, "block.1").tensors.push_back(from.tensors[0]);
    retotal(m);
  });
  expect_rejected("range past the end of its file", "range outside its file", [](ModelIndex& m) {
    auto& t = group(m, Component::kVae, "decoder").tensors[0];
    t.offset = m.files[t.file].size - 1;
  });
  expect_rejected("huge offset", "range outside its file", [](ModelIndex& m) {
    group(m, Component::kVae, "decoder").tensors[0].offset = std::uint64_t{1} << 62;
  });
  expect_rejected("bad file id", "bad file id", [](ModelIndex& m) {
    group(m, Component::kVae, "decoder").tensors[0].file = static_cast<std::uint32_t>(m.files.size());
  });
  expect_stale("missing text-encoder layer", "group count", [](ModelIndex& m) {
    m.groups.erase(m.groups.begin() + 3);  // embed_tokens, layers.0, layers.1, [layers.2]
  });
  expect_stale("wrong layer type", "group text_encoder/layers.0", [](ModelIndex& m) {
    group(m, Component::kTextEncoder, "layers.0").layer_type = "full_attention";
  });
  expect_stale("extra DiT block", "group count", [](ModelIndex& m) {
    WeightGroup g = group(m, Component::kTransformer, "block.1");
    g.name = "block.2";
    g.tensors.clear();
    m.groups.push_back(g);
    retotal(m);
  });
  expect_stale("tensor moved between groups", "group transformer/block.0", [](ModelIndex& m) {
    auto& from = group(m, Component::kTransformer, "block.0");
    group(m, Component::kTransformer, "block.1").tensors.push_back(from.tensors.back());
    from.tensors.pop_back();
    retotal(m);
  });
  expect_stale("dtype differs from the header", "tensor 'layers.0.proj.q.weight'", [](ModelIndex& m) {
    group(m, Component::kTransformer, "block.0").tensors[0].dtype = DType::kF16;  // same size as BF16
  });
  expect_stale("shape differs from the header", "tensor 'layers.0.proj.q.weight'", [](ModelIndex& m) {
    group(m, Component::kTransformer, "block.0").tensors[0].shape = {2, 8};
  });
  EXPECT_EQ(checked, 24);

  // Type errors anywhere in the document are IndexErrors too.
  std::string text = read_text(b.root() / "good.mossidx");
  text.replace(text.find("\"size\": "), 8, "\"size\": \"x\", \"_\": ");
  write_text(b.root() / "typed.mossidx", text);
  EXPECT_THROW(load_index(b.root() / "typed.mossidx"), IndexError);
}

std::string index_error(const fs::path& bundle, const BuildOptions& options) {
  try {
    build_index(bundle, options);
  } catch (const IndexError& e) {
    return e.what();
  }
  return "";
}

// Components a component file replaces are left out (BuildOptions::skip): none of their weight files is opened, so
// their folders need only their configs, which stay required. A component not skipped fails exactly as before.
TEST(ModelIndex, SkippedComponentsNeedOnlyTheirConfigs) {
  Bundle b("mi_skip");
  b.build();
  const ModelIndex full = build_index(b.root());
  const ComponentSet te = ComponentSet{}.add(Component::kTextEncoder);
  const ComponentSet te_dit = ComponentSet{te}.add(Component::kTransformer);
  // The skipped components' weights unreadable (never opened), then gone.
  write_text(b.root() / "text_encoder/model-00001-of-00002.safetensors", "not a safetensors file");
  write_text(b.root() / "transformer/diffusion_pytorch_model.safetensors", "not a safetensors file");
  EXPECT_THROW(build_index(b.root()), FormatError);
  for (int gone = 0; gone < 2; ++gone) {
    SCOPED_TRACE(gone ? "weights removed" : "weights unreadable");
    const ModelIndex idx = build_index(b.root(), {.skip = te_dit});
    EXPECT_TRUE(idx.groups_of(Component::kTextEncoder).empty());
    EXPECT_TRUE(idx.groups_of(Component::kTransformer).empty());
    ASSERT_EQ(idx.groups.size(), 1u);
    ASSERT_EQ(idx.files.size(), 1u);
    EXPECT_EQ(idx.files[0].path, "vae/diffusion_pytorch_model.safetensors");
    const WeightGroup* vae = full.find_group(Component::kVae, "decoder");
    ASSERT_EQ(idx.groups[0].tensors.size(), vae->tensors.size());
    for (std::size_t i = 0; i < vae->tensors.size(); ++i) {
      EXPECT_EQ(idx.groups[0].tensors[i].name, vae->tensors[i].name);
      EXPECT_EQ(idx.groups[0].tensors[i].offset, vae->tensors[i].offset);
      EXPECT_EQ(idx.file_path(idx.groups[0].tensors[i].file), full.file_path(vae->tensors[i].file));
    }
    ASSERT_EQ(idx.excluded.size(), 2u);  // encoder., quant_conv.
    for (const auto& e : idx.excluded) EXPECT_EQ(e.component, Component::kVae) << e.prefix;
    for (const char* rel : {"text_encoder/model-00001-of-00002.safetensors", "text_encoder/model-00002-of-00002.safetensors",
                            "text_encoder/model.safetensors.index.json", "transformer/diffusion_pytorch_model.safetensors"}) {
      fs::remove(b.root() / rel);
    }
  }
  // Every component skipped: configs only (the VAE's weights unreadable, then gone, as the others').
  const ComponentSet all = ComponentSet{te_dit}.add(Component::kVae);
  write_text(b.root() / "vae/diffusion_pytorch_model.safetensors", "not a safetensors file");
  EXPECT_TRUE(build_index(b.root(), {.skip = all}).groups.empty());
  EXPECT_TRUE(build_index(b.root(), {.skip = all}).files.empty());
  fs::remove_all(b.root() / "vae");
  EXPECT_TRUE(build_index(b.root(), {.skip = all}).groups.empty());

  // A component not skipped: today's error.
  const std::string no_weights = "expected one .safetensors file or a shard index, found 0 files and no index";
  EXPECT_NE(index_error(b.root(), {}).find("text_encoder: " + no_weights), std::string::npos);
  EXPECT_NE(index_error(b.root(), {.skip = te}).find("transformer: " + no_weights), std::string::npos);
  // Configs stay required, also for a skipped component. The message names the file with native separators.
  const std::string te_config_file = (fs::path("text_encoder") / "config.json").string();
  const std::string dit_config_file = (fs::path("transformer") / "config.json").string();
  const std::string te_config = read_text(b.root() / "text_encoder/config.json");
  fs::remove(b.root() / "text_encoder/config.json");
  EXPECT_NE(index_error(b.root(), {.skip = te_dit}).find(te_config_file), std::string::npos);
  write_text(b.root() / "text_encoder/config.json", R"({"text_config":{}})");
  EXPECT_NE(index_error(b.root(), {.skip = te_dit}).find("missing text_config.layer_types"), std::string::npos);
  write_text(b.root() / "text_encoder/config.json", te_config);
  fs::remove(b.root() / "transformer/config.json");
  EXPECT_NE(index_error(b.root(), {.skip = te_dit}).find(dit_config_file), std::string::npos);
}

// A .mossidx built for the whole bundle serves a run that skips components: their stored files, groups and exclusions
// are dropped unread (their files may be gone) and the rest keep their checksums under renumbered file ids. An index
// built without a component serves only runs that skip it.
TEST(ModelIndex, LoadIndexSkipsComponents) {
  const ComponentSet te = ComponentSet{}.add(Component::kTextEncoder);
  const ComponentSet dit = ComponentSet{}.add(Component::kTransformer);
  {
    Bundle b("mi_loadskip");
    b.build();
    const ModelIndex full = build_index(b.root(), {.checksum = true});
    write_index(full, b.root() / "moss.mossidx");
    fs::remove(b.root() / "text_encoder/model-00002-of-00002.safetensors");
    EXPECT_THROW(load_index(b.root() / "moss.mossidx"), IndexError);  // missing weights for text_encoder
    const ModelIndex loaded = load_index(b.root() / "moss.mossidx", {}, te);
    const ModelIndex fresh = build_index(b.root(), {.skip = te});
    ASSERT_EQ(loaded.files.size(), 2u);  // transformer, vae
    for (std::size_t i = 0; i < fresh.files.size(); ++i) EXPECT_EQ(loaded.files[i].path, fresh.files[i].path);
    ASSERT_EQ(loaded.groups.size(), fresh.groups.size());
    for (const WeightGroup& g : loaded.groups) {
      EXPECT_NE(g.component, Component::kTextEncoder);
      const WeightGroup* f = full.find_group(g.component, g.name);
      ASSERT_NE(f, nullptr);
      EXPECT_EQ(g.checksum, f->checksum) << g.name;
      ASSERT_EQ(g.tensors.size(), f->tensors.size());
      for (std::size_t t = 0; t < g.tensors.size(); ++t) {
        EXPECT_EQ(loaded.file_path(g.tensors[t].file), full.file_path(f->tensors[t].file)) << g.tensors[t].name;
      }
      EXPECT_EQ(checksum_group(loaded, g), *g.checksum) << g.name;
    }
    ASSERT_EQ(loaded.excluded.size(), fresh.excluded.size());
    for (const auto& e : loaded.excluded) EXPECT_NE(e.component, Component::kTextEncoder) << e.prefix;
    // The components not skipped are still checked: a changed DiT file is stale.
    const fs::path dit_file = b.root() / "transformer/diffusion_pytorch_model.safetensors";
    fs::last_write_time(dit_file, fs::last_write_time(dit_file) + std::chrono::hours(1));
    EXPECT_THROW(load_index(b.root() / "moss.mossidx", {}, te), StaleIndexError);
    EXPECT_NO_THROW(load_index(b.root() / "moss.mossidx", {}, ComponentSet{te}.add(Component::kTransformer)));
  }
  {
    Bundle b("mi_partialidx");
    b.build();
    write_index(build_index(b.root(), {.skip = te}), b.root() / "moss.mossidx");
    EXPECT_NO_THROW(load_index(b.root() / "moss.mossidx", {}, te));
    EXPECT_THROW(load_index(b.root() / "moss.mossidx"), StaleIndexError);  // the bundle's text encoder is not in it
    EXPECT_THROW(load_index(b.root() / "moss.mossidx", {}, dit), StaleIndexError);

    // A kept group's tensor in a skipped component's file: refused before any range is trusted.
    ModelIndex evil = build_index(b.root());
    std::uint32_t dit_file = 0;
    while (evil.files[dit_file].path.rfind("transformer/", 0) != 0) ++dit_file;
    group(evil, Component::kTextEncoder, "embed_tokens").tensors[0].file = dit_file;
    write_index(evil, b.root() / "evil.mossidx");
    try {
      load_index(b.root() / "evil.mossidx", {}, dit);
      FAIL() << "expected IndexError";
    } catch (const StaleIndexError& e) {
      FAIL() << e.what();
    } catch (const IndexError& e) {
      EXPECT_NE(std::string(e.what()).find("component left out"), std::string::npos) << e.what();
    }
  }
}

// Config, shard-index and .mossidx files come from a downloaded bundle: size is checked before
// reading and the DOM's value count while parsing.
TEST(ModelIndex, RejectsOversizedAndOverValuedJson) {
  Bundle b("mi_caps");
  b.build();
  ASSERT_NO_THROW(build_index(b.root()));
  write_index(build_index(b.root()), b.root() / "moss.mossidx");

  auto many_values = [](std::size_t n) {  // {"x":[0,0,...]} with n + 2 values
    std::string t = "{\"x\":[0";
    t.reserve(2 * n + 16);
    for (std::size_t i = 1; i < n; ++i) t += ",0";
    return t + "]}";
  };
  auto expect_index_error = [](const std::function<void()>& fn, const char* needle) {
    try {
      fn();
      ADD_FAILURE() << "expected IndexError";
    } catch (const StaleIndexError& e) {
      ADD_FAILURE() << "not staleness: " << e.what();
    } catch (const IndexError& e) {
      EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
  };
  // Restores a file after the test mutates it.
  auto with_file = [&](const fs::path& p, const std::string& text, const std::function<void()>& fn) {
    const std::string saved = read_text(p);
    write_text(p, text);
    fn();
    write_text(p, saved);
  };
  auto oversize = [](std::uintmax_t limit) { return std::string(static_cast<std::size_t>(limit) + 1, ' '); };

  const fs::path te_cfg = b.root() / "text_encoder/config.json";
  const fs::path dit_cfg = b.root() / "transformer/config.json";
  const fs::path shard_idx = b.root() / "text_encoder/model.safetensors.index.json";
  const fs::path mossidx = b.root() / "moss.mossidx";

  with_file(te_cfg, oversize(kMaxConfigBytes), [&] { expect_index_error([&] { build_index(b.root()); }, "exceeds the limit"); });
  with_file(dit_cfg, oversize(kMaxConfigBytes), [&] { expect_index_error([&] { build_index(b.root()); }, "exceeds the limit"); });
  with_file(te_cfg, many_values(kMaxConfigValues), [&] { expect_index_error([&] { build_index(b.root()); }, "too many values"); });
  with_file(dit_cfg, many_values(kMaxConfigValues), [&] { expect_index_error([&] { build_index(b.root()); }, "too many values"); });
  with_file(shard_idx, oversize(kMaxShardIndexBytes), [&] { expect_index_error([&] { build_index(b.root()); }, "exceeds the limit"); });
  with_file(shard_idx, many_values(kMaxShardIndexValues), [&] { expect_index_error([&] { build_index(b.root()); }, "too many values"); });
  with_file(mossidx, oversize(kMaxMossIdxBytes), [&] { expect_index_error([&] { load_index(mossidx); }, "exceeds the limit"); });
  with_file(mossidx, many_values(kMaxMossIdxValues), [&] { expect_index_error([&] { load_index(mossidx); }, "too many values"); });
  EXPECT_NO_THROW(build_index(b.root()));
  EXPECT_NO_THROW(load_index(mossidx));
}

TEST(ModelIndex, RejectsOutOfRangeLayerCounts) {
  Bundle b("mi_layers");
  b.build();
  auto expect_rejected = [&](const std::string& message) {
    try {
      build_index(b.root());
      ADD_FAILURE() << "expected IndexError";
    } catch (const IndexError& e) {
      EXPECT_NE(std::string(e.what()).find(message), std::string::npos) << e.what();
    }
  };
  // Rejected from the config alone, before group names are generated from the count.
  for (const char* cfg : {R"({"num_layers":1000000000000})", R"({"num_layers":1025})", R"({"num_layers":-1})"}) {
    SCOPED_TRACE(cfg);
    write_text(b.root() / "transformer/config.json", cfg);
    expect_rejected("is outside [0, 1024]");
  }
  for (const char* cfg : {R"({"num_layers":"2"})", R"({"num_layers":2.5})", "{}"}) {
    SCOPED_TRACE(cfg);
    write_text(b.root() / "transformer/config.json", cfg);
    expect_rejected("missing or non-integer num_layers");
  }
  write_text(b.root() / "transformer/config.json", R"({"num_layers":2})");
  ASSERT_NO_THROW(build_index(b.root()));

  std::string many = "[";
  for (int i = 0; i < 1025; ++i) many += std::string(i ? "," : "") + "\"full_attention\"";
  write_text(b.root() / "text_encoder/config.json", R"({"text_config":{"layer_types":)" + many + "]}}");
  expect_rejected("more than 1024 layer_types");
  write_text(b.root() / "text_encoder/config.json", R"({"text_config":{"layer_types":[1,2,3]}})");
  expect_rejected("holds a non-string");
}

TEST(ModelIndex, MalformedShardIndexAndUnreadableBundleAreIndexErrors) {
  Bundle b("mi_shardtype");
  b.build();
  const fs::path idx = b.root() / "text_encoder/model.safetensors.index.json";
  std::string text = read_text(idx);
  text.replace(text.find("\"model-00001-of-00002.safetensors\""), 34, "17");
  write_text(idx, text);
  EXPECT_THROW(build_index(b.root()), IndexError);

  // A component path that exists but is not a directory.
  Bundle c("mi_notdir");
  c.build();
  fs::remove_all(c.root() / "vae");
  write_text(c.root() / "vae", "not a directory");
  EXPECT_THROW(build_index(c.root()), IndexError);

#ifndef _WIN32
  // A component directory that cannot be listed: a filesystem_error inside, an IndexError outside
  // (root ignores permission bits, so there is nothing to check then).
  Bundle d("mi_noperm");
  d.build();
  write_index(build_index(d.root()), d.root() / "moss.mossidx");
  fs::permissions(d.root() / "vae", fs::perms::none);
  if (::geteuid() != 0) {
    EXPECT_THROW(build_index(d.root()), IndexError);
    EXPECT_THROW(load_index(d.root() / "moss.mossidx"), IndexError);
  }
  fs::permissions(d.root() / "vae", fs::perms::owner_all);
#endif
}

TEST(ModelIndex, LayerNumbersMustBeCanonical) {
  for (const char* alias : {"layers.01.extra.weight", "layers.00.extra.weight", "layers.-0.extra.weight", "layers.+1.extra.weight"}) {
    Bundle b("mi_alias");
    b.build();
    write_safetensors(b.root() / "transformer/diffusion_pytorch_model.safetensors",
                      {{"layers.0.proj.q.weight", DType::kBF16, {4, 4}},
                       {"layers.1.proj.q.weight", DType::kBF16, {4, 4}},
                       {alias, DType::kBF16, {4}},
                       {"text_embedder.proj.weight", DType::kBF16, {4, 4}},
                       {"video_embedder.weight", DType::kBF16, {4, 2}}});
    try {
      build_index(b.root());
      ADD_FAILURE() << "expected IndexError for " << alias;
    } catch (const IndexError& e) {
      EXPECT_NE(std::string(e.what()).find("unclassified"), std::string::npos) << e.what();
    }
  }
}

TEST(ModelIndex, SummaryOfIndexWithoutBlocks) {
  std::ostringstream os;
  print_summary(ModelIndex{}, os);
  EXPECT_NE(os.str().find("DiT blocks: 0\n"), std::string::npos) << os.str();

  ModelIndex only_globals;
  WeightGroup g;
  g.component = Component::kTransformer;
  g.name = "globals";
  only_globals.groups.push_back(g);
  std::ostringstream os2;
  print_summary(only_globals, os2);
  EXPECT_NE(os2.str().find("DiT blocks: 0\n"), std::string::npos) << os2.str();
}

TEST(ModelIndex, NonAsciiPathsRoundTrip) {
  // CJK + emoji (UTF-8 source, compiled with /utf-8 on MSVC), in the bundle dir and in file names.
  const std::string vae_name = "模型🙂.safetensors";
  const std::string te_name = "文本 编码器-00002 🚀.safetensors";
  Bundle b("mi_utf8_目录🙂");
  b.build();
  fs::rename(b.root() / "vae/diffusion_pytorch_model.safetensors", b.root() / "vae" / mi::util::path_from_utf8(vae_name));
  fs::rename(b.root() / "text_encoder/model-00002-of-00002.safetensors", b.root() / "text_encoder" / mi::util::path_from_utf8(te_name));
  const fs::path shard_index = b.root() / "text_encoder/model.safetensors.index.json";
  std::string text = read_text(shard_index);
  for (std::size_t p; (p = text.find("model-00002-of-00002.safetensors")) != std::string::npos;) text.replace(p, 32, te_name);
  write_text(shard_index, text);

  ModelIndex built = build_index(b.root(), {.checksum = true});
  write_index(built, b.root() / "moss.mossidx");
  EXPECT_NE(read_text(b.root() / "moss.mossidx").find("\"vae/" + vae_name + "\""), std::string::npos);
  ModelIndex loaded = load_index(b.root() / "moss.mossidx");
  std::vector<std::string> paths;
  for (const auto& f : loaded.files) paths.push_back(f.path);
  EXPECT_NE(std::find(paths.begin(), paths.end(), "vae/" + vae_name), paths.end());
  EXPECT_NE(std::find(paths.begin(), paths.end(), "text_encoder/" + te_name), paths.end());
  for (std::uint32_t i = 0; i < loaded.files.size(); ++i) EXPECT_TRUE(fs::is_regular_file(loaded.file_path(i))) << loaded.files[i].path;
  for (const auto& g : loaded.groups) EXPECT_EQ(checksum_group(loaded, g), *g.checksum) << g.name;

  // Messages carry the UTF-8 name.
  fs::remove(b.root() / "vae" / mi::util::path_from_utf8(vae_name));
  try {
    load_index(b.root() / "moss.mossidx");
    ADD_FAILURE() << "expected IndexError";
  } catch (const IndexError& e) {  // missing weights for vae
    EXPECT_NE(std::string(e.what()).find(vae_name), std::string::npos) << e.what();
  }
}

TEST(ModelIndex, SummaryMentionsComponents) {
  Bundle b("mi_sum");
  b.build();
  std::ostringstream os;
  print_summary(build_index(b.root()), os);
  EXPECT_NE(os.str().find("[text_encoder]"), std::string::npos);
  EXPECT_NE(os.str().find("DiT blocks: 2"), std::string::npos);
  EXPECT_NE(os.str().find("excluded encoder."), std::string::npos);
}

// ---- Real bundle (headers only; checksum only for the 100 MB-class VAE decoder group) ---------

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}

TEST(ModelIndexReal, BaseBundleMatchesKnownArchitecture) {
  const fs::path bundle = weights_root() / "base";
  if (!fs::is_directory(bundle / "transformer")) GTEST_SKIP() << "weights not present at " << bundle;
  ModelIndex idx = build_index(bundle);

  // --- DiT ---
  auto dit = idx.groups_of(Component::kTransformer);
  std::int64_t dit_params = 0;
  std::uint64_t dit_bytes = 0;
  for (const auto* g : dit) {
    dit_params += g->total_params;
    dit_bytes += g->total_bytes;
  }
  // Per block, from the module definitions: q,k,v,o (3072^2 each), ffn_up_gate (16384x3072),
  // ffn_down (3072x8192) = 113,246,208 matrix params, plus norm1/norm2 (3072 each) and q/k norms (128 each).
  const std::int64_t matrix = 4 * 3072 * 3072 + 16384 * 3072 + 3072 * 8192;
  EXPECT_EQ(matrix, 113246208);
  const std::int64_t block_params = matrix + 2 * 3072 + 2 * 128;
  std::size_t blocks = 0;
  for (const auto* g : dit) {
    if (g->name.rfind("block.", 0) != 0) continue;
    ++blocks;
    EXPECT_EQ(g->total_params, block_params) << g->name;
    EXPECT_EQ(g->total_bytes, static_cast<std::uint64_t>(block_params) * 2) << g->name;
    EXPECT_EQ(g->tensors.size(), 10u) << g->name;
  }
  EXPECT_EQ(blocks, 42u);
  EXPECT_EQ(42 * matrix, 4756340736);
  EXPECT_EQ(dit_params, 5082308624);        // 5,082,308,608 bf16 + 16 f32 (rope.bands)
  EXPECT_EQ(dit_bytes, 10164617280u);       // equals safetensors metadata total_size
  EXPECT_NE(idx.find_group(Component::kTransformer, "globals"), nullptr);
  EXPECT_NE(idx.find_group(Component::kTransformer, "text_embedder"), nullptr);

  // --- Text encoder ---
  auto te = idx.groups_of(Component::kTextEncoder);
  int linear = 0, full = 0;
  std::int64_t te_params = 0;
  for (const auto* g : te) {
    te_params += g->total_params;
    linear += g->layer_type == "linear_attention";
    full += g->layer_type == "full_attention";
  }
  EXPECT_EQ(linear, 24);
  EXPECT_EQ(full, 8);
  EXPECT_EQ(te.size(), 34u);  // embed_tokens + 32 layers + final_norm
  EXPECT_EQ(te_params, 7936684544);
  EXPECT_EQ(te_params + 456010480, 8392695024);  // + vision tower = checkpoint metadata total_parameters
  // Layers 3,7,...,31 are the full-attention ones.
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(idx.find_group(Component::kTextEncoder, "layers." + std::to_string(i))->layer_type,
              i % 4 == 3 ? "full_attention" : "linear_attention");
  }
  EXPECT_EQ(idx.find_group(Component::kTextEncoder, "embed_tokens")->total_params, 248320ll * 4096);
  bool visual_excluded = false;
  for (const auto& e : idx.excluded) visual_excluded |= e.component == Component::kTextEncoder && e.prefix == "visual.";
  EXPECT_TRUE(visual_excluded);
  for (const auto* g : te) {
    for (const auto& t : g->tensors) EXPECT_EQ(t.name.find("visual."), std::string::npos) << t.name;
  }

  // --- VAE ---
  const WeightGroup* dec = idx.find_group(Component::kVae, "decoder");
  ASSERT_NE(dec, nullptr);
  for (const auto& t : dec->tensors) {
    EXPECT_TRUE(t.name.rfind("encoder.", 0) != 0 && t.name.rfind("quant_conv.", 0) != 0) << t.name;
  }
  bool has_bn = false, has_post = false;
  for (const auto& t : dec->tensors) {
    has_bn |= t.name == "bn.running_var";
    has_post |= t.name == "post_quant_conv.weight";
  }
  EXPECT_TRUE(has_bn);
  EXPECT_TRUE(has_post);
  EXPECT_EQ(dec->total_params, 27947492);

  // XXH64 of the decoder group matches an independent hash over mmap'd bytes.
  std::uint64_t got = checksum_group(idx, *dec);
  mi::Xxh64 ref;
  std::map<std::uint32_t, MappedFile> maps;
  for (const auto& t : dec->tensors) {
    auto it = maps.find(t.file);
    if (it == maps.end()) it = maps.emplace(t.file, MappedFile(idx.file_path(t.file))).first;
    ref.update(it->second.data() + t.offset, t.nbytes);
  }
  EXPECT_EQ(got, ref.digest());

  // A checksummed index of the real bundle round-trips (only the VAE group is hashed here).
  idx.groups[idx.groups.size() - 1].checksum = got;
  TempDir tmp("mi_real");
  write_index(idx, tmp.path() / "real.mossidx");
  // load_index re-reads every safetensors header and config to cross-check the index; record what
  // that costs next to a plain build (warm page cache).
  auto ms_since = [](std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  auto t0 = std::chrono::steady_clock::now();
  build_index(bundle);
  const double build_ms = ms_since(t0);
  t0 = std::chrono::steady_clock::now();
  ModelIndex back = load_index(tmp.path() / "real.mossidx", bundle);
  const double load_ms = ms_since(t0);
  RecordProperty("build_index_ms", std::to_string(build_ms));
  RecordProperty("load_index_ms", std::to_string(load_ms));
  std::printf("real bundle: build_index %.1f ms, load_index (with cross-check) %.1f ms\n", build_ms, load_ms);
  EXPECT_EQ(back.groups.size(), idx.groups.size());
  EXPECT_EQ(back.groups.back().checksum, got);
}

}  // namespace
