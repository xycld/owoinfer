// Lane D (P7a): generation metadata in PNG text chunks: round trip through the reader, an independent decoder for the
// pixels (stb_image), a 9000-character Chinese prompt, and the chunk formats.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "pipeline/png_writer.h"
#include "util/json.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

namespace {

using mi::pipeline::GenerationMetadata;
using mi::pipeline::PngText;

std::vector<std::uint8_t> test_image(int w, int h) {
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(w * h * 3));
  for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0xff);
  return rgb;
}

std::string long_chinese(std::size_t chars) {
  const std::string unit = "清晨的阳光透过薄雾洒在古老的石桥上，招牌上写着：“欢迎光临”。";  // 31 characters
  std::string s;
  std::size_t n = 0;
  while (n < chars) {
    s += unit;
    n += 31;
  }
  // Cut at a character boundary: every character above is 3 bytes in UTF-8.
  return s.substr(0, chars * 3);
}

GenerationMetadata sample_meta() {
  GenerationMetadata m;
  m.engine_version = "0.7.0";
  m.tier = "high";
  m.models = {{"dit", "moss_image_sft_int8_convrot.safetensors", std::string(64, 'a')},
              {"text_encoder", "qwen3.5_9b_int8_convrot.safetensors", std::string(64, 'b')},
              {"vae", "flux2-vae.safetensors", ""}};
  m.loras = {{"style, v2", std::string(64, 'c'), 0.35}, {"detail", "", 0.6}};
  m.prompt = "A \"quoted\" fox,\nsecond line\twith tab";
  m.negative_prompt = "blurry";
  m.sampler = "dpmpp_2m_sde";
  m.scheduler = "sgm_uniform";
  m.steps = 30;
  m.cfg = 3.5;
  m.shift = 5;
  m.seed = 9007199254740991ull;
  m.width = 1024;
  m.height = 768;
  m.extra = {{"noise", "philox"}};
  return m;
}

TEST(PngMetadata, RoundTripAndPixelsIntact) {
  const int w = 37, h = 23;
  const auto rgb = test_image(w, h);
  const GenerationMetadata meta = sample_meta();
  const std::vector<PngText> chunks = mi::pipeline::generation_text_chunks(meta);
  ASSERT_EQ(chunks.size(), 2u);
  const std::vector<std::uint8_t> png = mi::pipeline::encode_png_rgb8(rgb, w, h, chunks);

  const std::vector<PngText> back = mi::pipeline::read_png_text(png);
  ASSERT_EQ(back.size(), 2u);
  EXPECT_EQ(back[0].keyword, "generation");
  EXPECT_EQ(back[0].text, chunks[0].text);
  EXPECT_EQ(back[1].keyword, "parameters");
  EXPECT_EQ(back[1].text, chunks[1].text);

  // The JSON parses and carries every field.
  const mi::json::Value j = mi::json::parse(back[0].text);
  EXPECT_EQ(j.at("schema").as_string(), "mossinfer.generation/1");
  EXPECT_EQ(j.at("engine").at("version").as_string(), "0.7.0");
  EXPECT_EQ(j.at("engine").at("tier").as_string(), "high");
  ASSERT_EQ(j.at("models").as_array().size(), 3u);
  EXPECT_EQ(j.at("models").as_array()[0].at("sha256").as_string(), std::string(64, 'a'));
  EXPECT_EQ(j.at("models").as_array()[2].find("sha256"), nullptr);
  ASSERT_EQ(j.at("loras").as_array().size(), 2u);
  EXPECT_EQ(j.at("loras").as_array()[0].at("name").as_string(), "style, v2");
  EXPECT_EQ(j.at("loras").as_array()[0].at("strength").as_double(), 0.35);
  EXPECT_EQ(j.at("prompt").as_string(), meta.prompt);
  EXPECT_EQ(j.at("negative_prompt").as_string(), "blurry");
  EXPECT_EQ(j.at("sampler").as_string(), "dpmpp_2m_sde");
  EXPECT_EQ(j.at("scheduler").as_string(), "sgm_uniform");
  EXPECT_EQ(j.at("steps").as_int(), 30);
  EXPECT_EQ(j.at("cfg").as_double(), 3.5);
  EXPECT_EQ(j.at("shift").as_double(), 5.0);
  EXPECT_EQ(j.at("seed").as_int(), 9007199254740991);
  EXPECT_EQ(j.at("width").as_int(), 1024);
  EXPECT_EQ(j.at("height").as_int(), 768);
  EXPECT_EQ(j.at("extra").at("noise").as_string(), "philox");

  // Pixels through an independent decoder.
  int dw = 0, dh = 0, dc = 0;
  stbi_uc* px = stbi_load_from_memory(png.data(), static_cast<int>(png.size()), &dw, &dh, &dc, 3);
  ASSERT_NE(px, nullptr) << stbi_failure_reason();
  EXPECT_EQ(dw, w);
  EXPECT_EQ(dh, h);
  EXPECT_EQ(std::memcmp(px, rgb.data(), rgb.size()), 0);
  stbi_image_free(px);

  // Without metadata the file is the plain image.
  EXPECT_TRUE(mi::pipeline::read_png_text(mi::pipeline::encode_png_rgb8(rgb, w, h)).empty());
}

TEST(PngMetadata, ParametersText) {
  const std::string t = mi::pipeline::generation_parameters_text(sample_meta());
  const std::string want =
      "A \"quoted\" fox,\nsecond line\twith tab\nNegative prompt: blurry\n"
      "Steps: 30, Sampler: DPM++ 2M SDE, Schedule type: SGM Uniform, CFG scale: 3.5, Shift: 5, Seed: 9007199254740991, "
      "Size: 1024x768, Model hash: aaaaaaaaaa, Model: moss_image_sft_int8_convrot.safetensors, "
      "Lora hashes: \"style, v2: cccccccccc, detail: unknown\", noise: philox, Version: moss-infer 0.7.0";
  EXPECT_EQ(t, want);
  // ASCII text goes into tEXt, other text into iTXt.
  const auto png = mi::pipeline::encode_png_rgb8(test_image(2, 2), 2, 2,
                                                 std::vector<PngText>{{"parameters", t}, {"generation", "中文"}});
  const std::string bytes(png.begin(), png.end());
  EXPECT_NE(bytes.find("tEXtparameters"), std::string::npos);
  EXPECT_NE(bytes.find(std::string("iTXtgeneration\0\0\0\0\0", 19)), std::string::npos);
}

TEST(PngMetadata, LargeChinesePrompt) {
  GenerationMetadata meta = sample_meta();
  meta.prompt = long_chinese(9000);
  meta.negative_prompt = "模糊，低质量";
  const auto rgb = test_image(16, 16);
  const auto png = mi::pipeline::encode_png_rgb8(rgb, 16, 16, mi::pipeline::generation_text_chunks(meta));
  const auto back = mi::pipeline::read_png_text(png);
  ASSERT_EQ(back.size(), 2u);
  const mi::json::Value j = mi::json::parse(back[0].text);
  EXPECT_EQ(j.at("prompt").as_string(), meta.prompt);
  EXPECT_EQ(j.at("prompt").as_string().size(), 27000u);
  EXPECT_EQ(j.at("negative_prompt").as_string(), meta.negative_prompt);
  EXPECT_EQ(back[1].text.substr(0, meta.prompt.size()), meta.prompt);
  int dw = 0, dh = 0, dc = 0;
  stbi_uc* px = stbi_load_from_memory(png.data(), static_cast<int>(png.size()), &dw, &dh, &dc, 3);
  ASSERT_NE(px, nullptr);
  EXPECT_EQ(std::memcmp(px, rgb.data(), rgb.size()), 0);
  stbi_image_free(px);
}

TEST(PngMetadata, NamesArePrivateAndStringsAreSanitized) {
  GenerationMetadata m = sample_meta();
  const std::string dit_path("/home/someone/models/caf\xe9 \x00" "dit.safetensors", 42);  // Latin-1 byte and NUL
  m.models = {{"dit", dit_path, "abc"},
              {"vae", "C:\\Users\\someone\\vae.safetensors", ""}};
  m.loras = {{"/srv/loras/style.safetensors", "", 1.0}};
  m.prompt = std::string("bad \xff byte and NUL \0 here", 25);
  m.extra = {{"key, with: comma", "v"}};
  const auto rgb = test_image(3, 3);
  // The write must not fail on such strings, and the image must stay intact.
  const auto png = mi::pipeline::encode_png_rgb8(rgb, 3, 3, mi::pipeline::generation_text_chunks(m));
  const auto back = mi::pipeline::read_png_text(png);
  ASSERT_EQ(back.size(), 2u);
  const std::string fffd = "\xef\xbf\xbd";
  const mi::json::Value j = mi::json::parse(back[0].text);
  EXPECT_EQ(j.at("models").as_array()[0].at("name").as_string(), std::string("caf") + fffd + " " + fffd + "dit.safetensors");
  EXPECT_EQ(j.at("models").as_array()[1].at("name").as_string(), "vae.safetensors");
  EXPECT_EQ(j.at("loras").as_array()[0].at("name").as_string(), "style.safetensors");
  EXPECT_EQ(j.at("prompt").as_string(), "bad " + fffd + " byte and NUL " + fffd + " here");
  EXPECT_EQ(j.at("extra").at("key, with: comma").as_string(), "v");
  const std::string& t = back[1].text;
  EXPECT_EQ(t.find("/home"), std::string::npos);
  EXPECT_EQ(t.find("someone"), std::string::npos);
  EXPECT_NE(t.find("Model: caf" + fffd + " " + fffd + "dit.safetensors"), std::string::npos);
  EXPECT_NE(t.find("bad " + fffd + " byte and NUL " + fffd + " here\n"), std::string::npos);
  EXPECT_NE(t.find(", \"key, with: comma\": v"), std::string::npos);
  EXPECT_NE(t.find("Lora hashes: \"style.safetensors: unknown\""), std::string::npos);
  EXPECT_EQ(t.find('\0'), std::string::npos);
  int dw = 0, dh = 0, dc = 0;
  stbi_uc* px = stbi_load_from_memory(png.data(), static_cast<int>(png.size()), &dw, &dh, &dc, 3);
  ASSERT_NE(px, nullptr);
  EXPECT_EQ(std::memcmp(px, rgb.data(), rgb.size()), 0);
  stbi_image_free(px);
}

TEST(PngMetadata, PromptsCanBeLeftOutAndEmptyNegativeIsOmitted) {
  GenerationMetadata m = sample_meta();
  m.include_prompts = false;
  const mi::json::Value j = mi::json::parse(mi::pipeline::generation_json(m));
  EXPECT_EQ(j.find("prompt"), nullptr);
  EXPECT_EQ(j.find("negative_prompt"), nullptr);
  const std::string t = mi::pipeline::generation_parameters_text(m);
  EXPECT_EQ(t.rfind("Steps: 30, ", 0), 0u) << t;
  EXPECT_EQ(t.find("fox"), std::string::npos);
  EXPECT_EQ(t.find("Negative prompt"), std::string::npos);
  m.include_prompts = true;
  m.negative_prompt.clear();
  const std::string t2 = mi::pipeline::generation_parameters_text(m);
  EXPECT_EQ(t2.find("Negative prompt"), std::string::npos);
  EXPECT_NE(t2.find("with tab\nSteps: 30, "), std::string::npos);
  EXPECT_EQ(mi::json::parse(mi::pipeline::generation_json(m)).at("negative_prompt").as_string(), "");
}

TEST(PngMetadata, ReaderAndArgumentChecks) {
  const auto rgb = test_image(4, 4);
  EXPECT_THROW(mi::pipeline::encode_png_rgb8(rgb, 4, 4, std::vector<PngText>{{"", "x"}}), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::encode_png_rgb8(rgb, 4, 4, std::vector<PngText>{{" lead", "x"}}), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::encode_png_rgb8(rgb, 4, 4, std::vector<PngText>{{std::string(80, 'k'), "x"}}),
               std::invalid_argument);
  EXPECT_THROW(mi::pipeline::encode_png_rgb8(rgb, 4, 4, std::vector<PngText>{{"k", std::string("\xc3\x28", 2)}}),
               std::invalid_argument);
  EXPECT_THROW(mi::pipeline::encode_png_rgb8(rgb, 4, 4, std::vector<PngText>{{"k", std::string("a\0b", 3)}}),
               std::invalid_argument);
  auto png = mi::pipeline::encode_png_rgb8(rgb, 4, 4, std::vector<PngText>{{"comment", "caf\xc3\xa9"}});
  EXPECT_EQ(mi::pipeline::read_png_text(png).at(0).text, "caf\xc3\xa9");
  png[40] ^= 1;  // inside the text chunk: CRC mismatch
  EXPECT_THROW(mi::pipeline::read_png_text(png), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::read_png_text(std::vector<std::uint8_t>{1, 2, 3}), std::invalid_argument);
  // A Latin-1 tEXt chunk is returned as UTF-8 (built by hand: keyword "k", text 0xE9).
  std::vector<std::uint8_t> latin = mi::pipeline::encode_png_rgb8(rgb, 4, 4, std::vector<PngText>{{"k", "x"}});
  const std::size_t at = 8 + 25 + 8 + 2;  // signature, IHDR chunk, tEXt length + type, "k\0"
  ASSERT_EQ(latin[at], 'x');
  latin[at] = 0xe9;
  // Fix the CRC over type + data ("tEXt" "k\0\xe9").
  std::uint32_t c = 0xffffffffu;
  for (std::size_t i = at - 6; i <= at; ++i) {
    c ^= latin[i];
    for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xedb88320u ^ (c >> 1) : c >> 1;
  }
  c = ~c;
  for (int k = 0; k < 4; ++k) latin[at + 1 + static_cast<std::size_t>(k)] = static_cast<std::uint8_t>(c >> (24 - 8 * k));
  EXPECT_EQ(mi::pipeline::read_png_text(latin).at(0).text, "\xc3\xa9");
}

}  // namespace
