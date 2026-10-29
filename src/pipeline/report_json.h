// The generation report (`moss-infer generate --json-report`, mi_result_report_json(), moss-server's `done` event):
// one renderer for the CLI and the C API, so both documents stay field for field the same.
// Locale-independent: numbers are written with std::to_chars and the stream uses the classic locale, so a host
// application's setlocale() or std::locale::global() never puts commas or digit grouping into the JSON.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace mi::pipeline {

struct Request;
struct Result;

// JSON string literal, quotes included. Ill-formed UTF-8 (e.g. a non-UTF-8 file name) becomes U+FFFD per bad byte
// (Unicode Table 3-7: no overlongs, surrogates or code points above U+10FFFF), so the document stays valid JSON.
std::string json_escape(std::string_view s);

// A JSON number with %.6g precision (std::chars_format::general, 6 digits); "null" for NaN and infinities.
std::string json_number(double d);

// [22, 24]
std::string json_layers(const std::vector<int>& v);

// One generation's report fields, without the enclosing braces, split around the image path so a caller can render
// the report once and fill in the image later: head + json_escape(image) + tail. head ends with `"image": `; tail
// starts with `, "latents": `.
struct ReportParts {
  std::string head, tail;
};
ReportParts report_parts(const Request& q, const Result& r, std::string_view latent_path);

// The report's "references" object (reference images: count, max_edge, sizes, tokens, cache_bytes, prefills,
// encode_schedule, warnings), or "" when no reference image ran (the report then has no "references" key; an edit
// adapter's warning without references is Result::reference_warnings alone).
std::string references_json(const Request& q, const Result& r);

// head + json_escape(image) + tail.
std::string report_fields(const Request& q, const Result& r, std::string_view image, std::string_view latent);

}  // namespace mi::pipeline
