// bf16 travels as its bits in 32-bit words, so these shaders need neither 16-bit storage nor 16-bit arithmetic. As in the
// engine's kernels every step is one fp32 operation rounded once, so results are declared precise: contraction into a
// fused multiply-add would change which way a value rounds.
#ifndef MI_VK_COMMON_GLSL
#define MI_VK_COMMON_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Words { uvec2 d[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Floats { float d[]; };

vec4 f4(uvec2 w) {
  return uintBitsToFloat(uvec4(w.x << 16, w.x & 0xffff0000u, w.y << 16, w.y & 0xffff0000u));
}

// fp32 to bf16 bits, round to nearest even. A NaN stays a NaN: the rounding carry could otherwise wrap it to -0.
uint bf(float v) {
  const uint u = floatBitsToUint(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x0040u;
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
uvec2 bf4(vec4 v) { return uvec2(bf(v.x) | (bf(v.y) << 16), bf(v.z) | (bf(v.w) << 16)); }
vec4 rbf4(vec4 v) { return f4(bf4(v)); }

// exp(-2 |x|) stays in range where GLSL's tanh may divide infinities; beyond 12 the result rounds to 1 in fp32 anyway.
vec4 tanh_safe(vec4 x) {
  const vec4 e = exp(-2.0 * min(abs(x), vec4(12.0)));
  precise const vec4 t = (1.0 - e) / (1.0 + e);
  return sign(x) * t;
}

vec4 situ(vec4 h, vec4 g) {
  const float inv25 = 1.0 / 25.0;  // torch divides by a CPU scalar as a multiply by its fp32 reciprocal
  precise const vec4 g_quarter = g * 0.25;
  const vec4 tanh_g = rbf4(tanh_safe(rbf4(g_quarter)));
  precise const vec4 four_tanh = 4.0 * tanh_g;
  precise const vec4 sigmoid = 1.0 / (1.0 + exp(-g));
  precise const vec4 left = rbf4(four_tanh) * rbf4(sigmoid);
  precise const vec4 h_scaled = h * inv25;
  const vec4 tanh_h = rbf4(tanh_safe(rbf4(h_scaled)));
  precise const vec4 right = 25.0 * tanh_h;
  precise const vec4 product = rbf4(left) * rbf4(right);
  return product;
}

#endif
