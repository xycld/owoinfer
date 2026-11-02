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

#endif
