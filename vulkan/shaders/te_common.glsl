// What the text encoder's shaders add to common.glsl: 32-bit words of two bf16, single bf16 reads, and fp32 vectors.
#ifndef MI_VK_TE_COMMON_GLSL
#define MI_VK_TE_COMMON_GLSL

#include "common.glsl"

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Uints { uint d[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer Vec4f { vec4 d[]; };

vec2 f2(uint w) { return uintBitsToFloat(uvec2(w << 16, w & 0xffff0000u)); }
uint bf2(vec2 v) { return bf(v.x) | (bf(v.y) << 16); }

// Element `index` of a bf16 array at a 4-byte aligned address.
float bf16_at(uint64_t base, uint64_t index) {
  const uint w = Uints(base + (index >> 1) * 4ul).d[0];
  return uintBitsToFloat((index & 1ul) == 0ul ? w << 16 : w & 0xffff0000u);
}

// fp32 element `index` of an array: GLSL indexes with 32 bits, so 64-bit offsets go into the address.
float f32_at(uint64_t base, uint64_t index) { return Floats(base + index * 4ul).d[0]; }
void f32_store(uint64_t base, uint64_t index, float v) { Floats(base + index * 4ul).d[0] = v; }

// exp(x) within about 1 fp32 ULP, where Vulkan's exp may be off by 3 + 2|x| ULP: x = n ln2 + r with ln2 in two parts,
// |r| <= ln2 / 2, exp(r) by its degree-7 Taylor polynomial (truncation below 0.1 ULP), times 2^n. Results below 2^-126
// flush to zero.
float exp_f(float x) {
  if (x < -87.3) return 0.0;
  if (x > 88.7) return uintBitsToFloat(0x7f800000u);
  const float n = roundEven(x * 1.4426950408889634);
  precise const float r_hi = fma(n, -0.693145751953125, x);  // ln2's leading 16 bits: n * them is exact
  precise const float r = fma(n, -1.4286068203094172e-06, r_hi);
  precise float p = 1.0 / 5040.0;
  p = fma(p, r, 1.0 / 720.0);
  p = fma(p, r, 1.0 / 120.0);
  p = fma(p, r, 1.0 / 24.0);
  p = fma(p, r, 1.0 / 6.0);
  p = fma(p, r, 0.5);
  p = fma(p, r, 1.0);
  p = fma(p, r, 1.0);
  const int e = int(n);
  // 2^e as two factors, so that e = -126 .. -127 and 128 stay representable.
  return (p * uintBitsToFloat(uint(127 + e / 2) << 23)) * uintBitsToFloat(uint(127 + e - e / 2) << 23);
}

// torch's fp32 silu and sigmoid: x / (1 + exp(-x)) and 1 / (1 + exp(-x)).
float silu_f(float x) {
  precise const float den = 1.0 + exp(-x);
  precise const float r = x / den;
  return r;
}
float sigmoid_f(float x) {
  precise const float den = 1.0 + exp(-x);
  precise const float r = 1.0 / den;
  return r;
}

#endif
