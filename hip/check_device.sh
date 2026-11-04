#!/bin/sh
# Static check of an AMD device build of one kernel source (block_kernels.hip, dit_kernels.hip, te_kernels.hip), from its
# assembly and linked code object:
#   check_device.sh <llvm bin dir> <arch> <file.s> <code object>
# Prints each kernel's v_wmma_f32_16x16x16_bf16 count and resources, and the code object's undefined symbols. Fails if
# there are undefined symbols, if a kernel's name is not one of the kernels below or its WMMA count is not the one listed
# for <arch> (a walk that lost its WMMAs on one path, packed or row-major, lowers the count), if a kernel uses scratch or
# spills to it (its fragments did not stay in registers), if a kernel's wave size or maximum flat workgroup size is not 32
# (the kernels depend on one workgroup being one wave, mma_leaf.h), if a kernel uses a dynamic stack, if a kernel in the
# metadata went unchecked, or if the code object cannot be read.
# HIP_BLOCK_WMMA_COUNTS=report: for another ROCm, whose clang may emit other counts. A kernel whose count differs prints
# "COUNT MISMATCH: <kernel> n vs expected m" and, if it still holds some WMMA, does not fail the check. A GEMM, SiTU or
# attention kernel with none, and every other condition above, still fails it.
set -eu
llvm_bin=$1
arch=$2
asm=$3
code_object=$4

# WMMA instructions per kernel as ROCm 7.2.4's clang emits this source: the GEMM and SiTU kernels 6 on each path, packed
# and row-major; attention 16 in its packed score walk and the rest in the row-major walk and P x V, unrolled further on
# gfx1100; the text encoder's causal attention 4 in its score walk and 16 in P x V. A change of source or compiler that
# moves them must update this table.
case $arch in
  gfx1100) attention=48 ;;
  gfx1200 | gfx1201) attention=34 ;;
  *)
    echo "check_device.sh: no expected WMMA counts for $arch"
    exit 1
    ;;
esac

report=0
case ${HIP_BLOCK_WMMA_COUNTS:-} in
  "") ;;
  report) report=1 ;;
  *)
    echo "check_device.sh: HIP_BLOCK_WMMA_COUNTS must be unset or report (got '$HIP_BLOCK_WMMA_COUNTS')"
    exit 1
    ;;
esac

"$llvm_bin/llvm-cxxfilt" <"$asm" | awk -v attention="$attention" -v report="$report" '
  BEGIN {
    ns = "hip_block::(anonymous namespace)::"
    want[ns "pack_kernel(" ns "PackArgs)"] = 0
    want[ns "pack_attention_kernel(" ns "PackAttnArgs)"] = 0
    want[ns "rms_norm_modulate_kernel(" ns "NormArgs)"] = 0
    want[ns "qk_norm_rope_kernel(" ns "QkArgs)"] = 0
    want["void " ns "gemm_kernel<(" ns "Epilogue)0>(" ns "GemmArgs)"] = 12
    want["void " ns "gemm_kernel<(" ns "Epilogue)1>(" ns "GemmArgs)"] = 12
    want[ns "gemm_situ_kernel(" ns "GemmArgs)"] = 12
    want[ns "attention_kernel(" ns "AttnArgs)"] = attention
    # dit_kernels.hip: the bias epilogues of the same GEMM kernel, the text refiner norm, SiLU and SiTU-GLU.
    want["void " ns "gemm_kernel<(" ns "Epilogue)2>(" ns "GemmArgs)"] = 12
    want["void " ns "gemm_kernel<(" ns "Epilogue)3>(" ns "GemmArgs)"] = 12
    want[ns "rms_norm_kernel(" ns "PlainNormArgs)"] = 0
    want[ns "silu_kernel(" ns "UnaryArgs)"] = 0
    want[ns "situ_glu_kernel(" ns "SituGluArgs)"] = 0
    want[ns "rope_table_kernel(" ns "RopeArgs)"] = 0
    # te_kernels.hip: the text encoder norms, GatedDeltaNet, causal attention and SiLU product; its GEMMs are the
    # gemm_kernel instances above.
    want[ns "te_rms_norm_kernel(" ns "TeNormArgs)"] = 0
    want[ns "te_rope_table_kernel(" ns "RopeArgs)"] = 0
    want[ns "te_conv_silu_kernel(" ns "ConvArgs)"] = 0
    want[ns "te_gates_kernel(" ns "GatesArgs)"] = 0
    want[ns "te_gdn_qk_kernel(" ns "GdnArgs)"] = 0
    want[ns "te_gdn_chunk_kernel(" ns "GdnArgs)"] = 0
    want[ns "te_gdn_solve_kernel(" ns "GdnArgs)"] = 0
    want[ns "te_gdn_state_kernel(" ns "GdnArgs)"] = 0
    want[ns "te_attention_kernel(" ns "TeAttnArgs)"] = 20
    want[ns "te_silu_mul_kernel(" ns "SiluMulArgs)"] = 0
  }
  /^[^ \t.#;].*:[ \t]*(;.*)?$/ { name = $0; sub(/:[ \t]*(;.*)?$/, "", name); current = name }
  /v_wmma_f32_16x16x16_bf16/ { wmma[current]++; total++ }
  /^[ \t]*\.max_flat_workgroup_size:/ { max_wg = $2 }
  /^[ \t]*\.name:/ { kernel = $0; sub(/^[ \t]*\.name:[ \t]*/, "", kernel) }
  /^[ \t]*\.group_segment_fixed_size:/ { lds = $2 }
  /^[ \t]*\.private_segment_fixed_size:/ { scratch = $2 }
  /^[ \t]*\.vgpr_count:/ { vgpr = $2 }
  /^[ \t]*\.vgpr_spill_count:/ { spill = $2 }
  /^[ \t]*\.symbol:/ { kernels++ }
  /^[ \t]*\.uses_dynamic_stack:/ { dynamic = $2 }
  /^[ \t]*\.wavefront_size:/ {
    # .wavefront_size is the last field the assembler prints for a kernel it is reporting on.
    checked++
    expected = (kernel in want) ? want[kernel] : "none"
    printf "%-5d wmma (expected %s) | wave %s | max workgroup %s | %3d VGPRs, %d spilled | LDS %6d B | scratch %d B | %s\n", wmma[kernel], expected, $2, max_wg, vgpr, spill, lds, scratch, kernel
    if (expected == "none") bad = 1
    else if (wmma[kernel] + 0 != expected) {
      mismatch[++mismatches] = sprintf("COUNT MISMATCH: %s %d vs expected %d", kernel, wmma[kernel], expected)
      if (!report || wmma[kernel] + 0 == 0) bad = 1  # no WMMA at all is not a toolchain difference
    }
    if (scratch == "" || spill == "" || scratch != 0 || spill != 0) bad = 1
    if (dynamic != "false") bad = 1
    scratch = spill = dynamic = ""  # a field missing from the next kernel must not inherit this value
    if ($2 != 32 || max_wg != 32) bad = 1
  }
  END {
    for (i = 1; i <= mismatches; ++i) print mismatch[i]
    printf "%d v_wmma_f32_16x16x16_bf16 in total, %d of %d kernels checked%s\n", total, checked, kernels, report ? "; counts not enforced (report mode)" : ""
    exit (kernels == 0 || checked != kernels || bad)
  }
'

undefined=$("$llvm_bin/llvm-nm" -u "$code_object")
if [ -n "$undefined" ]; then
  echo "undefined symbols in $code_object:"
  echo "$undefined"
  exit 1
fi
echo "no undefined symbols in $code_object"
