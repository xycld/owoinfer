#!/bin/sh
# Static check of an AMD device build, from its assembly and linked code object:
#   check_device.sh <llvm bin dir> <file.s> <code object>
# Prints each kernel's v_wmma_f32_16x16x16_bf16 count and resources, and the code object's undefined symbols. Fails if
# there are undefined symbols, if a GEMM or attention kernel holds no WMMA instruction, if a kernel uses scratch or spills
# to it (its fragments did not stay in registers), if a kernel's wave size or maximum flat workgroup size is not 32 (the
# kernels depend on one workgroup being one wave, mma_leaf.h), if a kernel uses a dynamic stack, if a kernel in the metadata
# went unchecked, or if the code object cannot be read.
set -eu
llvm_bin=$1
asm=$2
code_object=$3

"$llvm_bin/llvm-cxxfilt" <"$asm" | awk '
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
    printf "%-5d wmma | wave %s | max workgroup %s | %3d VGPRs, %d spilled | LDS %6d B | scratch %d B | %s\n", wmma[kernel], $2, max_wg, vgpr, spill, lds, scratch, kernel
    if (scratch == "" || spill == "" || scratch != 0 || spill != 0) bad = 1
    if (dynamic != "false") bad = 1
    scratch = spill = dynamic = ""  # a field missing from the next kernel must not inherit this value
    if ($2 != 32 || max_wg != 32) bad = 1
    if (kernel ~ /gemm|attention/ && wmma[kernel] == 0) bad = 1
  }
  END {
    printf "%d v_wmma_f32_16x16x16_bf16 in total, %d of %d kernels checked\n", total, checked, kernels
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
