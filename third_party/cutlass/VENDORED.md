# Vendored CUTLASS (C++ headers only)

- Upstream: https://github.com/NVIDIA/cutlass, tag v4.7.1, commit cb4247394dd82148787aed73e5dc7cef33cbf862
- Kept: `include/`, `tools/util/include/`, `LICENSE.txt` (BSD-3-Clause, covers the C++ templates).
- Removed: everything else, including the CUTLASS Python DSLs and their NVIDIA EULA (not used here).
- Patched: `include/cute/stride.hpp` carries upstream PR #3198 (open as of 2026-10-22: CompactSeq replaces the
  dependent member alias CompactLambda<Major>::seq, which MSVC rejects with C3545). Type-level only: the selected
  sequence, and so the generated code, is unchanged.
