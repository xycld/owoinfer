# stb (vendored)

- Upstream: https://github.com/nothings/stb
- Commit: `013ac3beddff3dbffafd5177e7972067cd2b5083` (2024-05-31, last change to `stb_image.h`)
- Kept: `stb_image.h` (v2.30, unmodified, sha256 `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3`)
  and `LICENSE` (public domain / MIT, choose either). Everything else in the repository was left out.
- Use:
  - Engine: `src/pipeline/pipeline_image.cpp` decodes init images and masks (PNG and JPEG only, `STBI_ONLY_PNG` +
    `STBI_ONLY_JPEG`, limited to 32768 px per side). Include path set by `cmake/modules/pipeline_image.cmake`.
  - Tests: `tests/unit/test_flux2_vae.cpp` decodes the reference PNGs. Include path set by `cmake/modules/flux2_vae.cmake`.
  - Both use a system include directory (the header is not warning-clean).
