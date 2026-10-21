# stb (vendored, tests only)

- Upstream: https://github.com/nothings/stb
- Commit: `013ac3beddff3dbffafd5177e7972067cd2b5083` (2024-05-31, last change to `stb_image.h`)
- Kept: `stb_image.h` (v2.30, unmodified, sha256 `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3`)
  and `LICENSE` (public domain / MIT, choose either). Everything else in the repository was left out.
- Use: `tests/unit/test_flux2_vae.cpp` decodes the reference PNGs (`STBI_ONLY_PNG`). Not linked into the engine.
  Include path set by `cmake/modules/flux2_vae.cmake` as a system directory (the header is not warning-clean).
