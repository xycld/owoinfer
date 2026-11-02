#include "gemm.h"

#include <stdexcept>
#include <string>

namespace mi::vk {
namespace {

const std::uint32_t kCoopSpirv[] = {
#include "gemm_coop.spv.inc"
};
const std::uint32_t kPlainSpirv[] = {
#include "gemm_plain.spv.inc"
};

constexpr std::uint32_t kCoopTile = 64;
constexpr std::uint32_t kPlainTileRows = 8, kPlainTileCols = 32;
constexpr std::uint32_t kCoopSubgroup = 32;

struct Push {
  std::uint64_t a, b, c;
  std::uint32_t m, n, k, lda, ldb, ldc;
};
struct PlainPush {
  std::uint64_t a, b, out, gate;
  std::uint32_t m, n, k, lda, ldb, ldo;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("vulkan gemm: " + message);
}

std::uint32_t tiles(std::uint32_t size, std::uint32_t tile) { return (size + tile - 1) / tile; }

}  // namespace

const char* gemm_path_name(GemmPath path) { return path == GemmPath::kCoop ? "coop" : "plain"; }

Gemm::Gemm(Device& device) : Gemm(device, device.info().caps.coop ? GemmPath::kCoop : GemmPath::kPlain) {}

Gemm::Gemm(Device& device, GemmPath path) : path_(path) {
  if (path == GemmPath::kCoop) {
    if (!device.info().caps.coop) {
      throw std::runtime_error(device.info().name + " cannot run the cooperative-matrix GEMM: " +
                               device.info().caps.coop_reason);
    }
    coop_ = std::make_unique<Pipeline>(device, kCoopSpirv, sizeof(kCoopSpirv), sizeof(Push), kCoopSubgroup);
  } else {
    plain_f16_ = std::make_unique<Pipeline>(device, kPlainSpirv, sizeof(kPlainSpirv), sizeof(PlainPush), 0,
                                            std::vector<std::uint32_t>{0, 0});
    plain_bf16_ = std::make_unique<Pipeline>(device, kPlainSpirv, sizeof(kPlainSpirv), sizeof(PlainPush), 0,
                                             std::vector<std::uint32_t>{1, 0});
  }
}

Gemm::~Gemm() = default;

void Gemm::record(VkCommandBuffer cb, const GemmArgs& args, InputType input) const {
  require(args.a != nullptr && args.b != nullptr && args.c != nullptr, "null buffer");
  require(args.m > 0 && args.n > 0 && args.k > 0, "M, N and K must be positive");
  require(args.lda >= args.k && args.ldb >= args.k && args.ldc >= args.n, "a stride is smaller than its row");
  require(args.lda % 4 == 0 && args.ldb % 4 == 0, "lda and ldb must be multiples of 4");
  require(args.k % (path_ == GemmPath::kCoop ? 16 : 4) == 0,
          std::string("K must be a multiple of ") + (path_ == GemmPath::kCoop ? "16" : "4"));
  require((args.m - 1ull) * args.lda + args.k <= args.a->bytes() / 2, "A is too small");
  require((args.n - 1ull) * args.ldb + args.k <= args.b->bytes() / 2, "B is too small");
  require((args.m - 1ull) * args.ldc + args.n <= args.c->bytes() / 4, "C is too small");
  require(path_ == GemmPath::kPlain || input == InputType::kF16, "the cooperative-matrix path takes f16 inputs");
  require(path_ == GemmPath::kCoop || args.n % 4 == 0, "the plain path needs N to be a multiple of 4");

  if (path_ == GemmPath::kCoop) {
    const Push push{args.a->address(), args.b->address(), args.c->address(), args.m, args.n, args.k, args.lda, args.ldb,
                    args.ldc};
    coop_->dispatch(cb, &push, tiles(args.n, kCoopTile), tiles(args.m, kCoopTile));
  } else {
    const PlainPush push{args.a->address(), args.b->address(), args.c->address(), 0, args.m, args.n, args.k, args.lda,
                         args.ldb, args.ldc};
    const Pipeline& pipeline = input == InputType::kBF16 ? *plain_bf16_ : *plain_f16_;
    pipeline.dispatch(cb, &push, tiles(args.n, kPlainTileCols), tiles(args.m, kPlainTileRows));
  }
  compute_barrier(cb);
}

}  // namespace mi::vk
