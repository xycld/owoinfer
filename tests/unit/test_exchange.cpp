// runtime/exchange.h: the component step's device-pointer checks, the two-event stream handoff, and the pinned staging
// (layout, growth under a host budget, a host-mode round trip).
#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "core/cuda_check.h"
#include "runtime/exchange.h"
#include "test_runtime_kernels.h"

namespace {

using mi::runtime::DevicePointerCheck;
using mi::runtime::PinnedStaging;
using mi::runtime::StagingError;
using mi::runtime::StagingLayout;
using mi::runtime::StreamHandoff;

constexpr std::size_t kMiB = std::size_t{1} << 20;

// The message of the std::invalid_argument `f` throws ("" when it does not throw).
template <class F>
std::string refusal(F&& f) {
  try {
    f();
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

bool contains(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// ---- layout (no device) -------------------------------------------------------------------------------------------

TEST(Exchange, StagingLayoutAlignsEveryTensor) {
  const StagingLayout l = StagingLayout::plan(1000, 3, 600, 0);
  EXPECT_EQ(l.x, 0u);
  EXPECT_EQ(l.out[0], 1024u);
  EXPECT_EQ(l.out[1], 1024u + 768u);
  EXPECT_EQ(l.out[2], 1024u + 2 * 768u);
  EXPECT_EQ(l.control_bytes, 0u);
  EXPECT_EQ(l.bytes, 1024u + 3 * 768u);
  const StagingLayout c = StagingLayout::plan(2 * kMiB, 8, kMiB, 100);
  EXPECT_EQ(c.control, 2 * kMiB);  // x, control, outputs
  EXPECT_EQ(c.out[0], 2 * kMiB + 256);
  EXPECT_EQ(c.out[7], 2 * kMiB + 256 + 7 * kMiB);
  EXPECT_EQ(c.bytes, 2 * kMiB + 256 + 8 * kMiB);
  EXPECT_THROW(StagingLayout::plan(1000, 0, 600, 0), std::invalid_argument);
  EXPECT_THROW(StagingLayout::plan(1000, 9, 600, 0), std::invalid_argument);
  EXPECT_THROW(StagingLayout::plan(0, 1, 600, 0), std::invalid_argument);
  EXPECT_THROW(StagingLayout::plan(1000, 1, 0, 0), std::invalid_argument);
}

// ---- pointer checks -----------------------------------------------------------------------------------------------

class GpuExchange : public ::testing::Test {
 protected:
  void SetUp() override {
    MI_CUDA_CHECK(cudaSetDevice(0));
    MI_CUDA_CHECK(cudaFree(nullptr));  // the runtime holds the primary context from here on
  }
};

TEST_F(GpuExchange, AcceptsDeviceMemoryInsideItsAllocation) {
  void* d = nullptr;
  MI_CUDA_CHECK(cudaMalloc(&d, kMiB));
  const DevicePointerCheck check(0);
  auto* p = static_cast<std::byte*>(d);
  EXPECT_NO_THROW(check.check(d, kMiB, "x"));
  EXPECT_NO_THROW(check.check(p + 4096, kMiB - 4096, "out[0]"));
  const std::string past = refusal([&] { check.check(p + 4096, kMiB, "out[1]"); });
  EXPECT_TRUE(contains(past, "out[1] needs 1048576 bytes, but only 1044480 are mapped")) << past;
  EXPECT_TRUE(contains(refusal([&] { check.check(d, kMiB + 1, "x"); }), "only 1048576 are mapped"));
  MI_CUDA_CHECK(cudaFree(d));
}

TEST_F(GpuExchange, AcceptsStreamOrderedPoolMemory) {
  // cudaMallocAsync (torch's cudaMallocAsync backend): pool memory, not tied to a context the way cudaMalloc is.
  int pools = 0;
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, 0));
  if (pools == 0) GTEST_SKIP() << "no memory pools on this device";
  cudaStream_t s = nullptr;
  MI_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  void* d = nullptr;
  MI_CUDA_CHECK(cudaMallocAsync(&d, kMiB, s));
  MI_CUDA_CHECK(cudaStreamSynchronize(s));
  EXPECT_NO_THROW(DevicePointerCheck(0).check(d, kMiB, "x"));
  MI_CUDA_CHECK(cudaFreeAsync(d, s));
  MI_CUDA_CHECK(cudaStreamSynchronize(s));
  MI_CUDA_CHECK(cudaStreamDestroy(s));
}

TEST_F(GpuExchange, RefusesHostManagedAndOtherDevicesMemory) {
  const DevicePointerCheck check(0);
  std::vector<float> pageable(1024);
  EXPECT_TRUE(contains(refusal([&] { check.check(pageable.data(), 4096, "x"); }), "x is not CUDA memory"));
  void* pinned = nullptr;
  MI_CUDA_CHECK(cudaHostAlloc(&pinned, kMiB, cudaHostAllocDefault));
  EXPECT_TRUE(contains(refusal([&] { check.check(pinned, kMiB, "out[2]"); }), "out[2] is page-locked host memory"));
  MI_CUDA_CHECK(cudaFreeHost(pinned));
  void* managed = nullptr;
  MI_CUDA_CHECK(cudaMallocManaged(&managed, kMiB));
  EXPECT_TRUE(contains(refusal([&] { check.check(managed, kMiB, "control"); }), "control is managed"));
  MI_CUDA_CHECK(cudaFree(managed));
  EXPECT_TRUE(contains(refusal([&] { check.check(nullptr, 4, "x"); }), "x has no data"));
  void* d = nullptr;
  MI_CUDA_CHECK(cudaMalloc(&d, kMiB));
  const std::string other = refusal([&] { DevicePointerCheck(1).check(d, kMiB, "x"); });
  EXPECT_TRUE(contains(other, "x is memory of device 0")) << other;
  EXPECT_TRUE(contains(other, "device memory of device 1")) << other;
  MI_CUDA_CHECK(cudaFree(d));
}

TEST_F(GpuExchange, VmmTensorsMustBeMappedThroughout) {
#ifdef _WIN32
  GTEST_SKIP() << "driver API through dlopen: Linux only in this test";
#else
  // torch's expandable segments: allocations mapped back to back into one reservation, the rest of it unmapped. The
  // pointer attributes' range is the whole reservation here; the check must follow the mapped allocations instead.
  void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(lib, nullptr);
  const auto granularity = reinterpret_cast<decltype(&cuMemGetAllocationGranularity)>(dlsym(lib, "cuMemGetAllocationGranularity"));
  const auto reserve = reinterpret_cast<decltype(&cuMemAddressReserve)>(dlsym(lib, "cuMemAddressReserve"));
  const auto create = reinterpret_cast<decltype(&cuMemCreate)>(dlsym(lib, "cuMemCreate"));
  const auto map = reinterpret_cast<decltype(&cuMemMap)>(dlsym(lib, "cuMemMap"));
  const auto set_access = reinterpret_cast<decltype(&cuMemSetAccess)>(dlsym(lib, "cuMemSetAccess"));
  const auto unmap = reinterpret_cast<decltype(&cuMemUnmap)>(dlsym(lib, "cuMemUnmap"));
  const auto release = reinterpret_cast<decltype(&cuMemRelease)>(dlsym(lib, "cuMemRelease"));
  const auto address_free = reinterpret_cast<decltype(&cuMemAddressFree)>(dlsym(lib, "cuMemAddressFree"));
  const auto attribute = reinterpret_cast<decltype(&cuPointerGetAttribute)>(dlsym(lib, "cuPointerGetAttribute"));
  ASSERT_TRUE(granularity && reserve && create && map && set_access && unmap && release && address_free && attribute);
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  std::size_t g = 0;
  ASSERT_EQ(granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), CUDA_SUCCESS);
  CUdeviceptr base = 0;
  ASSERT_EQ(reserve(&base, 4 * g, 0, 0, 0), CUDA_SUCCESS);
  CUmemGenericAllocationHandle a = 0, b = 0;
  ASSERT_EQ(create(&a, g, &prop, 0), CUDA_SUCCESS);
  ASSERT_EQ(create(&b, g, &prop, 0), CUDA_SUCCESS);
  ASSERT_EQ(map(base, g, 0, a, 0), CUDA_SUCCESS);
  ASSERT_EQ(map(base + g, g, 0, b, 0), CUDA_SUCCESS);
  CUmemAccessDesc access{};
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ASSERT_EQ(set_access(base, 2 * g, &access, 1), CUDA_SUCCESS);
  std::size_t range = 0;
  ASSERT_EQ(attribute(&range, CU_POINTER_ATTRIBUTE_RANGE_SIZE, base + g), CUDA_SUCCESS);
  std::printf("[exchange] VMM: granularity %zu, the attributes' range at chunk B: %zu bytes (the reservation: %zu)\n", g,
              range, 4 * g);
  const auto at = [&](std::size_t offset) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(base + offset)); };
  const DevicePointerCheck check(0);
  EXPECT_NO_THROW(check.check(at(0), 2 * g, "x"));              // both chunks
  EXPECT_NO_THROW(check.check(at(g / 2), g, "out[0]"));         // across the chunk boundary
  const std::string past = refusal([&] { check.check(at(g + g / 2), g, "out[1]"); });  // into the unmapped part
  EXPECT_TRUE(contains(past, ("out[1] needs " + std::to_string(g) + " bytes, but only " + std::to_string(g / 2) +
                              " are mapped").c_str()))
      << past;
  EXPECT_TRUE(contains(refusal([&] { check.check(at(2 * g + 16), 64, "control"); }), "control is not CUDA memory"));
  EXPECT_EQ(unmap(base, 2 * g), CUDA_SUCCESS);
  EXPECT_EQ(release(a), CUDA_SUCCESS);
  EXPECT_EQ(release(b), CUDA_SUCCESS);
  EXPECT_EQ(address_free(base, 4 * g), CUDA_SUCCESS);
#endif
}

TEST_F(GpuExchange, RefusesMemoryOfAnotherContext) {
#ifdef _WIN32
  GTEST_SKIP() << "driver API through dlopen: Linux only in this test";
#else
  // A context made with the driver API (not the runtime's primary one) and memory allocated in it.
  void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(lib, nullptr);
  using CtxCreate = CUresult (*)(CUcontext*, unsigned int, CUdevice);
  using CtxPop = CUresult (*)(CUcontext*);
  using CtxPush = CUresult (*)(CUcontext);
  using CtxDestroy = CUresult (*)(CUcontext);
  using MemAlloc = CUresult (*)(CUdeviceptr*, std::size_t);
  using MemFree = CUresult (*)(CUdeviceptr);
  const auto create = reinterpret_cast<CtxCreate>(dlsym(lib, "cuCtxCreate_v2"));
  const auto pop = reinterpret_cast<CtxPop>(dlsym(lib, "cuCtxPopCurrent_v2"));
  const auto push = reinterpret_cast<CtxPush>(dlsym(lib, "cuCtxPushCurrent_v2"));
  const auto destroy = reinterpret_cast<CtxDestroy>(dlsym(lib, "cuCtxDestroy_v2"));
  const auto alloc = reinterpret_cast<MemAlloc>(dlsym(lib, "cuMemAlloc_v2"));
  const auto mem_free = reinterpret_cast<MemFree>(dlsym(lib, "cuMemFree_v2"));
  ASSERT_TRUE(create && pop && push && destroy && alloc && mem_free);
  CUcontext ctx = nullptr;
  ASSERT_EQ(create(&ctx, 0, 0), CUDA_SUCCESS);  // made current
  CUdeviceptr d = 0;
  ASSERT_EQ(alloc(&d, kMiB), CUDA_SUCCESS);
  CUcontext popped = nullptr;
  ASSERT_EQ(pop(&popped), CUDA_SUCCESS);
  const std::string why =
      refusal([&] { DevicePointerCheck(0).check(reinterpret_cast<void*>(static_cast<std::uintptr_t>(d)), kMiB, "x"); });
  EXPECT_TRUE(contains(why, "x belongs to a CUDA context other than the device's primary context")) << why;
  ASSERT_EQ(push(ctx), CUDA_SUCCESS);
  EXPECT_EQ(mem_free(d), CUDA_SUCCESS);
  EXPECT_EQ(destroy(ctx), CUDA_SUCCESS);  // also pops it
  MI_CUDA_CHECK(cudaSetDevice(0));
#endif
}

// ---- the stream handoff -------------------------------------------------------------------------------------------

TEST_F(GpuExchange, HandoffOrdersBothWaysForEveryHostStreamHandle) {
  cudaStream_t created = nullptr, engine = nullptr;
  MI_CUDA_CHECK(cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking));
  MI_CUDA_CHECK(cudaStreamCreateWithFlags(&engine, cudaStreamNonBlocking));
  StreamHandoff handoff;
  mi::DeviceBuffer buf(4 * sizeof(int));
  int* d = static_cast<int*>(buf.get());
  for (const cudaStream_t host : {created, static_cast<cudaStream_t>(nullptr), cudaStreamLegacy, cudaStreamPerThread}) {
    const int zero[4] = {0, 0, 0, 0};
    buf.upload(zero, sizeof zero);
    // The host's slow producer; the engine must see its result.
    mi::testing::launch_spin(30'000'000, host);
    mi::testing::launch_add(d, 1, 41, host);
    handoff.acquire(host, engine);
    mi::testing::launch_copy_plus_one(d, d + 1, engine);  // 42 iff after the host's add
    mi::testing::launch_spin(20'000'000, engine);
    mi::testing::launch_add(d + 2, 1, 7, engine);
    handoff.release(engine, host);
    EXPECT_EQ(cudaStreamQuery(engine), cudaSuccess);  // the engine's work has finished on return
    mi::testing::launch_copy_plus_one(d + 2, d + 3, host);  // 8 iff after the engine's add
    MI_CUDA_CHECK(cudaStreamSynchronize(host));
    int out[4];
    buf.download(out, sizeof out);
    EXPECT_EQ(out[1], 42) << "host stream " << host;
    EXPECT_EQ(out[3], 8) << "host stream " << host;
  }
  MI_CUDA_CHECK(cudaStreamDestroy(created));
  MI_CUDA_CHECK(cudaStreamDestroy(engine));
}

// ---- pinned staging -----------------------------------------------------------------------------------------------

TEST_F(GpuExchange, StagingGrowsWithinTheBudgetAndNeverShrinks) {
  PinnedStaging s;
  EXPECT_EQ(s.bytes(), 0u);
  s.ensure(kMiB, 2 * kMiB);
  const mi::runtime::StagingBlock b = s.block();
  EXPECT_EQ(s.bytes(), kMiB);
  EXPECT_TRUE(b.owned);
  cudaPointerAttributes a{};
  MI_CUDA_CHECK(cudaPointerGetAttributes(&a, b.base));
  EXPECT_EQ(a.type, cudaMemoryTypeHost);
  s.ensure(kMiB / 2, 0);  // smaller: kept, whatever the budget
  EXPECT_EQ(s.block().base, b.base);
  try {
    s.ensure(3 * kMiB, 2 * kMiB);
    FAIL();
  } catch (const StagingError& e) {
    EXPECT_TRUE(contains(e.what(), "exceeds the host memory budget")) << e.what();
    EXPECT_EQ(e.requested, 3 * kMiB);
  }
  EXPECT_EQ(s.bytes(), kMiB);  // a refusal keeps the block
  s.ensure(2 * kMiB, 2 * kMiB);
  EXPECT_EQ(s.bytes(), 2 * kMiB);
  // A failed page-locked allocation: the same error, its own message; the staging is empty then.
  const std::size_t huge = std::size_t{1} << 50;
  try {
    s.ensure(huge, huge);
    FAIL();
  } catch (const StagingError& e) {
    EXPECT_TRUE(contains(e.what(), "page-locked host allocation failed")) << e.what();
  }
  EXPECT_EQ(s.bytes(), 0u);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  s.ensure(kMiB, kMiB);
  s.release();
  EXPECT_EQ(s.bytes(), 0u);
}

TEST_F(GpuExchange, HostModeRoundTripThroughTheLayout) {
  // x in by H2D from its staging view, a device op, the outputs back by D2H into theirs: one synchronization.
  const std::size_t n = 4096;
  const StagingLayout l = StagingLayout::plan(n * sizeof(int), 2, n * sizeof(int), 0);
  PinnedStaging s;
  s.ensure(l.bytes, l.bytes);
  std::byte* base = s.block().base;
  auto* x = reinterpret_cast<int*>(base + l.x);
  for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<int>(i);
  cudaStream_t engine = nullptr;
  MI_CUDA_CHECK(cudaStreamCreateWithFlags(&engine, cudaStreamNonBlocking));
  mi::DeviceBuffer dx(n * sizeof(int));
  MI_CUDA_CHECK(cudaMemcpyAsync(dx.get(), x, n * sizeof(int), cudaMemcpyHostToDevice, engine));
  mi::testing::launch_add(static_cast<int*>(dx.get()), static_cast<int>(n), 5, engine);
  for (int k = 0; k < 2; ++k) {
    MI_CUDA_CHECK(cudaMemcpyAsync(base + l.out[static_cast<std::size_t>(k)], dx.get(), n * sizeof(int),
                                  cudaMemcpyDeviceToHost, engine));
  }
  MI_CUDA_CHECK(cudaStreamSynchronize(engine));
  for (int k = 0; k < 2; ++k) {
    const auto* o = reinterpret_cast<const int*>(base + l.out[static_cast<std::size_t>(k)]);
    for (std::size_t i = 0; i < n; ++i) ASSERT_EQ(o[i], static_cast<int>(i) + 5) << k << " " << i;
  }
  MI_CUDA_CHECK(cudaStreamDestroy(engine));
}

}  // namespace
