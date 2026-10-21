#include "models/moss_dit/dit_weights.h"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "core/cuda_check.h"
#include "models/moss_dit/moss_dit.h"

namespace mi::moss_dit {
namespace {

using weights::Component;
using weights::TensorMap;
using weights::WeightGroup;

constexpr std::int64_t kQkvDim = 3 * kDim;
constexpr std::int64_t kUpGateDim = 2 * kFfnDim;
// A wait on the compute stream shorter than this is event overhead, not a stall.
constexpr double kStallMs = 0.05;

const WeightGroup& block_group(const weights::ModelIndex& index, int block) {
  const WeightGroup* g = index.find_group(Component::kTransformer, "block." + std::to_string(block));
  if (g == nullptr) throw std::runtime_error("mi::moss_dit: transformer group 'block." + std::to_string(block) + "' missing from the index");
  return *g;
}

std::string prefix(int block) { return "layers." + std::to_string(block) + "."; }

std::vector<std::string> fused_qkv(int block) {
  const std::string p = prefix(block);
  return {p + "proj.q.weight", p + "proj.k.weight", p + "proj.v.weight"};
}

// The tensors make_views() reads, with their shapes: checked up front for blocks that are only streamed.
void validate_block(const WeightGroup& g, int block) {
  const std::string p = prefix(block);
  const std::pair<const char*, std::vector<std::int64_t>> specs[] = {
      {"norm1.weight", {kDim}},          {"norm2.weight", {kDim}},          {"q_norm.weight", {kHeadDim}},
      {"k_norm.weight", {kHeadDim}},     {"proj.q.weight", {kDim, kDim}},   {"proj.k.weight", {kDim, kDim}},
      {"proj.v.weight", {kDim, kDim}},   {"proj.o.weight", {kDim, kDim}},   {"proj.ffn_up_gate.weight", {kUpGateDim, kDim}},
      {"proj.ffn_down.weight", {kDim, kFfnDim}}};
  for (const auto& [name, shape] : specs) {
    const auto it = std::find_if(g.tensors.begin(), g.tensors.end(), [&](const auto& t) { return t.name == p + name; });
    if (it == g.tensors.end()) throw std::runtime_error("mi::moss_dit: weight '" + p + name + "' missing");
    if (it->dtype != weights::DType::kBF16 || it->shape != shape) {
      throw std::runtime_error("mi::moss_dit: weight '" + p + name + "' has an unexpected type/shape");
    }
  }
}

void check_residency(const Residency& r) {
  if (r.resident_blocks < 0 || r.resident_blocks > kNumBlocks) {
    throw std::invalid_argument("mi::moss_dit: resident_blocks must be in [0, 42]");
  }
  if (r.streamed() && (r.buffers < 2 || r.reader_threads < 1 || r.ring_slots < 1 || r.chunk_bytes == 0)) {
    throw std::invalid_argument("mi::moss_dit: streaming needs >= 2 buffers and positive reader_threads, ring_slots, chunk_bytes");
  }
}

std::size_t ring_slot_bytes(const Residency& r) {
  return runtime::align_up(r.chunk_bytes + weights::kDirectAlign, weights::kDirectAlign);
}

int reader_count(const Residency& r) { return r.source == StreamSource::kFile ? r.reader_threads : 1; }

// Device block buffers a pass carves: no more than there are streamed blocks.
int buffer_count(const Residency& r) { return r.streamed() ? std::min(r.buffers, kNumBlocks - r.resident_blocks) : 0; }

// Submits the stream's enqueued work before this thread blocks on a host-side wait. WDDM batches submissions in the
// driver: work another thread (or the GPU) is waited for must not sit unsubmitted. cudaErrorNotReady (still running)
// is the expected answer, and real errors surface at the next checked call.
void flush(cudaStream_t s) { static_cast<void>(cudaStreamQuery(s)); }

float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  MI_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

}  // namespace

const char* stream_source_name(StreamSource s) {
  switch (s) {
    case StreamSource::kFile: return "file";
    case StreamSource::kHostPinned: return "host_pinned";
  }
  return "?";
}

std::bitset<kNumBlocks> resident_mask(int resident_blocks) {
  std::bitset<kNumBlocks> m;
  const int s = kNumBlocks - std::clamp(resident_blocks, 0, kNumBlocks);
  for (int i = 0; i < kNumBlocks; ++i) {
    if ((i + 1) * s / kNumBlocks == i * s / kNumBlocks) m.set(static_cast<std::size_t>(i));
  }
  return m;
}

void StreamStats::add(const StreamStats& o) {
  streamed_blocks += o.streamed_blocks;
  bytes += o.bytes;
  copy_s += o.copy_s;
  stall_s += o.stall_s;
  stalls += o.stalls;
  forward_s += o.forward_s;
}

namespace detail {

WeightGroup with_front(const WeightGroup& g, const std::vector<std::string>& names) {
  WeightGroup out = g;
  out.tensors.clear();
  for (const auto& n : names) {
    auto it = std::find_if(g.tensors.begin(), g.tensors.end(), [&](const auto& t) { return t.name == n; });
    if (it == g.tensors.end()) throw std::runtime_error("mi::moss_dit: tensor '" + n + "' missing from group " + g.name);
    out.tensors.push_back(*it);
  }
  for (const auto& t : g.tensors) {
    if (std::find(names.begin(), names.end(), t.name) == names.end()) out.tensors.push_back(t);
  }
  return out;
}

Tensor take(const TensorMap& m, const std::string& name, std::initializer_list<std::int64_t> shape) {
  auto it = m.find(name);
  if (it == m.end()) throw std::runtime_error("mi::moss_dit: weight '" + name + "' missing");
  const Tensor& t = it->second;
  bool ok = t.dtype == DType::kBF16 && t.rank == static_cast<int>(shape.size());
  int i = 0;
  for (std::int64_t d : shape) ok = ok && t.dim(i++) == d;
  if (!ok) throw std::runtime_error("mi::moss_dit: weight '" + name + "' has unexpected type/shape " + t.describe());
  return t;
}

Tensor fuse(const TensorMap& m, const std::vector<std::string>& names, std::int64_t rows, std::int64_t cols) {
  std::vector<Tensor> parts;
  for (const auto& n : names) parts.push_back(cols > 0 ? take(m, n, {rows, cols}) : take(m, n, {rows}));
  for (std::size_t i = 1; i < parts.size(); ++i) {
    if (static_cast<const char*>(parts[i].data) != static_cast<const char*>(parts[i - 1].data) + parts[i - 1].nbytes()) {
      throw std::runtime_error("mi::moss_dit: weights to fuse are not contiguous: " + names.front());
    }
  }
  const auto n = static_cast<std::int64_t>(parts.size()) * rows;
  return cols > 0 ? Tensor::view(parts[0].data, DType::kBF16, Device::kCUDA, {n, cols})
                  : Tensor::view(parts[0].data, DType::kBF16, Device::kCUDA, {n});
}

}  // namespace detail

// One reader thread's copy stream, ring and events.
struct BlockWeights::Reader {
  cudaStream_t stream = nullptr;
  std::unique_ptr<runtime::PinnedRing> ring;  // kFile
  std::vector<cudaEvent_t> copy_done;         // per buffer
  std::vector<cudaEvent_t> copy_begin, copy_end;  // timing, per sequence index
};

// Host-side handshake of one pass. CUDA waits bind to the record current when cudaStreamWaitEvent is called, so each
// side waits here until the other has recorded the event it is about to wait on: a reader refilling a buffer for
// sequence index j waits until block j - buffers was released (its compute-done event recorded); the compute thread
// waits until every reader recorded block j's copy-done event.
// abort is also read without the lock by the readers between chunks, and a reader that sees the cancel flag stops the
// pass, so a cancel lands within one chunk even while the compute thread waits for a block on slow storage.
struct BlockWeights::Sync {
  struct Aborted {};
  std::mutex mu;
  std::condition_variable cv;
  int released = 0;
  std::vector<int> enqueued;
  std::atomic<bool> abort{false};
  std::atomic<bool> cancel_seen{false};  // a reader stopped the pass for the cancel flag
  std::exception_ptr error;
  const std::atomic<bool>* cancel = nullptr;

  Sync(std::size_t readers, const std::atomic<bool>* cancel_flag) : enqueued(readers, 0), cancel(cancel_flag) {}

  bool cancelled() const { return cancel != nullptr && cancel->load(); }
  void fail(std::exception_ptr e) {
    std::lock_guard<std::mutex> lock(mu);
    if (!error) error = e;
    abort = true;
    cv.notify_all();
  }
  void stop() {
    std::lock_guard<std::mutex> lock(mu);
    abort = true;
    cv.notify_all();
  }
  // Between chunks (reader threads): throws Aborted once the pass stops or the cancel flag is set (then stopping it).
  void check() {
    if (abort.load()) throw Aborted{};
    if (cancelled()) {
      cancel_seen = true;
      stop();
      throw Aborted{};
    }
  }
  void wait_released(int n) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return abort.load() || released >= n; });
    if (abort.load()) throw Aborted{};
  }
  void set_released(int n) {
    std::lock_guard<std::mutex> lock(mu);
    released = n;
    cv.notify_all();
  }
  void set_enqueued(std::size_t reader, int n) {
    std::lock_guard<std::mutex> lock(mu);
    enqueued[reader] = n;
    cv.notify_all();
  }
  void wait_enqueued(int n) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return abort.load() || std::all_of(enqueued.begin(), enqueued.end(), [&](int e) { return e >= n; }); });
    if (error) std::rethrow_exception(error);
    if (abort.load()) {
      if (cancel_seen.load()) throw Cancelled();
      throw std::runtime_error("mi::moss_dit: block streaming aborted");
    }
  }
};

std::size_t BlockWeights::block_device_bytes(const weights::ModelIndex& index) {
  std::size_t bytes = 0;
  for (int i = 0; i < kNumBlocks; ++i) bytes = std::max(bytes, weights::group_device_bytes(block_group(index, i)));
  return bytes;
}

std::size_t BlockWeights::resident_bytes(const weights::ModelIndex& index, const Residency& residency) {
  check_residency(residency);
  const auto mask = resident_mask(residency.resident_blocks);
  std::size_t total = 0;
  for (int i = 0; i < kNumBlocks; ++i) {
    if (mask.test(static_cast<std::size_t>(i))) total += weights::group_device_bytes(block_group(index, i));
  }
  return total;
}

std::size_t BlockWeights::buffer_bytes(const weights::ModelIndex& index, const Residency& residency) {
  check_residency(residency);
  return static_cast<std::size_t>(buffer_count(residency)) * runtime::align_up(block_device_bytes(index));
}

std::size_t BlockWeights::host_bytes(const weights::ModelIndex& index, const Residency& residency) {
  check_residency(residency);
  if (!residency.streamed()) return 0;
  if (residency.source == StreamSource::kFile) {
    return static_cast<std::size_t>(residency.reader_threads) * static_cast<std::size_t>(residency.ring_slots) *
           ring_slot_bytes(residency);
  }
  const auto mask = resident_mask(residency.resident_blocks);
  std::size_t total = 0;
  for (int i = 0; i < kNumBlocks; ++i) {
    if (!mask.test(static_cast<std::size_t>(i))) total += weights::host_group_bytes(block_group(index, i));
  }
  return total;
}

BlockWeights::BlockWeights(const weights::ModelIndex& index, runtime::DeviceArena& arena, const Residency& residency,
                           const Upload& upload, const std::function<void(const WeightGroup&)>& before_group)
    : index_(index), arena_(arena), residency_(residency) {
  check_residency(residency);
  mask_ = resident_mask(residency.resident_blocks);
  nbuf_ = buffer_count(residency);
  block_bytes_ = runtime::align_up(block_device_bytes(index));
  seq_of_.assign(kNumBlocks, -1);
  resident_.resize(kNumBlocks);
  try {
    if (residency.streamed()) readers_ = std::make_unique<weights::ReaderCache>(index, residency.direct_io);
    for (int i = 0; i < kNumBlocks; ++i) {
      ordered_.push_back(detail::with_front(block_group(index, i), fused_qkv(i)));
      if (resident(i)) {
        resident_[static_cast<std::size_t>(i)] = make_views(i, upload(ordered_.back()));
        continue;
      }
      if (before_group) before_group(ordered_.back());
      validate_block(ordered_.back(), i);
      seq_of_[static_cast<std::size_t>(i)] = static_cast<int>(streamed_.size());
      streamed_.push_back(i);
      if (residency.source == StreamSource::kHostPinned) {
        host_.push_back(weights::load_group_to_host(*readers_, ordered_.back(), weights::HostMemory::kPinned));
      }
    }
    if (residency.streamed()) create_streaming_resources();
  } catch (...) {
    release();
    throw;
  }
}

BlockWeights::~BlockWeights() { release(); }

void BlockWeights::create_streaming_resources() {
  const auto n = static_cast<std::size_t>(reader_count(residency_));
  const std::size_t s = streamed_.size();
  for (std::size_t r = 0; r < n; ++r) {
    rd_.push_back(std::make_unique<Reader>());
    Reader& me = *rd_.back();
    MI_CUDA_CHECK(cudaStreamCreateWithFlags(&me.stream, cudaStreamNonBlocking));
    for (int b = 0; b < nbuf_; ++b) {
      MI_CUDA_CHECK(cudaEventCreateWithFlags(&me.copy_done.emplace_back(), cudaEventDisableTiming));
    }
    for (std::size_t j = 0; j < s; ++j) {
      MI_CUDA_CHECK(cudaEventCreate(&me.copy_begin.emplace_back()));
      MI_CUDA_CHECK(cudaEventCreate(&me.copy_end.emplace_back()));
    }
    if (residency_.source == StreamSource::kFile) {
      me.ring = std::make_unique<runtime::PinnedRing>(me.stream, static_cast<std::size_t>(residency_.ring_slots),
                                                      ring_slot_bytes(residency_));
    }
  }
  for (int b = 0; b < nbuf_; ++b) {
    MI_CUDA_CHECK(cudaEventCreateWithFlags(&compute_done_.emplace_back(), cudaEventDisableTiming));
  }
  MI_CUDA_CHECK(cudaEventCreateWithFlags(&buffers_free_, cudaEventDisableTiming));
  for (std::size_t j = 0; j < s; ++j) {
    MI_CUDA_CHECK(cudaEventCreate(&wait_begin_.emplace_back()));
    MI_CUDA_CHECK(cudaEventCreate(&wait_end_.emplace_back()));
  }
  MI_CUDA_CHECK(cudaEventCreate(&loop_begin_));
  MI_CUDA_CHECK(cudaEventCreate(&loop_end_));
}

void BlockWeights::release() noexcept {
  // Errors are not actionable here. Rings first (they wait for their own in-flight copies).
  for (auto& r : rd_) {
    r->ring.reset();
    if (r->stream != nullptr) cudaStreamSynchronize(r->stream);
    for (auto* v : {&r->copy_done, &r->copy_begin, &r->copy_end}) {
      for (cudaEvent_t e : *v) cudaEventDestroy(e);
    }
    if (r->stream != nullptr) cudaStreamDestroy(r->stream);
  }
  rd_.clear();
  for (auto* v : {&compute_done_, &wait_begin_, &wait_end_}) {
    for (cudaEvent_t e : *v) cudaEventDestroy(e);
    v->clear();
  }
  for (cudaEvent_t* e : {&buffers_free_, &loop_begin_, &loop_end_}) {
    if (*e != nullptr) cudaEventDestroy(*e);
    *e = nullptr;
  }
  host_.clear();
}

std::size_t BlockWeights::host_resident_bytes() const {
  std::size_t total = 0;
  for (const auto& g : host_) total += g.storage_bytes;
  return total;
}

BlockTensors BlockWeights::make_views(int block, const TensorMap& m) const {
  const std::string p = prefix(block);
  BlockTensors b;
  b.norm1 = detail::take(m, p + "norm1.weight", {kDim});
  b.norm2 = detail::take(m, p + "norm2.weight", {kDim});
  b.q_norm = detail::take(m, p + "q_norm.weight", {kHeadDim});
  b.k_norm = detail::take(m, p + "k_norm.weight", {kHeadDim});
  b.qkv = detail::fuse(m, fused_qkv(block), kDim, kDim);
  b.o = detail::take(m, p + "proj.o.weight", {kDim, kDim});
  b.up_gate = detail::take(m, p + "proj.ffn_up_gate.weight", {kUpGateDim, kDim});
  b.down = detail::take(m, p + "proj.ffn_down.weight", {kDim, kFfnDim});
  return b;
}

void BlockWeights::reader_loop(std::size_t r, const std::vector<std::vector<Chunk>>& chunks, Sync& sync) {
  Reader& me = *rd_[r];
  const int nbuf = nbuf_;
  const std::size_t stride = rd_.size();
  // The buffers were just carved out of the arena stack: work queued earlier on the compute stream may still use that
  // memory, so no copy may land before it is done.
  MI_CUDA_CHECK(cudaStreamWaitEvent(me.stream, buffers_free_, 0));
  for (std::size_t j = 0; j < chunks.size(); ++j) {
    const auto b = j % static_cast<std::size_t>(nbuf);
    if (j >= static_cast<std::size_t>(nbuf)) {
      flush(me.stream);
      sync.wait_released(static_cast<int>(j) - nbuf + 1);
      MI_CUDA_CHECK(cudaStreamWaitEvent(me.stream, compute_done_[b], 0));
    }
    MI_CUDA_CHECK(cudaEventRecord(me.copy_begin[j], me.stream));
    for (std::size_t i = r; i < chunks[j].size(); i += stride) {
      sync.check();
      if (hook_) hook_(static_cast<int>(r), streamed_[j]);
      const Chunk& c = chunks[j][i];
      if (residency_.source == StreamSource::kFile) {
        me.ring->copy_range(readers_->get(c.tensor->file), c.tensor->offset + c.rel, c.size, c.dst);
      } else {
        MI_CUDA_CHECK(cudaMemcpyAsync(c.dst, c.host, c.size, cudaMemcpyHostToDevice, me.stream));
      }
    }
    MI_CUDA_CHECK(cudaEventRecord(me.copy_end[j], me.stream));
    MI_CUDA_CHECK(cudaEventRecord(me.copy_done[b], me.stream));
    flush(me.stream);  // the compute stream waits on copy_done once the compute thread sees it enqueued
    sync.set_enqueued(r, static_cast<int>(j) + 1);
  }
}

void BlockWeights::collect_stats() {
  stats_pending_ = false;
  StreamStats s;
  s.streamed_blocks = static_cast<int>(streamed_.size());
  MI_CUDA_CHECK(cudaEventSynchronize(loop_end_));
  for (const auto& r : rd_) {
    if (!r->copy_end.empty()) MI_CUDA_CHECK(cudaEventSynchronize(r->copy_end.back()));
  }
  // Copy busy time: the union of the blocks' copy intervals (first reader start to last reader end of each block).
  double busy_ms = 0, cover_end = -1e30;
  for (std::size_t j = 0; j < streamed_.size(); ++j) {
    double begin = 1e30, end = -1e30;
    for (const auto& r : rd_) {
      begin = std::min<double>(begin, elapsed_ms(loop_begin_, r->copy_begin[j]));
      end = std::max<double>(end, elapsed_ms(loop_begin_, r->copy_end[j]));
    }
    begin = std::max(begin, cover_end);
    if (end > begin) busy_ms += end - begin;
    cover_end = std::max(cover_end, end);
    const double wait = elapsed_ms(wait_begin_[j], wait_end_[j]);
    s.stall_s += wait / 1e3;
    s.stalls += wait > kStallMs ? 1 : 0;
    for (const auto& t : ordered_[static_cast<std::size_t>(streamed_[j])].tensors) s.bytes += t.nbytes;
  }
  s.copy_s = busy_ms / 1e3;
  s.forward_s = elapsed_ms(loop_begin_, loop_end_) / 1e3;
  last_ = s;
  total_.add(s);
}

const StreamStats& BlockWeights::last_stats() {
  if (stats_pending_) collect_stats();
  return last_;
}

const StreamStats& BlockWeights::total_stats() {
  if (stats_pending_) collect_stats();
  return total_;
}

void BlockWeights::reset_total() {
  if (stats_pending_) collect_stats();
  total_ = StreamStats{};
}

BlockWeights::Pass::Pass(BlockWeights& w, cudaStream_t stream, const std::atomic<bool>* cancel) : w_(w), stream_(stream) {
  if (!w.residency_.streamed()) return;
  if (w.stats_pending_) w.collect_stats();  // the previous pass's events are about to be re-recorded
  const auto nbuf = static_cast<std::size_t>(w.nbuf_);
  std::vector<void*> bufs;
  for (std::size_t b = 0; b < nbuf; ++b) bufs.push_back(w.arena_.alloc_stack(w.block_bytes_));
  chunks_.resize(w.streamed_.size());
  std::vector<weights::UploadPiece> pieces;
  for (std::size_t j = 0; j < w.streamed_.size(); ++j) {
    const int block = w.streamed_[j];
    const WeightGroup& g = w.ordered_[static_cast<std::size_t>(block)];
    views_.push_back(w.make_views(block, weights::plan_group_upload(g, bufs[j % nbuf], w.block_bytes_, pieces)));
    for (const weights::UploadPiece& piece : pieces) {
      const std::byte* host = nullptr;
      if (!w.host_.empty()) host = static_cast<const std::byte*>(w.host_[j].tensors.at(piece.tensor->name).data);
      for (std::uint64_t rel = 0; rel < piece.tensor->nbytes; rel += w.residency_.chunk_bytes) {
        const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(w.residency_.chunk_bytes, piece.tensor->nbytes - rel));
        chunks_[j].push_back({piece.tensor, rel, n, piece.dst + rel, host == nullptr ? nullptr : host + rel});
      }
    }
  }
  w.sync_ = std::make_unique<Sync>(w.rd_.size(), cancel);
  int device = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  // Recorded before the readers start, so their waits bind to these records.
  MI_CUDA_CHECK(cudaEventRecord(w.buffers_free_, stream));
  MI_CUDA_CHECK(cudaEventRecord(w.loop_begin_, stream));
  try {
    for (std::size_t r = 0; r < w.rd_.size(); ++r) {
      threads_.emplace_back([this, r, device]() {
        Sync& sync = *w_.sync_;
        try {
          MI_CUDA_CHECK(cudaSetDevice(device));
          w_.reader_loop(r, chunks_, sync);
        } catch (const Sync::Aborted&) {
        } catch (...) {
          sync.fail(std::current_exception());
        }
      });
    }
  } catch (...) {
    abort();
    throw;
  }
}

BlockWeights::Pass::~Pass() {
  if (!finished_) abort();
}

void BlockWeights::Pass::abort() noexcept {
  finished_ = true;
  if (!w_.residency_.streamed()) return;
  if (w_.sync_) w_.sync_->stop();
  for (std::thread& t : threads_) t.join();
  threads_.clear();
  // Copies may still target buffers the caller is about to roll back; kernels may still read them.
  for (const auto& r : w_.rd_) cudaStreamSynchronize(r->stream);
  cudaStreamSynchronize(stream_);
  w_.sync_.reset();
  w_.stats_pending_ = false;
}

const BlockTensors& BlockWeights::Pass::acquire(int block) {
  if (w_.resident(block)) return w_.resident_[static_cast<std::size_t>(block)];
  const int j = w_.seq_of_[static_cast<std::size_t>(block)];
  const auto b = static_cast<std::size_t>(j % w_.nbuf_);
  // Recorded before the host waits for the readers to enqueue the block: the stall then covers the whole time the
  // compute stream is ready and the block is not (a transfer-bound reader enqueues its last chunk only just before
  // the copy completes, so a wait recorded after the enqueue would miss most of it).
  MI_CUDA_CHECK(cudaEventRecord(w_.wait_begin_[static_cast<std::size_t>(j)], stream_));
  flush(stream_);  // the previous blocks' kernels run (and release their buffers) while this thread waits
  w_.sync_->wait_enqueued(j + 1);
  for (const auto& r : w_.rd_) MI_CUDA_CHECK(cudaStreamWaitEvent(stream_, r->copy_done[b], 0));
  MI_CUDA_CHECK(cudaEventRecord(w_.wait_end_[static_cast<std::size_t>(j)], stream_));
  return views_[static_cast<std::size_t>(j)];
}

void BlockWeights::Pass::release(int block) {
  if (w_.resident(block)) return;
  const int j = w_.seq_of_[static_cast<std::size_t>(block)];
  MI_CUDA_CHECK(cudaEventRecord(w_.compute_done_[static_cast<std::size_t>(j % w_.nbuf_)], stream_));
  w_.sync_->set_released(j + 1);
}

void BlockWeights::Pass::finish() {
  if (!w_.residency_.streamed()) {
    finished_ = true;
    return;
  }
  for (std::thread& t : threads_) t.join();
  threads_.clear();
  // A reader can still fail after the compute thread's last wait: not lost.
  if (w_.sync_->error) {
    const std::exception_ptr e = w_.sync_->error;
    abort();
    std::rethrow_exception(e);
  }
  MI_CUDA_CHECK(cudaEventRecord(w_.loop_end_, stream_));
  w_.sync_.reset();
  w_.stats_pending_ = true;
  finished_ = true;
}

}  // namespace mi::moss_dit
