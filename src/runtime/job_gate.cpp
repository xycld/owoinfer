#include "runtime/job_gate.h"

namespace mi::runtime {

JobGate::Job::~Job() { gate_.finish_job(); }

JobGate::Job JobGate::begin(Kind kind) {
  std::lock_guard<std::mutex> lock(mu_);
  if (ending_) {
    if (kind == Kind::kEngine) throw Busy("a component run is ending");
    throw Refused(ending_session_ ? "the session is closing" : "the run is ending");
  }
  if (kind != Kind::kEngine && !session_open_) throw Refused("no session is open");
  if (kind == Kind::kRun && !run_open_) throw Refused("no run is open");
  if (busy_) throw Busy("another call is running on this engine");
  if (kind != Kind::kRun && run_open_) throw Busy("a component run is open: end it first (mi_run_end)");
  busy_ = true;
  job_thread_ = std::this_thread::get_id();
  cancel_.store(false);
  return Job(*this);
}

void JobGate::finish_job() {
  std::lock_guard<std::mutex> lock(mu_);
  cancel_.store(false);
  busy_ = false;
  job_thread_ = {};
  cv_.notify_all();
}

void JobGate::open_session() {
  std::lock_guard<std::mutex> lock(mu_);
  if (!busy_ || job_thread_ != std::this_thread::get_id()) throw std::logic_error("JobGate::open_session outside its job");
  session_open_ = true;
}

void JobGate::open_run() {
  std::lock_guard<std::mutex> lock(mu_);
  if (!busy_ || job_thread_ != std::this_thread::get_id()) throw std::logic_error("JobGate::open_run outside its job");
  run_open_ = true;
}

bool JobGate::session_open() const {
  std::lock_guard<std::mutex> lock(mu_);
  return session_open_;
}

bool JobGate::run_open() const {
  std::lock_guard<std::mutex> lock(mu_);
  return run_open_;
}

bool JobGate::start_ending(std::unique_lock<std::mutex>& lock, bool session) {
  if (busy_ && job_thread_ == std::this_thread::get_id()) {
    throw Busy(session ? "the session cannot be closed from a callback of its running call"
                       : "the run cannot be ended from a callback of its running call");
  }
  cv_.wait(lock, [&] { return !ending_; });  // a second ender waits for the first
  if (!(session ? session_open_ : run_open_)) return false;
  ending_ = true;
  ending_session_ = session;
  if (busy_) cancel_.store(true);
  cv_.wait(lock, [&] { return !busy_; });
  busy_ = true;  // the ender's own job: nothing else starts while it frees
  job_thread_ = std::this_thread::get_id();
  return true;
}

void JobGate::finish_ending(bool session) {
  std::lock_guard<std::mutex> lock(mu_);
  run_open_ = false;
  if (session) session_open_ = false;
  ending_ = false;
  ending_session_ = false;
  cancel_.store(false);
  busy_ = false;
  job_thread_ = {};
  cv_.notify_all();
}

bool JobGate::end_run(const std::function<void()>& free_state) {
  {
    std::unique_lock<std::mutex> lock(mu_);
    if (!start_ending(lock, false)) return false;
  }
  try {
    free_state();
  } catch (...) {
    finish_ending(false);
    throw;
  }
  finish_ending(false);
  return true;
}

bool JobGate::end_session(const std::function<void()>& free_run, const std::function<void()>& free_session) {
  bool run = false;
  {
    std::unique_lock<std::mutex> lock(mu_);
    if (!start_ending(lock, true)) return false;
    run = run_open_;
  }
  try {
    if (run) free_run();
    free_session();
  } catch (...) {
    finish_ending(true);
    throw;
  }
  finish_ending(true);
  return true;
}

}  // namespace mi::runtime
