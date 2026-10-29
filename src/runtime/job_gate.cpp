#include "runtime/job_gate.h"

namespace mi::runtime {

JobGate::Job::~Job() { gate_.finish_job(); }

JobGate::Job JobGate::begin(Kind kind) {
  std::lock_guard<std::mutex> lock(mu_);
  if (closed_) throw Refused("the engine is being freed");
  if (ending_) {
    if (kind == Kind::kEngine) throw Busy("a component run is ending");
    throw Refused(ending_session_ ? "the session is closing" : "the run is ending");
  }
  if (kind != Kind::kEngine && !session_open_) throw Refused("no session is open");
  if (kind == Kind::kRun && !run_open_) throw Refused("no run is open");
  if (busy_) throw Busy("another call is running on this engine");
  if (kind != Kind::kRun && run_open_) throw Busy("a component run is open: end it first (mi_run_end)");
  busy_ = true;
  job_kind_ = kind;
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

void JobGate::read_run(const std::function<void()>& read) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (ending_ && !ending_session_) throw Refused("the run is ending");
  if (ending_) throw Refused("the session is closing");
  if (!run_open_) throw Refused("no run is open");
  read();
}

void JobGate::locked(const std::function<void()>& f) const {
  std::lock_guard<std::mutex> lock(mu_);
  f();
}

bool JobGate::start_ending(std::unique_lock<std::mutex>& lock, bool session) {
  if (busy_ && job_thread_ == std::this_thread::get_id()) {
    throw Busy(session ? "the session cannot be closed from a callback of its running call"
                       : "the run cannot be ended from a callback of its running call");
  }
  // While waiting this thread is counted: close_engine() frees the gate only after every waiter has left.
  ++waiters_;
  struct Leave {
    JobGate& g;
    ~Leave() {
      --g.waiters_;
      g.cv_.notify_all();
    }
  } leave{*this};
  // A second ender waits for the first; a run ending also waits for a session call in flight (a begin_run), so the
  // run that call opens is ended too.
  cv_.wait(lock, [&] { return closed_ || (!ending_ && !(!session && busy_ && job_kind_ == Kind::kSession)); });
  if (closed_ || !(session ? session_open_ : run_open_)) return false;
  ending_ = true;
  ending_session_ = session;
  // Only the component calls the ending ends are cancelled: a run ending cancels a step, a session closing any session
  // or run call; a job outside the session (a generate) is waited for, except in the engine's teardown.
  if (busy_ && (teardown_ || job_kind_ == Kind::kRun || (session && job_kind_ == Kind::kSession))) cancel_.store(true);
  cv_.wait(lock, [&] { return !busy_; });
  busy_ = true;  // the ender's own job: nothing else starts while it frees
  job_kind_ = session ? Kind::kSession : Kind::kRun;
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
  struct Finish {
    JobGate& g;
    ~Finish() { g.finish_ending(false); }
  } finish{*this};
  free_state();
  return true;
}

bool JobGate::end_session(const std::function<void()>& free_run, const std::function<void()>& free_session) {
  bool run = false;
  {
    std::unique_lock<std::mutex> lock(mu_);
    if (!start_ending(lock, true)) return false;
    run = run_open_;
  }
  struct Finish {
    JobGate& g;
    ~Finish() { g.finish_ending(true); }
  } finish{*this};
  try {
    if (run) free_run();
  } catch (...) {
    free_session();  // the session's own state is freed even when freeing its run throws
    throw;
  }
  free_session();
  return true;
}

void JobGate::begin_teardown() {
  std::lock_guard<std::mutex> lock(mu_);
  teardown_ = true;
  if (busy_) cancel_.store(true);
}

bool JobGate::on_job_thread() const {
  std::lock_guard<std::mutex> lock(mu_);
  return busy_ && job_thread_ == std::this_thread::get_id();
}

void JobGate::close_engine() {
  std::unique_lock<std::mutex> lock(mu_);
  if (busy_ && job_thread_ == std::this_thread::get_id()) {
    throw Busy("the engine cannot be freed from a callback of its running call");
  }
  closed_ = true;
  if (busy_) cancel_.store(true);
  cv_.notify_all();
  cv_.wait(lock, [&] { return !busy_ && waiters_ == 0; });
}

}  // namespace mi::runtime
