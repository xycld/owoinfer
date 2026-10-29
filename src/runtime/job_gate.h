// The one gate every job of an engine passes (component API design §4): at most one job at a time (generate, preload,
// release, and every component session call), the cancel flag the running job watches, and the open component run and
// session with their ending sequence. One gate per engine, so "a job runs" and "a run is open" are one state under one
// mutex: a generate can never start while a session call runs, nor a session call while a generate runs.
//
// The ending sequence (end_run, end_session):
//   1. under the mutex, the ending is marked: from then on every new session or run call is refused at once
//      (Refused: "the run is ending" / "the session is closing"), checked in the same critical section as the busy
//      test, so no call can start between the mark and the wait;
//   2. still under the mutex, a job in flight gets the cancel flag (its start cleared it, so the cancel lands on it);
//   3. the ender waits until no job runs (a job's end clears busy and notifies under the mutex, as its last access);
//   4. it frees the state (exclusively: the gate counts it as a job), then clears the ending and the open flag.
// A second ender arriving meanwhile waits for the first ending to complete, then does its own part (nothing when
// nothing is open any more). An ender on the running job's own thread (from its callback) is refused with Busy and
// does nothing.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace mi::runtime {

class JobGate {
 public:
  // Another job runs, a component run is open (for a job that needs none), or an ender called from the running job's
  // own thread: MI_ERR_BUSY.
  struct Busy : std::runtime_error {
    using std::runtime_error::runtime_error;
  };
  // No session or run is open, or it is ending: MI_ERR_INVALID_ARGUMENT.
  struct Refused : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
  };

  enum class Kind {
    kEngine,   // outside the session (generate, preload, release): not while a component run is open
    kSession,  // a session call between runs (encode, VAE, run begin): needs the session, no open run
    kRun,      // a call on the open run (step, staging adoption): needs the run, refused once its ending began
  };

  // The running job: busy from construction (JobGate::begin) to destruction; the cancel flag is cleared at both.
  class Job {
   public:
    ~Job();
    Job(const Job&) = delete;
    Job& operator=(const Job&) = delete;

   private:
    friend class JobGate;
    explicit Job(JobGate& gate) : gate_(gate) {}
    JobGate& gate_;
  };

  JobGate() = default;
  JobGate(const JobGate&) = delete;
  JobGate& operator=(const JobGate&) = delete;

  // Starts a job of `kind` on this thread, or throws Busy / Refused (nothing started).
  [[nodiscard]] Job begin(Kind kind);

  // The flag the running job's models watch. cancel() sets it: it stops the job running now; between jobs it is a
  // no-op (the next job's start clears it).
  const std::atomic<bool>* cancel_flag() const { return &cancel_; }
  void cancel() { cancel_.store(true); }

  // State changes made by the running job on this thread (std::logic_error otherwise): a session opened or a run begun.
  void open_session();
  void open_run();

  bool session_open() const;
  bool run_open() const;

  // The ending sequence above for the open run (end_run) or the session, its run first (end_session). `free_state`
  // frees what is ended; it runs with no job in flight. Returns false (nothing done) when nothing was open.
  bool end_run(const std::function<void()>& free_state);
  bool end_session(const std::function<void()>& free_run, const std::function<void()>& free_session);

 private:
  void finish_job();
  // Steps 1-3 for `session` (else the run); false when it is not open. Leaves the gate busy with the ender's job.
  bool start_ending(std::unique_lock<std::mutex>& lock, bool session);
  void finish_ending(bool session);

  mutable std::mutex mu_;
  std::condition_variable cv_;
  bool busy_ = false;
  std::thread::id job_thread_;
  bool session_open_ = false;
  bool run_open_ = false;
  bool ending_ = false;
  bool ending_session_ = false;
  std::atomic<bool> cancel_{false};
};

}  // namespace mi::runtime
