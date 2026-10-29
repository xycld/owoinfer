// runtime/job_gate.h: one job at a time, the open run and session, the cancel flag, and the ending sequence (design
// §4): an ender cancels the job in flight and waits for it, new calls are refused once the ending began, a second ender
// waits for the first, and an ender on the running job's thread is refused.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "runtime/job_gate.h"

namespace {

using mi::runtime::JobGate;
using Kind = JobGate::Kind;
using namespace std::chrono_literals;

// A gate with an open session (and run).
void open(JobGate& g, bool run) {
  JobGate::Job job = g.begin(Kind::kEngine);
  g.open_session();
  if (run) g.open_run();
}

template <class E, class F>
std::string refusal(F&& f) {
  try {
    f();
  } catch (const E& e) {
    return e.what();
  }
  return "no refusal";
}

TEST(JobGate, OneJobAtATime) {
  JobGate g;
  {
    JobGate::Job a = g.begin(Kind::kEngine);
    EXPECT_EQ(refusal<JobGate::Busy>([&] { JobGate::Job b = g.begin(Kind::kEngine); }), "another call is running on this engine");
  }
  JobGate::Job c = g.begin(Kind::kEngine);  // free again
}

TEST(JobGate, SessionAndRunStates) {
  JobGate g;
  EXPECT_EQ(refusal<JobGate::Refused>([&] { JobGate::Job j = g.begin(Kind::kSession); }), "no session is open");
  EXPECT_THROW(g.open_session(), std::logic_error);  // only from a job
  open(g, false);
  EXPECT_TRUE(g.session_open());
  EXPECT_EQ(refusal<JobGate::Refused>([&] { JobGate::Job j = g.begin(Kind::kRun); }), "no run is open");
  { JobGate::Job j = g.begin(Kind::kSession); }
  { JobGate::Job j = g.begin(Kind::kEngine); }  // generate between runs
  {
    JobGate::Job j = g.begin(Kind::kSession);
    g.open_run();
  }
  EXPECT_TRUE(g.run_open());
  EXPECT_EQ(refusal<JobGate::Busy>([&] { JobGate::Job j = g.begin(Kind::kEngine); }),
            "a component run is open: end it first (mi_run_end)");
  EXPECT_THROW({ JobGate::Job j = g.begin(Kind::kSession); }, JobGate::Busy);
  { JobGate::Job j = g.begin(Kind::kRun); }
  int freed = 0;
  EXPECT_TRUE(g.end_run([&] { ++freed; }));
  EXPECT_EQ(freed, 1);
  EXPECT_FALSE(g.run_open());
  EXPECT_FALSE(g.end_run([&] { ++freed; }));  // nothing open: MI_OK, nothing done
  EXPECT_EQ(freed, 1);
  EXPECT_EQ(refusal<JobGate::Refused>([&] { JobGate::Job j = g.begin(Kind::kRun); }), "no run is open");
}

TEST(JobGate, CancelFlagLandsOnTheRunningJobOnly) {
  JobGate g;
  g.cancel();  // between jobs: cleared by the next start
  {
    JobGate::Job j = g.begin(Kind::kEngine);
    EXPECT_FALSE(g.cancel_flag()->load());
    g.cancel();
    EXPECT_TRUE(g.cancel_flag()->load());
  }
  EXPECT_FALSE(g.cancel_flag()->load());  // cleared at the end too
}

TEST(JobGate, EndingCancelsTheJobInFlightAndRefusesNewCalls) {
  for (int iteration = 0; iteration < 200; ++iteration) {
    JobGate g;
    open(g, true);
    std::atomic<bool> started{false}, freed{false}, ran_after_free{false};
    std::thread step([&] {
      JobGate::Job j = g.begin(Kind::kRun);
      started = true;
      while (!g.cancel_flag()->load()) std::this_thread::yield();  // a step until its cancel
      if (freed) ran_after_free = true;
    });
    while (!started) std::this_thread::yield();
    std::thread ender([&] { EXPECT_TRUE(g.end_run([&] { freed = true; })); });
    ender.join();
    step.join();
    EXPECT_TRUE(freed);
    EXPECT_FALSE(ran_after_free);
    EXPECT_FALSE(g.run_open());
    EXPECT_TRUE(g.session_open());
  }
}

TEST(JobGate, CallsDuringTheEndingAreRefusedAndASecondEnderWaits) {
  JobGate g;
  open(g, true);
  std::promise<void> in_free;
  std::promise<void> release_free;
  std::shared_future<void> release = release_free.get_future().share();
  std::thread first([&] {
    EXPECT_TRUE(g.end_run([&] {
      in_free.set_value();
      release.wait();
    }));
  });
  in_free.get_future().wait();
  EXPECT_EQ(refusal<JobGate::Refused>([&] { JobGate::Job j = g.begin(Kind::kRun); }), "the run is ending");
  EXPECT_EQ(refusal<JobGate::Refused>([&] { JobGate::Job j = g.begin(Kind::kSession); }), "the run is ending");
  EXPECT_THROW({ JobGate::Job j = g.begin(Kind::kEngine); }, JobGate::Busy);
  std::atomic<bool> second_done{false};
  int second_freed = 0;
  std::thread second([&] {
    EXPECT_FALSE(g.end_run([&] { ++second_freed; }));  // waits, then finds nothing open
    second_done = true;
  });
  std::this_thread::sleep_for(20ms);
  EXPECT_FALSE(second_done);
  release_free.set_value();
  first.join();
  second.join();
  EXPECT_EQ(second_freed, 0);
  EXPECT_EQ(refusal<JobGate::Refused>([&] { JobGate::Job j = g.begin(Kind::kRun); }), "no run is open");
}

TEST(JobGate, EnderOnTheRunningJobsThreadIsBusy) {
  JobGate g;
  open(g, true);
  int freed = 0;
  {
    JobGate::Job j = g.begin(Kind::kRun);
    EXPECT_THROW(g.end_run([&] { ++freed; }), JobGate::Busy);
    EXPECT_THROW(g.end_session([&] { ++freed; }, [&] { ++freed; }), JobGate::Busy);
  }
  EXPECT_EQ(freed, 0);
  EXPECT_TRUE(g.run_open());
}

TEST(JobGate, SessionEndEndsItsRunFirst) {
  JobGate g;
  open(g, true);
  std::string order;
  EXPECT_TRUE(g.end_session([&] { order += "run "; }, [&] { order += "session"; }));
  EXPECT_EQ(order, "run session");
  EXPECT_FALSE(g.session_open());
  EXPECT_FALSE(g.run_open());
  EXPECT_EQ(refusal<JobGate::Refused>([&] { JobGate::Job j = g.begin(Kind::kSession); }), "no session is open");
  EXPECT_FALSE(g.end_session([] {}, [] {}));
}

TEST(JobGate, FailedFreeStillEndsTheRun) {
  JobGate g;
  open(g, true);
  EXPECT_THROW(g.end_run([] { throw std::runtime_error("cuda"); }), std::runtime_error);
  EXPECT_FALSE(g.run_open());
  JobGate::Job j = g.begin(Kind::kSession);  // usable afterwards
}

// The review's probe: mi_engine_free (end_session, close_engine, then the gate freed) while another thread's end_run
// waits for the session's ending. close_engine returns only once that waiter has left the gate (ASan: no use after
// free; the ASan build of this file runs it).
TEST(JobGate, CloseEngineWaitsForEveryWaiterBeforeTheFree) {
  for (int iteration = 0; iteration < 500; ++iteration) {
    auto g = std::make_unique<JobGate>();
    open(*g, true);
    std::atomic<bool> in_free{false}, second_started{false};
    JobGate* raw = g.get();
    std::thread first([&] {
      raw->end_session(
          [&] {
            in_free = true;
            while (!second_started) std::this_thread::yield();
            for (int k = 0; k < 2000; ++k) std::this_thread::yield();
          },
          [] {});
      raw->close_engine();
    });
    while (!in_free) std::this_thread::yield();
    std::thread second([&] {
      second_started = true;
      EXPECT_FALSE(raw->end_run([] {}));  // the session's ending ended the run; or the gate closed
    });
    first.join();
    g.reset();
    second.join();
  }
  JobGate g;
  open(g, false);
  g.close_engine();
  EXPECT_NE(refusal<JobGate::Refused>([&] { JobGate::Job j = g.begin(Kind::kEngine); }).find("being freed"), std::string::npos);
  EXPECT_FALSE(g.end_session([] {}, [] {}));
}

// A session closing (or a run ending) waits for a job outside the session (a generate) without cancelling it.
TEST(JobGate, EndingsNeverCancelAGenerate) {
  for (const bool session : {true, false}) {
    JobGate g;
    open(g, false);
    std::atomic<bool> started{false}, saw_cancel{false};
    std::thread generate([&] {
      JobGate::Job j = g.begin(Kind::kEngine);
      started = true;
      for (int k = 0; k < 200; ++k) {
        if (g.cancel_flag()->load()) saw_cancel = true;
        std::this_thread::sleep_for(100us);
      }
    });
    while (!started) std::this_thread::yield();
    if (session) {
      EXPECT_TRUE(g.end_session([] {}, [] {}));
    } else {
      EXPECT_FALSE(g.end_run([] {}));  // no run open: nothing to end, after the generate finished
    }
    generate.join();
    EXPECT_FALSE(saw_cancel);
  }
}

// The engine's teardown (begin_teardown): the generate in flight is cancelled at once, and the teardown's end_session
// waits only for it to stop; on_job_thread is true on the job's thread only.
TEST(JobGate, TeardownCancelsAGenerate) {
  JobGate g;
  open(g, false);
  EXPECT_FALSE(g.on_job_thread());
  std::atomic<bool> started{false}, saw_cancel{false}, on_thread{false};
  std::thread generate([&] {
    JobGate::Job j = g.begin(Kind::kEngine);
    on_thread = g.on_job_thread();
    started = true;
    for (int k = 0; k < 100000 && !saw_cancel; ++k) {
      if (g.cancel_flag()->load()) saw_cancel = true;
      std::this_thread::sleep_for(100us);
    }
  });
  while (!started) std::this_thread::yield();
  EXPECT_FALSE(g.on_job_thread());
  g.begin_teardown();
  EXPECT_TRUE(g.end_session([] {}, [] {}));
  generate.join();
  EXPECT_TRUE(on_thread);
  EXPECT_TRUE(saw_cancel);
  g.close_engine();
}

// end_run racing a begin_run: the ender waits for the session call in flight and ends the run it opened.
TEST(JobGate, EndRunEndsTheRunABeginRunInFlightOpens) {
  for (int iteration = 0; iteration < 100; ++iteration) {
    JobGate g;
    open(g, false);
    std::atomic<bool> started{false}, freed{false};
    std::thread begin_run([&] {
      JobGate::Job j = g.begin(Kind::kSession);
      started = true;
      std::this_thread::sleep_for(200us);
      g.open_run();
    });
    while (!started) std::this_thread::yield();
    EXPECT_TRUE(g.end_run([&] { freed = true; }));
    begin_run.join();
    EXPECT_TRUE(freed);
    EXPECT_FALSE(g.run_open());
  }
}

TEST(JobGate, ReadRunNeedsAnOpenRunNotEnding) {
  JobGate g;
  open(g, false);
  EXPECT_EQ(refusal<JobGate::Refused>([&] { g.read_run([] {}); }), "no run is open");
  {
    JobGate::Job j = g.begin(Kind::kSession);
    g.open_run();
  }
  int reads = 0;
  g.read_run([&] { ++reads; });
  EXPECT_EQ(reads, 1);
  g.end_run([&] { EXPECT_EQ(refusal<JobGate::Refused>([&] { g.read_run([] {}); }), "the run is ending"); });
}

}  // namespace
