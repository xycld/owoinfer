// runtime/job_gate.h: one job at a time, the open run and session, the cancel flag, and the ending sequence (design
// §4): an ender cancels the job in flight and waits for it, new calls are refused once the ending began, a second ender
// waits for the first, and an ender on the running job's thread is refused.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
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

}  // namespace
