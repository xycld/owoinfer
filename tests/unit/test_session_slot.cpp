// runtime/session_slot.h: the engine's component session object, its users, replacement and teardown (CPU; also run
// under TSan, see the P2 report).
#include "runtime/session_slot.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using mi::runtime::JobGate;
using Slot = mi::runtime::SessionSlot<int>;

// Blocks until `flag` is set (another thread's progress), yielding.
void until(const std::atomic<bool>& flag) {
  while (!flag.load()) std::this_thread::yield();
}

// Gives another thread time to reach a wait (it must still be blocked afterwards).
void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(20)); }

TEST(SessionSlot, RefOnAnEmptySlot) {
  Slot slot;
  EXPECT_THROW(Slot::Ref{slot}, JobGate::Refused);
  const Slot::Ref ref(slot, false);
  EXPECT_FALSE(ref);
  EXPECT_EQ(ref.get(), nullptr);
  slot.wait_idle();  // nothing held
}

TEST(SessionSlot, ReplaceDestroysTheOldObjectFirstAndARefSeesTheCurrentOne) {
  Slot slot;
  slot.replace([] {}, [] { return std::make_unique<int>(1); });
  {
    const Slot::Ref ref(slot);
    ASSERT_TRUE(ref);
    EXPECT_EQ(*ref.get(), 1);
  }
  std::vector<int> order;
  struct Tracked {
    std::vector<int>& order;
    int id;
    ~Tracked() { order.push_back(-id); }
  };
  mi::runtime::SessionSlot<Tracked> tracked;
  tracked.replace([] {}, [&] { return std::make_unique<Tracked>(order, 1); });
  tracked.replace([] {}, [&] {
    order.push_back(2);  // the old object is gone before the new one is made
    return std::make_unique<Tracked>(order, 2);
  });
  ASSERT_GE(order.size(), 2u);
  EXPECT_EQ(order[order.size() - 2], -1);
  EXPECT_EQ(order.back(), 2);
}

TEST(SessionSlot, ReplaceWaitsForEveryRef) {
  Slot slot;
  slot.replace([] {}, [] { return std::make_unique<int>(1); });
  std::atomic<bool> held{false}, release{false}, replaced{false};
  std::thread user([&] {
    const Slot::Ref ref(slot);
    held.store(true);
    until(release);
    EXPECT_EQ(*ref.get(), 1);  // still the object it found
  });
  until(held);
  std::thread opener([&] {
    slot.replace([] {}, [] { return std::make_unique<int>(2); });
    replaced.store(true);
  });
  settle();
  EXPECT_FALSE(replaced.load());
  release.store(true);
  user.join();
  opener.join();
  EXPECT_TRUE(replaced.load());
  const Slot::Ref ref(slot);
  EXPECT_EQ(*ref.get(), 2);
}

TEST(SessionSlot, AWaiterIsCountedDuringAReplacement) {
  Slot slot;
  std::atomic<bool> making{false}, finish{false}, got{false}, idle{false};
  std::thread opener([&] {
    slot.replace([] {}, [&] {
      making.store(true);
      until(finish);
      return std::make_unique<int>(7);
    });
  });
  until(making);
  std::atomic<bool> drop{false};
  std::thread caller([&] {
    const Slot::Ref ref(slot);  // waits for the replacement, counted
    EXPECT_EQ(*ref.get(), 7);
    got.store(true);
    until(drop);
  });
  std::thread freer([&] {
    slot.begin_freeing();
    slot.wait_idle();
    idle.store(true);
  });
  settle();
  EXPECT_FALSE(got.load());
  EXPECT_FALSE(idle.load());
  finish.store(true);  // the replacement that began before the freeing completes
  until(got);
  settle();
  EXPECT_FALSE(idle.load());  // the caller still holds its ref
  drop.store(true);
  opener.join();
  caller.join();
  freer.join();
  EXPECT_TRUE(idle.load());
}

TEST(SessionSlot, FreeingRefusesLaterReplacements) {
  Slot slot;
  slot.replace([] {}, [] { return std::make_unique<int>(1); });
  slot.begin_freeing();
  EXPECT_THROW(slot.replace([] {}, [] { return std::make_unique<int>(2); }), JobGate::Refused);
  const Slot::Ref ref(slot);  // a ref on the last object still works (the gate refuses the closed session's calls)
  EXPECT_EQ(*ref.get(), 1);
}

TEST(SessionSlot, BusyFromCheckOrAConcurrentReplacement) {
  Slot slot;
  EXPECT_THROW(slot.replace([] { throw JobGate::Busy("a session is open"); }, [] { return std::make_unique<int>(1); }),
               JobGate::Busy);
  EXPECT_FALSE(Slot::Ref(slot, false));
  std::atomic<bool> making{false}, finish{false};
  std::thread opener([&] {
    slot.replace([] {}, [&] {
      making.store(true);
      until(finish);
      return std::make_unique<int>(1);
    });
  });
  until(making);
  EXPECT_THROW(slot.replace([] {}, [] { return std::make_unique<int>(2); }), JobGate::Busy);
  finish.store(true);
  opener.join();
  EXPECT_EQ(*Slot::Ref(slot).get(), 1);
}

TEST(SessionSlot, AFailedMakeLeavesTheSlotEmptyAndUsable) {
  Slot slot;
  slot.replace([] {}, [] { return std::make_unique<int>(1); });
  EXPECT_THROW(slot.replace([] {}, []() -> std::unique_ptr<int> { throw std::invalid_argument("bad"); }),
               std::invalid_argument);
  EXPECT_FALSE(Slot::Ref(slot, false));
  slot.replace([] {}, [] { return std::make_unique<int>(3); });
  EXPECT_EQ(*Slot::Ref(slot).get(), 3);
  slot.wait_idle();
}

}  // namespace
