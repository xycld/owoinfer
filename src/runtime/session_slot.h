// The slot of an engine's component session object (mi_engine's session): the calls that use the object, its
// replacement by a new session, and the engine's teardown. A call holds the object through a Ref for its whole length,
// so it never reaches another session than the one it found; a replacement waits for every Ref to go, and a call that
// arrives meanwhile waits for the replacement (counted, so a teardown never frees under it). Which session is open is
// the JobGate's state; the slot only keeps the object alive (a closed one stays until the next replacement).
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>

#include "runtime/job_gate.h"

namespace mi::runtime {

template <class T>
class SessionSlot {
 public:
  // A call's hold on the object. Empty when the slot never held one (`required`: JobGate::Refused, "no session is
  // open"). Waits while a replacement runs.
  class Ref {
   public:
    explicit Ref(const SessionSlot& slot, bool required = true) : slot_(slot) {
      std::unique_lock<std::mutex> lock(slot_.mu_);
      if (slot_.replacing_) {
        ++slot_.waiters_;  // counted while waiting: wait_idle() waits for it to leave
        slot_.cv_.wait(lock, [&] { return !slot_.replacing_; });
        --slot_.waiters_;
        slot_.cv_.notify_all();
      }
      if (!slot_.object_) {
        if (required) throw JobGate::Refused("no session is open");
        return;
      }
      object_ = slot_.object_.get();
      ++slot_.users_;
    }
    ~Ref() {
      if (object_ == nullptr) return;
      const std::lock_guard<std::mutex> lock(slot_.mu_);
      --slot_.users_;
      slot_.cv_.notify_all();
    }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    explicit operator bool() const { return object_ != nullptr; }
    T* operator->() const { return object_; }
    T* get() const { return object_; }

   private:
    const SessionSlot& slot_;
    T* object_ = nullptr;
  };

  SessionSlot() = default;
  SessionSlot(const SessionSlot&) = delete;
  SessionSlot& operator=(const SessionSlot&) = delete;

  // Replaces the object by make()'s: JobGate::Refused once begin_freeing() ran; JobGate::Busy while another
  // replacement runs, or from `check` (called under the slot's mutex, e.g. "a session is open"). Waits for every Ref to
  // go, destroys the old object (its destructor must find nothing open), then stores the new one. A make() that throws
  // leaves the slot empty.
  template <class Check, class Make>
  void replace(Check&& check, Make&& make) {
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (freeing_) throw JobGate::Refused("the engine is being freed");
      if (replacing_) throw JobGate::Busy("a session is being opened on this engine");
      check();
      replacing_ = true;
      cv_.wait(lock, [&] { return users_ == 0; });
    }
    struct Done {
      SessionSlot& s;
      ~Done() {
        const std::lock_guard<std::mutex> lock(s.mu_);
        s.replacing_ = false;
        s.cv_.notify_all();
      }
    } done{*this};
    object_.reset();
    object_ = make();
  }

  // The engine's teardown has begun: later replacements are refused.
  void begin_freeing() {
    const std::lock_guard<std::mutex> lock(mu_);
    freeing_ = true;
  }
  // Returns once no Ref is held, none waits and no replacement runs.
  void wait_idle() const {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return users_ == 0 && waiters_ == 0 && !replacing_; });
  }

 private:
  mutable std::mutex mu_;
  mutable std::condition_variable cv_;
  std::unique_ptr<T> object_;
  mutable int users_ = 0;
  mutable int waiters_ = 0;
  bool replacing_ = false;
  bool freeing_ = false;
};

}  // namespace mi::runtime
