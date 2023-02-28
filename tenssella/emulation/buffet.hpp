/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

// A simple serial emulator will not work because the decoupled loop nests will
// run ahead and violate buffet synchronization. There are two approaches:
// 1. Use a host-level multithreaded emulator with each transfer block running
//    in a different host thread. Read/Update locks are implemented with host-
//    level mutex locks.
// 2. Two-pass approach. The first pass runs through the decoupled loop nests
//    and populates a hierarchical spacce-time tree with events that must
//    happen at each hardware unit at each (space,time) coordinate. The second
//    pass walks through the space-time event tree and emulates the events.
//    Read/Update locks are implemented as flags and assertions. Note that we
//    schedule a ReadIU and its corresponding Update at the same time
//    coordinate. As weird as this is, it works because space-time trees are
//    decoupled for different tensors--except at the arithmetic unit, where the
//    Read/Update does happen atomically.
// We are going with approach 1.

#include <map>
#include <string>
#include <mutex>
#include <condition_variable>

template<typename T>
class Buffet
{
 private:
  enum class State
  {
    Empty, Ready, Locked
  };

  friend std::ostream& operator << (std::ostream& out, const State& state)
  {
    // TODO: move this outside this function.
    std::map<State, std::string> StateName =
      {
        { State::Empty, "Empty" },
        { State::Ready, "Ready" },
        { State::Locked, "Locked" }
      };

    out << StateName.at(state);
    return out;
  }

  // Ugh, this is broken because the CVs will only guard a state transition
  // and not the entire variable.

  struct EntrySynchronizer
  {
    State state = State::Empty;
    std::mutex mutex;
    std::condition_variable cv_state;
    size_t time_stamp;
  };

 private:
  std::string name_;
  std::mutex mutex_;
  std::map<std::string, EntrySynchronizer> entry_synchronizers_;
  std::map<std::string, T> contents_;
  std::size_t fail_count_ = 0;

  EntrySynchronizer& GetSynchronizer(std::string addr)
  {
    // Lock global buffet data structures.
    const std::lock_guard<std::mutex> lock(mutex_);

    // Synchronizers are instantiated on-demand by the first thread that touches
    // this address.
    if (entry_synchronizers_.find(addr) == entry_synchronizers_.end())
    {
      entry_synchronizers_[addr]; // Creates a new lock.
      entry_synchronizers_.at(addr).time_stamp = 0; //init timestamp to be 0
    }
    return entry_synchronizers_.at(addr);
  }

  T& GetContent(std::string addr)
  {
    // Lock global buffet data structures.
    const std::lock_guard<std::mutex> lock(mutex_);

    // Entries are instantiated on-demand by the first thread that touches
    // this address.
    if (contents_.find(addr) == contents_.end())
    {
      contents_[addr]; // Creates a new entry.
    }

    // Perform the fill.
    return contents_.at(addr);
  }

 public:
  Buffet() {}

  Buffet(const Buffet& other) :
      name_(other.name_),
      mutex_(),
      entry_synchronizers_(other.entry_synchronizers_),
      contents_(other.contents_)
  { }

  Buffet(std::string name) : name_(name) {}

  size_t getTimeStamp(std::string addr) {
    auto& entry_synchronizer = GetSynchronizer(addr);
    return entry_synchronizer.time_stamp;
  }

  void setTimeStamp(std::string addr, size_t t) {
    auto& entry_synchronizer = GetSynchronizer(addr);
    entry_synchronizer.time_stamp = t;
  }

  size_t getMaxTimeStamp() {
    size_t max_t = 0;
    for (auto& it: entry_synchronizers_) {
      max_t = std::max(max_t, it.second.time_stamp);
    }
    return max_t;
  }

  void fill(std::string addr, const T& val)
  {
    // Get/allocate the synchronization data for this entry.
    auto& entry_synchronizer = GetSynchronizer(addr);

    // Block until this location is in Empty state.
    std::unique_lock<std::mutex> entry_lock(entry_synchronizer.mutex);
    entry_synchronizer.cv_state.wait(entry_lock, [&](){ return entry_synchronizer.state == State::Empty; });

    // Perform the fill.
    GetContent(addr) = val;

    std::thread::id tid = std::this_thread::get_id();
    TRACE_LOCK(2);
    TRACE(2) << "[" << std::hex << tid << std::dec << "]    buffet " << name_ << " FILL " << addr << " = " << val << std::endl;
    TRACE_UNLOCK(2);

    // Switch to Ready state.
    entry_synchronizer.state = State::Ready;

    // Unlock the entry.
    entry_synchronizer.mutex.unlock();

    // Notify any read/read-iu threads that may have been waiting on me.
    entry_synchronizer.cv_state.notify_all();
  }

  T read_iu(std::string addr)
  {
    // Get/allocate the synchronization data for this entry.
    auto& entry_synchronizer = GetSynchronizer(addr);

    // Block until this location is in Ready state.
    std::unique_lock<std::mutex> entry_lock(entry_synchronizer.mutex);
    entry_synchronizer.cv_state.wait(entry_lock, [&](){ return entry_synchronizer.state == State::Ready; });

    // Perform the read.
    T val = GetContent(addr);

    std::thread::id tid = std::this_thread::get_id();
    TRACE_LOCK(2);
    TRACE(2) << "[" << std::hex << tid << std::dec << "]    buffet " << name_ << " READ_IU " << addr << " = " << val << std::endl;
    TRACE_UNLOCK(2);

    // Since this is a Read-IU, switch to Locked state.
    entry_synchronizer.state = State::Locked;

    // Unlock the entry.
    entry_lock.unlock();

    // Notify any updater threads that may have been waiting on me.
    entry_synchronizer.cv_state.notify_all();

    // Done.
    return val;
  }

  T read(std::string addr)
  {
    // Get/allocate the synchronization data for this entry.
    auto& entry_synchronizer = GetSynchronizer(addr);

    // Block until this location is in Ready state.
    std::unique_lock<std::mutex> entry_lock(entry_synchronizer.mutex);
    entry_synchronizer.cv_state.wait(entry_lock, [&](){ return entry_synchronizer.state == State::Ready; });

    // Since this is a simple Read, do not change states.

    // Perform the read.
    T val = GetContent(addr);

    std::thread::id tid = std::this_thread::get_id();
    TRACE_LOCK(2);
    TRACE(2) << "[" << std::hex << tid << std::dec << "]    buffet " << name_ << " READ " << addr << " = " << val << std::endl;
    TRACE_UNLOCK(2);

    // Unlock the entry.
    entry_lock.unlock();

    // Done.
    return val;
  }

  void shrink(std::string addr)
  {
    // Get/allocate the synchronization data for this entry.
    auto& entry_synchronizer = GetSynchronizer(addr);

    // Block until this location is in Ready state.
    std::unique_lock<std::mutex> entry_lock(entry_synchronizer.mutex);
    entry_synchronizer.cv_state.wait(entry_lock, [&](){ return entry_synchronizer.state == State::Ready; });

    // Change state to empty.
    entry_synchronizer.state = State::Empty;

    std::thread::id tid = std::this_thread::get_id();
    TRACE_LOCK(2);
    TRACE(2) << "[" << std::hex << tid << std::dec << "]    buffet " << name_ << " SHRINK " << addr << std::endl;
    TRACE_UNLOCK(2);

    // Unlock the entry.
    entry_lock.unlock();

    // Notify any threads that may have been waiting on the empty state.
    entry_synchronizer.cv_state.notify_all();

    // Done.
  }

  T drain(std::string addr)
  {
    // Get/allocate the synchronization data for this entry.
    auto& entry_synchronizer = GetSynchronizer(addr);

    // Block until this location is in Ready state.
    std::unique_lock<std::mutex> entry_lock(entry_synchronizer.mutex);
    entry_synchronizer.cv_state.wait(entry_lock, [&](){ return entry_synchronizer.state == State::Ready; });

    // Change state to empty.
    entry_synchronizer.state = State::Empty;

    // Perform the read.
    T val = GetContent(addr);

    std::thread::id tid = std::this_thread::get_id();
    TRACE_LOCK(2);
    TRACE(2) << "[" << std::hex << tid << std::dec << "]    buffet " << name_ << " DRAIN " << addr << " = " << val << std::endl;
    TRACE_UNLOCK(2);

    // Unlock the entry.
    entry_lock.unlock();

    // Notify any threads that may have been waiting on the empty state.
    entry_synchronizer.cv_state.notify_all();

    // Done.
    return val;
  }

  void update(std::string addr, const T& val)
  {
    // Get/allocate the synchronization data for this entry.
    auto& entry_synchronizer = GetSynchronizer(addr);

    // Block until this entry is in Locked state.
    std::unique_lock<std::mutex> entry_lock(entry_synchronizer.mutex);
    entry_synchronizer.cv_state.wait(entry_lock, [&](){ return entry_synchronizer.state == State::Locked; });

    // Perform the update.
    GetContent(addr) = val;

    std::thread::id tid = std::this_thread::get_id();
    TRACE_LOCK(2);
    TRACE(2) << "[" << std::hex << tid << std::dec << "]    buffet " << name_ << " UPDATE " << addr << " = " << val << std::endl;
    TRACE_UNLOCK(2);

    // Switch to ready state.
    entry_synchronizer.state = State::Ready;

    // Unlock the entry.
    entry_lock.unlock();

    // Notify any read/read_iu threads that may have been waiting on ready state.
    entry_synchronizer.cv_state.notify_all();
  }

  void reduce_update(std::string addr, const T& val)
  {
    // Get/allocate the synchronization data for this entry.
    auto& entry_synchronizer = GetSynchronizer(addr);

    // Block until this entry is in Locked state.
    std::unique_lock<std::mutex> entry_lock(entry_synchronizer.mutex);
    entry_synchronizer.cv_state.wait(entry_lock, [&](){ return entry_synchronizer.state == State::Locked; });

    // Perform the reduce-update.
    GetContent(addr) += val;

    std::thread::id tid = std::this_thread::get_id();
    TRACE_LOCK(2);
    TRACE(2) << "[" << std::hex << tid << std::dec << "]    buffet " << name_ << " REDUCE-UPDATE " << addr << " = " << val << std::endl;
    TRACE_UNLOCK(2);

    // Switch to ready state.
    entry_synchronizer.state = State::Ready;

    // Unlock the entry.
    entry_lock.unlock();

    // Notify any read/read_iu threads that may have been waiting on ready state.
    entry_synchronizer.cv_state.notify_all();
  }

  void validate(std::string addr, const T& val)
  {
    // Get/allocate the synchronization data for this entry.
    auto& entry_synchronizer = GetSynchronizer(addr);

    // Block until this location is in Ready state.
    std::unique_lock<std::mutex> entry_lock(entry_synchronizer.mutex);
    if (entry_synchronizer.state != State::Ready)
    {
      std::thread::id tid = std::this_thread::get_id();
      TRACE_LOCK(0);
      TRACE(0) << "[" << std::hex << tid << std::dec << "]    buffet " << name_ << " ERROR: VALIDATE "
               << addr << "/" << val << " in non-ready state = " << entry_synchronizer.state << std::endl;
      TRACE_UNLOCK(0);
      std::exit(1);
    }

    // entry_synchronizer.cv_state.wait(entry_lock, [&](){ return entry_synchronizer.state == State::Ready; });

    // Since we perform a simple Read here, do not change states.

    // Read the target and validate.
    T val_dst = GetContent(addr);

    std::thread::id tid = std::this_thread::get_id();
    TRACE_LOCK(1);
    TRACE(1) << "[" << std::hex << tid << std::dec << "]    buffet " << name_ << " VALIDATE " << addr << " = " << val_dst << " ";
    if (val == val_dst)
    {
      TRACE(1) << "PASS";
    }
    else
    {
      TRACE(1) << "FAIL (expected " << val << ")";
      fail_count_++;
    }
    TRACE(1) << std::endl;
    TRACE_UNLOCK(1);

    // Unlock the entry.
    entry_lock.unlock();
  }

  std::size_t FailCount()
  {
    return fail_count_;
  }

  void Dump()
  {
    std::thread::id tid = std::this_thread::get_id();
    TRACE_LOCK(0);
    TRACE(0) << "  [" << std::hex << tid << std::dec << "] buffet " << name_ << " DUMP" << std::endl;

    for (auto& e: contents_)
    {
      auto addr = e.first;
      auto val = e.second;
      auto& sync = entry_synchronizers_.at(addr);
      if (sync.state != State::Empty)
      {
        TRACE(0) << "    [" << addr << "]: " << sync.state << ": " << val << std::endl;
      }
    }

    TRACE_UNLOCK(0);
  }
};

