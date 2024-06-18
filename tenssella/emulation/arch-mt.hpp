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

#include <fstream>
#include <map>
#include <mutex>
#include <csignal>

std::mutex global_lock;

//int TRACE_LEVEL = (char* trace_level = getenv("TENSSELLA_EMU_TRACE_LEVEL")) == NULL ? 0 : atoi(trace_level);
std::ofstream NULL_STREAM;
std::ofstream TRACE_STREAM;
int TRACE_LEVEL;

std::ostream& TRACE(int level)
{
  if (level <= TRACE_LEVEL)
    return std::cerr;
  else
    return NULL_STREAM;
}

void TRACE_LOCK(int level)
{
  if (level <= TRACE_LEVEL)
    global_lock.lock();
}

void TRACE_UNLOCK(int level)
{
  if (level <= TRACE_LEVEL)
    global_lock.unlock();
}




#include "buffet.hpp"

// Ugh... Forward declaration of arch hierarchy
template<typename T> class Arch;
template<typename T> Arch<T>* arch_;

void __attribute__((constructor)) init();

#include "transfer-engine.hpp"

template<typename T>
class PhysicalLevel
{
 private:
  std::mutex mutex_;
  std::string name_ = "";
  std::map<int, Buffet<T>> buffets_;
  std::map<int, TransferEngine<T>> transfer_engines_;

 public:
  //PhysicalLevel() {}
  PhysicalLevel(const PhysicalLevel& other) :
      mutex_(),
      name_(other.name_),
      buffets_(other.buffets_),
      transfer_engines_(other.transfer_engines_)
  { }

  PhysicalLevel(std::string name) : name_(name) {}

  Buffet<T>& operator [](int space_coord)
  {
    // Buffets are instantiated on-demand by the first thread that touches
    // this coordinate.
    mutex_.lock();
    if (buffets_.find(space_coord) == buffets_.end())
    {
      char buffet_name[256];
      sprintf(buffet_name, "%s[%d].buffet", name_.c_str(), space_coord);
      //buffets_[space_coord] = Buffet<T>(std::string(buffet_name));
      //buffets_.emplace(space_coord, Buffet<T>(std::string(buffet_name)));
      buffets_.emplace(space_coord, std::string(buffet_name));
    }
    mutex_.unlock();
    return buffets_[space_coord];
  }

  TransferEngine<T>& operator ()(int space_coord)
  {
    // TransferEngines are instantiated on-demand by the first thread that touches
    // this coordinate.
    mutex_.lock();
    if (transfer_engines_.find(space_coord) == transfer_engines_.end())
    {
      char transfer_engine_name[256];
      sprintf(transfer_engine_name, "%s[%d].transfer_engine", name_.c_str(), space_coord);
      transfer_engines_.emplace(space_coord, std::string(transfer_engine_name));
    }
    mutex_.unlock();
    return transfer_engines_[space_coord];
  }

  size_t GetLatency(int sid) {
    auto it = buffets_.find(sid);
    if (it == buffets_.end()) {
      std::cerr << "ERROR: could not find buffet [" << name_ << "] with sid = " << sid << std::endl;
      assert(false);
    }
    return it->second.getMaxTimeStamp();
  }

  void Optimizations()
  {
    for (auto& kv: transfer_engines_)
    {
      kv.second.Optimizations();
    }
  }

  void Run()
  {
    for (auto& kv: transfer_engines_)
    {
      kv.second.Run();
    }
  }

  void Wait()
  {
    for (auto& kv: transfer_engines_)
    {
      kv.second.Wait();
    }
  }

  void Dump()
  {
    mutex_.lock();
    std::cerr << "Level " << name_ << " dumping buffets:" << std::endl;
    for (auto& b: buffets_)
    {
      b.second.Dump();
    }
    mutex_.unlock();
  }
};

template<typename T>
class Arch
{
 private:
  std::mutex mutex_;
  std::map<std::string, PhysicalLevel<T>> levels_;

 public:
  PhysicalLevel<T>& operator [](std::string level_name)
  {
    // Levels are instantiated on-demand by the first thread that touches
    // this level.
    mutex_.lock();
    if (levels_.find(level_name) == levels_.end())
    {
      //levels_[level_name] = PhysicalLevel<T>(level_name);
      //levels_.emplace(level_name, PhysicalLevel<T>(level_name));
      levels_.emplace(level_name, level_name);
    }
    mutex_.unlock();
    return levels_.at(level_name);
  }

  void Run()
  {
    for (auto& kv: levels_) {
      kv.second.Optimizations();
    }
    for (auto& kv: levels_)
    {
      kv.second.Run();
    }
  }

  void Wait()
  {
    for (auto& kv: levels_)
    {
      kv.second.Wait();
    }
  }

  void Reset()
  {
    // Destroy all levels *except* those called __val__.
    mutex_.lock();
    for (auto it = levels_.begin(); it != levels_.end(); it++)
    {
      if (it->first.compare("__val__") != 0)
        it = levels_.erase(it);
    }
    mutex_.unlock();
  }

  void PrintLatency(std::string level) {
    mutex_.lock();
    auto it = levels_.find(level);
    if (it == levels_.end()) {
      std::cerr << "ERROR: could not find output buffet level -- " << level << std::endl;
      assert(false);
    } else {
      auto latency = it->second.GetLatency(0);
      std::cerr << std::endl << "Test Emulation Latency = " << latency << std::endl;
    }
    mutex_.unlock();
  }

  void PrintValidationResult()
  {
    // Find the buffet level called __val__.
    mutex_.lock();
    auto it = levels_.find("__val__");
    if (it == levels_.end())
    {
      std::cerr << "ERROR: could not find validation buffet __val__." << std::endl;
      assert(false);
    }
    else
    {
      std::size_t fail_count = it->second[0].FailCount();
      if (fail_count == 0)
        std::cerr << "Validation PASSED." << std::endl;
      else {
        std::cerr << "Validation FAILED with " << fail_count << " errors." << std::endl;
        assert(false);
      }
    }
    mutex_.unlock();
  }

  void Dump()
  {
    mutex_.lock();
    for (auto it = levels_.begin(); it != levels_.end(); it++)
    {
      it->second.Dump();
    }
    mutex_.unlock();
  }
};

void handler(int s)
{
  (void) s;
  (*arch_<float>).Dump();
  exit(1);
}

void register_handler()
{
  struct sigaction action;
  action.sa_handler = handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  sigaction(SIGINT, &action, NULL);
}

void init_tracing()
{
  NULL_STREAM.setstate(std::ios_base::badbit);
  char* trace_level = getenv("TENSSELLA_EMU_TRACE_LEVEL");
  if (trace_level != NULL)
    TRACE_LEVEL = atoi(trace_level);
  else
    TRACE_LEVEL = 0;
}

void init()
{
  register_handler();
  init_tracing();
}
