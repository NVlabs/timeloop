/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <thread>
#include <mutex>
#include <random>

#include "model/engine.hpp"
#include "model/sparse-optimization-info.hpp"
#include "search/search.hpp"

struct EvaluationResult
{
  bool valid = false;
  Mapping mapping;
  model::Topology::Stats stats;

  bool UpdateIfBetter(const EvaluationResult& other, const std::vector<std::string>& metrics);
};

//--------------------------------------------//
//              Failure Tracking              //
//--------------------------------------------//

enum class FailClass
{
  Fanout,
  Capacity
};

std::ostream& operator << (std::ostream& out, const FailClass& fail_class);

struct FailInfo
{
  uint128_t count = 0;
  Mapping mapping;
  std::string reason;
};

//--------------------------------------------//
//               Mapper Thread                //
//--------------------------------------------//

class MapperThread
{
 public:
  struct Stats
  {
    EvaluationResult thread_best;
    EvaluationResult index_factor_best;    
    std::map<FailClass, std::map<unsigned, FailInfo>> fail_stats;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;

    Stats();

    void UpdateFails(FailClass fail_class, std::string fail_reason, unsigned level, const Mapping& mapping);
  };

 private:
  // Configuration information sent from main thread.
  unsigned thread_id_;
  search::SearchAlgorithm* search_;
  mapspace::MapSpace* mapspace_;
  std::mutex* mutex_;
  uint128_t search_size_;
  std::uint32_t timeout_;
  std::uint32_t victory_condition_;
  uint128_t sync_interval_;
  uint128_t log_interval_;  
  bool log_oaves_;
  bool log_stats_;
  bool log_suboptimal_;
  std::ostream& log_stream_;
  std::ostream& oaves_csv_file_;
  bool live_status_;
  bool diagnostics_on_;
  bool penalize_consecutive_bypass_fails_;
  std::vector<std::string> optimization_metrics_;
  model::Engine::Specs arch_specs_;
  problem::Workload &workload_;
  sparse::SparseOptimizationInfo* sparse_optimizations_;
  EvaluationResult* best_;
    
  // Thread-local data (stats etc.).
  std::thread thread_;
  Stats stats_;
  
 public:
  MapperThread(
    unsigned thread_id,
    search::SearchAlgorithm* search,
    mapspace::MapSpace* mapspace,
    std::mutex* mutex,
    uint128_t search_size,
    std::uint32_t timeout,
    std::uint32_t victory_condition,
    uint128_t sync_interval,    
    uint128_t log_interval,
    bool log_oaves,
    bool log_stats,
    bool log_suboptimal,
    std::ostream& log_stream,
    std::ostream& oaves_csv_file,
    bool live_status,
    bool diagnostics_on,
    bool penalize_consecutive_bypass_fails,
    std::vector<std::string> optimization_metrics,
    model::Engine::Specs arch_specs,
    problem::Workload &workload,
    sparse::SparseOptimizationInfo* sparse_optimizations,
    EvaluationResult* best
    );

  void Start();

  void Join();

  const Stats& GetStats() const;

  void Run();

  void PrintStats(model::Topology& topology, EvaluationResult& result);

  void PrintOAVESStats(model::Topology& topology, EvaluationResult& result);
};
