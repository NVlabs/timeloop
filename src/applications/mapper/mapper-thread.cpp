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

#include <ncurses.h>

#include "applications/mapper/mapper-thread.hpp"

bool gTerminate = false;

enum class Betterness
{
  Better,
  SlightlyBetter,
  SlightlyWorse,
  Worse
};

static std::uint64_t SumStats(problem::PerDataSpace<std::uint64_t>& data, problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces)
{
  if (pv != problem::GetShape()->NumDataSpaces)
  {
    return data.at(pv);
  }
  else
  {
    std::uint64_t stat = 0;
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      stat += SumStats(data, problem::Shape::DataSpaceID(pvi));
    }
    return stat;
  }
}

static double Cost(const model::Topology::Stats& stats, const std::string metric)
{
  double cost;
  if (metric == "delay")
  {
    cost = static_cast<double>(stats.cycles);
  }
  else if (metric == "energy")
  {
    cost = stats.energy;
  }
  else if (metric == "last-level-accesses")
  {
    cost = stats.last_level_accesses;
  }
  else if (metric.compare(0, 9, "accesses-") == 0)
  {
    unsigned level = unsigned(atoi(metric.substr(9).c_str()));
    cost = stats.accesses.at(level);
  }
  else
  {
    assert(metric == "edp");
    cost = (stats.energy * stats.cycles);
  }
  return cost;
}

static Betterness IsBetterRecursive_(const model::Topology::Stats& candidate, const model::Topology::Stats& incumbent,
                                     const std::vector<std::string>::const_iterator metric,
                                     const std::vector<std::string>::const_iterator end)
{
  const double tolerance = 0.001;

  double candidate_cost = Cost(candidate, *metric);
  double incumbent_cost = Cost(incumbent, *metric);

  // Compute % improvement relative to incumbent. We need to
  // special-case cost == 0 to avoid a divide-by-zero error. Note that
  // cost == 0 is a legitimate cost for a mapping. Also note that lower
  // cost is better.
  double absolute_improvement = incumbent_cost - candidate_cost;
  double relative_improvement = incumbent_cost == 0 ?
    (candidate_cost == 0 ? 0 : absolute_improvement / candidate_cost) :
    absolute_improvement / incumbent_cost;

  if (fabs(relative_improvement) > tolerance)
  {
    // We have a clear winner.
    if (relative_improvement > 0)
      return Betterness::Better;
    else
      return Betterness::Worse;
  }
  else
  {
    // Within tolerance range, try to recurse.
    if (std::next(metric) == end)
    {
      // Base case. NOTE! Equality is categorized as SlightlyWorse (prefers incumbent).
      if (relative_improvement > 0)
        return Betterness::SlightlyBetter;
      else
        return Betterness::SlightlyWorse;
    }
    else
    {
      // Recursive call.
      Betterness lsm = IsBetterRecursive_(candidate, incumbent, std::next(metric), end);
      if (lsm == Betterness::Better || lsm == Betterness::Worse)
        return lsm;
      // NOTE! Equality is categorized as SlightlyWorse (prefers incumbent).
      else if (relative_improvement > 0)
        return Betterness::SlightlyBetter;
      else
        return Betterness::SlightlyWorse;
    }      
  }
}

static inline bool IsBetter(const model::Topology::Stats& candidate, const model::Topology::Stats& incumbent,
                            const std::vector<std::string>& metrics)
{
  Betterness b = IsBetterRecursive_(candidate, incumbent, metrics.begin(), metrics.end());
  return (b == Betterness::Better || b == Betterness::SlightlyBetter);
}

bool EvaluationResult::UpdateIfBetter(const EvaluationResult& other, const std::vector<std::string>& metrics)
{
  bool updated = false;
  if (other.valid &&
      (!valid || IsBetter(other.stats, stats, metrics)))
  {
    valid = true;
    mapping = other.mapping;
    stats = other.stats;
    updated = true;
  }
  return updated;
}

//--------------------------------------------//
//              Failure Tracking              //
//--------------------------------------------//

std::map<FailClass, std::string> FailClassToString =
{
  { FailClass::Fanout, "Fanout" },
  { FailClass::Capacity, "Capacity" }
};

std::ostream& operator << (std::ostream& out, const FailClass& fail_class)
{
  out << FailClassToString.at(fail_class);
  return out;
}

//--------------------------------------------//
//               Mapper Thread                //
//--------------------------------------------//

MapperThread::Stats::Stats() :
    distribution(0.0,1.0)
{
}

void MapperThread::Stats::UpdateFails(FailClass fail_class, std::string fail_reason, unsigned level, const Mapping& mapping)
{
  // Find the data corresponding to this fail class.
  auto fail_bucket_it = fail_stats.find(fail_class);
  if (fail_bucket_it == fail_stats.end())
  {
    // We've never seen this fail class before.
    std::map<unsigned, FailInfo> fail_bucket;
    fail_bucket[level] = { .count = 1, .mapping = mapping, .reason = fail_reason };
    fail_stats[fail_class] = fail_bucket;
  }
  else
  {
    // We've seen this fail class, see if this level has
    // failed in this class.
    auto& fail_bucket = fail_bucket_it->second;
    auto fail_info_it = fail_bucket.find(level);
    if (fail_info_it == fail_bucket.end())
    {
      // No, this is the first time this level has failed in
      // this fail class, create a new entry.
      fail_bucket[level] = { .count = 1, .mapping = mapping, .reason = fail_reason };
    }
    else
    {
      // This level has already failed in this class,
      // increment its count.
      fail_info_it->second.count += 1;
 
      // p(x) = prob. that I switch to x when it arrives
      // p(0) = 1

      // P(x) = prob. that x is finally selected.
      // 1/N = P(0) = p(0).(1-p(1)).(1-p(2))...(1-p(N-1))
      // 1/N = P(1) =        (p(1)).(1-p(2))...(1-p(N-1))

      // p(x).(1-p(x+1)) = p(x+1)
      // ...
      // => p(x+1) = p(x) / [1+p(x)]
      // ...
      // => p(x) = 1/(1+x)

      // Compute the probability of switching (we've already computed count=x+1)
      double prob = 1 / fail_info_it->second.count.convert_to<double>();

      // Probabilistically update the mapping.
      double roll = distribution(generator);
      if (roll < prob)
      {
        fail_info_it->second.mapping = mapping;
        fail_info_it->second.reason = fail_reason;
      }
    }
  }
}

MapperThread::MapperThread(
  unsigned thread_id,
  search::SearchAlgorithm* search,
  mapspace::MapSpace* mapspace,
  std::mutex* mutex,
  uint128_t search_size,
  std::uint32_t timeout,
  std::uint32_t victory_condition,
  uint128_t sync_interval,  
  uint128_t log_interval,
  bool log_index_factor_best,
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
  ) :
    thread_id_(thread_id),
    search_(search),
    mapspace_(mapspace),
    mutex_(mutex),
    search_size_(search_size),
    timeout_(timeout),
    victory_condition_(victory_condition),
    sync_interval_(sync_interval),
    log_interval_(log_interval), 
    log_index_factor_best_(log_index_factor_best),
    log_oaves_(log_oaves),
    log_stats_(log_stats),
    log_suboptimal_(log_suboptimal),
    log_stream_(log_stream),
    oaves_csv_file_(oaves_csv_file),
    live_status_(live_status),
    diagnostics_on_(diagnostics_on),
    penalize_consecutive_bypass_fails_(penalize_consecutive_bypass_fails),
    optimization_metrics_(optimization_metrics),
    arch_specs_(arch_specs),
    workload_(workload),
    sparse_optimizations_(sparse_optimizations),
    best_(best),
    thread_(),
    stats_()
{
}

void MapperThread::Start()
{
  // We can do this because std::thread is movable.
  thread_ = std::thread(&MapperThread::Run, this);
}

void MapperThread::Join()
{
  thread_.join();
}

const MapperThread::Stats& MapperThread::GetStats() const
{
  return stats_;
}

void MapperThread::Run()
{
  uint128_t total_mappings = 0;
  uint128_t valid_mappings = 0;
  uint128_t invalid_mappings_mapcnstr = 0;
  uint128_t invalid_mappings_eval = 0;
  std::uint32_t mappings_since_last_best_update = 0;

  const int ncurses_line_offset = 6;
      
  model::Engine engine;
  engine.Spec(arch_specs_);

  mapspace::ID prev_mapping_id;

  // =================
  // Main mapper loop.
  // =================
  while (true)
  {
    if (live_status_)
    {
      std::stringstream msg;

      msg << std::setw(3) << thread_id_ << std::setw(11) << total_mappings
          << std::setw(11) << (total_mappings - valid_mappings)  << std::setw(11) << valid_mappings
          << std::setw(11) << invalid_mappings_mapcnstr + invalid_mappings_eval
          << std::setw(11) << mappings_since_last_best_update;

      if (valid_mappings > 0)
      {
        msg << std::setw(10) << OUT_FLOAT_FORMAT << std::setprecision(2) << (stats_.thread_best.stats.utilization * 100) << "%"
            << std::setw(11) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats_.thread_best.stats.energy /
          stats_.thread_best.stats.algorithmic_computes;
      }

      mutex_->lock();
      mvaddstr(thread_id_ + ncurses_line_offset, 0, msg.str().c_str());
      refresh();
      mutex_->unlock();
    }

    // Termination conditions.
    bool terminate = false;

    if (gTerminate)
    {
      mutex_->lock();
      log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: "
                  << "global termination flag activated, terminating search."
                  << std::endl;
      mutex_->unlock();
      terminate = true;
    }

    if (search_size_ > 0 && valid_mappings == search_size_)
    {
      mutex_->lock();
      log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << search_size_
                  << " valid mappings found, terminating search."
                  << std::endl;
      mutex_->unlock();
      terminate = true;
    }

    if (victory_condition_ > 0 && mappings_since_last_best_update == victory_condition_)
    {
      mutex_->lock();
      log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << victory_condition_
                  << " suboptimal mappings found since the last upgrade, terminating search."
                  << std::endl;
      mutex_->unlock();
      terminate = true;
    }
        
    if ((invalid_mappings_mapcnstr + invalid_mappings_eval) > 0 &&
        (invalid_mappings_mapcnstr + invalid_mappings_eval) == timeout_)
    {
      mutex_->lock();
      log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << timeout_
                  << " invalid mappings (" << invalid_mappings_mapcnstr << " fanout, "
                  << invalid_mappings_eval << " capacity) found since the last valid mapping, "
                  << "terminating search." << std::endl;
      mutex_->unlock();
      terminate = true;
    }

    // Try to obtain the next mapping from the search algorithm.
    mapspace::ID mapping_id;
    if (!search_->Next(mapping_id))
    {
      mutex_->lock();
      log_stream_ << "[" << std::setw(3) << thread_id_ << "] STATEMENT: "
                  << "search algorithm is done, terminating search."
                  << std::endl;        
      mutex_->unlock();
      terminate = true;
    }

    if((log_index_factor_best_ || log_oaves_) && terminate && stats_.index_factor_best.valid)
    {
      auto topology =  engine.GetTopology();

      // print performance 
      if (log_index_factor_best_)
        PrintStats(topology, stats_.index_factor_best);
      if (log_oaves_)
        PrintOAVESStats(topology, stats_.index_factor_best);
      // reset the best for next permutation/bypassing
      stats_.index_factor_best.valid = false;
    }

    // Terminate.
    if (terminate)
    {
      if (live_status_)
      {
        mutex_->lock();
        mvaddstr(thread_id_ + ncurses_line_offset, 0, "-");
        refresh();
        mutex_->unlock();
      }
      break;
    }

    //
    // Periodically sync thread_best with global best.
    //
    if (total_mappings != 0 && sync_interval_ > 0 && total_mappings % sync_interval_ == 0)
    {
      mutex_->lock();
          
      // Sync from global best to thread_best.
      bool global_pulled = false;
      if (best_->valid)
      {
        if (stats_.thread_best.UpdateIfBetter(*best_, optimization_metrics_))
        {
          global_pulled = true;
        }
      }

      // Sync from thread_best to global best.
      if (stats_.thread_best.valid && !global_pulled)
      {
        best_->UpdateIfBetter(stats_.thread_best, optimization_metrics_);
      }
          
      mutex_->unlock();
    }

    //
    // Check if the only change vs. the previous mapping was in the Bypass
    // dimension. This is useful later.
    //
    bool only_bypass_changed = false;
    if (total_mappings > 1)
    {
      bool match = true;
      for (unsigned idim = 0; idim < unsigned(mapspace::Dimension::Num); idim++)
      {
        if (mapspace::Dimension(idim) != mapspace::Dimension::DatatypeBypass)
          match &= (mapping_id[idim] == prev_mapping_id[idim]);
      }
      only_bypass_changed = match;
    }
    prev_mapping_id = mapping_id;

    //
    // Begin Mapping. We do this in several stages with increasing algorithmic
    // complexity and attempt to bail out as quickly as possible at each stage.
    //
    bool success = true;

    // Stage 1: Construct a mapping from the mapping ID. This step can fail
    //          because the space of *legal* mappings isn't dense (unfortunately),
    //          so a mapping ID may point to an illegal mapping.
    Mapping mapping;

    auto construction_status = mapspace_->ConstructMapping(mapping_id, &mapping, !diagnostics_on_);
    success &= std::accumulate(construction_status.begin(), construction_status.end(), true,
                               [](bool cur, const mapspace::Status& status)
                               { return cur && status.success; });

    total_mappings++;

    if (!success)
    {
      invalid_mappings_mapcnstr++;
      if (diagnostics_on_)
      {
        for (unsigned level = 0; level < construction_status.size(); level++)
          if (!construction_status.at(level).success)
            stats_.UpdateFails(FailClass::Fanout, construction_status.at(level).fail_reason, level, mapping);
      }
      search_->Report(search::Status::MappingConstructionFailure);
      continue;
    }

    // Stage 2: (Re)Configure a hardware model to evaluate the mapping
    //          on, and run some lightweight pre-checks that the
    //          model can use to quickly reject a nest.
    //engine.Spec(arch_specs_);
    auto status_per_level = engine.PreEvaluationCheck(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
    success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus& status)
                               { return cur && status.success; });

    if (!success)
    {
      // Pre-evaluation failed.
      // If the only change in this mapping vs. the previous mapping was in
      // its dataspace bypass scheme, then we may not want to make this
      // failure count towards the timeout termination trigger.
      if (penalize_consecutive_bypass_fails_ || !only_bypass_changed)
      {
        invalid_mappings_eval++;
      }

      if (diagnostics_on_)
      {
        for (unsigned level = 0; level < status_per_level.size(); level++)
          if (!status_per_level.at(level).success)
            stats_.UpdateFails(FailClass::Capacity, status_per_level.at(level).fail_reason, level, mapping);
      }
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // Stage 3: Heavyweight evaluation.
    status_per_level = engine.Evaluate(mapping, workload_, sparse_optimizations_, !diagnostics_on_);
    success &= std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus& status)
                               { return cur && status.success; });
    if (!success)
    {
      // Evaluation failed.
      // If the only change in this mapping vs. the previous mapping was in
      // its dataspace bypass scheme, then we may not want to make this
      // failure count towards the timeout termination trigger.
      if (penalize_consecutive_bypass_fails_ || !only_bypass_changed)
      {
        invalid_mappings_eval++;
      }

      if (diagnostics_on_)
      {
        for (unsigned level = 0; level < status_per_level.size(); level++)
          if (!status_per_level.at(level).success)
            stats_.UpdateFails(FailClass::Capacity, status_per_level.at(level).fail_reason, level, mapping);
      }
      search_->Report(search::Status::EvalFailure);
      continue;
    }

    // SUCCESS!!!
    // Output results at log interval
    auto topology =  engine.GetTopology();
    auto stats = topology.GetStats();
    EvaluationResult result = { true, mapping, stats };

    if (log_index_factor_best_ && total_mappings != 0 && (stats_.index_factor_best.valid && (SumStats(stats_.index_factor_best.stats.tile_sizes[0]) != SumStats(stats.tile_sizes[0]))))
    {

      // print performance 
      PrintStats(topology, stats_.index_factor_best);

      // reset the best for next permutation/bypassing
      stats_.index_factor_best.valid = false;             
    }

    if (log_oaves_ && total_mappings != 0 && (stats_.index_factor_best.valid && (SumStats(stats_.index_factor_best.stats.tile_sizes[0]) != SumStats(stats.tile_sizes[0]))))
    {
      // print performance 
      PrintOAVESStats(topology, stats_.index_factor_best);

      // reset the best for next permutation/bypassing
      stats_.index_factor_best.valid = false;             
    }

    valid_mappings++;
    if (log_stats_)
    {
      mutex_->lock();
      log_stream_ << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                  << " " << invalid_mappings_mapcnstr + invalid_mappings_eval << std::endl;
      mutex_->unlock();
    }        
    invalid_mappings_mapcnstr = 0;
    invalid_mappings_eval = 0;
    search_->Report(search::Status::Success, Cost(stats, optimization_metrics_.at(0)));

    bool is_sparse_topology = !sparse_optimizations_->no_optimization_applied;
    if (log_suboptimal_ && total_mappings != 0 && log_interval_ > 0 && total_mappings % log_interval_ == 0)    
    {

      PrintStats(topology, result);      

      mutex_->lock();
      if (is_sparse_topology)
      {      
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]" 
                  << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization 
                  << " | pJ/Algorithmic-Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                  << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                  << " | " << mapping.PrintCompact()
                  << std::endl;
      }
      else
      {
        log_stream_ << "[" << std::setw(3) << thread_id_ << "]" 
                  << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization 
                  << " | pJ/Compute = " << std::setw(4) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                  << " | " << mapping.PrintCompact()
                  << std::endl;
      }
      mutex_->unlock();
    }

    // update index factor best
    stats_.index_factor_best.UpdateIfBetter(result, optimization_metrics_);

    // Is the new mapping "better" than the previous best mapping?
    if (stats_.thread_best.UpdateIfBetter(result, optimization_metrics_))
    {
      if (log_stats_)
      {
        // FIXME: improvement only captures the primary stat.
        double improvement = stats_.thread_best.valid ?
          (Cost(stats_.thread_best.stats, optimization_metrics_.at(0)) - Cost(stats, optimization_metrics_.at(0))) /
          Cost(stats_.thread_best.stats, optimization_metrics_.at(0)) : 1.0;
        mutex_->lock();
        log_stream_ << "[" << thread_id_ << "] UPDATE " << total_mappings << " " << valid_mappings
                    << " " << mappings_since_last_best_update << " " << improvement << std::endl;
        mutex_->unlock();
      }
        
      if (!log_suboptimal_)
      {
        mutex_->lock();
        if (is_sparse_topology)
        {      
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]" 
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization 
                    << " | pJ/Algorithmic-Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.algorithmic_computes
                    << " | pJ/Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << std::endl;
        }
        else
        {
          log_stream_ << "[" << std::setw(3) << thread_id_ << "]" 
                    << " Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << stats.utilization 
                    << " | pJ/Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << stats.energy / stats.actual_computes
                    << " | " << mapping.PrintCompact()
                    << std::endl;
        }        mutex_->unlock();
      }

      mappings_since_last_best_update = 0;
    }
    else
    {
      // If the only change in this mapping vs. the previous mapping was in
      // its dataspace bypass scheme, then we may not want to make this
      // failure count towards the timeout termination trigger.
      if (penalize_consecutive_bypass_fails_ || !only_bypass_changed)
      {
        mappings_since_last_best_update++;
      }
    }
  } // while ()
}

void MapperThread::PrintStats(model::Topology& topology, EvaluationResult& result)
{
  mutex_->lock();
  // print performance 
  if (result.valid) {
    std::cout << "---------------------------" << std::endl;
    std::string indent = "    ";
    std::cout << "=== Buffer Utilization ===" << std::endl;

    for (unsigned storage_level_id = 0; storage_level_id < topology.NumStorageLevels(); storage_level_id++)
    {
      unsigned inv_storage_level = topology.NumStorageLevels() - 1 - storage_level_id;
      std::shared_ptr<model::BufferLevel> buffer_level = topology.GetStorageLevel(inv_storage_level);
      std::cout << buffer_level->Name() << ":";
      for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
      {       
        auto pv = problem::Shape::DataSpaceID(pvi);
        auto utilized_instances = result.stats.utilized_instances.at(inv_storage_level).at(pv);
        auto utilized_capacity = result.stats.utilized_capacities.at(inv_storage_level).at(pv) * utilized_instances; 
        std::cout << " " << utilized_capacity;
      }
      std::cout << std::endl;
    }
    std::cout << "=== Operational Intensity ===" << std::endl;

    std::uint64_t total_min_traffic = 0;
    std::uint64_t total_output_size = 0;

    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);

      std::uint64_t utilized_capacity = -1;
      for (unsigned storage_level_id = 0; storage_level_id < topology.NumStorageLevels(); storage_level_id++)
      {
        unsigned inv_storage_level = topology.NumStorageLevels() - 1 - storage_level_id;
        utilized_capacity = result.stats.utilized_capacities.at(inv_storage_level).at(pv);
        // use the last non-bypassed level with capacity size not equal to 0
        if (utilized_capacity > 0)
        {
          break;
        }
      }  
      assert(utilized_capacity > 0);
      total_min_traffic += utilized_capacity;      
      if (problem::GetShape()->IsReadWriteDataSpace.at(pv)) {
        total_output_size += utilized_capacity;
      }
    }
    // std::cout <<  "total_min_traffic " << total_min_traffic << std::endl;
    // std::cout <<  "total_output_size " << total_output_size << std::endl;

    // out << indent << std::left << std::setw(70) << "Total elementwise ops";
    uint64_t total_elementwise_ops = result.stats.actual_computes;
    // out << ": " << total_elementwise_ops << std::endl; 

    // out << indent << std::left << std::setw(70) << "Total reduction ops";
    uint64_t total_reduction_ops = 0;

    if (tiling::gEnableFirstReadElision){
      total_reduction_ops = result.stats.actual_computes - total_output_size;
    } else{
      total_reduction_ops = result.stats.actual_computes;
    }
    // std::cout << ": " << total_reduction_ops << std::endl;  
    std::cout << indent << std::left << std::setw(70) << "Total ops";
    uint64_t total_ops = total_elementwise_ops + total_reduction_ops;
    std::cout << ": " << total_ops << std::endl;
    // std::cout << indent << std::left << std::setw(70) << "Total memory accesses required";
    // std::cout << ": " << total_min_traffic << std::endl; 
    // unsigned inv_storage_level = topology.NumStorageLevels() - 1;
    // std::shared_ptr<model::BufferLevel> buffer_level = topology.GetStorageLevel(inv_storage_level);      
    // auto op_per_byte = float(total_ops) / (buffer_level->GetSpecs().word_bits.Get() * total_min_traffic / 8);
    // std::cout << indent << std::left << std::setw(70) << "Optimal Op per Byte";
    // std::cout << ": " << op_per_byte << std::endl;

    for (unsigned i = 0; i < topology.NumStorageLevels(); i++)
    {        
      std::shared_ptr<model::BufferLevel> buffer_level = topology.GetStorageLevel(i);
      // auto stats = buffer_level->GetStats();
      std::cout << "--- " << buffer_level->Name() << " ---" << std::endl;

      auto scalar_accesses_space = result.stats.accesses.at(i);
      uint64_t total_scalar_access = SumStats(scalar_accesses_space);

      float op_per_byte = -1;

      if (total_scalar_access > 0)
      {
        // std::cout << indent << std::left << std::setw(70) << "Total accesses";
        // std::cout << ": " << total_scalar_access << std::endl;
        op_per_byte = float(total_ops) / (buffer_level->GetSpecs().word_bits.Get() * total_scalar_access / 8);
        // std::cout << indent << std::left << std::setw(70) << "Op per Byte";
        std::cout <<  total_scalar_access << " " << op_per_byte << std::endl;
      } else {  
          std::cout << "0 -1" << std::endl;
      }
    }         
    std::cout << "=== Summary ===" << std::endl;

    // std::cout << "GFLOPs (@1GHz): " << float(total_ops)  / result.stats.cycles << std::endl;
    // std::cout << "Utilization: " << result.stats.utilization << std::endl;
    std::cout << "Cycles: " << result.stats.cycles << std::endl;
    std::cout << "Energy: " << result.stats.energy / 1000000 << " uJ" << std::endl;
    // std::cout << "EDP(J*cycle): " << std::scientific << float(result.stats.cycles) * result.stats.energy / 1e12 << std::fixed << std::endl;
    // std::cout << "Area: " << result.stats.area / 1000000 << " mm^2" << std::endl;
    // std::cout << std::endl;
    std::cout << "=== Mapping ===" << std::endl;
    std::cout << result.mapping.PrintCompact() << std::endl; 
    std::cout << std::endl;
    mutex_->unlock();
  }    
}

void MapperThread::PrintOAVESStats(model::Topology& topology, EvaluationResult& result)
{
  mutex_->lock();
  // print performance
  if (result.valid && topology.NumStorageLevels() > 0) {
    // get the buffer utilization of the innermost memory level
    unsigned storage_level_id = 0;
    std::uint64_t total_utilization = 0;
    std::vector<std::uint64_t> utilizations;
    std::shared_ptr<model::BufferLevel> buffer_level = topology.GetStorageLevel(storage_level_id);

    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {       
      auto pv = problem::Shape::DataSpaceID(pvi);
      auto utilized_instances = result.stats.utilized_instances.at(storage_level_id).at(pv);
      auto utilized_capacity = result.stats.utilized_capacities.at(storage_level_id).at(pv) * utilized_instances; 

      auto utilized_capacity_byte = utilized_capacity * buffer_level->GetSpecs().word_bits.Get() / 8;
      utilizations.push_back(utilized_capacity_byte);
      total_utilization += utilized_capacity_byte;
    }

    std::uint64_t total_min_traffic = 0;
    std::uint64_t total_output_size = 0;

    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);

      std::uint64_t utilized_capacity = -1;
      for (unsigned storage_level_id = 0; storage_level_id < topology.NumStorageLevels(); storage_level_id++)
      {
        unsigned inv_storage_level = topology.NumStorageLevels() - 1 - storage_level_id;
        utilized_capacity = result.stats.utilized_capacities.at(inv_storage_level).at(pv);
        // use the last non-bypassed level with capacity size not equal to 0
        if (utilized_capacity > 0)
        {
          break;
        }
      }
      assert(utilized_capacity > 0);
      total_min_traffic += utilized_capacity;      
      if (problem::GetShape()->IsReadWriteDataSpace.at(pv)) {
        total_output_size += utilized_capacity;
      }
    }
    uint64_t total_elementwise_ops = result.stats.actual_computes;
    uint64_t total_reduction_ops = 0;

    if (tiling::gEnableFirstReadElision){
      total_reduction_ops = result.stats.actual_computes - total_output_size;
    } else{
      total_reduction_ops = result.stats.actual_computes;
    }
    uint64_t total_ops = total_elementwise_ops + total_reduction_ops;

    // Assume the DRAM is the last level
    auto last_storage_level = topology.NumStorageLevels() - 1;
    buffer_level = topology.GetStorageLevel(last_storage_level);

    auto scalar_accesses_space = result.stats.accesses.at(last_storage_level);
    uint64_t total_scalar_access = 0;
    std::vector<uint64_t> scalar_accesses;

    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      uint64_t per_dataspace_access = scalar_accesses_space[pv];
      scalar_accesses.push_back(per_dataspace_access);
      total_scalar_access += per_dataspace_access;
    }

    float op_per_byte = -1;

    if (total_scalar_access > 0)
    {
      op_per_byte = float(total_ops) / (buffer_level->GetSpecs().word_bits.Get() * total_scalar_access / 8);
    }
    oaves_csv_file_ << total_utilization << "," << op_per_byte << "," << total_scalar_access;
    for (uint64_t utilization: utilizations) {
      oaves_csv_file_<< "," << utilization;
    }
    for (uint64_t scalar_access: scalar_accesses) {
      oaves_csv_file_<< "," << scalar_access;
    }
    oaves_csv_file_ << "," << result.mapping.PrintCompact() << std::endl;
    mutex_->unlock();
  }
}
