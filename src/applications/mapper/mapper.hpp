/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "mapspaces/mapspace-factory.hpp"
#include "search/search-factory.hpp"

#include <fstream>
#include <thread>
#include <mutex>
#include <iomanip>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/bitset.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

extern bool gTerminate;

struct EvaluationResult
{
  Mapping mapping;
  model::Engine engine;
};

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

class Application
{
 protected:
  problem::Workload workload_;

  model::Engine::Specs arch_specs_;
  mapspace::MapSpace* mapspace_;
  std::vector<mapspace::MapSpace*> split_mapspaces_;
  std::vector<search::SearchAlgorithm*> search_;

  uint128_t search_size_;
  std::uint32_t num_threads_;
  std::uint32_t timeout_;
  std::uint32_t victory_condition_;
  uint128_t sync_interval_;
  bool log_stats_;
  bool log_suboptimal_;

  std::vector<std::string> optimization_metrics_;

  char* cfg_string_;

  EvaluationResult best_;
  
 private:

  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0)
  {
    if(version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(workload_);
    }
  }

 public:

  Application(libconfig::Config& config)
  {
    try
    {
      // Problem configuration.
      libconfig::Setting& problem = config.lookup("problem");
      problem::ParseWorkload(problem, workload_);
      std::cout << "Problem configuration complete." << std::endl;

      // Architecture configuration.
      libconfig::Setting& arch = config.lookup("arch");
      arch_specs_ = model::Engine::ParseSpecs(arch);
      std::cout << "Architecture configuration complete." << std::endl;

      // Mapper (this application) configuration.
      libconfig::Setting& mapper = config.lookup("mapper");
      num_threads_ = std::thread::hardware_concurrency();
      if (mapper.lookupValue("num-threads", num_threads_))
      {
        std::cout << "Using threads = " << num_threads_ << std::endl;
      }
      else
      {
        std::cout << "Using all available hardware threads = " << num_threads_ << std::endl;
      }

      std::string metric;
      if (mapper.lookupValue("optimization-metric", metric))
      {
        optimization_metrics_ = { metric };
      }
      else if (mapper.exists("optimization-metrics"))
      {
        auto& metrics = mapper.lookup("optimization-metrics");
        assert(metrics.isArray());
        for (const std::string& m: metrics)
        {
          optimization_metrics_.push_back(m);
        }
      }
      else
      {
        optimization_metrics_ = { "edp" };
      }

      // Search size (divide between threads).
      std::uint32_t search_size = 0;
      mapper.lookupValue("search-size", search_size);
      mapper.lookupValue("search_size", search_size); // backwards compatibility.
      if (search_size > 0)
        search_size = 1 + (search_size - 1) / num_threads_;
      search_size_ = static_cast<uint128_t>(search_size);
    
      // Number of consecutive invalid mappings to trigger termination.
      timeout_ = 1000;
      mapper.lookupValue("timeout", timeout_);
      mapper.lookupValue("heartbeat", timeout_); // backwards compatibility.

      // Number of suboptimal valid mappings to trigger victory
      // (do NOT divide between threads).
      victory_condition_ = 500;
      mapper.lookupValue("victory-condition", victory_condition_);

      // Inter-thread sync interval.
      std::uint32_t sync_interval = 0;
      mapper.lookupValue("sync-interval", sync_interval);
      sync_interval_ = static_cast<uint128_t>(sync_interval);
    
      // Misc.
      log_stats_ = false;
      mapper.lookupValue("log-stats", log_stats_);    
      log_suboptimal_ = false;
    
      mapper.lookupValue("log-suboptimal", log_suboptimal_);
      mapper.lookupValue("log-all", log_suboptimal_); // backwards compatibility.
      std::cout << "Mapper configuration complete." << std::endl;

      // MapSpace configuration.
      libconfig::Setting& mapspace = config.lookup("mapspace");
      mapspace_ = mapspace::ParseAndConstruct(mapspace, arch_specs_, workload_);
      split_mapspaces_ = mapspace_->Split(num_threads_);
      std::cout << "Mapspace construction complete." << std::endl;

      // Search configuration.
      libconfig::Setting& search = config.lookup("mapper");
      for (unsigned t = 0; t < num_threads_; t++)
      {
        search_.push_back(search::ParseAndConstruct(search, split_mapspaces_.at(t), t));
      }
      std::cout << "Search configuration complete." << std::endl;
    }
    catch (const libconfig::SettingTypeException& e)
    {
      std::cerr << "ERROR: setting type exception at: " << e.getPath() << std::endl;
      exit(1);
    }
    catch (const libconfig::SettingNotFoundException& e)
    {
      std::cerr << "ERROR: setting not found: " << e.getPath() << std::endl;
      exit(1);
    }
    catch (const libconfig::SettingNameException& e)
    {
      std::cerr << "ERROR: setting name exception at: " << e.getPath() << std::endl;
      exit(1);
    }
    
    // Store the complete configuration in a string.
    std::size_t len;
    FILE* cfg_stream = open_memstream(&cfg_string_, &len);
    config.write(cfg_stream);
    fclose(cfg_stream);
  }

  // This class does not support being copied
  Application(const Application&) = delete;
  Application& operator=(const Application&) = delete;

  ~Application()
  {
    if (mapspace_)
    {
      delete mapspace_;
    }

    for (auto& search: search_)
    {
      if (search)
      {
        delete search;
      }
    }
  }

  static double Cost(const model::Engine& engine, const std::string metric)
  {
    if (metric == "delay")
    {
      return static_cast<double>(engine.Cycles());
    }
    else if (metric == "energy")
    {
      return engine.Energy();
    }
    else if (metric == "last-level-accesses")
    {
      auto num_storage_levels = engine.GetTopology().NumStorageLevels();
      return engine.GetTopology().GetStorageLevel(num_storage_levels-1)->Accesses();
    }
    else
    {
      assert(metric == "edp");
      return (engine.Energy() * engine.Cycles());
    }
  }

  enum class Betterness
  {
    Better,
    SlightlyBetter,
    SlightlyWorse,
    Worse
  };
  
  static inline bool IsBetter(const model::Engine& candidate, const model::Engine& incumbent,
                              const std::string metric)
  {
    std::vector<std::string> metrics = { metric };
    return IsBetter(candidate, incumbent, metrics);
  }

  static inline bool IsBetter(const model::Engine& candidate, const model::Engine& incumbent,
                              std::vector<std::string> metrics)
  {
    Betterness b = IsBetterRecursive_(candidate, incumbent, metrics.begin(), metrics.end());
    return (b == Betterness::Better || b == Betterness::SlightlyBetter);
  }

  static Betterness IsBetterRecursive_(const model::Engine& candidate, const model::Engine& incumbent,
                                       const std::vector<std::string>::iterator metric,
                                       const std::vector<std::string>::iterator end)
  {
    const double tolerance = 0.001;

    double candidate_cost = Cost(candidate, *metric);
    double incumbent_cost = Cost(incumbent, *metric);

    double relative_improvement = incumbent_cost == 0 ? 1.0 :
      (incumbent_cost - candidate_cost) / incumbent_cost;

    if (abs(relative_improvement) > tolerance)
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

  class MapperThread
  {
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
    bool log_stats_;
    bool log_suboptimal_;
    std::vector<std::string> optimization_metrics_;
    model::Engine::Specs arch_specs_;
    problem::Workload &workload_;
    EvaluationResult* best_;

    // Thread-local data.
    std::thread thread_;
    EvaluationResult thread_best_;    

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
      bool log_stats,
      bool log_suboptimal,
      std::vector<std::string> optimization_metrics,
      model::Engine::Specs arch_specs,
      problem::Workload &workload,
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
        log_stats_(log_stats),
        log_suboptimal_(log_suboptimal),
        optimization_metrics_(optimization_metrics),
        arch_specs_(arch_specs),
        workload_(workload),
        best_(best),
        thread_()
    {
    }

    void Start()
    {
      // We can do this because std::thread is movable.
      thread_ = std::thread(&MapperThread::Run, this);
    }

    void Join()
    {
      thread_.join();
    }

    Mapping& BestMapping()
    {
      return thread_best_.mapping;
    }

    model::Engine& BestMappedEngine()
    {
      return thread_best_.engine;
    }

    void Run()
    {
      uint128_t total_mappings = 0;
      uint128_t valid_mappings = 0;
      uint128_t invalid_mappings = 0;
      std::uint32_t mappings_since_last_best_update = 0;
      
      model::Engine engine;

      // Main loop: Keep ask the search pattern generator to generate an index into each
      // mapping sub-space, and repeat until it refuses.
      mapspace::ID mapping_id;
      while (search_->Next(mapping_id) && !gTerminate)
      {
        // Termination conditions.
        if (search_size_ > 0 && valid_mappings == search_size_)
        {
          mutex_->lock();
          std::cerr << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << search_size_
                    << " valid mappings found, terminating search."
                    << std::endl;
          mutex_->unlock();
          break;
        }

        if (victory_condition_ > 0 && mappings_since_last_best_update == victory_condition_)
        {
          mutex_->lock();
          std::cerr << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << victory_condition_
                    << " suboptimal mappings found since the last upgrade, terminating search."
                    << std::endl;
          mutex_->unlock();
          break;
        }
        
        if (invalid_mappings > 0 && invalid_mappings == timeout_)
        {
          mutex_->lock();
          std::cerr << "[" << std::setw(3) << thread_id_ << "] STATEMENT: " << timeout_
                    << " invalid mappings found since the last valid mapping, terminating search."
                    << std::endl;
          mutex_->unlock();
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
          if (best_->engine.IsSpecced())
          {
            if (!thread_best_.engine.IsSpecced() || IsBetter(best_->engine, thread_best_.engine, optimization_metrics_))
            {
              thread_best_.mapping = best_->mapping;
              thread_best_.engine = best_->engine;
              global_pulled = true;
            }
          }

          // Sync from thread_best to global best.
          if (thread_best_.engine.IsSpecced() && !global_pulled)
          {
            if (!best_->engine.IsSpecced() || IsBetter(thread_best_.engine, best_->engine, optimization_metrics_))
            {
              best_->mapping = thread_best_.mapping;
              best_->engine = thread_best_.engine;
            }            
          }
          
          mutex_->unlock();
        }

        //
        // Begin Mapping. We do this in several stages with increasing algorithmic
        // complexity and attempt to bail out as quickly as possible at each stage.
        //
        bool success = true;

        // Stage 1: Construct a mapping from the mapping ID. This step can fail
        //          because the space of *legal* mappings isn't dense (unfortunately),
        //          so a mapping ID may point to an illegal mapping.
        Mapping mapping;

        success &= mapspace_->ConstructMapping(mapping_id, &mapping);
        total_mappings++;

        if (!success)
        {
          invalid_mappings++;
          search_->Report(search::Status::MappingConstructionFailure);
          continue;
        }

        // Stage 2: (Re)Configure a hardware model to evaluate the mapping
        //          on, and run some lightweight pre-checks that the
        //          model can use to quickly reject a nest.
        engine.Spec(arch_specs_);
        success &= engine.PreEvaluationCheck(mapping, workload_);
        if (!success)
        {
          invalid_mappings++;
          search_->Report(search::Status::EvalFailure);
          continue;
        }

        // Stage 3: Heavyweight evaluation.
        success &= engine.Evaluate(mapping, workload_);
        if (!success)
        {
          invalid_mappings++;
          search_->Report(search::Status::EvalFailure);
          continue;
        }

        // SUCCESS!!!
        valid_mappings++;
        if (log_stats_)
        {
          mutex_->lock();
          std::cerr << "[" << thread_id_ << "] INVALID " << total_mappings << " " << valid_mappings
                    << " " << invalid_mappings << std::endl;
          mutex_->unlock();
        }        
        invalid_mappings = 0;
        search_->Report(search::Status::Success, Cost(engine, optimization_metrics_.at(0)));

        if (log_suboptimal_)
        {
          auto num_storage_levels = engine.GetTopology().NumStorageLevels();
          auto num_maccs = engine.GetTopology().GetArithmeticLevel()->MACCs();
          mutex_->lock();
          std::cerr << "[" << thread_id_ << "] Utilization = " << engine.Utilization() << " pJ/MACC = "
                    << engine.Energy() / num_maccs << " LL-accesses/MACC = "
                    << double(engine.GetTopology().GetStorageLevel(num_storage_levels-1)->Accesses()) / num_maccs
                    << " CapUtil = ";
          for (unsigned i = 0; i < num_storage_levels; i++)
            std::cerr << engine.GetTopology().GetStorageLevel(i)->CapacityUtilization() << " ";
          std::cerr << std::endl;
          mutex_->unlock();
        }

        // Is the new mapping "better" than the previous best mapping?
        if (!thread_best_.engine.IsSpecced() || IsBetter(engine, thread_best_.engine, optimization_metrics_))
        {
          if (log_stats_)
          {
            // FIXME: improvement only captures the primary stat.
            double improvement = thread_best_.engine.IsSpecced() ?
              (Cost(thread_best_.engine, optimization_metrics_.at(0)) - Cost(engine, optimization_metrics_.at(0))) /
              Cost(thread_best_.engine, optimization_metrics_.at(0)) : 1.0;
            mutex_->lock();
            std::cerr << "[" << thread_id_ << "] UPDATE " << total_mappings << " " << valid_mappings
                      << " " << mappings_since_last_best_update << " " << improvement << std::endl;
            mutex_->unlock();
          }
          
          thread_best_.mapping = mapping;
          thread_best_.engine = engine;
          
          if (!log_suboptimal_)
          {
            mutex_->lock();
            std::cerr << "[" << std::setw(3) << thread_id_ << "]" 
                      << " Utilization = " << std::setw(4) << std::fixed << std::setprecision(2) << engine.Utilization() 
                      << " | pJ/MACC = " << std::setw(8) << std::fixed << std::setprecision(3) << engine.Energy() /
              engine.GetTopology().GetArithmeticLevel()->MACCs() << std::endl;
            mutex_->unlock();
          }

          mappings_since_last_best_update = 0;
        }
        else
        {
          mappings_since_last_best_update++;
        }
      } // while ()
      
      //
      // End Mapping.
      //
    }
  };

  // Main mapper's Run function.
  void Run()
  {
    // Prepare the threads.
    std::mutex mutex;
    std::vector<MapperThread*> threads_;
    for (unsigned t = 0; t < num_threads_; t++)
    {
      threads_.push_back(new MapperThread(t, search_.at(t),
                                          split_mapspaces_.at(t),
                                          &mutex,
                                          search_size_,
                                          timeout_,
                                          victory_condition_,
                                          sync_interval_,
                                          log_stats_,
                                          log_suboptimal_,
                                          optimization_metrics_,
                                          arch_specs_,
                                          workload_,
                                          &best_));
    }

    // Launch the threads.
    for (unsigned t = 0; t < num_threads_; t++)
    {
      threads_.at(t)->Start();
    }

    // Wait for the threads to join.
    for (unsigned t = 0; t < num_threads_; t++)
    {
      threads_.at(t)->Join();
    }

    // Select the best mapping from each thread.
    Mapping best_mapping;
    model::Engine best_mapped_engine;
    for (unsigned t = 0; t < num_threads_; t++)
    {
      auto& mapping = threads_.at(t)->BestMapping();
      auto& engine = threads_.at(t)->BestMappedEngine();
      if (!best_mapped_engine.IsSpecced() ||
          (engine.IsSpecced() && IsBetter(engine, best_mapped_engine, optimization_metrics_)))
      {
          best_mapping = mapping;
          best_mapped_engine = engine;
      }
    }

    std::cout << std::endl;

    for (unsigned t = 0; t < num_threads_; t++)
    {
      delete threads_.at(t);
      threads_.at(t) = nullptr;
    }

    if (best_mapped_engine.IsEvaluated())
    {
      std::cout << best_mapping << std::endl;
      std::cout << best_mapped_engine << std::endl;
    }

    // Printing the Timeloop Mapping to an XML file
    std::ofstream ofs("timeLoopOutput.xml");
    boost::archive::xml_oarchive ar(ofs);
    ar << BOOST_SERIALIZATION_NVP(best_mapped_engine);
    ar << BOOST_SERIALIZATION_NVP(best_mapping);
    const Application* a = this;
    ar << BOOST_SERIALIZATION_NVP(a);

    // Create an output cfg starting with the original cfg contents.
    libconfig::Config config;
    config.readString(cfg_string_);
    free(cfg_string_);
    libconfig::Setting& root = config.getRoot();

    // Update the mapper constraints.
    libconfig::Setting& mapper = root.lookup("mapper");
    if (mapper.exists("num-threads"))
      mapper["num-threads"] = 1;
    else
      mapper.add("num-threads", libconfig::Setting::TypeInt) = 1;

    // Delete the mapspace constraint.
    root.remove("mapspace");

    // Create a new mapspace constraint.
    libconfig::Setting& mapspace = root.add("mapspace", libconfig::Setting::TypeGroup);
    
    // Format the best mapping as libconfig constraints.
    best_mapping.FormatAsConstraints(mapspace);

    config.writeFile("out.cfg");
  }
};

