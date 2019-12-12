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

#include <fstream>
#include <thread>
#include <mutex>
#include <iomanip>
#include <ncurses.h>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/bitset.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "util/accelergy_interface.hpp"
#include "mapspaces/mapspace-factory.hpp"
#include "search/search-factory.hpp"
#include "compound-config/compound-config.hpp"
#include "mapper-thread.hpp"

//--------------------------------------------//
//                Mapper                 //
//--------------------------------------------//

struct MapResult
{
  std::string config_name_;
  Mapping best_mapping_;
  model::Engine best_mapped_engine_;

  void PrintResults()
  {
      std::cout << config_name_ << "Summary stats for best mapping found by mapper:" << std::endl; 
      std::cout << "  Utilization = " << std::setw(4) << std::fixed << std::setprecision(2)
                << best_mapped_engine_.Utilization() << " | pJ/MACC = " << std::setw(8)
                << std::fixed << std::setprecision(3) << best_mapped_engine_.Energy() /
        best_mapped_engine_.GetTopology().MACCs() << std::endl;
  }
};

class Mapper 
{
 protected:
   
  //solver specs
  problem::Workload workload_;
  model::Engine::Specs arch_specs_;
  mapspace::MapSpace* mapspace_;

  //solver results
  MapResult best_result_;

  std::vector<mapspace::MapSpace*> split_mapspaces_;
  std::vector<search::SearchAlgorithm*> search_;


  uint128_t search_size_;
  std::uint32_t num_threads_;
  std::uint32_t timeout_;
  std::uint32_t victory_condition_;
  uint128_t sync_interval_;
  bool log_stats_;
  bool log_suboptimal_;
  bool live_status_;
  bool diagnostics_on_;
  bool emit_whoop_nest_;

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
  
  Mapper(std::string name, config::CompoundConfigNode arch, config::CompoundConfigNode problem)
  {
    best_result_.config_name_ = name;

    // Problem configuration.
    std::cout << "Starting mapper for : \"" << name << "\"" << std::endl;

    std::cout << "Problem YAML : " << problem.getYNode() << "" << std::endl;
   
    auto problem_spec = problem.lookup("problem");    
    problem::ParseWorkload(problem_spec, workload_);
    std::cout << "Problem configuration complete.." << std::endl;


    // Architecture configuration.
    auto arch_spec = arch.lookup("architecture");
    if (arch.exists("arch")) {
      arch_spec = arch.lookup("arch");
    } else if (arch.exists("architecture")) {
      arch_spec = arch.lookup("architecture");
    }
    arch_specs_ = model::Engine::ParseSpecs(arch_spec);
    std::cout << "Architecture configuration complete." << std::endl;

    //embed ERT in arch YAML
    if (arch.exists("ERT")) {
      auto ert = arch.lookup("ERT");
      std::cout << "Found Accelergy ERT (energy reference table), replacing internal energy model." << std::endl;
      arch_specs_.topology.ParseAccelergyERT(ert);
    } else {
#ifdef USE_ACCELERGY
      // Call accelergy ERT with all input files
      if (arch.exists("subtree") || arch.exists("local")) {
        accelergy::invokeAccelergy(config->inFiles);
        auto ertConfig = new config::CompoundConfig("ERT.yaml");
        auto ert = ertConfig->getRoot().lookup("ERT");
        std::cout << "Generate Accelergy ERT (energy reference table) to replace internal energy model." << std::endl;
        arch_specs_.topology.ParseAccelergyERT(ert);
      }
#endif
    }
    std::cout << "Energy Tables complete." << std::endl;

    // Mapper (this application) configuration.
    auto mapper = arch.lookup("mapper");
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
      mapper.lookupArrayValue("optimization-metrics", optimization_metrics_);
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
    live_status_ = false;
    mapper.lookupValue("live-status", live_status_);
    diagnostics_on_ = false;
    mapper.lookupValue("diagnostics", diagnostics_on_);
    emit_whoop_nest_ = false;
    mapper.lookupValue("emit-whoop-nest", emit_whoop_nest_);    
    std::cout << "Mapper configuration complete." << std::endl;

    // MapSpace configuration.
    if (arch.exists("mapspace"))
    {
      // MapSpace configuration is combined with arch in same node
      auto mapspace = arch.lookup("mapspace");
      mapspace_ = mapspace::ParseAndConstruct(mapspace, arch_specs_, workload_); 
    } 
    else if (arch.exists("mapspace_constraints"))
    {
      // MapSpace configuration is combined with arch in same node
      auto mapspace = arch.lookup("mapspace_constraints");
      mapspace_ = mapspace::ParseAndConstruct(mapspace, arch_specs_, workload_); 
    } 
    else
    {
      std::cerr << "ERROR: found neither \"mapspace\" nor \"mapspace_constraints\" "
                << "directive. To run the mapper without any constraints set "
                << "mapspace_constraints as an empty list []." << std::endl;
      exit(1);
    }

    split_mapspaces_ = mapspace_->Split(num_threads_);
    std::cout << "Mapspace construction complete." << std::endl;

    // Search configuration.
    auto search = arch.lookup("mapper");
    for (unsigned t = 0; t < num_threads_; t++)
    {
      search_.push_back(search::ParseAndConstruct(search, split_mapspaces_.at(t), t));
    }
    std::cout << "Search configuration complete." << std::endl;
    // Store the complete configuration in a string.
    cfg_string_ = nullptr;
  }

  MapResult GetResults()
  {
    return best_result_;
  }


  // This class does not support being copied
  Mapper() {};
  //Mapper(const Mapper&) = delete;
  //Mapper& operator=(const Mapper&) = delete;

  ~Mapper()
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



  // ---------------
  // Run the mapper.
  // ---------------
  void Run()
  {
    // Output file names.
    const std::string out_prefix = "timeloop-mapper.";
    const std::string log_file_name = out_prefix + "log";
    const std::string stats_file_name = out_prefix + "stats.txt";
    const std::string xml_file_name = out_prefix + "map+stats.xml";
    const std::string map_txt_file_name = out_prefix + "map.txt";
    const std::string map_cfg_file_name = out_prefix + "map.cfg";
    const std::string map_cpp_file_name = out_prefix + "map.cpp";
    
    // Prepare live status/log stream.
    std::ofstream log_file;

    // std::streambuf* streambuf_cout = std::cout.rdbuf(); 
    std::streambuf* streambuf_cerr = std::cerr.rdbuf();

    if (live_status_)
    {
      log_file.open(log_file_name);
      // std::cout.rdbuf(log_file.rdbuf());
      std::cerr.rdbuf(log_file.rdbuf());
  
      initscr();
      cbreak();
      noecho();
      clear();

      std::stringstream line0, line1, line2, line3, line4, line5;
      line0 << "================================================================================";
      line1 << "                                TIMELOOP MAPPER";
      line2 << "================================================================================";
      line3 << std::setw(3) << "TID" << std::setw(11) << "Total" << std::setw(11) << "Invalid"
            << std::setw(11) << "Valid" <<  std::setw(11) << "Consec." << std::setw(11) << "Last"
            << std::setw(11) << "Opt.util" << std::setw(11) << "Opt.energy";
      line4 << std::setw(3) << " " << std::setw(11) << " " << std::setw(11) << " "
            << std::setw(11) << " " <<  std::setw(11) << "invalid" << std::setw(11) << "update";
      line5 << "--------------------------------------------------------------------------------";
      mvaddstr(0, 0, line0.str().c_str());
      mvaddstr(1, 0, line1.str().c_str());
      mvaddstr(2, 0, line2.str().c_str());      
      mvaddstr(3, 0, line3.str().c_str());
      mvaddstr(4, 0, line4.str().c_str());
      mvaddstr(5, 0, line5.str().c_str());      
      refresh();
    }

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
                                          live_status_ ? log_file : std::cerr,
                                          live_status_,
                                          diagnostics_on_,
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

    // Close log and end curses.
    if (live_status_)
    {
      // std::cout.rdbuf(streambuf_cout);
      std::cerr.rdbuf(streambuf_cerr);
      log_file.close();

      mvaddstr(LINES-1, 0, "Press any key to exit.");
      getch();
      endwin();
    }

    // Diagnostics.
    if (diagnostics_on_)
    {
      // Aggregate diagnostic data from all threads.
      std::vector<uint128_t> eval_fail_counts(arch_specs_.topology.NumLevels(), 0);
      std::vector<EvaluationResult> eval_fail_sample_results(arch_specs_.topology.NumLevels());
      unsigned worst_eval_fail_level_id = 0;
      uint128_t worst_eval_fail_count = 0;

      for (unsigned t = 0; t < num_threads_; t++)
      {
        auto& thread_counts = threads_.at(t)->InvalidEvalCounts();
        auto& thread_sample_results = threads_.at(t)->InvalidEvalSampleResults();
        for (unsigned level_id = 0; level_id < arch_specs_.topology.NumLevels(); level_id++)
        {
          // Pick up a sample mapping from the first thread with non-zero fails at this level.
          if (eval_fail_counts.at(level_id) == 0 && thread_counts.at(level_id) != 0)
            eval_fail_sample_results.at(level_id) = thread_sample_results.at(level_id);

          eval_fail_counts.at(level_id) += thread_counts.at(level_id);
        }
      }

      for (unsigned level_id = 0; level_id < arch_specs_.topology.NumLevels(); level_id++)
      {
        if (eval_fail_counts.at(level_id) > worst_eval_fail_count)
        {
          worst_eval_fail_count = eval_fail_counts.at(level_id);
          worst_eval_fail_level_id = level_id;
        }
      }

      // Print.
      std::cout << std::endl;
      std::cout << "===============================================" << std::endl;
      std::cout << "               BEGIN DIAGNOSTICS               " << std::endl;
      std::cout << "-----------------------------------------------" << std::endl;
      std::cout << "Per-level buffer capacity failure counts: " << std::endl;
      for (unsigned level_id = 0; level_id < arch_specs_.topology.NumLevels(); level_id++)
      {
        if (eval_fail_counts.at(level_id) > 0)
        {
          std::cout << std::setw(24) << arch_specs_.topology.GetLevel(level_id)->level_name
                    << ": " << eval_fail_counts.at(level_id) << std::endl;
        }
      }
      
      if (worst_eval_fail_count > 0)
      {
        std::cout << std::endl << "Level with most failures: "
                  << arch_specs_.topology.GetLevel(worst_eval_fail_level_id)->level_name
                  << ": " << worst_eval_fail_count << std::endl;
        std::cout << "Sample failed mapping: " << std::endl;

        auto& engine = eval_fail_sample_results.at(worst_eval_fail_level_id).engine;
        auto& mapping = eval_fail_sample_results.at(worst_eval_fail_level_id).mapping;
        if (!engine.IsEvaluated())
        {
          engine.Evaluate(mapping, workload_, false);
        }
        mapping.PrettyPrint(std::cout, arch_specs_.topology.StorageLevelNames(),
                            engine.GetTopology().TileSizes());
      }

      std::cout << "-----------------------------------------------" << std::endl;
      std::cout << "                 END DIAGNOSTICS               " << std::endl;
      std::cout << "===============================================" << std::endl;
    }

    // Select the best mapping from each thread.
    //Mapping best_mapping_;
    //model::Engine best_mapped_engine_;
    for (unsigned t = 0; t < num_threads_; t++)
    {
      auto& mapping = threads_.at(t)->BestMapping();
      auto& engine = threads_.at(t)->BestMappedEngine();
      if (!best_result_.best_mapped_engine_.IsSpecced() ||
          (engine.IsSpecced() && IsBetter(engine, best_result_.best_mapped_engine_, optimization_metrics_)))
      {
          best_result_.best_mapping_ = mapping;
          best_result_.best_mapped_engine_ = engine;
      }
    }



    std::cout << std::endl;

    for (unsigned t = 0; t < num_threads_; t++)
    {
      delete threads_.at(t);
      threads_.at(t) = nullptr;
    }

    if (best_result_.best_mapped_engine_.IsEvaluated())
    {
      std::ofstream map_txt_file(map_txt_file_name);
      best_result_.best_mapping_.PrettyPrint(map_txt_file, arch_specs_.topology.StorageLevelNames(), best_result_.best_mapped_engine_.GetTopology().TileSizes());
      map_txt_file.close();

      std::ofstream stats_file(stats_file_name);
      stats_file << best_result_.best_mapped_engine_ << std::endl;
      stats_file.close();

      if (emit_whoop_nest_)
      {
        std::ofstream map_cpp_file(map_cpp_file_name);
        best_result_.best_mapping_.PrintWhoopNest(map_cpp_file, arch_specs_.topology.StorageLevelNames(),
                                    best_result_.best_mapped_engine_.GetTopology().TileSizes(),
                                    best_result_.best_mapped_engine_.GetTopology().UtilizedInstances());
        map_cpp_file.close();
      }

      std::cout << "Summary stats for best mapping found by mapper:" << std::endl; 
      std::cout << "  Utilization = " << std::setw(4) << std::fixed << std::setprecision(2)
                << best_result_.best_mapped_engine_.Utilization() << " | pJ/MACC = " << std::setw(8)
                << std::fixed << std::setprecision(3) << best_result_.best_mapped_engine_.Energy() /
        best_result_.best_mapped_engine_.GetTopology().MACCs() << std::endl;
    }
    else
    {
      std::cout << "MESSAGE: no valid mappings found within search criteria. Some suggestions:" << std::endl;
      std::cout << "(1) Observe each mapper thread's termination message. If it terminated due to" << std::endl
                << "    consecutive failed mappings, it will tell you the number of mappings that" << std::endl
                << "    failed because of a spatial fanout violation and the number that failed" << std::endl
                << "    because of a buffer capacity violation." << std::endl;
      std::cout << "(2) Check your architecture configuration (especially mapspace constraints)." << std::endl
                << "    Try to find the offending constraints that are likely to have caused the" << std::endl
                << "    above violations, and disable those constraints." << std::endl;
      std::cout << "(3) Try other search algorithms, and relax the termination criteria:" << std::endl
                << "    victory-condition, timeout and/or search-size." << std::endl;
      if (!diagnostics_on_)
      {
        std::cout << "(4) Enable mapper's diagnostics (mapper.diagnostics = True) to track and emit " << std::endl
                  << "    more information about failed mappings." << std::endl;
      }
    }

    // Print the engine stats and mapping to an XML file
    std::ofstream ofs(xml_file_name);
    boost::archive::xml_oarchive ar(ofs);
    ar << BOOST_SERIALIZATION_NVP(best_result_.best_mapped_engine_);
    ar << BOOST_SERIALIZATION_NVP(best_result_.best_mapping_);
    const Mapper* a = this;
    ar << BOOST_SERIALIZATION_NVP(a);

    if (!cfg_string_)  return; // empty because input was yml

    // Create an output cfg starting with the original cfg contents.
    libconfig::Config config;
    config.readString(cfg_string_);
    free(cfg_string_);
    libconfig::Setting& root = config.getRoot();

    // Update the mapper constraints.
    libconfig::Setting& mapper = root.lookup("mapper");

    if (mapper.exists("algorithm"))
      mapper["algorithm"] = "exhaustive";
    else
      mapper.add("algorithm", libconfig::Setting::TypeString) = "exhaustive";

    if (mapper.exists("num-threads"))
      mapper["num-threads"] = 1;
    else
      mapper.add("num-threads", libconfig::Setting::TypeInt) = 1;

    if (mapper.exists("search_size"))
      mapper.remove("search_size");

    if (mapper.exists("search-size"))
      mapper["search-size"] = 1;
    else
      mapper.add("search-size", libconfig::Setting::TypeInt) = 1;

    // Delete the mapspace constraint.
    root.remove("mapspace");

    // Create a new mapspace constraint.
    libconfig::Setting& mapspace = root.add("mapspace", libconfig::Setting::TypeGroup);
    
    // Format the best mapping as libconfig constraints.
    best_result_.best_mapping_.FormatAsConstraints(mapspace);

    config.writeFile(map_cfg_file_name.c_str());
  }
};

