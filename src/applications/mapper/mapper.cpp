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

#include <fstream>
#include <thread>
#include <mutex>
#include <iomanip>
#include <ncurses.h>

#include "util/accelergy_interface.hpp"

#include "applications/mapper/mapper.hpp"

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

template <class Archive>
void Application::serialize(Archive& ar, const unsigned int version)
{
  if(version == 0)
  {
    ar& BOOST_SERIALIZATION_NVP(workload_);
  }
}

Application::Application(config::CompoundConfig* config,
                         std::string output_dir,
                         std::string name) :
    name_(name)
{
  auto rootNode = config->getRoot();

  // Problem configuration.
  auto problem = rootNode.lookup("problem");
  problem::ParseWorkload(problem, workload_);
  std::cout << "Problem configuration complete." << std::endl;

  // Mapper (this application) configuration.
  auto mapper = rootNode.lookup("mapper");
  std::string semi_qualified_prefix = name;
  mapper.lookupValue("out_prefix", semi_qualified_prefix);
  out_prefix_ = output_dir + "/" + semi_qualified_prefix;

  // Architecture configuration.
  config::CompoundConfigNode arch;
  if (rootNode.exists("arch"))
  {
    arch = rootNode.lookup("arch");
  }
  else if (rootNode.exists("architecture"))
  {
    arch = rootNode.lookup("architecture");
  }
  arch_specs_ = model::Engine::ParseSpecs(arch);

  if (rootNode.exists("ERT"))
  {
    auto ert = rootNode.lookup("ERT");
    std::cout << "Found Accelergy ERT (energy reference table), replacing internal energy model." << std::endl;
    arch_specs_.topology.ParseAccelergyERT(ert);
    if (rootNode.exists("ART")){ // Nellie: well, if the users have the version of Accelergy that generates ART
      auto art = rootNode.lookup("ART");
      std::cout << "Found Accelergy ART (area reference table), replacing internal area model." << std::endl;
      arch_specs_.topology.ParseAccelergyART(art);  
    }
  }
  else
  {
#ifdef USE_ACCELERGY
    // Call accelergy ERT with all input files
    if (arch.exists("subtree") || arch.exists("local"))
    {
      accelergy::invokeAccelergy(config->inFiles, semi_qualified_prefix, output_dir);
      std::string ertPath = out_prefix_ + ".ERT.yaml";
      auto ertConfig = new config::CompoundConfig(ertPath.c_str());
      auto ert = ertConfig->getRoot().lookup("ERT");
      std::cout << "Generate Accelergy ERT (energy reference table) to replace internal energy model." << std::endl;
      arch_specs_.topology.ParseAccelergyERT(ert);
        
      std::string artPath = out_prefix_ + ".ART.yaml";
      auto artConfig = new config::CompoundConfig(artPath.c_str());
      auto art = artConfig->getRoot().lookup("ART");
      std::cout << "Generate Accelergy ART (area reference table) to replace internal area model." << std::endl;
      arch_specs_.topology.ParseAccelergyART(art);
    }
#endif
  }

  std::cout << "Architecture configuration complete." << std::endl;

  // Sparse optimizations
  config::CompoundConfigNode sparse_optimizations;
  if (rootNode.exists("sparse_optimizations"))
    sparse_optimizations = rootNode.lookup("sparse_optimizations");
  
  sparse_optimizations_ = new sparse::SparseOptimizationInfo(sparse::ParseAndConstruct(sparse_optimizations, arch_specs_));
  // characterize workload on whether it has metadata
  workload_.SetDefaultDenseTensorFlag(sparse_optimizations_->compression_info.all_ranks_default_dense);
  
  std::cout << "Sparse optimization configuration complete." << std::endl;

  // Mapper (this application) configuration. (the rest)

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

  penalize_consecutive_bypass_fails_ = false;
  mapper.lookupValue("penalize-consecutive-bypass-fails", penalize_consecutive_bypass_fails_);

  emit_whoop_nest_ = false;
  mapper.lookupValue("emit-whoop-nest", emit_whoop_nest_);    

  std::cout << "Mapper configuration complete." << std::endl;

  // MapSpace configuration.
  config::CompoundConfigNode arch_constraints;
  config::CompoundConfigNode mapspace;

  // Architecture constraints.
  if (arch.exists("constraints"))
    arch_constraints = arch.lookup("constraints");
  else if (rootNode.exists("arch_constraints"))
    arch_constraints = rootNode.lookup("arch_constraints");
  else if (rootNode.exists("architecture_constraints"))
    arch_constraints = rootNode.lookup("architecture_constraints");

  // Mapspace constraints.
  if (rootNode.exists("mapspace"))
    mapspace = rootNode.lookup("mapspace");
  else if (rootNode.exists("mapspace_constraints"))
    mapspace = rootNode.lookup("mapspace_constraints");
  // else
  // {
  //   std::cerr << "ERROR: found neither \"mapspace\" nor \"mapspace_constraints\" "
  //             << "directive. To run the mapper without any constraints set "
  //             << "mapspace_constraints as an empty list []." << std::endl;
  //   exit(1);
  // }

  bool filter_spatial_fanout = sparse_optimizations_->action_spatial_skipping_info.size() == 0;
  mapspace_ = mapspace::ParseAndConstruct(mapspace, arch_constraints, arch_specs_, workload_, filter_spatial_fanout);
  split_mapspaces_ = mapspace_->Split(num_threads_);

  std::cout << "Mapspace construction complete." << std::endl;

  // Search configuration.
  auto search = rootNode.lookup("mapper");
  for (unsigned t = 0; t < num_threads_; t++)
  {
    search_.push_back(search::ParseAndConstruct(search, split_mapspaces_.at(t), t));
  }
  std::cout << "Search configuration complete." << std::endl;
  // Store the complete configuration in a string.
  if (config->hasLConfig())
  {
    std::size_t len;
    FILE* cfg_stream = open_memstream(&cfg_string_, &len);
    auto& lconfig = config->getLConfig();
    lconfig.write(cfg_stream);
    fclose(cfg_stream);
  }
  else
  {
    cfg_string_ = nullptr;
  }

}

Application::~Application()
{
  if (mapspace_)
  {
    delete mapspace_;
  }

  if (sparse_optimizations_)
  {
    delete sparse_optimizations_;
  }

  for (auto& search: search_)
  {
    if (search)
    {
      delete search;
    }
  }
}

EvaluationResult Application::GetGlobalBest()
{
  return global_best_;
}

// ---------------
// Run the mapper.
// ---------------
void Application::Run()
{
  // Output file names.
  std::string log_file_name = out_prefix_ + ".log";
  std::string stats_file_name = out_prefix_ + ".stats.txt";
  std::string xml_file_name = out_prefix_ + ".map+stats.xml";
  std::string map_txt_file_name = out_prefix_ + ".map.txt";
  // std::string map_yaml_file_name = out_prefix_ + ".constraints.yaml";
  std::string map_cfg_file_name = out_prefix_ + ".map.cfg";
  std::string map_cpp_file_name = out_prefix_ + ".map.cpp";
    
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
                                        penalize_consecutive_bypass_fails_,
                                        optimization_metrics_,
                                        arch_specs_,
                                        workload_,
                                        sparse_optimizations_,
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
    std::map<FailClass, std::map<unsigned, FailInfo>> fail_stats;
      
    for (unsigned t = 0; t < num_threads_; t++)
    {
      for (auto& i: threads_.at(t)->GetStats().fail_stats)
      {
        auto& thread_fail_class = i.first;
        auto& thread_fail_bucket = i.second;

        auto fail_bucket_it = fail_stats.find(thread_fail_class);
        if (fail_bucket_it == fail_stats.end())
        {
          // We've never seen this fail class before.
          fail_stats[thread_fail_class] = thread_fail_bucket;
        }
        else
        {
          auto& fail_bucket = fail_bucket_it->second;
            
          // We've seen this fail class. Walk through each level in this fail bucket.
          for (auto& j: thread_fail_bucket)
          {
            auto& thread_fail_level_id = j.first;
            auto& thread_fail_info = j.second;

            auto fail_info_it = fail_bucket.find(thread_fail_level_id);
            if (fail_info_it == fail_bucket.end())
            {
              // We haven't seen this level within this fail bucket.
              fail_bucket[thread_fail_level_id] = thread_fail_info;
            }
            else
            {
              // We've seen this level within this fail bucket.
              fail_info_it->second.count += thread_fail_info.count;
            }
          }
        }
      }
    }
        

    // Print.
    std::cout << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "               BEGIN DIAGNOSTICS               " << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    for (auto& i: fail_stats)
    {
      auto& fail_class = i.first;
      auto& fail_bucket = i.second;
        
      std::cout << "Fail class: " << fail_class << std::endl;
      for (auto& j: fail_bucket)
      {
        std::cout << std::endl;
        std::cout << "  Level: " << arch_specs_.topology.GetLevel(j.first)->level_name << std::endl;
        std::cout << "    Fail count: " << j.second.count << std::endl;
        std::cout << "    Sample mapping that experienced this fail class:" << std::endl;

        auto& mapping = j.second.mapping;

        model::Engine engine;
        engine.Spec(arch_specs_);
        engine.Evaluate(mapping, workload_, sparse_optimizations_, false);
        mapping.PrettyPrint(std::cout, arch_specs_.topology.StorageLevelNames(),
                            engine.GetTopology().GetStats().utilized_capacities,
                            engine.GetTopology().GetStats().tile_sizes, "      ");

        std::cout << "    Fail reason: " << j.second.reason << std::endl;
        std::cout << std::endl;
      }
    }
      
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "                 END DIAGNOSTICS               " << std::endl;
    std::cout << "===============================================" << std::endl;
  }

  // Select the best mapping from each thread.
  for (unsigned t = 0; t < num_threads_; t++)
  {
    auto& thread_best = threads_.at(t)->GetStats().thread_best;
    global_best_.UpdateIfBetter(thread_best, optimization_metrics_);
  }

  std::cout << std::endl;

  for (unsigned t = 0; t < num_threads_; t++)
  {
    delete threads_.at(t);
    threads_.at(t) = nullptr;
  }

  if (global_best_.valid)
  {
    std::ofstream map_txt_file(map_txt_file_name);
    global_best_.mapping.PrettyPrint(map_txt_file, arch_specs_.topology.StorageLevelNames(),
                                     global_best_.stats.utilized_capacities,
                                     global_best_.stats.tile_sizes);
    map_txt_file.close();

    // std::ofstream map_yaml_file(map_yaml_file_name);
    // global_best_.mapping.PrintAsConstraints(map_yaml_file_name);
    // map_yaml_file.close();

    // Re-evaluate the mapping so that we get a live engine with complete specs and stats
    // that can be printed out hierarchically.
    model::Engine engine;
    engine.Spec(arch_specs_);
    engine.Evaluate(global_best_.mapping, workload_, sparse_optimizations_);

    std::ofstream stats_file(stats_file_name);
    stats_file << engine << std::endl;
    stats_file.close();

    if (emit_whoop_nest_)
    {
      std::ofstream map_cpp_file(map_cpp_file_name);
      global_best_.mapping.PrintWhoopNest(map_cpp_file, arch_specs_.topology.StorageLevelNames(),
                                          global_best_.stats.tile_sizes,
                                          global_best_.stats.utilized_instances);
      map_cpp_file.close();
    }

    std::cout << "Summary stats for best mapping found by mapper:" << std::endl;
    std::cout << "  Utilization = " << std::setw(4) << std::fixed << std::setprecision(2)
              << global_best_.stats.utilization << " | pJ/Algorithmic-Compute = " << std::setw(8)
              << std::fixed << std::setprecision(3) << global_best_.stats.energy /
      global_best_.stats.algorithmic_computes
              << " | pJ/Compute = " << std::setw(12)
              << std::fixed << std::setprecision(3) << global_best_.stats.energy /
      global_best_.stats.actual_computes << std::endl;

    // Print the engine stats and mapping to an XML file
    std::ofstream ofs(xml_file_name);
    boost::archive::xml_oarchive ar(ofs);
    ar << boost::serialization::make_nvp("engine", engine);
    ar << boost::serialization::make_nvp("mapping", global_best_.mapping);
    const Application* a = this;
    ar << BOOST_SERIALIZATION_NVP(a);
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

  if (!cfg_string_)  return; // empty because input was yml

  // Create an output cfg starting with the original cfg contents.
  libconfig::Config config;
  config.readString(cfg_string_);
  free(cfg_string_);
  libconfig::Setting& root = config.getRoot();

#ifdef EMIT_OPT_AS_CONSTRAINTS
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

  if (global_best_.valid)
  {
    // Create a new mapspace constraint.
    libconfig::Setting& mapspace = root.add("mapspace", libconfig::Setting::TypeGroup);
    
    // Format the best mapping as libconfig constraints.
    global_best_.mapping.FormatAsConstraints(mapspace);
  }
#else
  // We used to create a set of mapper constraints to fit exactly one mapping,
  // which could then be provided to timeloop-mapper.
  // We now create a single mapping which can be fed to timeloop-model.
  root.remove("mapper");
  root.remove("mapspace");

  if (global_best_.valid)
  {
    // Create a new mapping.
    libconfig::Setting& mapping = root.add("mapping", libconfig::Setting::TypeList);
    
    // Format the best mapping as a libconfig spec.
    global_best_.mapping.FormatAsLibConfig(mapping, arch_specs_.topology.StorageLevelNames());
  }
#endif

  config.writeFile(map_cfg_file_name.c_str());
}
