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

#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/bitset.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "mapspaces/mapspace-factory.hpp"
#include "search/search-factory.hpp"
#include "compound-config/compound-config.hpp"
#include "applications/mapper/mapper-thread.hpp"
#include "model/sparse-optimization-parser.hpp"

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

namespace application
{

class Mapper
{
 public:
  std::string name_;

  /**
   * @brief Contains string versions of the output that used to print to file
   *   during a call to `Run()`. Instead, this is returned so that `main()` can
   *   print to file instead.
   */
  struct Result
  {
    std::string mapping_cpp_string;
    std::string mapping_yaml_string;
    std::string mapping_string;
    std::string stats_string;
    std::string tensella_string;
    std::string xml_mapping_stats_string;
    std::string oaves_string;
  };

 protected:

  problem::Workload workload_;

  model::Engine::Specs arch_specs_;
  mapspace::MapSpace* mapspace_;
  std::vector<mapspace::MapSpace*> split_mapspaces_;
  std::vector<search::SearchAlgorithm*> search_;
  sparse::SparseOptimizationInfo* sparse_optimizations_;

  uint128_t search_size_;
  std::uint32_t num_threads_;
  std::uint32_t timeout_;
  std::uint32_t victory_condition_;
  std::int32_t max_temporal_loops_in_a_mapping_;
  uint128_t sync_interval_;
  uint128_t log_interval_;

  bool log_stats_;
  bool log_oaves_;
  bool log_oaves_mappings_;
  bool log_suboptimal_;
  bool live_status_;
  bool diagnostics_on_;
  bool penalize_consecutive_bypass_fails_;
  bool emit_whoop_nest_;
  std::string out_prefix_;

  std::vector<std::string> optimization_metrics_;

  char* cfg_string_;

  EvaluationResult best_;
  EvaluationResult global_best_;

 private:

  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0);

 public:

  Mapper(config::CompoundConfig* config,
         std::string output_dir = ".",
         std::string name = "timeloop-mapper");

  // This class does not support being copied
  Mapper(const Mapper&) = delete;
  Mapper& operator=(const Mapper&) = delete;

  ~Mapper();

  EvaluationResult GetGlobalBest();

  Mapper::Result Run();
};

} // namespace application