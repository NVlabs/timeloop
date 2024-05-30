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
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "mapping/parser.hpp"
#include "mapping/arch-properties.hpp"
#include "mapping/constraints.hpp"
#include "compound-config/compound-config.hpp"
#include "model/sparse-optimization-parser.hpp"

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

namespace application
{

class Model
{
 public:
  std::string name_;

  struct Stats
  {
    double energy;
    double cycles;

    std::string stats_string;
    std::string map_string;
    std::string xml_map_and_stats_string;
    std::string tensella_string;
  };

 protected:
  // Critical state.
  problem::Workload workload_;
  model::Engine::Specs arch_specs_;

  // Many of the following submodules are dynamic objects because
  // we can only instantiate them after certain config files have
  // been parsed.

  // The mapping.
  Mapping* mapping_;

  // Abstract representation of the architecture.
  ArchProperties* arch_props_;

  // Constraints.
  mapping::Constraints* constraints_;  
  
  // Application flags/config.
  bool verbose_ = false;
  bool auto_bypass_on_failure_ = false;
  std::string out_prefix_;

  // Sparse optimization
  sparse::SparseOptimizationInfo* sparse_optimizations_;

 private:

  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0);

 public:

  Model(config::CompoundConfig* config,
        std::string output_dir = ".",
        std::string name = "timeloop-model");

  // This class does not support being copied
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  ~Model();

  // Run the evaluation.
  Stats Run();
};


} // namespace application