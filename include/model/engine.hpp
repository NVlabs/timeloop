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

#include <cstdlib>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "model/model-base.hpp"
#include "model/arithmetic.hpp"
#include "model/topology.hpp"
#include "model/sparse-optimization-info.hpp"
#include "mapping/mapping.hpp"
#include "loop-analysis/nest-analysis.hpp"
#include "compound-config/compound-config.hpp"

namespace model
{

class Engine : public Module
{
 public:
  struct Specs
  {
    Topology::Specs topology;
  };
  
 private:
  // Specs.
  Specs specs_;

  // Organization.
  Topology topology_;

  // Utilities.
  analysis::NestAnalysis nest_analysis_;

  // Serialization.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(topology_);
    }
  }

 public:

  // The hierarchical ParseSpecs functions are static and do not
  // affect the internal specs_ data structure, which is set by
  // the dynamic Spec() call later.
  static Specs ParseSpecs(config::CompoundConfigNode setting);

  void Spec(Specs specs);

  const Topology& GetTopology() const;

  std::vector<EvalStatus> PreEvaluationCheck(const Mapping& mapping, problem::Workload& workload, sparse::SparseOptimizationInfo* sparse_optimizations, bool break_on_failure = true);

  std::vector<EvalStatus> Evaluate(Mapping& mapping, problem::Workload& workload, sparse::SparseOptimizationInfo* sparse_optimizations, bool break_on_failure = true);
  
  double Energy() const;
  double Area() const;
  std::uint64_t Cycles() const;
  double Utilization() const;

  friend std::ostream& operator << (std::ostream& out, Engine& engine);
};

} // namespace model
