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

#include <map>

#include "mapping/arch-properties.hpp"

namespace mapping
{

//--------------------------------------------//
//                 Constraints                //
//--------------------------------------------//

class Constraints
{
 protected:

  // Abstract representation of the architecture.
  const ArchProperties& arch_props_;

  // Workload.
  const problem::Workload& workload_;  
 
  // The constraints.
  std::map<unsigned, std::map<problem::Shape::DimensionID, int>> factors_;
  std::map<unsigned, std::map<problem::Shape::DimensionID, int>> max_factors_;
  std::map<unsigned, std::vector<problem::Shape::DimensionID>> permutations_;
  std::map<unsigned, std::uint32_t> spatial_splits_;
  std::map<unsigned, double> confidence_thresholds_;
  problem::PerDataSpace<std::string> bypass_strings_;
  double min_parallelism_;
  bool min_parallelism_isset_;  

 public:
  Constraints() = delete;

  Constraints(const ArchProperties& arch_props,
              const problem::Workload& workload);

  const std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& Factors() const;
  const std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& MaxFactors() const;
  const std::map<unsigned, std::vector<problem::Shape::DimensionID>>& Permutations() const;
  const std::map<unsigned, std::uint32_t>& SpatialSplits() const;  
  const problem::PerDataSpace<std::string>& BypassStrings() const;
  double MinParallelism() const;
  const std::map<unsigned, double>& ConfidenceThresholds() const;

  // Create a constraints object from a given mapping object. The resultant
  // constraints will *only* be satisfied by that mapping.
  void Generate(Mapping* mapping);
  
  // Check if a given Constraints object is a subset (i.e., more constrained).
  bool operator >= (const Constraints& other) const;

  // Check if a given mapping satisfies these constraints.
  bool SatisfiedBy(Mapping* mapping) const;

  // Parse user-provided constraints.
  void Parse(config::CompoundConfigNode config);

  // Parse a list of constraints.
  void ParseList(config::CompoundConfigNode constraints);

  // Parse a single user constraint.
  void ParseSingleConstraint(
    config::CompoundConfigNode target,
    config::CompoundConfigNode constraint,
    config::CompoundConfigNode attributes);

  // FindTargetTilingLevel()
  unsigned FindTargetTilingLevel(config::CompoundConfigNode constraint, std::string type);

  // Parsers.
  std::map<problem::Shape::DimensionID, int> ParseFactors(config::CompoundConfigNode constraint);
  std::map<problem::Shape::DimensionID, int> ParseMaxFactors(config::CompoundConfigNode constraint);
  std::vector<problem::Shape::DimensionID> ParsePermutations(config::CompoundConfigNode constraint);
  void ParseDatatypeBypassSettings(config::CompoundConfigNode constraint, unsigned level);
};

} // namespace mapping
