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

#include <vector>
#include <map>
#include <numeric>

#include "util/numeric.hpp"
#include "workload/shape-models/problem-shape.hpp"
#include "workload/util/per-problem-dimension.hpp"

namespace mapspace
{

//--------------------------------------------//
//           IndexFactorizationSpace          //
//--------------------------------------------//

class IndexFactorizationSpace
{
 private:
  problem::PerFlattenedDimension<Factors> dimension_factors_;
  CartesianCounterDynamic tiling_counter_;

 public:
  IndexFactorizationSpace();

  void Init(const problem::Workload &workload,
            std::map<problem::Shape::FlattenedDimensionID, std::uint64_t> cofactors_order,
            std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>> prefactors =
            std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>>(),
            std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>> maxfactors =
            std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>>());

  unsigned long GetFactor(uint128_t nest_id, problem::Shape::FlattenedDimensionID dim, unsigned level);

  uint128_t Size() const;
};

//--------------------------------------------//
//      ResidualIndexFactorizationSpace       //
//--------------------------------------------//

class ResidualIndexFactorizationSpace
{
 private:
  problem::PerFlattenedDimension<ResidualFactors> dimension_factors_;
  CartesianCounterDynamic tiling_counter_;

 public:
  ResidualIndexFactorizationSpace();

  void Init(const problem::Workload &workload,
            std::map<problem::Shape::FlattenedDimensionID, std::uint64_t> cofactors_order,
            std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>> prefactors,
            std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>> maxfactors,
            std::vector<unsigned long int> s_fan = {},
            std::vector<unsigned long int> s_index = {}
            );

  std::vector<unsigned long> GetFactor(uint128_t nest_id, problem::Shape::FlattenedDimensionID dim, unsigned level);

  uint128_t Size() const;
};

//--------------------------------------------//
//              PermutationSpace              //
//--------------------------------------------//

class PermutationSpace
{
 private:
  std::uint64_t num_levels_;
  struct Pattern
  {
    std::vector<problem::Shape::FlattenedDimensionID> baked_prefix;
    std::vector<problem::Shape::FlattenedDimensionID> permutable_suffix;
  };
  std::map<unsigned, Pattern> patterns_;
  std::vector<problem::Shape::FlattenedDimensionID> canonical_pattern_;
  std::map<unsigned, std::uint64_t> size_;    
  Factoradic<problem::Shape::FlattenedDimensionID> factoradic_;

 public:
  PermutationSpace();

  void Init(uint64_t num_levels);
  void InitLevelCanonical(uint64_t level);
  void InitLevel(uint64_t level, std::vector<problem::Shape::FlattenedDimensionID> user_prefix,
                 std::vector<problem::Shape::FlattenedDimensionID> pruned_dimensions = {});

  std::vector<std::vector<problem::Shape::FlattenedDimensionID>> GetPatterns(uint128_t id);

  uint128_t Size() const;
};

//--------------------------------------------//
//              SpatialSplitSpace             //
//--------------------------------------------//

class SpatialSplitSpace
{
 private:
  // Ugh. The number of levels given to us is the total number
  // of tiling levels. Of these, only a subset are spatial. We
  // need to remember (a) which of these are spatial, and (b)
  // which of the spatial ones have user-specified splits.
  std::uint64_t num_levels_;
  std::map<unsigned, bool> is_user_specified_;
  std::map<unsigned, std::uint32_t> user_splits_;
  std::map<unsigned, std::size_t> size_;
  std::map<unsigned, unsigned> unit_factors_;

  uint64_t n_;
  bool is_fixed_;
  uint64_t fixed_;
  
 public:
  SpatialSplitSpace();

  void Init(uint64_t num_levels);
  void InitLevel(uint64_t level, unsigned unit_factors = 0);
  void InitLevelUserSpecified(uint64_t level, std::uint32_t user_split);

  std::map<unsigned, std::uint32_t> GetSplits(uint128_t id);

  uint128_t Size() const;
};

} // namespace mapspace
