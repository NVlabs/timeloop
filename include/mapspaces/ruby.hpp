/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include <set>

#include "util/numeric.hpp"
#include "util/misc.hpp"
#include "workload/shape-models/problem-shape.hpp"
#include "mapspaces/mapspace-base.hpp"
#include "mapspaces/subspaces.hpp"
#include "compound-config/compound-config.hpp"
#include "mapping/arch-properties.hpp"
#include "mapping/constraints.hpp"

namespace mapspace
{

//--------------------------------------------//
//                Ruby MapSpace               //
//--------------------------------------------//

class Ruby : public MapSpace
{
 protected:

  // Sub-spaces.
  RubyPermutationSpace permutation_space_;
  ResidualIndexFactorizationSpace index_factorization_space_;
  SpatialSplitSpace spatial_split_space_;
  std::vector<tiling::CompoundMaskNest> datatype_bypass_nest_space_;

  // Splits of this mapspace (used for parallelizing).
  std::vector<Ruby*> splits_;
  std::uint64_t split_id_;
  std::uint64_t num_parent_splits_;
  
  // Abstract representation of the architecture.
  ArchProperties arch_props_;

  // Constraints.
  mapping::Constraints constraints_;

  // Filter Fanout
  bool filter_spatial_fanout_;
  
 public:

  //
  // Ruby() - Derived mapspace classes may wish to call with skip_init
  //          set to true and then explicitly call Init(), possibly with
  //          customized pre-mapped tiles.
  //
  Ruby(
    config::CompoundConfigNode config,
    config::CompoundConfigNode arch_constraints,
    model::Engine::Specs arch_specs,
    const problem::Workload& workload,
    bool filter_spatial_fanout = true,
    bool skip_init = false);
  Ruby(const Ruby& other) = default;
  ~Ruby();
  
  //------------------------------------------//
  //        Initialization and Setup          // 
  //------------------------------------------//
  
  void Init(config::CompoundConfigNode config, config::CompoundConfigNode arch_constraints);  
  void InitIndexFactorizationSpace();
  void InitLoopPermutationSpace(std::map<unsigned, std::vector<problem::Shape::FlattenedDimensionID>> pruned_dimensions = {});
  void InitSpatialSpace(std::map<unsigned, unsigned> unit_factors = {});
  void InitDatatypeBypassNestSpace();
  void InitPruned(uint128_t index_factorization_id);

  // Split the mapspace (used for parallelization).
  std::vector<MapSpace*> Split(std::uint64_t num_splits);
  void InitSplit(std::uint64_t split_id, uint128_t split_if_size, std::uint64_t num_parent_splits);
  bool IsSplit();

  //------------------------------------------//
  //           Mapping Construction           // 
  //------------------------------------------//
  
  std::vector<Status> ConstructMapping(
    mapspace::ID mapping_id,
    Mapping* mapping,
    bool break_on_failure = true);

  // Mapping Construction
  void InitSubnests(loop::NestConfig& subnests);
  void PermuteSubnests(uint128_t mapping_permutation_id, loop::NestConfig& subnests);
  void AssignIndexFactors(uint128_t mapping_index_factorization_id, loop::NestConfig& subnests);
  std::vector<Status> AssignSpatialTilingDirections(uint128_t mapping_spatial_id,
                                                    loop::NestConfig& subnests,
                                                    tiling::CompoundMaskNest datatype_bypass_nest,
                                                    bool break_on_failure);
  Status AssignSpatialTilingDirections_Level_Expand(std::uint32_t spatial_split,
                                                    std::vector<loop::Descriptor>& level_nest,
                                                    unsigned tiling_level_id,
                                                    double& fanout_utilization);
  tiling::CompoundMaskNest ConstructDatatypeBypassNest(uint128_t mapping_datatype_bypass_id);

  //------------------------------------------//
  //                 Parsing                  // 
  //------------------------------------------//
  
  void Parse(config::CompoundConfigNode config, config::CompoundConfigNode arch_constraints);
};

} // namespace mapspace
