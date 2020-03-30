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

#include <iterator>
#include <mutex>
#include <regex>

#include "util/numeric.hpp"
#include "util/misc.hpp"
#include "workload/problem-shape.hpp"
#include "mapspaces/mapspace-base.hpp"
#include "mapspaces/subspaces.hpp"
#include "compound-config/compound-config.hpp"
#include "mapping/arch-properties.hpp"

namespace mapspace
{

//--------------------------------------------//
//                Uber MapSpace               //
//--------------------------------------------//

class Uber : public MapSpace
{
 protected:
  // Sub-spaces.
  PermutationSpace permutation_space_;
  IndexFactorizationSpace index_factorization_space_;
  SpatialSplitSpace spatial_split_space_;
  std::vector<tiling::CompoundMaskNest> datatype_bypass_nest_space_;

  // Splits of this mapspace (used for parallelizing).
  std::vector<Uber*> splits_;
  std::uint64_t split_id_;
  std::uint64_t num_parent_splits_;
  
  // Parsed metadata needed to construct subspaces.
  std::map<unsigned, std::map<problem::Shape::DimensionID, int>> user_factors_;
  std::map<unsigned, std::map<problem::Shape::DimensionID, int>> user_max_factors_;
  std::map<unsigned, std::vector<problem::Shape::DimensionID>> user_permutations_;
  std::map<unsigned, std::uint32_t> user_spatial_splits_;
  problem::PerDataSpace<std::string> user_bypass_strings_;

  // Abstract representation of the architecture.
  ArchProperties arch_props_;

  // Minimum parallelism (constraint).
  double min_parallelism_;
  bool min_parallelism_isset_;
  
 public:

  //
  // Uber() - Derived mapspace classes may wish to call with skip_init
  //          set to true and then explicitly call Init(), possibly with
  //          customized pre-mapped tiles.
  //
  Uber(
    config::CompoundConfigNode config,
    model::Engine::Specs arch_specs,
    const problem::Workload& workload,
    bool skip_init = false) :
      MapSpace(arch_specs, workload),
      split_id_(0),
      num_parent_splits_(0),
      arch_props_(arch_specs),
      min_parallelism_(0.0)
  {
    if (!skip_init)
    {
      Init(config);
    }
  }

  //------------------------------------------//
  //        Initialization and Setup          // 
  //------------------------------------------//
  
  //
  // Init() - called by derived classes or by constructor.
  //
  void Init(config::CompoundConfigNode config)
  {
    // Setup Map space.
    user_factors_.clear();
    user_permutations_.clear();
    user_spatial_splits_.clear();
    user_bypass_strings_.clear();

    // Parse config.
    ParseUserConfig(config, user_factors_, user_max_factors_, user_permutations_,
                    user_spatial_splits_, user_bypass_strings_);

    // Setup all the mapping sub-spaces.
    InitIndexFactorizationSpace(user_factors_, user_max_factors_);
    InitLoopPermutationSpace(user_permutations_);
    InitSpatialSpace(user_spatial_splits_);
    InitDatatypeBypassNestSpace(user_bypass_strings_);

    // FIXME: optimization: add a "deferred" flag which bypasses
    // PermutationSpace initialization if it's going to be re-initialized
    // during a later InitPruned() call.

    // Sanity checks.
    for (int i = 0; i < int(mapspace::Dimension::Num); i++)
    {
      std::cout << "Mapspace Dimension [" << mapspace::Dimension(i)
                << "] Size: " << size_[i] << std::endl;
    }
    
    // Check for integer overflow in the above multiplications.
    uint128_t de_cumulative_prod = Size();
    for (int i = 0; i < int(mapspace::Dimension::Num); i++)
    {
      de_cumulative_prod /= size_[i];
    }
    if (de_cumulative_prod != 1)
    {
      std::cerr << "ERROR: overflow detected: mapspace size appears to be "
                << "greater than 2^128. Please add some mapspace constraints."
                << std::endl;
      exit(1);
    }
  }
  
  //
  // InitIndexFactorizationSpace()
  //
  void InitIndexFactorizationSpace(std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_factors,
                                   std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_max_factors)
  {
    assert(user_factors.size() <= arch_props_.TilingLevels());

    // We'll initialize the index_factorization_space_ object here. To do that, we first
    // need to determine the number of factors that *each* problem dimension needs to be
    // split up into. In other words, this is the order of the cofactors vector for each
    // problem dimension.
    
    std::map<problem::Shape::DimensionID, std::uint64_t> cofactors_order;
    for (unsigned i = 0; i < unsigned(problem::GetShape()->NumDimensions); i++)
    {
      // Factorize each problem dimension into num_tiling_levels partitions.
      cofactors_order[problem::Shape::DimensionID(i)] = arch_props_.TilingLevels();
    }

    // Next, for each problem dimension, we need to tell the index_factorization_space_
    // object if any of the cofactors have been given a fixed, min or max value by
    // the user. 

    std::map<problem::Shape::DimensionID, std::map<unsigned, unsigned long>> prefactors;
    std::map<problem::Shape::DimensionID, std::map<unsigned, unsigned long>> maxfactors;
    std::vector<bool> exhausted_um_loops(int(problem::GetShape()->NumDimensions), false);

    // Find user-specified fixed factors.
    for (unsigned level = 0; level < arch_props_.TilingLevels(); level++)
    {
      auto it = user_factors.find(level);
      if (it != user_factors.end())
      {
        // Some factors exist for this level.        
        for (auto& factor : it->second)
        {
          auto& dimension = factor.first;
          auto& end = factor.second;
          if (end == -1)
          {
            assert(!exhausted_um_loops[int(dimension)]);
            exhausted_um_loops[int(dimension)] = true;
          }
          else
          {
            prefactors[dimension][level] = end;
          }
        }
      }
    }

    // Find user-specified max factors.
    for (unsigned level = 0; level < arch_props_.TilingLevels(); level++)
    {
      auto it = user_max_factors.find(level);
      if (it != user_max_factors.end())
      {
        // Some max factors exist for this level.        
        for (auto& factor : it->second)
        {
          auto& dimension = factor.first;
          auto& max = factor.second;
          maxfactors[dimension][level] = max;
        }
      }
    }

    // We're now ready to initialize the object.
    index_factorization_space_.Init(workload_, cofactors_order, prefactors, maxfactors);

    // Update the size of the mapspace.
    size_[int(mapspace::Dimension::IndexFactorization)] = index_factorization_space_.Size();
  }

  //
  // InitLoopPermutationSpace()
  //
  void InitLoopPermutationSpace(std::map<unsigned, std::vector<problem::Shape::DimensionID>>& user_permutations,
                                std::map<unsigned, std::vector<problem::Shape::DimensionID>> pruned_dimensions = {})
  {
    permutation_space_.Init(arch_props_.TilingLevels());
    
    for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
    {
      // Extract the user-provided pattern for this level.
      std::vector<problem::Shape::DimensionID> user_prefix;
      auto it = user_permutations.find(level);
      if (it != user_permutations.end())
      {
        user_prefix = it->second;
      }

      bool use_canonical_permutation = false;
      if (arch_props_.IsSpatial(level))
      {
        use_canonical_permutation = user_prefix.empty() && !arch_props_.IsSpatial2D(level);
      }
      else
      {
        use_canonical_permutation = (level == 0); // || level == arch_props_.TilingLevels()-1); // FIXME: last level?
      }
      
      if (use_canonical_permutation)
      {
        // Permutations do not matter; use canonical pattern.
        permutation_space_.InitLevelCanonical(level);
      }
      else
      {
        // Initialize the permutation space for this level using the
        // user-provided pattern. If this pattern is empty or incomplete,
        // it exposes a permutation space. This logic is handled by the
        // permutation space object itself.
        auto it = pruned_dimensions.find(level);
        if (it != pruned_dimensions.end())
          permutation_space_.InitLevel(level, user_prefix, it->second);
        else
          permutation_space_.InitLevel(level, user_prefix);
      }
    }    

    size_[int(mapspace::Dimension::LoopPermutation)] = permutation_space_.Size();
  }

  //
  // InitSpatialSpace()
  //
  void InitSpatialSpace(std::map<unsigned, std::uint32_t>& user_spatial_splits,
                        std::map<unsigned, unsigned> unit_factors = {})
  {
    // Given a spatial permutation, this indicates where the changeover from X
    // to Y dimension occurs. Obviously, this is limited by hardware fanout
    // capabilities at this spatial level.
    spatial_split_space_.Init(arch_props_.TilingLevels());
    
    for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
    {
      if (arch_props_.IsSpatial(level))
      {
        // Extract the user-provided split point for this level.
        auto it = user_spatial_splits.find(level);
        if (it != user_spatial_splits.end())
        {
          spatial_split_space_.InitLevelUserSpecified(level, it->second);
        }
        else
        {
          auto ituf = unit_factors.find(level);
          if (ituf != unit_factors.end())
            spatial_split_space_.InitLevel(level, ituf->second);
          else
            spatial_split_space_.InitLevel(level);
        }
      }
    }

    size_[int(mapspace::Dimension::Spatial)] = spatial_split_space_.Size();
  }

  void InitDatatypeBypassNestSpace(problem::PerDataSpace<std::string> user_bypass_strings =
                                   problem::PerDataSpace<std::string>(""))
  {
    // The user_mask input is a set of per-datatype strings. Each string has a length
    // equal to num_storage_levels, and contains the characters 0 (bypass), 1 (keep),
    // or X (evaluate both).
    
    // A CompoundMaskNest is effectively a PerDataSpace<std::bitset<MaxTilingLevels>>.
    // The datatype_bypass_nest_space_ is a vector of CompoundMaskNests.

    // First, seed the space with a single mask with each bit set to 1.
    assert(datatype_bypass_nest_space_.empty());
    tiling::CompoundMaskNest seed_mask_nest;
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      for (unsigned level = 0; level < arch_specs_.topology.NumStorageLevels(); level++)
      {
        seed_mask_nest.at(pvi).set(level);
      }
    }
    datatype_bypass_nest_space_.push_back(seed_mask_nest);

    // Now parse the user strings and edit/expand the space as necessary.
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);

      // Start parsing the user mask string.
      assert(user_bypass_strings.at(pv).length() <= arch_specs_.topology.NumStorageLevels());

      // The first loop runs the length of the user-specified string.
      unsigned level = 0;
      for (; level < user_bypass_strings.at(pv).length(); level++)
      {
        char spec = user_bypass_strings.at(pv).at(level);
        switch (spec)
        {
          case '0':
          {
            for (auto& compound_mask_nest: datatype_bypass_nest_space_)
            {
              compound_mask_nest.at(pvi).reset(level);
            }
            break;
          }
            
          case '1':
          {
            for (auto& compound_mask_nest: datatype_bypass_nest_space_)
            {
              compound_mask_nest.at(pvi).set(level);
            }
            break;
          }
            
          case 'X':
          {
            auto copy = datatype_bypass_nest_space_;
            for (auto& compound_mask_nest: datatype_bypass_nest_space_)
            {
              compound_mask_nest.at(pvi).reset(level);
            }
            for (auto& compound_mask_nest: copy)
            {
              compound_mask_nest.at(pvi).set(level);
            }
            datatype_bypass_nest_space_.insert(datatype_bypass_nest_space_.end(),
                                               copy.begin(), copy.end());
            break;
          }
                        
          default:
          {
            assert(false);
            break;
          }
        }
      }

      // We've exhausted the user-specified string, but we may have
      // more levels that are un-specified. We consider these 'X's.
      // However, we'll leave the outermost level at 1 (unless the
      // user already overrode that in the provided string).
      for (; level < arch_specs_.topology.NumStorageLevels()-1; level++)
      {
        auto copy = datatype_bypass_nest_space_;
        for (auto& compound_mask_nest: datatype_bypass_nest_space_)
        {
          compound_mask_nest.at(pvi).reset(level);
        }
        for (auto& compound_mask_nest: copy)
        {
          compound_mask_nest.at(pvi).set(level);
        }
        datatype_bypass_nest_space_.insert(datatype_bypass_nest_space_.end(),
                                           copy.begin(), copy.end());
      }
    } // for (pvi)

    size_[int(mapspace::Dimension::DatatypeBypass)] = datatype_bypass_nest_space_.size();
  }


  Uber(const Uber& other) = default;
  
  //
  // Split the mapspace (used for parallelization).
  //
  std::vector<MapSpace*> Split(std::uint64_t num_splits)
  {
    assert(size_[int(mapspace::Dimension::IndexFactorization)] > 0);
    assert(num_splits > 0);

    uint128_t split_size = 1 + (size_[int(mapspace::Dimension::IndexFactorization)] - 1) / num_splits;
    uint128_t split_residue = (split_size * num_splits) - size_[int(mapspace::Dimension::IndexFactorization)];

    std::cout << "Mapspace split! Per-split Mapping Dimension ["
              << mapspace::Dimension::IndexFactorization
              << "] Size: " << split_size
              << " Residue: " << split_residue << std::endl;

    std::vector<Uber*> splits;
    std::vector<MapSpace*> retval;
    for (unsigned i = 0; i < num_splits; i++)
    {
      Uber* mapspace = new Uber(*this);

      // Last <residue> splits have 1-smaller size.
      uint128_t if_size = (i < num_splits - split_residue) ? split_size : split_size - 1;
      mapspace->InitSplit(i, if_size, num_splits);
      
      splits.push_back(mapspace);
      retval.push_back(static_cast<MapSpace*>(mapspace));
    }

    splits_ = splits;
    return retval;
  }

  void InitSplit(std::uint64_t split_id, uint128_t split_if_size, std::uint64_t num_parent_splits)
  {
    split_id_ = split_id;
    size_[int(mapspace::Dimension::IndexFactorization)] = split_if_size;
    num_parent_splits_ = num_parent_splits;
  }

  bool IsSplit()
  {
    return (splits_.size() > 0);
  }

  ~Uber()
  {
    for (auto split : splits_)
    {
      delete split;
    }
    splits_.clear();
  }

  void InitPruned(uint128_t index_factorization_id)
  {
    assert(!IsSplit());

    // Each split knows its private IF size and should never generate an out-of-range ID.
    if (index_factorization_id >= size_[int(mapspace::Dimension::IndexFactorization)])
    {
      std::cerr << "if size = " << size_[int(mapspace::Dimension::IndexFactorization)] << std::endl;
      std::cerr << "if id = " << index_factorization_id << std::endl;
      std::cerr << "split id = " << split_id_ << std::endl;
      assert(false);
      return;
    }
    
    // Find global index factorization id (across all splits).
    uint128_t mapping_index_factorization_id = index_factorization_id * num_parent_splits_ + split_id_;

    // Create a set of pruned dimensions (one per tiling level).
    std::map<unsigned, std::vector<problem::Shape::DimensionID>> pruned_dimensions;
    std::map<unsigned, unsigned> unit_factors;

    // Extract the index factors resulting from this ID for all loops at all levels.
    for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
    {
      // We won't prune spatial dimensions with user-specificed
      // spatial splits, because pruning re-orders the dimensions, which
      // changes the user-intended spatial split point. There's probably
      // a smarter way to do this, but we'll use the easy way out for now.
      if (user_spatial_splits_.find(level) == user_spatial_splits_.end())
      {
        for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumDimensions); idim++)
        { 
          auto dim = problem::Shape::DimensionID(idim);
          auto factor = index_factorization_space_.GetFactor(
            mapping_index_factorization_id, dim, level);
          if (factor == 1)
          {
            pruned_dimensions[level].push_back(dim);
          }
        }
      }
      unit_factors[level] = pruned_dimensions[level].size();
    }

    // Re-initialize the Permutation and Spatial Split sub-spaces.
    InitLoopPermutationSpace(user_permutations_, pruned_dimensions);
    InitSpatialSpace(user_spatial_splits_, unit_factors);
  }


  //------------------------------------------//
  //           Mapping Construction           // 
  //------------------------------------------//
  
  //
  // ConstructMapping()
  //   Given a multi-dimensional mapping ID within this map space,
  //   construct a full mapping.
  //
  bool ConstructMapping(
      mapspace::ID mapping_id,
      Mapping* mapping)
  {
    // FIXME: add a cache so that if any of the mapping subspace IDs are the same as the
    // last call, we can avoid some re-computing.

    assert(!IsSplit());

    // A set of subnests, one for each tiling level.
    loop::NestConfig subnests(arch_props_.TilingLevels());

    // We will construct the mapping in several stages. At each stage,
    // we will provide a private ID that indexes into the sub-space
    // for that stage.

    // The last split-ID may overflow because the search algorithm may not know
    // that it drew the short straw.
    if (mapping_id[int(mapspace::Dimension::IndexFactorization)] >=
        size_[int(mapspace::Dimension::IndexFactorization)])
    {
      assert(false);
      return false;
    }
    
    // Find global index factorization id (across all splits).
    uint128_t mapping_index_factorization_id =
      mapping_id[int(mapspace::Dimension::IndexFactorization)] * num_parent_splits_ + split_id_;
    // uint128_t mapping_index_factorization_id =
    //   size_[int(mapspace::Dimension::IndexFactorization)] * split_id +
    //   mapping_id[int(mapspace::Dimension::IndexFactorization)];
    
    uint128_t mapping_permutation_id = mapping_id[int(mapspace::Dimension::LoopPermutation)];
    uint128_t mapping_spatial_id = mapping_id[int(mapspace::Dimension::Spatial)];
    uint128_t mapping_datatype_bypass_id = mapping_id[int(mapspace::Dimension::DatatypeBypass)];

    // === Stage 0 ===
    InitSubnests(subnests);

    // === Stage 1 ===
    PermuteSubnests(mapping_permutation_id, subnests);

    // === Stage 2 ===
    AssignIndexFactors(mapping_index_factorization_id, subnests);

    // === Stage 4 ===
    mapping->datatype_bypass_nest = ConstructDatatypeBypassNest(mapping_datatype_bypass_id);

    // FIXME: optimization: if we are using the mapspace in deferred/pruned mode then
    // the index factorization does not need to be re-processed.

    // We had to reverse the order of stage 4 and 3 because AssignSpatialTilingDirections
    // needs the datatype bypass nest to determine if a spatial fanout is possible or
    // not.
    
    // === Stage 3 ===
    bool success = AssignSpatialTilingDirections(mapping_spatial_id, subnests, mapping->datatype_bypass_nest);
    if (!success)
    {
      return false;
    }

    // Concatenate the subnests to form the final mapping nest.    
    std::uint64_t storage_level = 0;
    for (uint64_t i = 0; i < arch_props_.TilingLevels(); i++)
    {
      uint64_t num_subnests_added = 0;
      for (int dim = 0; dim < int(problem::GetShape()->NumDimensions); dim++)
      {
        // Ignore trivial factors
        // This reduces computation time by 1.5x on average.
        if (subnests[i][dim].start + subnests[i][dim].stride < subnests[i][dim].end)
        {
          mapping->loop_nest.AddLoop(subnests[i][dim]);
          num_subnests_added++;
        }
      }
      if (!arch_props_.IsSpatial(i))
      {
        if (num_subnests_added == 0)
        {
          // Add a trivial temporal nest to make sure
          // we have at least one subnest in each level.
          mapping->loop_nest.AddLoop(problem::Shape::DimensionID(int(problem::GetShape()->NumDimensions) - 1),
                                     0, 1, 1, spacetime::Dimension::Time);
        }
        mapping->loop_nest.AddStorageTilingBoundary();
        storage_level++;
      }
    }

    // Finalize mapping.
    mapping->id = mapping_id.Integer();
    
    return true;
  }

  //
  // Mapping Construction
  // Stage 0: Initialize subnests.
  //
  void InitSubnests(loop::NestConfig& subnests)
  {
    // Construct num_storage_levels loop-nest partitions and assign dimensions.
    // This is the only stage at which the invariant subnests[][dim].dimension == dim
    // will hold. The subnests will later get permuted, breaking the invariant.
    for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
    {
      auto spacetime_dim = arch_props_.IsSpatial(level)
        ? spacetime::Dimension::SpaceX // Placeholder.
        : spacetime::Dimension::Time;
        
      // Each partition has problem::GetShape()->NumDimensions loops.
      for (int idim = 0; idim < int(problem::GetShape()->NumDimensions); idim++)
      {
        loop::Descriptor loop;
        loop.dimension = problem::Shape::DimensionID(idim); // Placeholder.
        loop.start = 0;
        loop.end = 0;                              // Placeholder.
        loop.stride = 1;                           // FIXME.
        loop.spacetime_dimension = spacetime_dim;
        
        subnests.at(level).push_back(loop);
      }
    }
  }

  //
  // Mapping Construction
  // Stage 1: Permute Subnests.
  //
  void PermuteSubnests(uint128_t mapping_permutation_id, loop::NestConfig& subnests)
  {
    loop::NestConfig reordered(arch_props_.TilingLevels());
    
    // Obtain a pattern of loop variables for all levels.
    auto dimensions = permutation_space_.GetPatterns(mapping_permutation_id);
    assert(dimensions.size() == subnests.size());

    for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
    {
      // Re-order the subnest based on the pattern. 
      assert(dimensions[level].size() == subnests[level].size());
      for (unsigned i = 0; i < dimensions[level].size(); i++)
      {
        auto target_dim = dimensions[level][i];
        assert(subnests[level][int(target_dim)].dimension == target_dim);
        reordered[level].push_back(subnests[level][int(target_dim)]);
      }
    }

    subnests = reordered;
  }

  //
  // Mapping Construction
  // Stage 2: Assign Index Factors.
  //
  void AssignIndexFactors(uint128_t mapping_index_factorization_id, loop::NestConfig& subnests)
  {
    for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
    {
      for (auto& loop : subnests[level])
      {
        loop.end = int(index_factorization_space_.GetFactor(
                         mapping_index_factorization_id,
                         loop.dimension,
                         level));
      }
    }
  }

  //
  // Mapping Construction
  // Stage 3: Decide which of the spatial loop nests are along the space_x dimension.
  //
  bool AssignSpatialTilingDirections(uint128_t mapping_spatial_id,
                                     loop::NestConfig& subnests,
                                     tiling::CompoundMaskNest datatype_bypass_nest)
  {
    (void) datatype_bypass_nest;
    bool success = true;

    auto spatial_splits = spatial_split_space_.GetSplits(mapping_spatial_id);
    //auto datatype_bypass_masks = tiling::TransposeMasks(datatype_bypass_nest);
    
    double cumulative_fanout_utilization = 1.0;

    for (uint64_t level = 0; level < arch_props_.TilingLevels() && success; level++)
    {
      if (!arch_props_.IsSpatial(level))
      {
        continue;
      }
      
      // Note that spatial levels are never bypassed. Therefore, we do not have
      // to deal with the bypass mask.
      // auto& datatype_bypass_mask = datatype_bypass_masks.at(storage_level-1);

      success &= AssignSpatialTilingDirections_Level_Expand(
        spatial_splits.at(level),
        subnests[level],
        level,
        cumulative_fanout_utilization);
      
    } // for (level)
    
    success &= (cumulative_fanout_utilization >= min_parallelism_);
      
    return success;
  }

  bool AssignSpatialTilingDirections_Level_Expand(std::uint32_t spatial_split,
                                                  std::vector<loop::Descriptor>& level_nest,
                                                  unsigned tiling_level_id,
                                                  double& fanout_utilization)
  {
    // This version of the function assumes that spatial tiling will expand
    // the instances for *each* datatype exactly by the tiling parameters. For
    // example, if K=16 is a spatial factor, then 16 instances of the next
    // inner level will be created for Weights, Inputs and Outputs.
    
    bool success = true;

    unsigned storage_level_id = arch_props_.TilingToStorage(tiling_level_id);
    auto level_specs = arch_specs_.topology.GetStorageLevel(storage_level_id);

    std::size_t x_expansion = 1;
    std::size_t y_expansion = 1;
    
    // Based on the spatial mapping ID, split the level nest into two sections:
    // first X and then Y.
    for (unsigned i = 0; i < level_nest.size(); i++)
    {
      auto& loop = level_nest.at(i);
      
      assert(loop::IsSpatial(loop.spacetime_dimension));
      assert(loop.stride == 1);

      if (i < spatial_split)
      {
        // X
        x_expansion *= (loop.end - loop.start);
        loop.spacetime_dimension = spacetime::Dimension::SpaceX;
      }
      else
      {
        // Y
        y_expansion *= (loop.end - loop.start);
        loop.spacetime_dimension = spacetime::Dimension::SpaceY;
      }
    }

    std::size_t fanout_max;
    
    // if (level_specs->SharingType() == model::DataSpaceIDSharing::Shared)
    // {
    if (x_expansion > arch_props_.FanoutX(storage_level_id))
      success = false;
      
    if (y_expansion > arch_props_.FanoutY(storage_level_id))
      success = false;

    fanout_max = arch_props_.Fanout(storage_level_id);
    // }
    // else
    // {
    //   std::size_t x_fanout_max = 0;
    //   std::size_t y_fanout_max = 0;

    //   // The following loop is silly since we now only allow one fanout per level
    //   // (as opposed to a per-dataspace fanout for partitioned levels). However,
    //   // we will keep the code because we may need to move to a multiple-buffers
    //   // per level later. The loop will not be over data spaces but buffer
    //   // instances per level.

    //   for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    //   {
    //     // auto pv = problem::Shape::DataSpaceID(pvi);

    //     if (x_expansion > arch_props_.FanoutX(storage_level_id))
    //       success = false;

    //     if (y_expansion > arch_props_.FanoutY(storage_level_id))
    //       success = false;

    //     // Track max available (not utilized) fanout across all datatypes.
    //     x_fanout_max = std::max(x_fanout_max, arch_props_.FanoutX(storage_level_id));
    //     y_fanout_max = std::max(y_fanout_max, arch_props_.FanoutY(storage_level_id));
    //   }

    //   fanout_max = x_fanout_max * y_fanout_max;
    // }

    // Compute fanout utilization at this level.
    // Ignore bypass and partitioning. The only purpose of this is to accumulate
    // the level-wise utilizations to compute arithmetic utilization.
    fanout_utilization *= double(x_expansion) * double(y_expansion) / fanout_max;

    // if (!success)
    // {
    //   std::cerr << "Level: " << arch_props_.StorageLevelName(storage_level_id) << std::endl;
    //   std::cerr << "  X: ";
    //   std::cerr << " expansion = " << x_expansion << " fanout = " << arch_props_.FanoutX(storage_level_id) << std::endl;
    //   std::cerr << "  Y: ";
    //   std::cerr << " expansion = " << y_expansion << " fanout = " << arch_props_.FanoutY(storage_level_id) << std::endl;
    //   std::cerr << "  util = " << fanout_utilization << std::endl;
    //   std::cerr << std::endl;
    // }
    
    return success;
  }
  
  
  //
  // Mapping Construction
  // Stage 4: Construct datatype bypass nest.
  //
  tiling::CompoundMaskNest ConstructDatatypeBypassNest(uint128_t mapping_datatype_bypass_id)
  {
    assert(mapping_datatype_bypass_id < size_[int(mapspace::Dimension::DatatypeBypass)]);
    return datatype_bypass_nest_space_.at(int(mapping_datatype_bypass_id));
  }

  //------------------------------------------//
  //                Helper/Misc.              // 
  //------------------------------------------//
  
  //
  // Parse user config.
  //
  void ParseUserConfig(
    config::CompoundConfigNode config,
    std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_factors,
    std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_max_factors,
    std::map<unsigned, std::vector<problem::Shape::DimensionID>>& user_permutations,
    std::map<unsigned, std::uint32_t>& user_spatial_splits,
    problem::PerDataSpace<std::string>& user_bypass_strings)
  {
    // This is primarily a wrapper function written to handle various ways to
    // get to the list of constraints. The only reason there are multiple ways
    // to get to this list is because of backwards compatibility.
    if (config.isList())
    {
      // We're already at the constraints list.
      
      ParseUserConstraints(config, user_factors, user_max_factors, user_permutations,
                           user_spatial_splits, user_bypass_strings);
    }
    else
    {
      // Constraints can be specified either anonymously as a list, or indirected
      // via a string name.
      std::string name = "";
      if (config.exists("constraints"))
      {
        if (config.lookup("constraints").isList())
          name = "constraints";
        else if (config.lookupValue("constraints", name))
          name = std::string("constraints_") + name;
      }
      else if (config.exists("targets"))
      {
        if (config.lookup("targets").isList())
          name = "targets";
      }

      if (name == "")
        // No constraints specified, nothing to do.
        return;

      auto constraints = config.lookup(name);
      ParseUserConstraints(constraints, user_factors, user_max_factors, user_permutations,
                           user_spatial_splits, user_bypass_strings);
    }
  }  

  //
  // Parse user-provided constraints.
  //
  void ParseUserConstraints(
    config::CompoundConfigNode constraints,
    std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_factors,
    std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_max_factors,
    std::map<unsigned, std::vector<problem::Shape::DimensionID>>& user_permutations,
    std::map<unsigned, std::uint32_t>& user_spatial_splits,
    problem::PerDataSpace<std::string>& user_bypass_strings)
  {
    assert(constraints.isList());

    // Initialize user bypass strings to "XXXXX...1" (note the 1 at the end).
    // FIXME: there's probably a cleaner way/place to initialize this.
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      std::string xxx(arch_specs_.topology.NumStorageLevels(), 'X');
      xxx.back() = '1';
      user_bypass_strings[problem::Shape::DataSpaceID(pvi)] = xxx;
    }

    // Iterate over all the constraints/targets.
    int len = constraints.getLength();
    for (int i = 0; i < len; i++)
    {
      // Prepare a set of CompoundConfigNodes that we will be parsing. These
      // nodes can be in different places in the hierarchy for backwards-
      // compatibility reasons.
      config::CompoundConfigNode target, constraint, attributes;

      target = constraints[i];
      if (target.exists("constraints"))
      {
        auto constraints_list = target.lookup("constraints");
        assert(constraints_list.isList());
        
        for (int j = 0; j < constraints_list.getLength(); j++)
        {
          auto constraint = constraints_list[j];

          config::CompoundConfigNode attributes;
          if (constraint.exists("attributes"))
            attributes = constraint.lookup("attributes");
          else
            attributes = constraint; // Backwards compatibility.

          ParseSingleConstraint(target, constraint, attributes,        
                                user_factors, user_max_factors, user_permutations,
                                user_spatial_splits, user_bypass_strings);
        }
      }
      else // Backwards compatibility.
      {
        auto constraint = target;

        config::CompoundConfigNode attributes;
        if (constraint.exists("attributes"))
          attributes = constraint.lookup("attributes");
        else
          attributes = constraint; // Backwards compatibility.

        ParseSingleConstraint(target, constraint, attributes,        
                              user_factors, user_max_factors, user_permutations,
                              user_spatial_splits, user_bypass_strings);
      }
    }    
  }

  //
  // Parse a single user constraint.
  //
  void ParseSingleConstraint(
    config::CompoundConfigNode target,
    config::CompoundConfigNode constraint,
    config::CompoundConfigNode attributes,
    std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_factors,
    std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_max_factors,
    std::map<unsigned, std::vector<problem::Shape::DimensionID>>& user_permutations,
    std::map<unsigned, std::uint32_t>& user_spatial_splits,
    problem::PerDataSpace<std::string>& user_bypass_strings)
  {
      // Find out if this is a temporal constraint or a spatial constraint.
      std::string type;
      assert(constraint.lookupValue("type", type));

      auto level_id = FindTargetTilingLevel(target, type);

      if (type == "temporal" || type == "spatial")
      {
        auto level_factors = ParseUserFactors(attributes);
        for (auto& factor: level_factors)
        {
          if (user_factors[level_id].find(factor.first) != user_factors[level_id].end())
          {
            std::cerr << "ERROR: re-specification of factor for dimension "
                      << problem::GetShape()->DimensionIDToName.at(factor.first)
                      << " at level " << arch_props_.TilingLevelName(level_id)
                      << ". This may imply a conflict between architecture and "
                      << "mapspace constraints." << std::endl;
            exit(1);
          }
          user_factors[level_id][factor.first] = factor.second;
        }

        auto level_max_factors = ParseUserMaxFactors(attributes);
        for (auto& max_factor: level_max_factors)
        {
          if (user_max_factors[level_id].find(max_factor.first) != user_max_factors[level_id].end())
          {
            std::cerr << "ERROR: re-specification of max factor for dimension "
                      << problem::GetShape()->DimensionIDToName.at(max_factor.first)
                      << " at level " << arch_props_.TilingLevelName(level_id)
                      << ". This may imply a conflict between architecture and "
                      << "mapspace constraints." << std::endl;
            exit(1);
          }
          user_max_factors[level_id][max_factor.first] = max_factor.second;
        }

        auto level_permutations = ParseUserPermutations(attributes);
        if (level_permutations.size() > 0)
        {
          if (user_permutations[level_id].size() > 0)
          {
            std::cerr << "ERROR: re-specification of permutation at level "
                      << arch_props_.TilingLevelName(level_id)
                      << ". This may imply a conflict between architecture and "
                      << "mapspace constraints." << std::endl;
            exit(1);
          }
          user_permutations[level_id] = level_permutations;
        }

        if (type == "spatial")
        {
          std::uint32_t user_split;
          if (constraint.lookupValue("split", user_split))
          {
            if (user_spatial_splits.find(level_id) != user_spatial_splits.end())
            {
              std::cerr << "ERROR: re-specification of spatial split at level "
                        << arch_props_.TilingLevelName(level_id)
                        << ". This may imply a conflict between architecture and "
                        << "mapspace constraints." << std::endl;
              exit(1);
            }
            user_spatial_splits[level_id] = user_split;
          }
        }
      }
      else if (type == "datatype" || type == "bypass" || type == "bypassing")
      {
        // Error handling for re-spec conflicts are inside the parse function.
        ParseUserDatatypeBypassSettings(attributes,
                                        arch_props_.TilingToStorage(level_id),
                                        user_bypass_strings);
      }
      else if (type == "utilization" || type == "parallelism")
      {
        if (min_parallelism_isset_)
        {
          std::cerr << "ERROR: re-specification of min parallelism/utilization at level "
                    << arch_props_.TilingLevelName(level_id)
                    << ". This may imply a conflict between architecture and "
                    << "mapspace constraints." << std::endl;
          exit(1);
        }
        assert(attributes.lookupValue("min", min_parallelism_));
        min_parallelism_isset_ = true;
      }
      else
      {
        assert(false);
      }
  }

  //
  // FindTargetTilingLevel()
  //
  unsigned FindTargetTilingLevel(config::CompoundConfigNode constraint, std::string type)
  {
    auto num_storage_levels = arch_specs_.topology.NumStorageLevels();
    
    //
    // Find the target storage level. This can be specified as either a name or an ID.
    //
    std::string storage_level_name;
    unsigned storage_level_id;
    
    if (constraint.lookupValue("target", storage_level_name) ||
        constraint.lookupValue("name", storage_level_name))
    {
      // Find this name within the storage hierarchy in the arch specs.
      for (storage_level_id = 0; storage_level_id < num_storage_levels; storage_level_id++)
      {
        if (arch_specs_.topology.GetStorageLevel(storage_level_id)->level_name == storage_level_name)
          break;
      }
      if (storage_level_id == num_storage_levels)
      {
        std::cerr << "ERROR: target storage level not found: " << storage_level_name << std::endl;
        exit(1);
      }
    }
    else
    {
      int id;
      assert(constraint.lookupValue("target", id));
      assert(id >= 0 && id < int(num_storage_levels));
      storage_level_id = static_cast<unsigned>(id);
    }
    
    assert(storage_level_id < num_storage_levels);    

    //
    // Translate this storage ID to a tiling ID.
    //
    unsigned tiling_level_id;
    if (type == "temporal" || type == "datatype" || type == "bypass" || type == "bypassing")
    {
      // This should always succeed.
      tiling_level_id = arch_props_.TemporalToTiling(storage_level_id);
    }
    else if (type == "spatial")
    {
      // This will fail if this level isn't a spatial tiling level.
      try
      {
        tiling_level_id = arch_props_.SpatialToTiling(storage_level_id);
      }
      catch (const std::out_of_range& oor)
      {
        std::cerr << "ERROR: cannot find spatial tiling level associated with "
                  << "storage level " << arch_props_.StorageLevelName(storage_level_id)
                  << ". This is because the number of instances of the next-inner "
                  << "level ";
        if (storage_level_id != 0)
        {
          std::cerr << "(" << arch_props_.StorageLevelName(storage_level_id-1) << ") ";
        }
        std::cerr << "is the same as this level, which means there cannot "
                  << "be a spatial fanout." << std::endl;
        exit(1);
      }
    }
    else if (type == "utilization" || type == "parallelism")
    {
      // For now, we only allow parallelism to be specified for level 0.
      // Note that this is the level 0 storage, not arithmetic. Fanout from
      // level 0 to arithmetic is undefined. The parallelism specified here
      // is the cumulative fanout utilization all the way from the top of
      // the storage tree down to level 0.
      if (storage_level_id != 0)
      {
        std::cerr << "ERROR: parallelism cannot be constrained at level "
                  << storage_level_name << ". It must be constrained at the "
                  << "innermost storage level." << std::endl;
        exit(1);
      }
      tiling_level_id = arch_props_.TemporalToTiling(storage_level_id);
    }
    else
    {
      std::cerr << "ERROR: unrecognized constraint type: " << type << std::endl;
      exit(1);
    }

    return tiling_level_id;
  }

  //
  // Parse user factors.
  //
  std::map<problem::Shape::DimensionID, int> ParseUserFactors(config::CompoundConfigNode constraint)
  {
    std::map<problem::Shape::DimensionID, int> retval;

    std::string buffer;
    if (constraint.lookupValue("factors", buffer))
    {
      std::regex re("([A-Za-z]+)[[:space:]]*[=]*[[:space:]]*([0-9]+)", std::regex::extended);
      std::smatch sm;
      std::string str = std::string(buffer);

      while (std::regex_search(str, sm, re))
      {
        std::string dimension_name = sm[1];
        problem::Shape::DimensionID dimension;
        try
        {
          dimension = problem::GetShape()->DimensionNameToID.at(dimension_name);
        }
        catch (const std::out_of_range& oor)
        {
          std::cerr << "ERROR: parsing factors: " << buffer << ": dimension " << dimension_name
                    << " not found in problem shape." << std::endl;
          exit(1);
        }

        int end = std::stoi(sm[2]);
        if (end == 0)
        {
          std::cerr << "WARNING: Interpreting 0 to mean full problem dimension instead of residue." << std::endl;
          end = workload_.GetBound(dimension);
        }
        else if (end > workload_.GetBound(dimension))
        {
          std::cerr << "WARNING: Constraint " << dimension_name << "=" << end
                    << " exceeds problem dimension " << dimension_name << "="
                    << workload_.GetBound(dimension) << ". Setting constraint "
                    << dimension << "=" << workload_.GetBound(dimension) << std::endl;
          end = workload_.GetBound(dimension);
        }
        else
        {
          assert(end > 0);
        }

        // Found all the information we need to setup a factor!
        retval[dimension] = end;

        str = sm.suffix().str();
      }
    }

    return retval;
  }

  //
  // Parse user max factors.
  //
  std::map<problem::Shape::DimensionID, int> ParseUserMaxFactors(config::CompoundConfigNode constraint)
  {
    std::map<problem::Shape::DimensionID, int> retval;

    std::string buffer;
    if (constraint.lookupValue("factors", buffer))
    {
      std::regex re("([A-Za-z]+)[[:space:]]*<=[[:space:]]*([0-9]+)", std::regex::extended);
      std::smatch sm;
      std::string str = std::string(buffer);

      while (std::regex_search(str, sm, re))
      {
        std::string dimension_name = sm[1];
        problem::Shape::DimensionID dimension;
        try
        {
          dimension = problem::GetShape()->DimensionNameToID.at(dimension_name);
        }
        catch (const std::out_of_range& oor)
        {
          std::cerr << "ERROR: parsing factors: " << buffer << ": dimension " << dimension_name
                    << " not found in problem shape." << std::endl;
          exit(1);
        }

        int max = std::stoi(sm[2]);
        if (max <= 0)
        {
          std::cerr << "ERROR: max factor must be positive in constraint: " << buffer << std::endl;
          exit(1);
        }

        // Found all the information we need to setup a factor!
        retval[dimension] = max;

        str = sm.suffix().str();
      }
    }

    return retval;
  }

  //
  // Parse user permutations.
  //
  std::vector<problem::Shape::DimensionID> ParseUserPermutations(config::CompoundConfigNode constraint)
  {
    std::vector<problem::Shape::DimensionID> retval;
    
    std::string buffer;
    if (constraint.lookupValue("permutation", buffer))
    {
      std::istringstream iss(buffer);
      char token;
      while (iss >> token)
      {
        problem::Shape::DimensionID dimension;
        try
        {
          dimension = problem::GetShape()->DimensionNameToID.at(std::string(1, token));
        }
        catch (const std::out_of_range& oor)
        {
          std::cerr << "ERROR: parsing permutation: " << buffer << ": dimension " << token
                    << " not found in problem shape." << std::endl;
          exit(1);
        }
        retval.push_back(dimension);
      }
    }

    return retval;
  }

  //
  // Parse user datatype bypass settings.
  //
  void ParseUserDatatypeBypassSettings(config::CompoundConfigNode constraint,
                                       unsigned level,
                                       problem::PerDataSpace<std::string>& user_bypass_strings)
  {
    // Datatypes to "keep" at this level.
    if (constraint.exists("keep"))
    {
      std::vector<std::string> datatype_strings;
      constraint.lookupArrayValue("keep", datatype_strings);
      for (const std::string& datatype_string: datatype_strings)
      {
        problem::Shape::DataSpaceID datatype;
        try
        {
          datatype = problem::GetShape()->DataSpaceNameToID.at(datatype_string);
        }
        catch (std::out_of_range& oor)
        {
          std::cerr << "ERROR: parsing keep setting: data-space " << datatype_string
                    << " not found in problem shape." << std::endl;
          exit(1);
        }
        // FIXME: no error handling for overwriting last-level bypass setting.
        if (level != arch_specs_.topology.NumStorageLevels()-1 &&
            user_bypass_strings.at(datatype).at(level) != 'X')
        {
          std::cerr << "ERROR: re-specification of dataspace keep flag at level "
                    << arch_props_.StorageLevelName(level) << ". This may imply a "
                    << "conflict between architecture and mapspace constraints."
                    << std::endl;
          exit(1);          
        }
        user_bypass_strings.at(datatype).at(level) = '1';
      }
    }
      
    // Datatypes to "bypass" at this level.
    if (constraint.exists("bypass"))
    {
      std::vector<std::string> datatype_strings;
      constraint.lookupArrayValue("bypass", datatype_strings);
      for (const std::string& datatype_string: datatype_strings)
      {
        problem::Shape::DataSpaceID datatype;
        try
        {
          datatype = problem::GetShape()->DataSpaceNameToID.at(datatype_string);
        }
        catch (std::out_of_range& oor)
        {
          std::cerr << "ERROR: parsing bypass setting: data-space " << datatype_string
                    << " not found in problem shape." << std::endl;
          exit(1);
        }
        if (level != arch_specs_.topology.NumStorageLevels()-1 &&
            user_bypass_strings.at(datatype).at(level) != 'X')
        {
          std::cerr << "ERROR: re-specification of dataspace bypass flag at level "
                    << arch_props_.StorageLevelName(level) << ". This may imply a "
                    << "conflict between architecture and mapspace constraints."
                    << std::endl;
          exit(1);          
        }
        user_bypass_strings.at(datatype).at(level) = '0';
      }
    }
  }
};


} // namespace mapspace
