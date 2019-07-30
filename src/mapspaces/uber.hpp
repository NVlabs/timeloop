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

#include <iterator>
#include <mutex>
#include <regex>

#include "util/numeric.hpp"
#include "util/misc.hpp"
#include "workload/problem-shape.hpp"
#include "mapspaces/mapspace-base.hpp"
#include "mapspaces/subspaces.hpp"
#include "compound-config/compound-config.hpp"

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

  // Architecture specs
  uint64_t num_temporal_tiling_levels_;
  uint64_t num_spatial_tiling_levels_;
  uint64_t num_total_tiling_levels_;  // temporal + spatial
  
  std::vector<bool> spatial_mask_;       // across all levels
  std::vector<bool> twoD_spatial_mask_;  // across all levels

  // Maps to index between the different tiling spaces.
  std::map<unsigned, unsigned> temporal_to_tiling_map_;
  std::map<unsigned, unsigned> spatial_to_tiling_map_;
  std::map<unsigned, unsigned> tiling_to_storage_map_;

  // Minimum utilization (constraint).
  double min_utilization_;
  
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
      min_utilization_(0.0)
  {
    // Some pre-processing.
    InitSpatialMasks();

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

    // Parse user-provided constraints.
    ParseUserConstraints(config, user_factors_, user_max_factors_, user_permutations_,
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
      std::cout << "Mapping Dimension [" << mapspace::Dimension(i)
                << "] Size: " << size_[i] << std::endl;
    }
    
    // check for integer overflow in the above multiplications
    uint128_t de_cumulative_prod = Size();
    for (int i = 0; i < int(mapspace::Dimension::Num); i++)
    {
      de_cumulative_prod /= size_[i];
    }
    assert(de_cumulative_prod == 1);
  }
  
  //
  // InitSpatialMasks()
  //   Scan the arch specs to figure out how many temporal and spatial tiling
  //   levels we need to completely map onto the architecture.
  //
  void InitSpatialMasks()
  {
    auto num_storage_levels = arch_specs_.topology.NumStorageLevels();
    
    // one temporal partition for each storage level
    num_temporal_tiling_levels_ = num_storage_levels;

    uint64_t cur_tiling_level = 0;
    for (uint64_t i = 0; i < num_storage_levels; i++)
    {
      // Peek at the fanout in the arch specs to figure out if this is a
      // purely temporal level, a 1D spatial level or a 2D spatial level.

      // For partitioned levels, we have to look at all partitions. If
      // any of the partitions have a spatial fanout, then we treat
      // this as a spatial level.
      bool is_spatial = false;
      bool is_spatial_2D = false;

      auto& specs = *arch_specs_.topology.GetStorageLevel(i);
      auto lambda = [&] (problem::Shape::DataSpaceID pv)
        {
          if (specs.Fanout(pv).Get() > 1)
            is_spatial = true;
          if (specs.FanoutX(pv).Get() > 1 && specs.FanoutY(pv).Get() > 1)
            is_spatial_2D = true;
        };
      model::BufferLevel::ForEachDataSpaceID(lambda, specs.sharing_type);

      if (is_spatial)
      {
        // This is a spatial level.
        spatial_mask_.push_back(true);
        twoD_spatial_mask_.push_back(is_spatial_2D);
        spatial_to_tiling_map_[i] = cur_tiling_level;
        tiling_to_storage_map_[cur_tiling_level] = i;
        cur_tiling_level++;
      }
      
      // There is always a temporal level
      spatial_mask_.push_back(false);
      twoD_spatial_mask_.push_back(false);

      temporal_to_tiling_map_[i] = cur_tiling_level;
      tiling_to_storage_map_[cur_tiling_level] = i;
      cur_tiling_level++;      
    }

    num_total_tiling_levels_ = spatial_mask_.size();
    assert(twoD_spatial_mask_.size() == num_total_tiling_levels_);
  }  

  //
  // InitIndexFactorizationSpace()
  //
  void InitIndexFactorizationSpace(std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_factors,
                                   std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_max_factors)
  {
    assert(user_factors.size() <= num_total_tiling_levels_);

    // We'll initialize the index_factorization_space_ object here. To do that, we first
    // need to determine the number of factors that *each* problem dimension needs to be
    // split up into. In other words, this is the order of the cofactors vector for each
    // problem dimension.
    
    std::map<problem::Shape::DimensionID, std::uint64_t> cofactors_order;
    for (unsigned i = 0; i < unsigned(problem::GetShape()->NumDimensions); i++)
    {
      // Factorize each problem dimension into num_tiling_levels partitions.
      cofactors_order[problem::Shape::DimensionID(i)] = num_total_tiling_levels_;
    }

    // Next, for each problem dimension, we need to tell the index_factorization_space_
    // object if any of the cofactors have been given a fixed, min or max value by
    // the user. 

    std::map<problem::Shape::DimensionID, std::map<unsigned, unsigned long>> prefactors;
    std::map<problem::Shape::DimensionID, std::map<unsigned, unsigned long>> maxfactors;
    std::vector<bool> exhausted_um_loops(int(problem::GetShape()->NumDimensions), false);

    // Find user-specified fixed factors.
    for (unsigned level = 0; level < num_total_tiling_levels_; level++)
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
    for (unsigned level = 0; level < num_total_tiling_levels_; level++)
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
    permutation_space_.Init(num_total_tiling_levels_);
    
    for (uint64_t level = 0; level < num_total_tiling_levels_; level++)
    {
      // Extract the user-provided pattern for this level.
      std::vector<problem::Shape::DimensionID> user_prefix;
      auto it = user_permutations.find(level);
      if (it != user_permutations.end())
      {
        user_prefix = it->second;
      }

      bool use_canonical_permutation = false;
      if (IsSpatialTilingLevel(level))
      {
        use_canonical_permutation = user_prefix.empty() && !twoD_spatial_mask_[level];
      }
      else
      {
        use_canonical_permutation = (level == 0); // || level == num_total_tiling_levels_-1); // FIXME: last level?
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
    spatial_split_space_.Init(num_total_tiling_levels_);
    
    for (uint64_t level = 0; level < num_total_tiling_levels_; level++)
    {
      if (IsSpatialTilingLevel(level))
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

    // Some sanity checking.
    for (unsigned i = 0; i < arch_specs_.topology.NumStorageLevels(); i++)
    {
      auto& specs = *arch_specs_.topology.GetStorageLevel(i);
      auto lambda = [&] (problem::Shape::DataSpaceID pv)
        {
          assert(specs.FanoutX(pv).IsSpecified() && specs.FanoutY(pv).IsSpecified());
        };
      model::BufferLevel::ForEachDataSpaceID(lambda, specs.sharing_type);
    }
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

  // Uber(Uber* parent, std::uint64_t split_id, uint128_t split_residue) :
  //     permutation_space_(parent->permutation_space_),
  //     index_factorization_space_(parent->index_factorization_space_),
  //     spatial_split_space_(parent->spatial_split_space_),
  //     datatype_bypass_nest_space_(parent->datatype_bypass_nest_space_),
  //     split_residue_(split_residue),
  //     splits_(),
  //     user_factors_(parent->user_factors_),
  //     user_permutations_(parent->user_permutations_),
  //     user_spatial_splits_(parent->user_spatial_splits_),
  //     user_bypass_strings_(parent->user_bypass_strings_),
  //     num_temporal_tiling_levels_(parent->num_temporal_tiling_levels_),
  //     num_spatial_tiling_levels_(parent->num_spatial_tiling_levels_),
  //     num_total_tiling_levels_(parent->num_total_tiling_levels_),
  //     spatial_mask_(parent->spatial_mask_),
  //     twoD_spatial_mask_(parent->twoD_spatial_mask_),
  //     temporal_to_tiling_map_(parent->temporal_to_tiling_map_),
  //     spatial_to_tiling_map_(parent->spatial_to_tiling_map_),
  //     tiling_to_storage_map_(parent->tiling_to_storage_map_),
  //     min_utilization_(parent->min_utilization_)
  // {
    
  // }

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
    for (uint64_t level = 0; level < num_total_tiling_levels_; level++)
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
    loop::NestConfig subnests(num_total_tiling_levels_);

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
    for (uint64_t i = 0; i < num_total_tiling_levels_; i++)
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
      if (!IsSpatialTilingLevel(i))
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
    for (uint64_t level = 0; level < num_total_tiling_levels_; level++)
    {
      auto spacetime_dim = IsSpatialTilingLevel(level)
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
    loop::NestConfig reordered(num_total_tiling_levels_);
    
    // Obtain a pattern of loop variables for all levels.
    auto dimensions = permutation_space_.GetPatterns(mapping_permutation_id);
    assert(dimensions.size() == subnests.size());

    for (uint64_t level = 0; level < num_total_tiling_levels_; level++)
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
    for (uint64_t level = 0; level < num_total_tiling_levels_; level++)
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

    for (uint64_t level = 0; level < num_total_tiling_levels_ && success; level++)
    {
      if (!IsSpatialTilingLevel(level))
      {
        continue;
      }
      
      auto storage_level = tiling_to_storage_map_[level];
      auto level_specs = arch_specs_.topology.GetStorageLevel(storage_level);

      // We need to pass in the datatype bypass mask for the level receiving the
      // fanout from this level.
      // FIXME: we do not support fanout from storage level 0.
      //assert (storage_level >= 1);
      //auto& datatype_bypass_mask = datatype_bypass_masks.at(storage_level-1);

      success &= AssignSpatialTilingDirections_Level_Expand(
        spatial_splits.at(level),
        subnests[level],
        *level_specs,
        //datatype_bypass_mask,
        cumulative_fanout_utilization);

      // if (level_specs.SharingType() == model::Level::DataSpaceIDSharing::Partitioned)
      // {
      //   success &= AssignSpatialTilingDirections_PartitionedLevel(
      //     mapping_spatial_id, subnests[level], level_specs);
      // }
      // else
      // {
      //   success &= AssignSpatialTilingDirections_SharedLevel(
      //     mapping_spatial_id, subnests[level], level_specs);
      // }
      
    } // for (level)
    
    success &= (cumulative_fanout_utilization >= min_utilization_);
      
    return success;
  }

  bool AssignSpatialTilingDirections_Level_Expand(std::uint32_t spatial_split,
                                                  std::vector<loop::Descriptor>& level_nest,
                                                  model::BufferLevel::Specs& level_specs,
                                                //  tiling::CompoundMask& datatype_bypass_mask,
                                                  double& fanout_utilization)
  {
    // This version of the function assumes that spatial tiling will expand
    // the instances for *each* datatype exactly by the tiling parameters. For
    // example, if K=16 is a spatial factor, then 16 instances of the next
    // inner level will be created for Weights, Inputs and Outputs.
    //(void) datatype_bypass_mask;
    
    bool success = true;

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
    
    if (level_specs.SharingType() == model::BufferLevel::DataSpaceIDSharing::Shared)
    {
      if (x_expansion > level_specs.FanoutX().Get())
        success = false;
      
      if (y_expansion > level_specs.FanoutY().Get())
        success = false;

      fanout_max = level_specs.FanoutX().Get() * level_specs.FanoutY().Get();
    }
    else
    {
      std::size_t x_fanout_max = 0;
      std::size_t y_fanout_max = 0;

      for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
      {
        auto pv = problem::Shape::DataSpaceID(pvi);

        if (x_expansion > level_specs.FanoutX(pv).Get())
          success = false;

        if (y_expansion > level_specs.FanoutY(pv).Get())
          success = false;

        // Track max available (not utilized) fanout across all datatypes.
        x_fanout_max = std::max(x_fanout_max, level_specs.FanoutX(pv).Get());
        y_fanout_max = std::max(y_fanout_max, level_specs.FanoutY(pv).Get());
      }

      fanout_max = x_fanout_max * y_fanout_max;
    }

    // Compute fanout utilization at this level.
    // Ignore bypass and partitioning. The only purpose of this is to accumulate
    // the level-wise utilizations to compute arithmetic utilization.
    fanout_utilization *= double(x_expansion) * double(y_expansion) / fanout_max;

    // if (!success)
    // {
    //   std::cerr << "Level: " << level_specs.level_name << std::endl;
    //   std::cerr << "  X: ";
    //   std::cerr << " expansion = " << x_expansion << " fanout = " << level_specs.FanoutX().Get() << std::endl;
    //   std::cerr << "  Y: ";
    //   std::cerr << " expansion = " << y_expansion << " fanout = " << level_specs.FanoutY().Get() << std::endl;
    //   std::cerr << "  util = " << fanout_utilization << std::endl;
    //   std::cerr << std::endl;
    // }
    
    return success;
  }
  
  bool AssignSpatialTilingDirections_Level_MaxWS(std::uint32_t spatial_split,
                                                 std::vector<loop::Descriptor>& level_nest,
                                                 model::BufferLevel::Specs& level_specs,
                                                 tiling::CompoundMask& datatype_bypass_mask,
                                                 double& fanout_utilization)
  {

    bool success = true;

    problem::OperationPoint origin;
    problem::OperationPoint x_dimensions;
    problem::OperationPoint y_dimensions;

    x_dimensions.IncrementAllDimensions(); // initialize to { 1, 1, 1, ...}
    y_dimensions.IncrementAllDimensions(); // initialize to { 1, 1, 1, ...}

    std::size_t dimension_expansion = 1;
    
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
        x_dimensions[loop.dimension] = loop.end;
        loop.spacetime_dimension = spacetime::Dimension::SpaceX;
      }
      else
      {
        // Y
        y_dimensions[loop.dimension] = loop.end;
        loop.spacetime_dimension = spacetime::Dimension::SpaceY;
      }

      dimension_expansion *= loop.end;
    }

    // Look at the size of each datatype within the point sets. This tells us
    // the required fanout. Match this against the available fanout in the arch
    // spec.
    
    // ****** FIXME ****** this logic appears to be incorrect. It will compute
    // the number of distinct partitions that each datatype fans out to. However,
    // that is not what the hardware organization intended - it may have wanted
    // to keep copies of each datatype. In other words, any potential multicast
    // happens from *this* level down, not from the next level down. Fortunately,
    // the tile analysis stage seems to be doing the right thing later on.

    // origin gives us the low corner (inclusive) of the operation space.
    // x/y_dimensions gives the high corner (exclusive) of the operation space.
    // We need the inclusive high corner to build the operation space. See
    // OperationSpace constructor for details.
    problem::OperationPoint x_high = x_dimensions;
    x_high.IncrementAllDimensions(-1);
    problem::OperationPoint y_high = y_dimensions;
    y_high.IncrementAllDimensions(-1);

    problem::OperationSpace x_space(&workload_, origin, x_high);
    problem::OperationSpace y_space(&workload_, origin, y_high);
    
    auto x_sizes = x_space.GetSizes();
    auto y_sizes = y_space.GetSizes();

    std::size_t fanout_max;
    
    if (level_specs.SharingType() == model::BufferLevel::DataSpaceIDSharing::Shared)
    {
      // Shared level: required fanout is the max across all datatypes **kept at this level**.
      std::size_t x_max = 0;
      std::size_t y_max = 0;

      for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
      {
        if (datatype_bypass_mask.at(pvi))
        {
          x_max = std::max(x_max, x_sizes.at(problem::Shape::DataSpaceID(pvi)));
          y_max = std::max(y_max, y_sizes.at(problem::Shape::DataSpaceID(pvi)));
        }
      }

      if (x_max > level_specs.FanoutX().Get())
        success = false;
      
      if (y_max > level_specs.FanoutY().Get())
        success = false;

      fanout_max = level_specs.FanoutX().Get() * level_specs.FanoutY().Get();
    }
    else
    {
      // Partitioned: check each datatype.
      std::size_t x_fanout_max = 0;
      std::size_t y_fanout_max = 0;

      for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
      {
        auto pv = problem::Shape::DataSpaceID(pvi);
        if (x_sizes.at(pv) > level_specs.FanoutX(pv).Get())
          success = false;
        if (y_sizes.at(pv) > level_specs.FanoutY(pv).Get())
          success = false;

        // Track max available (not utilized) fanout across all datatypes.
        x_fanout_max = std::max(x_fanout_max, level_specs.FanoutX(pv).Get());
        y_fanout_max = std::max(y_fanout_max, level_specs.FanoutY(pv).Get());
      }

      fanout_max = x_fanout_max * y_fanout_max;
    }

    // Compute fanout utilization at this level.
    // Ignore bypass and partitioning. The only purpose of this is to accumulate
    // the level-wise utilizations to compute arithmetic utilization.
    fanout_utilization *= double(dimension_expansion) / fanout_max;

    // if (success)
    // {
    //   std::cerr << "X: ";
    //   for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    //     std::cerr << problem::Shape::DataSpaceID(pvi) << " = " << x_sizes.at(problem::Shape::DataSpaceID(pvi)) << " ";
    //   std::cerr << " max = " << x_max << " fanout = " << level_specs.FanoutX().Get() << std::endl;
    //   std::cerr << "Y: ";
    //   for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    //     std::cerr << problem::Shape::DataSpaceID(pvi) << " = " << y_sizes.at(problem::Shape::DataSpaceID(pvi)) << " ";
    //   std::cerr << " max = " << y_max << " fanout = " << level_specs.FanoutY().Get() << std::endl;
    //   std::cerr << "util = " << fanout_utilization << std::endl;
    //   std::cerr << std::endl;
    // }
    
    return success;
  }
  
  // *** DEPRECATED ***
  // AssignSpatialTilingDirections_SharedLevel()
  //   Each loop walks through a problem dimension. How does this affect
  //   the spatial fanout? Is it in the problem space or in each individual
  //   datatype's range for that problem dimension? For example, consider CNN's
  //   C dimension. If we have a partitioned level, this should not scale the
  //   child level's #instances (and therefore this level's fanout) for outputs.
  //   But what about CNN's P (output width) dimension? Will the fanout for inputs
  //   scale by P or (P + halo)?
  //
  //   For now, assume that we scale by problem dimension ***even if that dimension
  //   does not project into an operand datatype***. This will probably
  //   reject some valid spatial mappings.
  //
  bool AssignSpatialTilingDirections_SharedLevel(uint128_t& mapping_spatial_id,
                                                 std::vector<loop::Descriptor>& level_nest,
                                                 model::BufferLevel::Specs& level_specs)
  {
    // Until we re-enable flexible X/Y mesh, the spatial mapping ID doesn't serve
    // any purpose.
    (void)mapping_spatial_id;
    
    assert(level_specs.SharingType() == model::BufferLevel::DataSpaceIDSharing::Shared);

    bool success = true;
    
    // Assign the correct spatio-temporal type to each loop. Start with X,
    // then move to Y. Assume no loss in generality from doing this.
    problem::PerDataSpace<std::uint64_t> x_fanout(1);
    problem::PerDataSpace<std::uint64_t> y_fanout(1);

    bool x_exhausted = false;

    for (auto loop = level_nest.begin(); loop != level_nest.end() && success; loop++)
    {
      assert(loop::IsSpatial(loop->spacetime_dimension));

      if (!x_exhausted)
      {
        // Iterate through each data type and determine how much this loop
        // would cause that datatype's fanout to inflate.
        problem::PerDataSpace<std::uint64_t> upd_fanout;
        for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
        {
          auto pv = problem::Shape::DataSpaceID(pvi);
          if (true) // problem::IsSensitive(pv, loop->dimension))
          {
            upd_fanout[pv] = x_fanout[pv] * loop->end;
          }
        }

        // See if we can accommodate this loop within the X dimension.           
        // Compare this fanout vs. the available fanout in the arch spec.
        if (upd_fanout.Max() <= level_specs.FanoutX().Get())
        {
          // Yes, we can accept this loop.
          loop->spacetime_dimension = spacetime::Dimension::SpaceX;
          x_fanout = upd_fanout;
        }
        else
        {
          // Nope, we're done with X, switch to Y. Also rewind the iterator
          // so that we re-try this loop in the Y dimension.
          x_exhausted = true;
          --loop;
        }
      }
      else
      {
        // X is exhausted, we *have* to spill into Y.

        // Iterate through each data type and determine how much this loop
        // would cause that datatype's fanout to inflate.
        problem::PerDataSpace<std::uint64_t> upd_fanout;
        for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
        {
          auto pv = problem::Shape::DataSpaceID(pvi);
          if (true) // problem::IsSensitive(pv, loop->dimension))
          {
            upd_fanout[pv] = y_fanout[pv] * loop->end;
          }
        }

        // Compare this fanout vs. the available fanout in the arch spec.
        if (upd_fanout.Max() <= level_specs.FanoutY().Get())
        {
          // Yes, we can accept this loop.
          loop->spacetime_dimension = spacetime::Dimension::SpaceY;
          y_fanout = upd_fanout;
        }
        else
        {
          // We're out of dimensions. This mapping has failed.
          // std::cout << "FAIL x_fanout = " << x_fanout.Max()
          //           << " spec_x_fanout = " << level_specs.FanoutX().Get()
          //           << " need y_fanout = " << y_fanout.Max()
          //           << " spec_y_fanout = " << level_specs.FanoutY().Get()
          //           << std::endl;
          success = false;
        }
      } // x_exhausted.
    } // for (loop in level_nest)

    return success;
  }
  
  // *** DEPRECATED ***
  // AssignSpatialTilingDirections_PartitionedLevel()
  // FIXME: This function is incomplete. Logic needs to be thought through. Here's the
  //        challenge: Each loop walks through a problem dimension. How does this affect
  //        the spatial fanout? Is it in the problem space or in each individual
  //        datatype's range for that problem dimension? For example, consider CNN's
  //        C dimension. If we have a partitioned level, this should not scale the
  //        child level's #instances (and therefore this level's fanout) for outputs.
  //        But what about CNN's P (output width) dimension? Will the fanout for inputs
  //        scale by P or (P + halo)?
  //
  bool AssignSpatialTilingDirections_PartitionedLevel(uint128_t& mapping_spatial_id,
                                                      std::vector<loop::Descriptor>& level_nest,
                                                      model::BufferLevel::Specs& level_specs)
  {
    assert(level_specs.SharingType() == model::BufferLevel::DataSpaceIDSharing::Partitioned);

    (void)level_nest;
    assert(false);
    
    // Until we re-enable flexible X/Y mesh, the spatial mapping ID doesn't serve
    // any purpose.
    (void)mapping_spatial_id;

    bool success = true;

    // // Assign the correct spatio-temporal type to each loop. Start with X,
    // // then move to Y. Assume no loss in generality from doing this.
    // problem::PerDataSpace<std::uint64_t> x_fanout(1);
    // problem::PerDataSpace<std::uint64_t> y_fanout(1);

    // bool x_exhausted = false;

    // for (auto loop = level_nest.begin(); loop != level_nest.end && success; loop++)
    // {
    //   assert(loop::IsSpatial(loop->spacetime_dimension));

    //   if (!x_exhausted)
    //   {
    //     // Iterate through each data type and determine how much this loop
    //     // would cause that datatype's fanout to inflate.
    //     problem::PerDataSpace<std::uint64_t> upd_fanout;
    //     for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    //     {
    //       auto pv = problem::Shape::DataSpaceID(pvi);
    //       if (problem::IsSensitive(pv, loop->dimension))
    //       {
    //         upd_fanout[pv] = x_fanout[pv] * loop->end;
    //       }
    //     }

    //     // See if we can accommodate this loop within the X dimension.           
    //     // Compare this fanout vs. the available fanout in the arch spec.
    //     if (upd_fanout.Max() <= level_specs.FanoutX())
    //     {
    //       // Yes, we can accept this loop.
    //       loop->spacetime_dimension = spacetime::Dimension::SpaceX;
    //       x_fanout = upd_fanout;
    //     }
    //     else
    //     {
    //       // Nope, we're done with X, switch to Y. Also rewind the iterator
    //       // so that we re-try this loop in the Y dimension.
    //       x_exhausted = true;
    //       --loop;
    //     }
    //   }
    //   else
    //   {
    //     // X is exhausted, we *have* to spill into Y.

    //     // Iterate through each data type and determine how much this loop
    //     // would cause that datatype's fanout to inflate.
    //     problem::PerDataSpace<std::uint64_t> upd_fanout;
    //     for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    //     {
    //       auto pv = problem::Shape::DataSpaceID(pvi);
    //       if (problem::IsSensitive(pv, loop->dimension))
    //       {
    //         upd_fanout[pv] = y_fanout[pv] * loop->end;
    //       }
    //     }

    //     // Compare this fanout vs. the available fanout in the arch spec.
    //     if (upd_fanout.Max() <= level_specs.FanoutY())
    //     {
    //       // Yes, we can accept this loop.
    //       loop->spacetime_dimension = spacetime::Dimension::SpaceY;
    //       y_fanout = upd_fanout;
    //     }
    //     else
    //     {
    //       // We're out of dimensions. This mapping has failed.
    //       success = false;
    //     }
    //   } // x_exhausted.
    // } // for (loop in level_nest)

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
  // IsTwoDimensionalSpatialLevel()
  //
  bool IsTwoDimensionalSpatialLevel(int level)
  {
    assert(IsSpatialTilingLevel(level));
    return twoD_spatial_mask_[level];
  }

  //
  // IsSpatialTilingLevel()
  //
  bool IsSpatialTilingLevel(int level)
  {
    return spatial_mask_[level];
  }


  //
  // Parse user-provided constraints.
  //
  void ParseUserConstraints(
    config::CompoundConfigNode config,
    std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_factors,
    std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& user_max_factors,
    std::map<unsigned, std::vector<problem::Shape::DimensionID>>& user_permutations,
    std::map<unsigned, std::uint32_t>& user_spatial_splits,
    problem::PerDataSpace<std::string>& user_bypass_strings)
  {
    std::string name;
    if (!config.lookupValue("constraints", name))
    {
      // No constraint name specified, nothing to do.
      return;
    }

    // Find the name of the constraint.
    name = std::string("constraints_") + name;
    auto constraints = config.lookup(name);
    assert(constraints.isList());

    // Initialize user bypass strings to "XXXXX...1" (note the 1 at the end).
    // FIXME: there's probably a cleaner way/place to initialize this.
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      std::string xxx(arch_specs_.topology.NumStorageLevels(), 'X');
      xxx.back() = '1';
      user_bypass_strings[problem::Shape::DataSpaceID(pvi)] = xxx;
    }

    // Iterate over all the constraints.
    int len = constraints.getLength();
    for (int i = 0; i < len; i++)
    {
      auto constraint = constraints[i];
      // Find out if this is a temporal constraint or a spatial constraint.
      std::string type;
      assert(constraint.lookupValue("type", type));

      auto level_id = FindTargetTilingLevel(constraint, type);

      if (type == "temporal" || type == "spatial")
      {
        auto level_factors = ParseUserFactors(constraint);
        if (level_factors.size() > 0)
        {
          user_factors[level_id] = level_factors;
        }

        auto level_max_factors = ParseUserMaxFactors(constraint);
        if (level_max_factors.size() > 0)
        {
          user_max_factors[level_id] = level_max_factors;
        }

        auto level_permutations = ParseUserPermutations(constraint);
        if (level_permutations.size() > 0)
        {
          user_permutations[level_id] = level_permutations;
        }

        if (type == "spatial")
        {
          std::uint32_t user_split;
          if (constraint.lookupValue("split", user_split))
          {
            user_spatial_splits[level_id] = user_split;
          }
        }
      }
      else if (type == "datatype")
      {
        ParseUserDatatypeBypassSettings(constraint,
                                        tiling_to_storage_map_[level_id],
                                        user_bypass_strings);
      }
      else if (type == "utilization" || type == "parallelism")
      {
        assert(constraint.lookupValue("min", min_utilization_));
      }
      else
      {
        assert(false);
      }
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
    
    if (constraint.lookupValue("target", storage_level_name))
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
    if (type == "temporal" || type == "datatype")
    {
      // This should always succeed.
      tiling_level_id = temporal_to_tiling_map_.at(storage_level_id);
    }
    else if (type == "spatial")
    {
      // This will fail if this level isn't a spatial tiling level.
      auto it = spatial_to_tiling_map_.find(storage_level_id);
      if (it != spatial_to_tiling_map_.end())
      {
        tiling_level_id = it->second;
      }
      else
      {
        std::cerr << "ERROR: " << storage_level_name
                  << " is not a spatial tiling level (no fanout)."
                  << std::endl;
        exit(1);
      }
    }
    else if (type == "utilization" || type == "parallelism")
    {
      // For now, we only allow utilization to be specified for level 0.
      // Note that this is the level 0 storage, not arithmetic. Fanout from
      // level 0 to arithmetic is undefined. The utilization specified here
      // is the cumulative fanout utilization all the way from the top of
      // the storage tree down to level 0.
      if (storage_level_id != 0)
      {
        std::cerr << "ERROR: utilization cannot be constrained at level "
                  << storage_level_name << ". It must be constrained at the "
                  << "innermost storage level." << std::endl;
        exit(1);
      }
      tiling_level_id = temporal_to_tiling_map_.at(storage_level_id);
    }
    else
    {
      assert(false);
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
      std::regex re("([A-Za-z]+)[[:space:]]*([0-9]+)", std::regex::extended);
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
        user_bypass_strings.at(datatype).at(level) = '0';
      }
    }
  }
};


} // namespace mapspace
