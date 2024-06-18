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

#include "mapspaces/uber.hpp"

namespace mapspace
{

//--------------------------------------------//
//                Uber MapSpace               //
//--------------------------------------------//

//
// Uber() - Derived mapspace classes may wish to call with skip_init
//          set to true and then explicitly call Init(), possibly with
//          customized pre-mapped tiles.
//
Uber::Uber(
  config::CompoundConfigNode config,
  config::CompoundConfigNode arch_constraints,
  model::Engine::Specs arch_specs,
  const problem::Workload& workload,
  bool filter_spatial_fanout,
  bool skip_init) :
    
    MapSpace(arch_specs, workload),

    workload_(workload),
    permutation_space_(workload),
    index_factorization_space_(workload),
    spatial_split_space_(workload),
    
    split_id_(0),
    num_parent_splits_(0),
    arch_props_(arch_specs),
    constraints_(arch_props_, workload),
    filter_spatial_fanout_(filter_spatial_fanout)
{
  if (!skip_init)
  {
    Init(config, arch_constraints);
  }
}

Uber::~Uber()
{
  for (auto split : splits_)
  {
    delete split;
  }
  splits_.clear();
}

//------------------------------------------//
//        Initialization and Setup          // 
//------------------------------------------//
  
//
// Init() - called by derived classes or by constructor.
//
void Uber::Init(config::CompoundConfigNode config, config::CompoundConfigNode arch_constraints)
{
  // Parse user config.
  Parse(config, arch_constraints);

  // Setup all the mapping sub-spaces.
  InitIndexFactorizationSpace();
  InitLoopPermutationSpace();
  InitSpatialSpace();
  InitDatatypeBypassNestSpace();

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
void Uber::InitIndexFactorizationSpace()
{
  auto user_factors = constraints_.Factors();
  auto user_max_factors = constraints_.MaxFactors();
  auto user_min_factors = constraints_.MinFactors();

  assert(user_factors.size() <= arch_props_.TilingLevels());

  // We'll initialize the index_factorization_space_ object here. To do that, we first
  // need to determine the number of factors that *each* problem dimension needs to be
  // split up into. In other words, this is the order of the cofactors vector for each
  // problem dimension.
    
  std::map<problem::Shape::FlattenedDimensionID, std::uint64_t> cofactors_order;
  for (unsigned i = 0; i < unsigned(workload_.GetShape()->NumFlattenedDimensions); i++)
  {
    // Factorize each problem dimension into num_tiling_levels partitions.
    cofactors_order[problem::Shape::FlattenedDimensionID(i)] = arch_props_.TilingLevels();
  }

  // Next, for each problem dimension, we need to tell the index_factorization_space_
  // object if any of the cofactors have been given a fixed, min or max value by
  // the user. 

  std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>> prefactors;
  std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>> maxfactors;
  std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>> minfactors;
  std::vector<bool> exhausted_um_loops(int(workload_.GetShape()->NumFlattenedDimensions), false);

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

  // Find user-specified min factors.
  for (unsigned level = 0; level < arch_props_.TilingLevels(); level++)
  {
    auto it = user_min_factors.find(level);
    if (it != user_min_factors.end())
    {
      // Some min factors exist for this level.        
      for (auto& factor : it->second)
      {
        auto& dimension = factor.first;
        auto& min = factor.second;
        minfactors[dimension][level] = min;
      }
    }
  }

  // We're now ready to initialize the object.
  index_factorization_space_.Init(cofactors_order, prefactors, maxfactors, minfactors);

  // Update the size of the mapspace.
  size_[int(mapspace::Dimension::IndexFactorization)] = index_factorization_space_.Size();
}

//
// InitLoopPermutationSpace()
//
void Uber::InitLoopPermutationSpace(std::map<unsigned, std::vector<problem::Shape::FlattenedDimensionID>> pruned_dimensions)
{
  auto user_permutations = constraints_.Permutations();

  permutation_space_.Init(arch_props_.TilingLevels());
    
  for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
  {
    // Extract the user-provided pattern for this level.
    std::vector<problem::Shape::FlattenedDimensionID> user_prefix;
    std::vector<problem::Shape::FlattenedDimensionID> user_suffix;
    auto it = user_permutations.find(level);
    if (it != user_permutations.end())
    {
      user_prefix = it->second.first;
      user_suffix = it->second.second;
    }

    bool use_canonical_permutation = false;
    if (arch_props_.IsSpatial(level))
    {
      use_canonical_permutation = user_prefix.empty() && user_suffix.empty() && !arch_props_.IsSpatial2D(level);
    }
    else
    {
      use_canonical_permutation = (level == 0);
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
        permutation_space_.InitLevel(level, user_prefix, user_suffix, it->second);
      else
        permutation_space_.InitLevel(level, user_prefix, user_suffix);
    }
  }    

  size_[int(mapspace::Dimension::LoopPermutation)] = permutation_space_.Size();
}

//
// InitSpatialSpace()
//
void Uber::InitSpatialSpace(std::map<unsigned, unsigned> unit_factors)
{
  auto user_spatial_splits = constraints_.SpatialSplits();

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

void Uber::InitDatatypeBypassNestSpace()
{
  auto user_bypass_strings = constraints_.BypassStrings();

  // The user_mask input is a set of per-datatype strings. Each string has a length
  // equal to num_storage_levels, and contains the characters 0 (bypass), 1 (keep),
  // or X (evaluate both).
    
  // A CompoundMaskNest is effectively a PerDataSpace<std::bitset<MaxTilingLevels>>.
  // The datatype_bypass_nest_space_ is a vector of CompoundMaskNests.

  // First, seed the space with a single mask with each bit set to 1.
  assert(datatype_bypass_nest_space_.empty());
  tiling::CompoundMaskNest seed_mask_nest;
  for (unsigned pvi = 0; pvi < unsigned(workload_.GetShape()->NumDataSpaces); pvi++)
  {
    for (unsigned level = 0; level < arch_specs_.topology.NumStorageLevels(); level++)
    {
      seed_mask_nest.at(pvi).set(level);
    }
  }
  datatype_bypass_nest_space_.push_back(seed_mask_nest);

  // Now parse the user strings and edit/expand the space as necessary.
  for (unsigned pvi = 0; pvi < unsigned(workload_.GetShape()->NumDataSpaces); pvi++)
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

void Uber::InitPruned(uint128_t index_factorization_id)
{
  assert(!IsSplit());

  auto user_permutations = constraints_.Permutations();
  auto user_spatial_splits = constraints_.SpatialSplits();

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
  std::map<unsigned, std::vector<problem::Shape::FlattenedDimensionID>> pruned_dimensions;
  std::map<unsigned, unsigned> unit_factors;

  // Extract the index factors resulting from this ID for all loops at all levels.
  for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
  {
    // We won't prune spatial dimensions with user-specificed
    // spatial splits, because pruning re-orders the dimensions, which
    // changes the user-intended spatial split point. There's probably
    // a smarter way to do this, but we'll use the easy way out for now.
    if (user_spatial_splits.find(level) == user_spatial_splits.end())
    {
      for (unsigned idim = 0; idim < unsigned(workload_.GetShape()->NumFlattenedDimensions); idim++)
      { 
        auto dim = problem::Shape::FlattenedDimensionID(idim);
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
  InitLoopPermutationSpace(pruned_dimensions);
  InitSpatialSpace(unit_factors);
}

//
// Split the mapspace (used for parallelization).
//
std::vector<MapSpace*> Uber::Split(std::uint64_t num_splits)
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

void Uber::InitSplit(std::uint64_t split_id, uint128_t split_if_size, std::uint64_t num_parent_splits)
{
  split_id_ = split_id;
  size_[int(mapspace::Dimension::IndexFactorization)] = split_if_size;
  num_parent_splits_ = num_parent_splits;
}

bool Uber::IsSplit()
{
  return (splits_.size() > 0);
}


//------------------------------------------//
//           Mapping Construction           // 
//------------------------------------------//
  
//
// ConstructMapping()
//   Given a multi-dimensional mapping ID within this map space,
//   construct a full mapping.
//
std::vector<Status> Uber::ConstructMapping(
  mapspace::ID mapping_id,
  Mapping* mapping,
  bool break_on_failure)
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
    std::cerr << "FATAL ERROR: mapspace asked to construct a mapping with an out-of-bounds "
              << "IndexFactorization coordinate." << std::endl;
    assert(false);
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
  auto status = AssignSpatialTilingDirections(mapping_spatial_id,
                                              subnests,
                                              mapping->datatype_bypass_nest,
                                              break_on_failure);
  bool success = std::accumulate(status.begin(), status.end(), true,
                                 [](bool cur, const Status& status)
                                 { return cur && status.success; });
  if (break_on_failure && !success)
  {
    return status;
  }

  // Concatenate the subnests to form the final mapping nest.    
  for (uint64_t i = 0; i < arch_props_.TilingLevels(); i++)
  {
    uint64_t num_subnests_added = 0;
    for (int dim = 0; dim < int(workload_.GetShape()->NumFlattenedDimensions); dim++)
    {
      // Ignore trivial factors
      // This reduces computation time by 1.5x on average.
      if (subnests[i][dim].start + subnests[i][dim].stride < subnests[i][dim].end)
      {
        mapping->loop_nest.AddLoop(subnests[i][dim]);
        num_subnests_added++;
      }
      mapping->complete_loop_nest.AddLoop(subnests[i][dim]);
    }
    if (!arch_props_.IsSpatial(i))
    {
      if (num_subnests_added == 0)
      {
        // Add a trivial temporal nest to make sure
        // we have at least one subnest in each level.
        mapping->loop_nest.AddLoop(problem::Shape::FlattenedDimensionID(int(workload_.GetShape()->NumFlattenedDimensions) - 1),
                                   0, 1, 1, spacetime::Dimension::Time);
      }
      mapping->loop_nest.AddStorageTilingBoundary();
      mapping->complete_loop_nest.AddStorageTilingBoundary();
    }
  }

  mapping->confidence_thresholds = constraints_.ConfidenceThresholds();
  mapping->id = mapping_id.Integer();
  mapping->fanoutX_map = arch_props_.FanoutX();
  mapping->fanoutY_map = arch_props_.FanoutY();
  mapping->loop_nest.skew_descriptors = constraints_.Skews();
  mapping->loop_nest.no_link_transfer = constraints_.NoLinkTransfers();
  mapping->loop_nest.no_multicast = constraints_.NoMulticast();
  mapping->loop_nest.no_temporal_reuse = constraints_.NoTemporalReuse();
  mapping->loop_nest.rmw_first_update = constraints_.RMWOnFirstWriteback();
  mapping->loop_nest.no_coalesce = constraints_.no_coalesce();

  return status;
}

//
// Mapping Construction
// Stage 0: Initialize subnests.
//
void Uber::InitSubnests(loop::NestConfig& subnests)
{
  // Construct num_storage_levels loop-nest partitions and assign dimensions.
  // This is the only stage at which the invariant subnests[][dim].dimension == dim
  // will hold. The subnests will later get permuted, breaking the invariant.
  for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
  {
    auto spacetime_dim = arch_props_.IsSpatial(level)
      ? spacetime::Dimension::SpaceX // Placeholder.
      : spacetime::Dimension::Time;
        
    // Each partition has workload_.GetShape()->NumFlattenedDimensions loops.
    for (int idim = 0; idim < int(workload_.GetShape()->NumFlattenedDimensions); idim++)
    {
      loop::Descriptor loop;
      loop.dimension = problem::Shape::FlattenedDimensionID(idim); // Placeholder.
      loop.start = 0;
      loop.end = 0;                              // Placeholder.
      loop.residual_end = 0;                     // Placeholder.
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
void Uber::PermuteSubnests(uint128_t mapping_permutation_id, loop::NestConfig& subnests)
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
void Uber::AssignIndexFactors(uint128_t mapping_index_factorization_id, loop::NestConfig& subnests)
{
  std::map<problem::Shape::FlattenedDimensionID, int> remaining;
  for (int idim = 0; idim < int(problem::GetShape()->NumFlattenedDimensions); idim++)
  {
    auto dim = problem::Shape::FlattenedDimensionID(idim);
    remaining[dim] = workload_.GetFlattenedBound(dim);
  }


  for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
  {
    for (auto it = subnests[level].rbegin(); it != subnests[level].rend(); it++)
    {
      auto& loop = *it;
      loop.end = int(index_factorization_space_.GetFactor(
                       mapping_index_factorization_id,
                       loop.dimension,
                       level));
      if (loop.end > remaining[loop.dimension]) loop.end = remaining[loop.dimension];

      if(remaining[loop.dimension] % loop.end == 0) 
      {
        loop.residual_end = loop.end; // Perfect factorization.
        remaining[loop.dimension] /= loop.end;
      }
      else 
      {
        
        loop.residual_end = remaining[loop.dimension] % loop.end;
        remaining[loop.dimension] = ceil((double)remaining[loop.dimension] / (double) loop.end);
      }
    } 
  }

  // Create a modifiable map that reverse iterates through all loops for easy access.
  std::vector<loop::Descriptor> loops;
  for (int level = arch_props_.TilingLevels() - 1; level >= 0; level--)
  {
    for (auto it = subnests[level].rbegin(); it != subnests[level].rend(); it++)
    {
      loops.push_back(*it);
    }
  }
  // Assert that the imperfect factorization formula checks out
  for (int idim = 0; idim < int(problem::GetShape()->NumFlattenedDimensions); idim++)
  {
    auto dim = problem::Shape::FlattenedDimensionID(idim);
    assert(remaining[dim] == 1);

    // Check the recurrence relation
    int d0 = 1;
    // Two nested loops, i and j, over the reverse loop iterator above
    for (auto it = loops.begin(); it != loops.end(); it++)
    {
      auto& loop = *it;
      int curterm = 1;
      if(loop.dimension != dim) continue;
      for (auto it2 = it + 1; it2 != loops.end(); it2++)
      {
        auto& loop2 = *it2;
        if(loop2.dimension != dim) continue;
        curterm *= loop2.end;
      }
      d0 += curterm * (loop.residual_end - 1);
    }
    if(d0 != workload_.GetFlattenedBound(dim)) std::cout << "Dimension " << dim << " has a factorization error. d0 = " << d0 << " and bound = " << workload_.GetFlattenedBound(dim) << std::endl;
    if(d0 != workload_.GetFlattenedBound(dim))
    {
      int j = 0;
      for(auto loop3 = loops.begin(); loop3 != loops.end(); loop3++)
      {
        if(loop3->end == 1) continue;
        for(int q = 0; q < j; q++) std::cout << "  ";
        std::cout << "Loop " << j << ": "
                  << loop3->PrintCompact(workload_.GetShape()->FlattenedDimensionIDToName);
        // if(loop3->level == loop->level) std::cout << " <---";
        std::cout << std::endl;
        j++;
      }
      std::cout << "Exiting..." << std::endl;
      exit(1);
    }
  }
}

//
// Mapping Construction
// Stage 3: Decide which of the spatial loop nests are along the space_x dimension.
//
std::vector<Status> Uber::AssignSpatialTilingDirections(uint128_t mapping_spatial_id,
                                                        loop::NestConfig& subnests,
                                                        tiling::CompoundMaskNest datatype_bypass_nest,
                                                        bool break_on_failure)
{
  (void) datatype_bypass_nest;

  std::vector<Status> status(arch_props_.Specs().topology.NumLevels(),
                             { .success = true, .fail_reason = "" });

  auto spatial_splits = spatial_split_space_.GetSplits(mapping_spatial_id);
  //auto datatype_bypass_masks = tiling::TransposeMasks(datatype_bypass_nest);
    
  double cumulative_fanout_utilization = 1.0;

  for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
  {
    if (!arch_props_.IsSpatial(level))
    {
      continue;
    }
      
    // Note that spatial levels are never bypassed. Therefore, we do not have
    // to deal with the bypass mask.
    // auto& datatype_bypass_mask = datatype_bypass_masks.at(storage_level-1);

    auto s = AssignSpatialTilingDirections_Level_Expand(
      spatial_splits.at(level),
      subnests[level],
      level,
      cumulative_fanout_utilization);

    unsigned storage_level = arch_props_.TilingToStorage(level);
    unsigned topology_level = arch_props_.Specs().topology.StorageMap(storage_level);

    // Merge with existing failures at this level.
    if (status.at(topology_level).success)
      status.at(topology_level) = s;
    else
      status.at(topology_level).fail_reason += ", " + s.fail_reason;

    // success &= s.success;

    if (break_on_failure && !s.success)
      break;
      
  } // for (level)
    
  if (cumulative_fanout_utilization < constraints_.MinParallelism())
  {
    std::ostringstream fail_reason;
    fail_reason << "parallelism " << cumulative_fanout_utilization << " is less than "
                << "constrained min-parallelism " << constraints_.MinParallelism();

    // Report this as an arithmetic-level failure.
    unsigned topology_level = arch_props_.Specs().topology.ArithmeticMap();

    // Merge with existing failures at this level.
    if (status.at(topology_level).success)
      status.at(topology_level) = { false, fail_reason.str() };
    else
      status.at(topology_level).fail_reason += ", " + fail_reason.str();
  }

  return status;
}

Status Uber::AssignSpatialTilingDirections_Level_Expand(std::uint32_t spatial_split,
                                                        std::vector<loop::Descriptor>& level_nest,
                                                        unsigned tiling_level_id,
                                                        double& fanout_utilization)
{
  // This version of the function assumes that spatial tiling will expand
  // the instances for *each* datatype exactly by the tiling parameters. For
  // example, if K=16 is a spatial factor, then 16 instances of the next
  // inner level will be created for Weights, Inputs and Outputs.
  bool success = true;
  std::ostringstream fail_reason;

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
  if (filter_spatial_fanout_ && x_expansion > arch_props_.FanoutX(storage_level_id))
  {
    success = false;
    fail_reason << "mapped fanoutX " << x_expansion << " exceeds hardware fanoutX "
                << arch_props_.FanoutX(storage_level_id);
  }
      
  if (filter_spatial_fanout_ && y_expansion > arch_props_.FanoutY(storage_level_id))
  {
    success = false;
    fail_reason << "mapped fanoutY " << y_expansion << " exceeds hardware fanoutY "
                << arch_props_.FanoutY(storage_level_id);
  }

  fanout_max = arch_props_.Fanout(storage_level_id);

  // Compute fanout utilization at this level.
  // Ignore bypass and partitioning. The only purpose of this is to accumulate
  // the level-wise utilizations to compute arithmetic utilization.
  fanout_utilization *= double(x_expansion) * double(y_expansion) / fanout_max;

  Status status;
  status.success = success;
  status.fail_reason = fail_reason.str();

  return status;
}
  
  
//
// Mapping Construction
// Stage 4: Construct datatype bypass nest.
//
tiling::CompoundMaskNest Uber::ConstructDatatypeBypassNest(uint128_t mapping_datatype_bypass_id)
{
  assert(mapping_datatype_bypass_id < size_[int(mapspace::Dimension::DatatypeBypass)]);
  return datatype_bypass_nest_space_.at(int(mapping_datatype_bypass_id));
}

//------------------------------------------//
//                 Parsing                  // 
//------------------------------------------//
  
//
// Parse.
//
void Uber::Parse(config::CompoundConfigNode config, config::CompoundConfigNode arch_constraints)
{
  // Parse constraints.
  // We accept mapspace config and arch_constraints as separate configuration
  // trees, but as far as parsing is concerned we handle them in exactly the
  // same way. The underlying parsing methods are built to handle conflicts.
  constraints_.Parse(config);
  constraints_.Parse(arch_constraints);
}

} // namespace mapspace
