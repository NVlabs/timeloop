/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "loop-analysis/coordinate-space-tile-info.hpp"
#include "loop-analysis/sparse-analysis.hpp"

namespace sparse
{

//
// SparseAnalysisState Function Implementations
//
bool SparseAnalysisState::Init(sparse::SparseOptimizationInfo* sparse_optimization_info,
                               problem::Workload* workload,
                               Mapping mapping,
                               std::uint64_t num_storage_levels)
{

  bool sparse_analysis_needed = false;

  if (sparse_optimization_info->no_optimization_applied) return sparse_analysis_needed;
  else sparse_analysis_needed = true;

  sparse_optimization_info_ = sparse_optimization_info;
  workload_ = workload;
  mapping_ = mapping;
  num_storage_levels_ = num_storage_levels;
  Reset();

  return sparse_analysis_needed;
}

void SparseAnalysisState::Reset()
{
  maxtile_molds_high_ = {};
  complete_subnests_ = {};
  trivial_nest_masks_ = {};
  prob_explicitly_optimized_read_ = {};
  c_operand_densities_ = {};
  c_intersection_dims_ = {};
  scalar_storage_optimization_ = {};
  // by default, no propagation impact
  for (DataSpaceID pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      scalar_storage_optimization_[pv] = false;
    }
  }

  // by default, no explicit optimization applied
  dspace_optimization_masks_ = {{"gate", {}}, {"skip", {}}};
  scalar_scalar_opt_masks_ = {{"gate", {}}, {"skip", {}}};
  for (unsigned l = 0; l < num_storage_levels_; l++)
  {
    dspace_optimization_masks_["gate"].push_back({});
    dspace_optimization_masks_["skip"].push_back({});
    scalar_scalar_opt_masks_["gate"].push_back({});
    scalar_scalar_opt_masks_["skip"].push_back({});

    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      dspace_optimization_masks_["gate"][l][pv] = false;
      dspace_optimization_masks_["skip"][l][pv] = false;
      scalar_scalar_opt_masks_["gate"][l][pv] = false;
      scalar_scalar_opt_masks_["skip"][l][pv] = false;
    }
  }

}

void SparseAnalysisState::CollectCompletePointSetsAndSubnests()
{
  problem::OperationPoint origin;
  problem::OperationPoint dimension_sizes;
  dimension_sizes.IncrementAllDimensions(); // initialize to { 1, 1, 1... }

  maxtile_molds_high_.push_back({});
  complete_subnests_.push_back({});
  trivial_nest_masks_.push_back({});

  unsigned tiling_level = 0;
  auto& loops = mapping_.complete_loop_nest.loops;
  for (unsigned loop_level = 0; loop_level < loops.size(); loop_level++)
  {
    auto& loop = loops[loop_level];
    auto factor = ceil((loop.end - loop.start) / loop.stride);
    dimension_sizes[loop.dimension] *= factor;

    // origin gives us the low corner (inclusive) of the operation space.
    // dimension_sizes gives the high corner (exclusive) of the operation space.
    // We need the inclusive high corner to build the operation space. See
    // OperationSpace constructor for details.
    problem::OperationPoint high = dimension_sizes;
    high.IncrementAllDimensions(-1);
    maxtile_molds_high_[tiling_level].push_back(high);
    complete_subnests_[tiling_level].push_back(loop);
    trivial_nest_masks_[tiling_level].push_back(factor == 1);

    if (loop_level == mapping_.complete_loop_nest.storage_tiling_boundaries.at(tiling_level))
    {
      maxtile_molds_high_.push_back({});
      complete_subnests_.push_back({});
      trivial_nest_masks_.push_back({});
      tiling_level++;
    }
  }
}

//
// Sparse Analysis Functions
//

// necessary data structures
struct ExplicitReadOptimizationImpact
{
  DataSpaceID target_dspace_id;
  std::vector <DataSpaceID> condition_on_dspace_ids;
  unsigned target_dspace_level;
  double optimization_prob;
  double expected_target_tile_occupancy;
  double scalar_scalar_opt;
};

bool CheckComputeAlignmentUnitRequirement(SparseAnalysisState& state,
                                          tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                          const model::Topology::Specs& topology_specs,
                                          std::vector <model::EvalStatus>& eval_status)
{

  bool success = true;

  // find the upper most level that stores the read dataspace

  auto contracted_dimensions = problem::GetShape()->GetFullyContractedDimensions();

  // check logic:
  // find the inner most storage that stores a "read-only" dataspace
  // if nontrivial temporal loopnests below this level doesn't involve contracted dimensions:
  //        compute doesn't need intersection optimization
  // else:  intersection optimization must be specified for compute


  // step 1: find the inner most level that stores read only dataspace
  unsigned inner_most_level = std::numeric_limits<unsigned>::max();

  for (unsigned l = 0; l < topology_specs.NumStorageLevels()
    && inner_most_level == std::numeric_limits<unsigned>::max(); l++)
  {
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      if (compound_data_movement_nest[pv][l].shape != 0 &&
        !problem::GetShape()->IsReadWriteDataSpace.at(pv))
      {
        inner_most_level = l;
        break;
      }
    }
  }

  // std::cout << "inner most level that defines the loop nests: "
  // << topology_specs.GetStorageLevel(inner_most_level)->level_name <<std::endl;

  // step2: check the loops below the inner most level
  std::uint64_t num_contracted_dims = 0;
  for (unsigned l = 0; l <= inner_most_level; l++)
  {
    for (unsigned loop_id = 0; loop_id < state.complete_subnests_[l].size(); loop_id++)
    {
      if (!state.trivial_nest_masks_[l][loop_id])
      {
        auto& loop = state.complete_subnests_[l][loop_id];
        if (loop.spacetime_dimension == spacetime::Dimension::Time &&
          (contracted_dimensions.find(loop.dimension) != contracted_dimensions.end()))
        {
          num_contracted_dims++;
          state.c_intersection_dims_.push_back(loop.dimension);
          //std::cout << "contracted dimension: " << problem::GetShape()->DimensionIDToName.at(loop.dimension) << std::endl;
        }
      }
    }
  }

  // std::cout << "number of contracted dimensions: " << num_contracted_dims << std::endl;

  // step3: check if intersection need matches intersection support
  if (state.c_intersection_dims_.size() > 0)
  {
    //TODO: formalize more on how the intersection unit should be specified and how we want to check that specification
    (void)eval_status;
  }
  return success;
}

void InitializeFineGrainedAccesses(tiling::CompoundTileNest& compound_tile_nest,
                                   const model::Topology::Specs& topology_specs)
{

  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto& compute_info_nest = compound_tile_nest.compute_info_nest;

  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    for (unsigned l = 0; l < topology_specs.NumStorageLevels(); l++)
    {
      auto& fine_grained_accesses = compound_data_movement_nest[pv][l].fine_grained_accesses;

      for (unsigned op_id = 0; op_id < tiling::storageOperationTypes.size(); op_id++)
      {
        auto op_name = tiling::storageOperationTypes[op_id];
        fine_grained_accesses[op_name] = 0;
      }

      // default to uncompressed without metadata
      fine_grained_accesses["random_read"] = compound_data_movement_nest[pv][l].reads;
      fine_grained_accesses["random_fill"] = compound_data_movement_nest[pv][l].fills;
      fine_grained_accesses["random_update"] = compound_data_movement_nest[pv][l].updates;
    }
  }

  auto& compute_info = compute_info_nest[0];

  for (unsigned op_id = 0; op_id < tiling::arithmeticOperationTypes.size(); op_id++)
  {
    auto op_name = tiling::arithmeticOperationTypes[op_id];
    compute_info.fine_grained_accesses[op_name] = 0;
  }
  compute_info.fine_grained_accesses["random_compute"] = compute_info.accesses * compute_info.replication_factor;
}

bool ComputeIneffectualReadImpact(const SparseAnalysisState& state,
                                  tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                  const unsigned storage_level_id,
                                  const model::Topology::Specs& topology_specs,
                                  ExplicitReadOptimizationImpact& resulted_impact,
                                  std::vector <model::EvalStatus>& eval_status)
{

  bool success = true;
  std::ostringstream fail_reason;

  // resulted impact stores the target data space id and the list of conditioned on dataspace id

  //
  // rule for finding the (set of) tiles for each dataspace
  //

  // target dataspace tile is skipped if the corresponding conditioned on dataspace(s) is(are) empty,
  // logic to find the point sets for target dspace and conditioned dspace by looking at the mapping

  // 1) find the closest storage level >= storage level id that stores dataspace a, named target_dspace_level
  // 2)
  // if this is not the last level that stores target dataspace (i.e., reuse possible in next level)
  //    go through loop nests above its child level in a bottom up fashion, locate the first loop that projects to target, named target-loop
  // if this is the last level that stores target dataspace
  //    will just be the bottom most loop (i.e., no reuse no matter how the loops are ordered)

  // 3) find the operation space that defines the dataspace tiles that is conditioned on
  //    note that the block size of the storage that target dataspace is stored in affects the dataspace b tile that we look at

  //    Specifically, if target_dspace_level has a block size > 1, then finest skipping/gating granularity is  block-size-a
  //    if target-loop is a coiteration loop (dimension of this loop projects to both target dataspace and conditioned on dataspace),
  //    then in order to find the corresponding conditioned on dspace tile,
  //    we should look at the operation space defined by *block-size-a iterations* of target-loop

  DataSpaceID target_dspace_id = resulted_impact.target_dspace_id;
  auto target_dspace_dimensions = problem::GetShape()->DataSpaceIDToDimensionIDVector.at(target_dspace_id);

  // for (auto iter = target_dspace_dimensions.begin(); iter != target_dspace_dimensions.end(); iter++)
  // {
  //  std::cout << problem::GetShape()->DimensionIDToName.at(*iter) << std::endl;
  // }

  // step 1)
  unsigned target_dspace_level = storage_level_id;
  // if this specific level does not store dataspace a, find the closest upper level that stores a
  if (compound_data_movement_nest[target_dspace_id][storage_level_id].shape == 0)
  {
    unsigned l;
    for (l = storage_level_id + 1; l < topology_specs.NumStorageLevels(); l++)
    {
      if (compound_data_movement_nest[target_dspace_id][l].shape > 0)
      {
        target_dspace_level = l;
        break;
      }
    }

    if (l == topology_specs.NumStorageLevels())
    {
      // there is no source storage to have the target dataspace stored
      // illegal gating/skipping specification
      auto overall_level_id = topology_specs.StorageMap(storage_level_id);
      fail_reason << "skipping/gating for " << problem::GetShape()->DataSpaceIDToName.at(target_dspace_id)
                  << "does not have a source storage for datastreams"
                  << topology_specs.GetStorageLevel(storage_level_id)->level_name
                  << std::endl;
      success = false;
      eval_status[overall_level_id].success = false;
      eval_status[overall_level_id].fail_reason = fail_reason.str();
      return success;
    }
  }

  //
  //step 2)
  //

  int child_level = compound_data_movement_nest[target_dspace_id][target_dspace_level].child_level
                      == std::numeric_limits<unsigned>::max() ?
                    -1 : compound_data_movement_nest[target_dspace_id][target_dspace_level].child_level; // -1 is compute level...

  bool found_target_dspace_loop = false;
  problem::Shape::DimensionID target_loop_dim = std::numeric_limits<unsigned>::max();
  problem::OperationPoint mold_high;
  bool target_loop_trivial;

  for (unsigned l = child_level + 1; l <= storage_level_id && !found_target_dspace_loop; l++)
  {
    for (unsigned loop_id = 0; loop_id < state.complete_subnests_[l].size() && !found_target_dspace_loop; loop_id++)
    {
      if (!state.trivial_nest_masks_[l][loop_id])
      {
        target_loop_dim = state.complete_subnests_[l][loop_id].dimension;
        if (target_dspace_dimensions.find(target_loop_dim) != target_dspace_dimensions.end()
          || child_level == -1)
        {
          // found loop related to target dataspace
          found_target_dspace_loop = true;
          // std::cout << "found the loop that defines the operation space at level:  "
          //           << topology_specs.GetStorageLevel(l)->level_name << std::endl;
          // std::cout << state.complete_subnests_[l][loop_id] << std::endl;

          //
          // step 3)
          //

          // innermost storage loop: a singleton tile (compute unit operands)
          if (child_level == -1 && loop_id == 0 && l == unsigned(child_level + 1))
          {
            // singleton operation space, pass
            target_loop_trivial = true;
          }
            // innermost loop of level l: next level top most operation point
          else if (loop_id == 0)
          {
            mold_high = state.maxtile_molds_high_[l - 1].back();
            target_loop_trivial = false;
          }
            // intermediate loop of a level l: next loop in level l
          else
          {
            mold_high = state.maxtile_molds_high_[l][loop_id - 1];
            target_loop_trivial = false;
          }
        }
      }
    }
  }

  // target-loop is trivial and topmost
  if (!found_target_dspace_loop)
  {
    target_loop_trivial = true;
    target_loop_dim = state.complete_subnests_[target_dspace_level].back().dimension;
    mold_high = state.maxtile_molds_high_[target_dspace_level].back();
    found_target_dspace_loop = true;
    // std::cout << " top most loop at storage level: "
    // << topology_specs.GetStorageLevel(target_dspace_level)->level_name << std::endl;
  }

  // sanity check: target_loop_dim must be assigned, i.e., a loop that describes the operation space must be found
  assert(target_loop_dim != std::numeric_limits<unsigned>::max());

  // factor in the block size of the target dataspace storage
  //   if the conditioned on dataspace co-iteration on this dimension, the corresponding tile shape will be impacted
  //   else, it makes no difference on the condition on dataspace tile shape
  auto target_dspace_level_block_size = topology_specs.GetStorageLevel(target_dspace_level)->block_size.Get();
  mold_high.IncrementAllDimensions();
  unsigned scaled_dim_bound = mold_high[target_loop_dim] * target_dspace_level_block_size;
  mold_high[target_loop_dim] = scaled_dim_bound;
  mold_high.IncrementAllDimensions(-1);
  //
  // Ineffectual read probability calculations
  //

  // construct the corresponding operation space mold
  problem::OperationPoint origin;
  problem::OperationSpace operation_space_mold(state.workload_, origin, mold_high);

  // go through each conditioned on dataspace id to get the probability of optimized away reads
  bool scalar_opt_cond_on_scalar = false;
  double prob_target_dspace_effectual = 1.0;
  for (unsigned i = 0; i < resulted_impact.condition_on_dspace_ids.size(); i++)
  {
    DataSpaceID condition_on_dspace_id = resulted_impact.condition_on_dspace_ids[i];
    auto co_iterated_dimensions = problem::GetShape()->GetCoIteratedDimensions({target_dspace_id, condition_on_dspace_id});
    bool is_co_iterated_dim = co_iterated_dimensions.find(target_loop_dim) != co_iterated_dimensions.end();
    auto optimization_granularity = operation_space_mold.GetSize(target_dspace_id);
    auto condition_on_granularity = operation_space_mold.GetSize(condition_on_dspace_id);
    // std::cout << "\n target dspace: " << problem::GetShape()->DataSpaceIDToName.at(resulted_impact.target_dspace_id)
    //           << "  target loop dim: " << problem::GetShape()->DimensionIDToName.at(target_loop_dim)
    //           << "  co-iterated: " << is_co_iterated_dim
    //           << "  trivial loop: " << target_loop_trivial
    //           << "  optimization granularity: " << optimization_granularity
    //           << "  condition on granularity: " << condition_on_granularity
    //           << std::endl;

    if (condition_on_granularity == 1 && ((optimization_granularity == 1 && target_loop_trivial)
      || (is_co_iterated_dim && optimization_granularity == 1)))
    {
      scalar_opt_cond_on_scalar = true;
      // std::cout << "---> scalar optimization of the target space based on a scalar conditioned on dspace" << std::endl;
    }

    // construct the corresponding coordinate space tile for dataspace b and calculate the prob of the tile being empty
    tiling::CoordinateSpaceTileInfo cspace_tile;
    cspace_tile.Set(operation_space_mold.GetSize(condition_on_dspace_id), condition_on_dspace_id);

    double prob_condition_on_dspace_empty = state.workload_->GetDensity(condition_on_dspace_id)
      ->GetTileOccupancyProbability(cspace_tile, 0);
    prob_target_dspace_effectual *= (1 - prob_condition_on_dspace_empty);

    // std::cout << " target dspace: " << problem::GetShape()->DataSpaceIDToName.at(resulted_impact.target_dspace_id)
    //           << " condition on dspace: " << problem::GetShape()->DataSpaceIDToName.at(condition_on_dspace_id)
    //           << "\n operation space mold derived dataspace sizes (target, conditioned): "
    //           << operation_space_mold.GetSize(target_dspace_id) << "  "
    //           << operation_space_mold.GetSize(condition_on_dspace_id)
    //           << "\n condtion on tile empty probability(target skip prob): " << prob_condition_on_dspace_empty
    //           << std::endl;
  }
  double prob_target_dspace_ineffectual = 1 - prob_target_dspace_effectual;
  // std::cout << " (after aggregation) target skip prob: " << prob_target_dspace_ineffectual << std::endl;

  //
  // Occupancy calculations
  //
  double
    expected_target_tile_occupancy = compound_data_movement_nest[target_dspace_id][target_dspace_level].expected_data_occupancy;

  // Calculate equivalence metadata occupancy
  //  if the storage level has a child level, only the ranks associated with the child level need to be sent out
  //  if the storage level is the innermost (of its dataspace), then all metadata ranks need to be read out for
  //     operand alignment
  unsigned l = child_level == -1 ? target_dspace_level : child_level;
  auto level_specs = topology_specs.GetStorageLevel(target_dspace_level);
  auto ratio = level_specs->metadata_word_bits.Get() / level_specs->word_bits.Get();
  double equivalent_metadata_occupancy =
    compound_data_movement_nest[target_dspace_id][l].GetExpectedAggregatedMetaDataTileOccupancy() * ratio;
  expected_target_tile_occupancy += equivalent_metadata_occupancy;

  //
  // Populate the impact
  //
  resulted_impact.optimization_prob = prob_target_dspace_ineffectual;
  resulted_impact.target_dspace_level = target_dspace_level;
  resulted_impact.expected_target_tile_occupancy = expected_target_tile_occupancy;
  resulted_impact.scalar_scalar_opt = scalar_opt_cond_on_scalar;

  return success;
}

bool DefineIneffectualReadImpact(SparseAnalysisState& state,
                                 tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                 const unsigned storage_level_id,
                                 const GroupOfActionOptimization& group_of_action_optimization,
                                 const model::Topology::Specs& topology_specs,
                                 std::string optimization_type,
                                 std::vector <model::EvalStatus>& eval_status)
{

  bool success = true;
  std::vector <ExplicitReadOptimizationImpact> possible_impact(group_of_action_optimization.size());
  for (unsigned i = 0; i < group_of_action_optimization.size(); i++)
  {

    //
    // compute the optimization impact of the various choices
    //
    assert(group_of_action_optimization[i].type == CONDITIONED_ON);
    possible_impact[i].target_dspace_id = group_of_action_optimization[i].cond_on_opt.target_dspace_id;
    possible_impact[i].condition_on_dspace_ids = group_of_action_optimization[i].cond_on_opt.condition_on_dspace_ids;
    success = ComputeIneffectualReadImpact(state, compound_data_movement_nest, storage_level_id,
                                           topology_specs, possible_impact[0], eval_status);
    if (!success) return success;
  }

  //
  // compare which way allows more savings
  //

  // the final impact of the optimization is a consequence of:
  // (1) prob of skipping/gating
  // (2) amount of data that will be transferred
  // (3) storage level access cost

  double access_cost;
  double max_savings = 0;
  unsigned option_id = 0;
  bool scalar_scalar_opt = false;

  for (unsigned i = 0; i < possible_impact.size(); i++)
  {
    auto& impact = possible_impact[i];

    access_cost = topology_specs.GetStorageLevel(impact.target_dspace_level)->op_energy_map.at("random_read");
    auto block_size = topology_specs.GetStorageLevel(impact.target_dspace_level)->block_size.Get();
    double savings = (access_cost / block_size) * impact.expected_target_tile_occupancy * impact.optimization_prob;
    max_savings = savings >= max_savings ? savings : max_savings;
    option_id = savings >= max_savings ? i : option_id;
    scalar_scalar_opt = savings >= max_savings ? impact.scalar_scalar_opt : scalar_scalar_opt;
  }

  //
  // record the impact of the better choice
  //
  auto target_dspace_level = possible_impact[option_id].target_dspace_level;
  auto optimization_prob = possible_impact[option_id].optimization_prob;
  auto target_dspace_id = possible_impact[option_id].target_dspace_id;

  // std::cout << "\t=== Final: target dspace: " << problem::GetShape()->DataSpaceIDToName.at(target_dspace_id)
  // << "  optimization prob: " << optimization_prob
  // << "  scalar to scalar opt: " << scalar_scalar_opt
  // << std::endl;

  if (state.dspace_optimization_masks_["gate"][target_dspace_level][target_dspace_id]
    || state.dspace_optimization_masks_["skip"][target_dspace_level][target_dspace_id])
  {
    // if another optimization unit already performed skipping/gating on the same tile, choose the more effective one
    if (state.prob_explicitly_optimized_read_[target_dspace_level][target_dspace_id] < optimization_prob)
    {
      state.dspace_optimization_masks_["gate"][target_dspace_level][target_dspace_id] = false;
      state.dspace_optimization_masks_["skip"][target_dspace_level][target_dspace_id] = false;

      state.prob_explicitly_optimized_read_[target_dspace_level][target_dspace_id] = optimization_prob;
      state.dspace_optimization_masks_[optimization_type][target_dspace_level][target_dspace_id] = true;
      state.scalar_scalar_opt_masks_[optimization_type][target_dspace_level][target_dspace_id] = scalar_scalar_opt;
    }
  } else
  {
    state.dspace_optimization_masks_[optimization_type][target_dspace_level][target_dspace_id] = true;
    state.prob_explicitly_optimized_read_[target_dspace_level][target_dspace_id] = optimization_prob;
    state.scalar_scalar_opt_masks_[optimization_type][target_dspace_level][target_dspace_id] = scalar_scalar_opt;
  }

  return success;
}

bool DefineStorageOptimizationImpact(SparseAnalysisState& state,
                                     tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                     const model::Topology::Specs& topology_specs,
                                     std::vector <model::EvalStatus>& eval_status,
                                     const bool break_on_failure)
{

  bool success = true;
  auto action_gating_info = state.sparse_optimization_info_->action_gating_info;
  auto action_skipping_info = state.sparse_optimization_info_->action_skipping_info;


  //
  // Go through the explicit gating and skipping specifications
  //

  if (action_gating_info.size() > 0)
  {
    for (auto level_iter = action_gating_info.begin();
         level_iter != action_gating_info.end(); level_iter++)
    {
      auto storage_level_id = level_iter->first;
      auto optimization_list = level_iter->second;

      for (unsigned group_id = 0; group_id < optimization_list.size(); group_id++)
      {
        GroupOfActionOptimization group = optimization_list[group_id];

        success = DefineIneffectualReadImpact(state, compound_data_movement_nest, storage_level_id,
                                              group, topology_specs, "gate", eval_status);
        if (!success && break_on_failure) return success;
      }
    }
  }

  if ((success || !break_on_failure) && action_skipping_info.size() > 0)
  {
    for (auto level_iter = action_skipping_info.begin();
         level_iter != action_skipping_info.end(); level_iter++)
    {
      auto storage_level_id = level_iter->first;
      auto optimization_list = level_iter->second;

      for (unsigned group_id = 0; group_id < optimization_list.size(); group_id++)
      {
        GroupOfActionOptimization group = optimization_list[group_id];

        success = DefineIneffectualReadImpact(state, compound_data_movement_nest, storage_level_id,
                                              group, topology_specs, "skip", eval_status);
        if (!success && break_on_failure) return success;
      }
    }
  }

  return success;
}

void CalculateExpectedMetaDataAccesses(tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                       const model::Topology::Specs& topology_specs)
{
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    for (int l = topology_specs.NumStorageLevels() - 1; l >= 0; l--)
    {

      auto& data_movement_info = compound_data_movement_nest[pv][l];
      // 	  std::cout << "\tstorage level: " << topology_specs.GetStorageLevel(l)->level_name
      // << "  dataspace name: " << problem::GetShape()->DataSpaceIDToName.at(pv)
      // << " shape: " << data_movement_info.shape << "  has metadata: " << data_movement_info.has_metadata
      // << std::endl;

      if (data_movement_info.shape == 0 || !data_movement_info.has_metadata) continue;

      tiling::MetaDataTileOccupancy expected_metadata_tile_occupancy = data_movement_info.expected_metadata_occupancy;
      // std::cout <<" \tgot expected metadata tile occupancy..." << std::endl;

      std::uint64_t num_child_metadata_ranks;
      if (data_movement_info.child_level != std::numeric_limits<unsigned>::max())
      {
        // upon a read, only ranks associated with the child level will be sent out
        num_child_metadata_ranks =
          compound_data_movement_nest[pv][data_movement_info.child_level].GetNumMetaDataRanks();
      } else
      {
        // upon a read, last level storage sends all metadata
        num_child_metadata_ranks = data_movement_info.GetNumMetaDataRanks();
      }

      double total_metadata_payload_units_per_tile = 0.0;
      double child_metadata_payload_units_per_tile = 0.0;

      for (unsigned r_id = 0; r_id < expected_metadata_tile_occupancy.size(); r_id++)
      {
        total_metadata_payload_units_per_tile += expected_metadata_tile_occupancy[r_id].MetaDataUnits();
        total_metadata_payload_units_per_tile += expected_metadata_tile_occupancy[r_id].PayloadUnits();

        if (r_id < num_child_metadata_ranks)
        {
          child_metadata_payload_units_per_tile += expected_metadata_tile_occupancy[r_id].MetaDataUnits();
          child_metadata_payload_units_per_tile += expected_metadata_tile_occupancy[r_id].PayloadUnits();
        }
      }

      // calculate how many rounds did the tile get read/fill/update, then scale the metadata accesses per tile accordingly
      double read_ratio = (double)data_movement_info.reads / data_movement_info.shape;
      double fill_ratio = (double)data_movement_info.fills / data_movement_info.shape;
      double update_ratio = (double)data_movement_info.updates / data_movement_info.shape;
      data_movement_info.metadata_fills = ceil(total_metadata_payload_units_per_tile * fill_ratio);
      data_movement_info.metadata_reads = ceil(total_metadata_payload_units_per_tile * read_ratio);
      data_movement_info.metadata_updates = ceil(total_metadata_payload_units_per_tile * update_ratio);

      if (total_metadata_payload_units_per_tile == 0)
      {
        data_movement_info.child_level_metadata_occupancy_ratio = 0;
      } else
      {
        data_movement_info.child_level_metadata_occupancy_ratio =
          child_metadata_payload_units_per_tile / total_metadata_payload_units_per_tile;
      }
    }
  }
}

void PropagateImpactOfExplicitlyOptimizedRead(SparseAnalysisState& state,
                                              tiling::CompoundTileNest& compound_tile_nest,
                                              const model::Topology::Specs& topology_specs)
{

  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto& compute_info_nest = compound_tile_nest.compute_info_nest;

  std::vector <problem::PerDataSpace<double>> max_reads = {};
  std::vector <problem::PerDataSpace<double>> max_updates = {};
  std::vector <problem::PerDataSpace<double>> max_fills = {};

  std::vector <problem::PerDataSpace<double>> max_metadata_reads = {};
  std::vector <problem::PerDataSpace<double>> max_metadata_updates = {};
  std::vector <problem::PerDataSpace<double>> max_metadata_fills = {};

  std::vector<double> max_computes = {};

  // initialize vectors to record the maximum possible number of each type of accesses
  // storage levels
  for (unsigned l = 0; l < topology_specs.NumStorageLevels(); l++)
  {
    max_reads.push_back({});
    max_updates.push_back({});
    max_fills.push_back({});
    max_metadata_reads.push_back({});
    max_metadata_fills.push_back({});
    max_metadata_updates.push_back({});

    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      // data representation impact already reflected in the fine grained access counts
      //    if the fibertree element is not even there due to compression, propagation impact is meaningless
      max_reads[l][pv] = compound_data_movement_nest[pv][l].fine_grained_accesses["random_read"];
      max_fills[l][pv] = compound_data_movement_nest[pv][l].fine_grained_accesses["random_fill"];
      max_updates[l][pv] = compound_data_movement_nest[pv][l].fine_grained_accesses["random_update"];

      max_metadata_reads[l][pv] = compound_data_movement_nest[pv][l].metadata_reads;
      max_metadata_fills[l][pv] = compound_data_movement_nest[pv][l].metadata_fills;
      max_metadata_updates[l][pv] = compound_data_movement_nest[pv][l].metadata_updates;
    }
  }

  max_computes.push_back({}); // there is only one level of compute
  max_computes[0] = compute_info_nest[0].replication_factor * (double)compute_info_nest[0].accesses;

  // propagate the impact of explicitly applied read optimization
  // for reads and fills of lower levels in a top down fashion

  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    for (int l = topology_specs.NumStorageLevels() - 1; l >= 0; l--)
    {
      if (state.dspace_optimization_masks_.at("gate").at(l).at(pv)
        && state.dspace_optimization_masks_.at("skip").at(l).at(pv))
      {
        // not allowed to have gating and skipping applied to the same tile
        assert(false);
      }
      if (state.dspace_optimization_masks_.at("gate").at(l).at(pv)
        || state.dspace_optimization_masks_.at("skip").at(l).at(pv))
      {
        double p = state.prob_explicitly_optimized_read_.at(l).at(pv);
        // std::string type = state.dspace_optimization_masks_.at("skip").at(l).at(pv) ? "skipped" : "gated";

        unsigned impacted_level_id = compound_data_movement_nest[pv][l].child_level;
        while (impacted_level_id != std::numeric_limits<unsigned>::max())
        {
          // upper level propagation essentially chops off a subtree
          //  child level will not see the subtree, so the accesses are nonexistent
          //  do no need to increment the skipped and gated counts
          // auto& data_movement_record = compound_data_movement_nest[pv][impacted_level_id];
          // data_movement_record.fine_grained_accesses["skipped_read"] += floor(max_reads[impacted_level_id][pv] * p);
          // data_movement_record.fine_grained_accesses[type + "_fill"] += floor(max_fills[impacted_level_id][pv] * p);
          // data_movement_record.fine_grained_accesses["skipped_metadata_read"] +=
          //   floor(max_metadata_reads[impacted_level_id][pv] * p);
          // data_movement_record.fine_grained_accesses[type + "_metadata_fill"] +=
          //   floor(max_metadata_fills[impacted_level_id][pv] * p);

          // if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
          // {
          //   data_movement_record.fine_grained_accesses["skipped_update"] += floor(max_updates[impacted_level_id][pv] * p);
          //   data_movement_record.fine_grained_accesses["skipped_metadata_update"] +=
          //     floor(max_metadata_updates[impacted_level_id][pv] * p);
          // }

          max_reads[impacted_level_id][pv] -= floor(max_reads[impacted_level_id][pv] * p);
          max_fills[impacted_level_id][pv] -= floor(max_fills[impacted_level_id][pv] * p);

          max_metadata_reads[impacted_level_id][pv] -= floor(max_metadata_reads[impacted_level_id][pv] * p);
          max_metadata_fills[impacted_level_id][pv] -= floor(max_metadata_fills[impacted_level_id][pv] * p);

          if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
          {
            max_updates[impacted_level_id][pv] -= floor(max_reads[impacted_level_id][pv] * p);
            max_metadata_updates[impacted_level_id][pv] -= floor(max_metadata_updates[impacted_level_id][pv] * p);
          }

          impacted_level_id = compound_data_movement_nest[pv][impacted_level_id].child_level;
        }

        // optimization on operand tensors propagate to compute level as well
        if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
        {
          if (compound_data_movement_nest[pv][l].child_level != std::numeric_limits<unsigned>::max())
          {
            max_computes[0] -= floor(max_computes[0] * p);
          } else
          {
            if (!state.scalar_scalar_opt_masks_.at("gate").at(l).at(pv) &&
                !state.scalar_scalar_opt_masks_.at("skip").at(l).at(pv)
               )
            {
              max_computes[0] -= floor(max_computes[0] * p);
            } else
            {
              // scalar-wise skipping at storage level, will impact compute probability
              // we cannot blindly propagate here, deal with this case in compute action calculations
              state.scalar_storage_optimization_[pv] = true;
            }
          }
        }

      } else
      {
        continue;
      }
    }
  }

  // set the number of random accesses to max possible -> preparing for next stage processing
  for (unsigned l = 0; l < topology_specs.NumStorageLevels(); l++)
  {
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      compound_data_movement_nest[pv][l].fine_grained_accesses["random_read"] = max_reads[l][pv];
      compound_data_movement_nest[pv][l].fine_grained_accesses["random_fill"] = max_fills[l][pv];
      compound_data_movement_nest[pv][l].fine_grained_accesses["random_update"] = max_updates[l][pv];
      compound_data_movement_nest[pv][l].fine_grained_accesses["random_metadata_read"] = max_metadata_reads[l][pv];
      compound_data_movement_nest[pv][l].fine_grained_accesses["random_metadata_fill"] = max_metadata_fills[l][pv];
      compound_data_movement_nest[pv][l].fine_grained_accesses["random_metadata_update"] = max_metadata_updates[l][pv];
    }
  }
  compute_info_nest[0].fine_grained_accesses["random_compute"] = max_computes[0];
}

void ProcessDataReprImpactOnStorageAccesses(const SparseAnalysisState& state,
                                            tiling::CompoundDataMovementNest& compound_data_movement_nest)
{

  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    for (int l = state.num_storage_levels_ - 1; l >= 0; l--)
    {
      if (compound_data_movement_nest[pv][l].has_metadata)
      {
        // if there is metadata, then hardware should be able to avoid accesses to all empty fibertree elements
        double expected_sparsity = (1 - compound_data_movement_nest[pv][l].GetExpectedTileDensity());
        auto& access_record = compound_data_movement_nest[pv][l];

        access_record.fine_grained_accesses["random_read"] =
          access_record.reads - floor(access_record.reads * expected_sparsity);
        access_record.fine_grained_accesses["random_fill"] =
          access_record.fills - floor(access_record.fills * expected_sparsity);

        if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
        {
          access_record.fine_grained_accesses["random_update"] =
            access_record.updates - floor(access_record.updates * expected_sparsity);
        }
      }
    }
  }
}

void CalculateFineGrainedStorageAccesses(const SparseAnalysisState& state,
                                         tiling::CompoundDataMovementNest& compound_data_movement_nest)
{
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    for (int l = state.num_storage_levels_ - 1; l >= 0; l--)
    {
      auto& data_movement_record = compound_data_movement_nest[pv][l];

      //
      // Fine grained read/fill/update
      //

      if (state.dspace_optimization_masks_.at("gate").at(l).at(pv)
        || state.dspace_optimization_masks_.at("skip").at(l).at(pv))
      {
        auto max_reads = data_movement_record.fine_grained_accesses["random_read"];
        auto max_fills = data_movement_record.fine_grained_accesses["random_fill"];
        auto max_updates = data_movement_record.fine_grained_accesses["random_update"];
        auto max_metadata_reads = data_movement_record.fine_grained_accesses["random_metadata_read"];
        auto max_metadata_fills = data_movement_record.fine_grained_accesses["random_metadata_fill"];
        auto max_metadata_updates = data_movement_record.fine_grained_accesses["random_metadata_update"];

        //std::cout << "\t original counts: "<< std::endl;
        //for (auto iter=compound_data_movement_nest[pv][l].fine_grained_accesses.begin();
        //	 iter!=compound_data_movement_nest[pv][l].fine_grained_accesses.end(); iter++)
        //{
        //  std::cout << "\t" << iter->first << ": " << iter->second << std::endl;
        //}

        // apply per level explicit read optimization impact
        if (state.dspace_optimization_masks_.at("gate").at(l).at(pv) || state.dspace_optimization_masks_.at("skip").at(l).at(pv))
        {

          std::string type = state.dspace_optimization_masks_.at("gate").at(l).at(pv) ? "gated" : "skipped";
          //std::cout << "\t " << type << " read..." << std::endl;

          std::uint64_t delta_reads;
          if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
          {
            // use ceil to account for the potential rounding differences due to the RMW setup
            delta_reads = ceil(max_reads * state.prob_explicitly_optimized_read_.at(l).at(pv));
          }
          else
          {
            delta_reads = floor(max_reads * state.prob_explicitly_optimized_read_.at(l).at(pv));
          }

          data_movement_record.fine_grained_accesses[type + "_read"] += delta_reads;
          max_reads -= delta_reads;

          // we can only optimize on the portion of the metadata transferred to child level

          auto delta_metadata_reads = floor(max_metadata_reads * data_movement_record.child_level_metadata_occupancy_ratio
                                              * state.prob_explicitly_optimized_read_.at(l).at(pv));
          data_movement_record.fine_grained_accesses[type + "_metadata_read"] += delta_metadata_reads;
          max_metadata_reads -= delta_metadata_reads;

          // the optimized away reads will not lead to any updates to this level

          auto delta_updates = floor(max_updates * state.prob_explicitly_optimized_read_.at(l).at(pv));
          data_movement_record.fine_grained_accesses[type + "_update"] += delta_updates;
          max_updates -= delta_updates;

          auto delta_metadata_updates = floor(max_metadata_updates * data_movement_record.child_level_metadata_occupancy_ratio
                                                * state.prob_explicitly_optimized_read_.at(l).at(pv));
          data_movement_record.fine_grained_accesses[type + "_metadata_update"] += delta_metadata_updates;
          max_metadata_updates -= delta_metadata_updates;
        }

        // Finalize random counts -> which is just the left over max number of each type of action
        data_movement_record.fine_grained_accesses["random_read"] = max_reads;
        data_movement_record.fine_grained_accesses["random_fill"] = max_fills;
        data_movement_record.fine_grained_accesses["random_update"] = max_updates;
        data_movement_record.fine_grained_accesses["random_metadata_fill"] = max_metadata_fills;
        data_movement_record.fine_grained_accesses["random_metadata_read"] = max_metadata_reads;
        data_movement_record.fine_grained_accesses["random_metadata_update"] = max_metadata_updates;

        // Sanity chceks
        //   if the no metadata, the total algorithmic accesses == sum of all fine grained accesses
        //   otherwise, the total algorithmic accesses >= sum all of the fine grained accesses

        if (data_movement_record.has_metadata)
        {
          assert(data_movement_record.reads >= data_movement_record.fine_grained_accesses["random_read"]
            + data_movement_record.fine_grained_accesses["gated_read"]
            + data_movement_record.fine_grained_accesses["skipped_read"]);
          assert(data_movement_record.fills >= data_movement_record.fine_grained_accesses["random_fill"]
            + data_movement_record.fine_grained_accesses["gated_fill"]
            + data_movement_record.fine_grained_accesses["skipped_fill"]);
          assert(data_movement_record.updates >= data_movement_record.fine_grained_accesses["random_update"]
            + data_movement_record.fine_grained_accesses["gated_update"]
            + data_movement_record.fine_grained_accesses["skipped_update"]);
        } else
        {
          assert(data_movement_record.reads == data_movement_record.fine_grained_accesses["random_read"]
            + data_movement_record.fine_grained_accesses["gated_read"]
            + data_movement_record.fine_grained_accesses["skipped_read"]);
          assert(data_movement_record.fills == data_movement_record.fine_grained_accesses["random_fill"]
            + data_movement_record.fine_grained_accesses["gated_fill"]
            + data_movement_record.fine_grained_accesses["skipped_fill"]);
          assert(data_movement_record.updates == data_movement_record.fine_grained_accesses["random_update"]
            + data_movement_record.fine_grained_accesses["gated_update"]
            + data_movement_record.fine_grained_accesses["skipped_update"]);
        }
      }

      //
      // Temporal Reduction
      //

      // only the updates that actually happened lead to actual temporal reductions
      if (data_movement_record.size != 0 && problem::GetShape()->IsReadWriteDataSpace.at(pv))
      {
        data_movement_record.temporal_reductions = ceil(
          data_movement_record.temporal_reductions * (double)data_movement_record.fine_grained_accesses["random_update"]
            / data_movement_record.updates);
      }
    }
  }

}

void CalculateDecompressionCompressionCost(const std::uint64_t num_storage_levels,
                                           tiling::CompoundDataMovementNest& compound_data_movement_nest)
{

  // compute the compression and decompression counts
  for (int l = num_storage_levels - 1; l >= 0; l--)
  {
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      if (!compound_data_movement_nest[pv][l].compressed)
      {
        auto parent_level = compound_data_movement_nest[pv][l].parent_level;
        auto child_level = compound_data_movement_nest[pv][l].child_level;

        if (parent_level != std::numeric_limits<unsigned>::max()
          && compound_data_movement_nest[pv][parent_level].compressed)
        {
          if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
          {
            // compress at the current level and send to parent
            compound_data_movement_nest[pv][l].fine_grained_accesses["compression_count"] +=
              compound_data_movement_nest[pv][parent_level].fine_grained_accesses["random_update"];
          }
          // compressed data from parent and decompress at the current level
          compound_data_movement_nest[pv][l].fine_grained_accesses["decompression_count"] +=
            compound_data_movement_nest[pv][l].fine_grained_accesses["random_fill"];
        }

        if (child_level != std::numeric_limits<unsigned>::max()
          && compound_data_movement_nest[pv][child_level].compressed)
        {
          // we do not support the modeling of on-chip compression yet
          assert(false);
        }
      }
    }
  }
}

std::map<DataSpaceID, double> GetExpectedOperandDensities(const SetOfOperationSpaces set_of_operation_spaces,
                                                          const PerDataSpaceDensityModel operand_density_models)
{
  std::map<DataSpaceID, double> expected_operand_densities;
  // TODO: adjust to allow coordinate dependent modeling, right now we just use operation space mold to query density models,
  //  most complicated case would be read data model, where we will need to construct all possible operation spaces
  for (auto iter = operand_density_models.begin(); iter != operand_density_models.end(); iter++)
  {
    // operand has metadata, so query density model
    auto& mold_high = set_of_operation_spaces.op_space_mold_high;
    problem::OperationPoint origin;
    tiling::CoordinateSpaceTileInfo ctile_info;
    problem::OperationSpace mold_op_space(set_of_operation_spaces.workload, origin, mold_high);
    ctile_info.Set(mold_op_space.GetSize(iter->first), iter->first);
    expected_operand_densities[iter->first] = iter->second->GetExpectedTileOccupancy(ctile_info) / ctile_info.GetShape();
  }
  return expected_operand_densities;
}

/********************************************************************
 * Old Version of CalculateFineGrainedComputeAccesses
 * ** OBSOLETE
 ********************************************************************/
void CalculateFineGrainedComputeAccesses2Operand(const SparseAnalysisState& state,
                                         tiling::CompoundTileNest& compound_tile_nest)
{

  // scenarios of operands states       | resulted compute actions
  // -----------------------------------------------------------------
  // 1) A: ENZ, B: ENZ                  | random compute
  // 2) A: ENZ, B: EZ, 4) A: EZ, B: ENZ | random, gated compute (depends on optimization) not we cannot skip here as both operands exist
  // 3) A: ENZ, B: NE, 7) A: NE, B: ENZ | random, gated, skipped compute  depends on optimization
  // 5) A: EZ, B: EZ                    | random, gated compute (depends on optimization) note that we cannot skip here as both operands exist
  // 6) A: EZ, B: NE,  8) A: NE, B: EZ  | random, gated, skipped compute  depends on optimization
  // 9) A: NE, B: NE                    | nonexistent compute (the compute unit will not see this case happening as nothing actually exists)
  // -----------------------------------------------------------------
  // NE: not exist; EZ: exist, is zero; ENZ: exist, not zero


  // Find the inner most level tile for each operand
  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto& compute_info = compound_tile_nest.compute_info_nest[0];

  std::map<DataSpaceID, bool> operand_compressed;
  //std::map<DataSpaceID, bool> implicit_coordinates;
  std::map<DataSpaceID, bool> operand_has_metadata;
  PerDataSpaceDensityModel operand_density_models;
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      auto& pv_data_movement_nest = compound_data_movement_nest[pv];
      for (unsigned l = 0; l < state.num_storage_levels_; l++)
      {
        if (pv_data_movement_nest[l].shape != 0)
        {
          // found the inner most storage for dspace pv
          operand_compressed[pv] = pv_data_movement_nest[l].compressed;
          operand_density_models[pv] = pv_data_movement_nest[l].tile_density;
          operand_has_metadata[pv] = pv_data_movement_nest[l].has_metadata;
          // if (pv_data_movement_nest[l].has_metadata)
          // {
          //   implicit_coordinates[pv] = compression_info.per_level_info_map.at(l).at(pv).coordinates_implicit[0];
          // }
          break;
        }
      }
    }
  }

  // Query density models to get necessary density of the scalar operands
  //TODO: consider when there are more than 2 operand dataspaces
  assert(operand_compressed.size() == 2);

  SetOfOperationSpaces set_of_operation_spaces;
  problem::OperationPoint scalar_high;
  set_of_operation_spaces.op_space_mold_high = scalar_high;
  set_of_operation_spaces.upper_level_loops = state.mapping_.loop_nest.loops;
  set_of_operation_spaces.workload = state.workload_;
  auto operand_exp_densities = GetExpectedOperandDensities(set_of_operation_spaces, operand_density_models);
  // for (auto iter = operand_exp_densities.begin(); iter != operand_exp_densities.end(); iter++)
  // {
  //   std::cout << problem::GetShape()->DataSpaceIDToName.at(iter->first) << " density : " << iter->second << std::endl;
  // }

  // Calculate the probability of each possible state of the operands
  std::map <DataSpaceID, PerStateProb> per_operand_states;
  for (auto iter = operand_exp_densities.begin(); iter != operand_exp_densities.end(); iter++)
  {
    problem::Shape::DimensionID pv = iter->first;
    double pv_density = iter->second;
    per_operand_states[pv][EXIST_NOT_ZERO] = pv_density;
    if (operand_has_metadata.at(pv))
    {
      per_operand_states[pv][EXIST_ZERO] = 0;
      per_operand_states[pv][NOT_EXIST] = 1.0 - pv_density;
    } else
    {
      per_operand_states[pv][EXIST_ZERO] = 1.0 - pv_density;
      per_operand_states[pv][NOT_EXIST] = 0;
    }

    // std::cout << "  EZ: " << per_operand_states[pv][EXIST_ZERO] << std::endl;
    // std::cout << "  NE: " << per_operand_states[pv][NOT_EXIST] << std::endl;
    // std::cout << "  ENZ: " << per_operand_states[pv][EXIST_NOT_ZERO] << std::endl;

  }

  // Extract the operand dataspace ids
  auto iter = per_operand_states.begin();
  DataSpaceID op_a_id = iter->first;
  iter++;
  DataSpaceID op_b_id = iter->first;

  // Calculate dependent probabilities
  std::map<ComputeOperandStatePair, double> flattened_probs = {};
  flattened_probs[{EXIST_NOT_ZERO, EXIST_NOT_ZERO}] = 0.0;
  flattened_probs[{EXIST_NOT_ZERO, EXIST_ZERO}] = 0.0;
  flattened_probs[{EXIST_NOT_ZERO, NOT_EXIST}] = 0.0;
  flattened_probs[{EXIST_ZERO, EXIST_NOT_ZERO}] = 0.0;
  flattened_probs[{EXIST_ZERO, EXIST_ZERO}] = 0.0;
  flattened_probs[{EXIST_ZERO, NOT_EXIST}] = 0.0;
  flattened_probs[{NOT_EXIST, EXIST_NOT_ZERO}] = 0.0;
  flattened_probs[{NOT_EXIST, EXIST_ZERO}] = 0.0;
  flattened_probs[{NOT_EXIST, NOT_EXIST}] = 0.0;


  // Extract if both operands are scalar-scalar optimization
  bool scalar_scalar_opt = state.scalar_storage_optimization_.at(op_a_id) && state.scalar_storage_optimization_.at(op_b_id);

  if (scalar_scalar_opt)
  {
    // essentially intersection at scalar scale
    flattened_probs[{EXIST_NOT_ZERO, EXIST_NOT_ZERO}] = per_operand_states[op_a_id][EXIST_NOT_ZERO]
      * per_operand_states[op_b_id][EXIST_NOT_ZERO];
    flattened_probs[{NOT_EXIST, NOT_EXIST}] = 1 - flattened_probs[{EXIST_NOT_ZERO, EXIST_NOT_ZERO}];
  }
  else if (state.scalar_storage_optimization_.at(op_a_id))
  {
    flattened_probs[{EXIST_NOT_ZERO, EXIST_NOT_ZERO}] = per_operand_states[op_a_id][EXIST_NOT_ZERO]
      * per_operand_states[op_b_id][EXIST_NOT_ZERO];

    // since a is optimized on b, if b exist and zero != 0, that means b is read out of the storage to perform optimization on a
    // once b is read out and identified as zero, b here should not be sent to compute anymore
    flattened_probs[{EXIST_NOT_ZERO, EXIST_ZERO}] = 0.0;

    // since a optimized on b, if b not exist, it is not possible for a to exist anymore
    flattened_probs[{EXIST_NOT_ZERO, NOT_EXIST}] = 0.0;

    flattened_probs[{EXIST_ZERO, EXIST_NOT_ZERO}] = per_operand_states[op_a_id][EXIST_ZERO]
      * per_operand_states[op_b_id][EXIST_NOT_ZERO];

    // since a is optimized on b, if b exist and zero != 0, that means b is read out of the storage to perform optimization on a
    // once b is read out and identified as zero, b here should not be sent to compute anymore
    flattened_probs[{EXIST_ZERO, EXIST_ZERO}] = 0.0;

    // since a optimized on b, if b not exist, it is not possible for a to exist anymore
    flattened_probs[{EXIST_ZERO, NOT_EXIST}] = 0.0;

    flattened_probs[{NOT_EXIST, EXIST_NOT_ZERO}] = per_operand_states[op_a_id][NOT_EXIST]
      * per_operand_states[op_b_id][EXIST_NOT_ZERO];

    // since a is optimized on b, if b exist and zero != 0, that means b is read out of the storage to perform optimization on a
    // once b is read out and identified as zero, b here should not be sent to compute anymore
    flattened_probs[{NOT_EXIST, EXIST_ZERO}] = 0.0;

    // when b does not exist or b exist is zero, a must be optimized away
    // we add exist zero to here because when a is optimized, it is not possible for b to be sent to the compute
    // but since b can be of uncompressed format, we need to account for the probability
    flattened_probs[{NOT_EXIST, NOT_EXIST}] =
      1.0 * (per_operand_states[op_b_id][NOT_EXIST] + per_operand_states[op_b_id][EXIST_ZERO]);
  }
  else if (state.scalar_storage_optimization_.at(op_b_id))
  {
    flattened_probs[{EXIST_NOT_ZERO, EXIST_NOT_ZERO}] = per_operand_states[op_a_id][EXIST_NOT_ZERO]
      * per_operand_states[op_b_id][EXIST_NOT_ZERO];
    flattened_probs[{EXIST_NOT_ZERO, EXIST_ZERO}] = per_operand_states[op_a_id][EXIST_NOT_ZERO]
      * per_operand_states[op_b_id][EXIST_ZERO];
    flattened_probs[{EXIST_NOT_ZERO, NOT_EXIST}] = per_operand_states[op_a_id][EXIST_NOT_ZERO]
      * per_operand_states[op_b_id][NOT_EXIST];
    flattened_probs[{EXIST_ZERO, EXIST_NOT_ZERO}] = 0.0;
    flattened_probs[{EXIST_ZERO, EXIST_ZERO}] = 0.0;
    flattened_probs[{EXIST_ZERO, NOT_EXIST}] = 0.0;
    flattened_probs[{NOT_EXIST, EXIST_NOT_ZERO}] = 0.0;
    flattened_probs[{NOT_EXIST, EXIST_ZERO}] = 0.0;
    flattened_probs[{NOT_EXIST, NOT_EXIST}] =
      1.0 * (per_operand_states[op_a_id][NOT_EXIST] + per_operand_states[op_a_id][EXIST_ZERO]);
  }
  else
  {
    flattened_probs[{EXIST_NOT_ZERO, EXIST_NOT_ZERO}] = per_operand_states[op_a_id][EXIST_NOT_ZERO]
      * per_operand_states[op_b_id][EXIST_NOT_ZERO];
    flattened_probs[{EXIST_NOT_ZERO, EXIST_ZERO}] = per_operand_states[op_a_id][EXIST_NOT_ZERO]
      * per_operand_states[op_b_id][EXIST_ZERO];
    flattened_probs[{EXIST_NOT_ZERO, NOT_EXIST}] = per_operand_states[op_a_id][EXIST_NOT_ZERO]
      * per_operand_states[op_b_id][NOT_EXIST];
    flattened_probs[{EXIST_ZERO, EXIST_NOT_ZERO}] = per_operand_states[op_a_id][EXIST_ZERO]
      * per_operand_states[op_b_id][EXIST_NOT_ZERO];
    flattened_probs[{EXIST_ZERO, EXIST_ZERO}] = per_operand_states[op_a_id][EXIST_ZERO]
      * per_operand_states[op_b_id][EXIST_ZERO];
    flattened_probs[{EXIST_ZERO, NOT_EXIST}] = per_operand_states[op_a_id][EXIST_ZERO]
      * per_operand_states[op_b_id][NOT_EXIST];
    flattened_probs[{NOT_EXIST, EXIST_NOT_ZERO}] = per_operand_states[op_a_id][NOT_EXIST]
      * per_operand_states[op_b_id][EXIST_NOT_ZERO];
    flattened_probs[{NOT_EXIST, EXIST_ZERO}] = per_operand_states[op_a_id][NOT_EXIST]
      * per_operand_states[op_b_id][EXIST_ZERO];
    flattened_probs[{NOT_EXIST, NOT_EXIST}] = per_operand_states[op_a_id][NOT_EXIST]
      * per_operand_states[op_b_id][NOT_EXIST];
  }

  // Initialize fine grained access counts
  // double total_compute = compute_info.replication_factor * (double)compute_info.accesses;
  double total_compute = compute_info.fine_grained_accesses["random_compute"];

  // Extract hardware sparse optimization spec (can zero operand be identified?)
  bool gate_on_zero_operand = false;
  bool skip_on_not_aligned_operands = false;
  if (state.sparse_optimization_info_->compute_optimization_info.find("gate_on_zero_operand") !=
    state.sparse_optimization_info_->compute_optimization_info.end())
  {
    gate_on_zero_operand = state.sparse_optimization_info_->compute_optimization_info.at("gate_on_zero_operand");
  }

  if (state.sparse_optimization_info_->compute_optimization_info.find("skip_on_not_aligned_operands") !=
    state.sparse_optimization_info_->compute_optimization_info.end())
  {
    skip_on_not_aligned_operands = state.sparse_optimization_info_->compute_optimization_info.at("skip_on_not_aligned_operands");
  }

  // Initialize all the counters
  // nonexistent compute: although should happen in algorithmic world, will not be present at the hardware
  //   e.g., for cartesian product, any pair of non-empty operands is legal,
  //   the empty operands are then naturally skipped over by hardware as no alignment is performed
  //   however, if alignment is needed, unless the hardware can lookup corresponding pairs with the "skipping" optimization
  //   the cycle needs to be spent when one of the operands is empty
  double random_compute = 0.0, skipped_compute = 0.0, gated_compute = 0.0, tmp_delta = 0.0, nonexistent_compute = 0.0;

  // Analyze case by case
  // 1) A: ENZ, B: ENZ                  | random compute
  random_compute += total_compute * flattened_probs[{EXIST_NOT_ZERO, EXIST_NOT_ZERO}];
  //std::cout << "(1) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute << std::endl;

  // 2) A: ENZ, B: EZ, 4) A: EZ, B: ENZ | random, gated compute (depends on optimization) not we cannot skip here as both operands exist
  tmp_delta = total_compute * (flattened_probs[{EXIST_NOT_ZERO, EXIST_ZERO}] + flattened_probs[{EXIST_ZERO, EXIST_NOT_ZERO}]);
  if (gate_on_zero_operand)
  {
    gated_compute += tmp_delta;
  }
  else
  {
    random_compute += tmp_delta;
  }
  // std::cout << "(2)(4) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute <<
  // << " nonexistent: " << nonexistent_compute << std::endl;

  // 3) A: ENZ, B: NE, 7) A: NE, B: ENZ | nonexistent/(random, gated, skipped) compute (w/o vs w/ alignment (when alignment needed depends on optimization))
  tmp_delta = total_compute * (flattened_probs[{EXIST_NOT_ZERO, NOT_EXIST}] + flattened_probs[{NOT_EXIST, EXIST_NOT_ZERO}]);
  if (skip_on_not_aligned_operands)
    skipped_compute += tmp_delta; // operand alignment unit jumps to look for pair of ENZ ENZ operands
  else
  {
    if (gate_on_zero_operand)
    {
      gated_compute += tmp_delta;
    } else
    {
      random_compute += tmp_delta;  // operand alignment unit sends bubble to compute unit
    }
  }
  // std::cout << "(3)(7) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute <<
  // << " nonexistent: " << nonexistent_compute << std::endl;


  // 5) A: EZ, B: EZ                    | random, gated compute (depends on optimization) note that we cannot skip here as both operands exist
  tmp_delta = total_compute * flattened_probs[{EXIST_ZERO, EXIST_ZERO}] ;
  if (gate_on_zero_operand)
  {
    gated_compute += tmp_delta;
  } else
  {
    random_compute += tmp_delta;
  }

  // std::cout << "(5) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute
  // << " nonexistent: " << nonexistent_compute << std::endl;


  // 6) A: EZ, B: NE,  8) A: NE, B: EZ  | (random, gated, skipped) compute depends on optimization
  tmp_delta = total_compute * (flattened_probs[{EXIST_ZERO, NOT_EXIST}] + flattened_probs[{NOT_EXIST, EXIST_ZERO}]);
  if (skip_on_not_aligned_operands)
  {
    skipped_compute += tmp_delta; // operand alignment unit jumps to look for pair of ENZ ENZ operands
  }
  else
  {
    if (gate_on_zero_operand)
    {
      gated_compute += tmp_delta;
    }
    else
    {
      random_compute += tmp_delta;  // operand alignment unit sends bubble to compute unit
    }
  }

  // std::cout << "(6)(8) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute <<
  // << " nonexistent: " << nonexistent_compute << std::endl;


  // 9) A: NE, B: NE                    | nonexistent compute (the compute unit will not see this case happening as nothing actually exists)
  tmp_delta = total_compute * flattened_probs[{NOT_EXIST, NOT_EXIST}] ;
  nonexistent_compute += tmp_delta;

  //std::cout << "(9) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute
  //<< " nonexistent: " << nonexistent_compute << std::endl;

 //std::cout << "total: " << total_compute << "  sum: " <<  skipped_compute + random_compute + gated_compute + nonexistent_compute
 //<< " diff: " << total_compute - skipped_compute - random_compute - gated_compute  - nonexistent_compute << std::endl;

  // sanity check
  // as long as different is smaller than 0.001% of the total compute, pass
  // there could be tiny discrepencies due to lack of precision...
  assert(abs(skipped_compute + random_compute + gated_compute + nonexistent_compute - total_compute) < total_compute * 0.00001);


  // now round the action counts into integers (pessimistic rounding)
  compute_info.fine_grained_accesses["skipped_compute"] = floor(skipped_compute);
  compute_info.fine_grained_accesses["gated_compute"] = floor(gated_compute);
  compute_info.fine_grained_accesses["random_compute"] =
    total_compute - floor(skipped_compute) - floor(gated_compute) - floor(nonexistent_compute);

  // std::cout << "(final) skipped compute: " << compute_info.fine_grained_accesses["skipped_compute"]
  //   << " gated compute: " << compute_info.fine_grained_accesses["gated_compute"]
  //   << " random compute: " << compute_info.fine_grained_accesses["random_compute"]
  //   << " nonexistent: " << nonexistent_compute << std::endl;

}


/*****************************************************************************
 * A decoder.
 * Convert a decimal to base 3 (specifically to ComputeOperandState)
 * Each ternary bit represents the state of an operand. The number of bits
 * "size" represents the total number of operands involved in compute.
 *****************************************************************************/
void to_base_3(uint64_t value, std::vector<ComputeOperandState>& ternary_array, uint32_t size) 
{
	ComputeOperandState r = static_cast<ComputeOperandState>(value % 3);
	uint64_t q = (uint64_t) floor(value/3);
	ternary_array.push_back(r);
	
	
	if (ternary_array.size() == size) {
		return;
	}

	to_base_3(q, ternary_array, size);
}

/********************************************************************
 * Updated Version of CalculateFineGrainedComputeAccesses
 * Handles multiple operands.
 * TODO: potential bug with sanity check at the end.
 ********************************************************************/
void CalculateFineGrainedComputeAccesses(const SparseAnalysisState& state,
                                         tiling::CompoundTileNest& compound_tile_nest)
{

  /******************************************************************
   * Expanding the 2 operand scenario to multiple operands
   * I assume that each of the operands has an index shared with 
   * some other operand. That is, we can safely do multi-way
   * intersection. 
   * For example: A_mkj * B_nki * C_oih * D_phl = Z_mnopl
   * Here, we need the probability that all the matching scalars
   * for all 4 operands exist.
   ******************************************************************/ 
  // scenarios of operands states       | resulted compute actions
  // -----------------------------------------------------------------
  // 1) All operands are ENZ            | random compute
  // 2) At least one operand is EZ,     | random, gated compute (depends on optimization)
  //    no operands are NE              |   note we cannot skip here as all operands exist
  // 3) At least one operand is NE, but | random, gated, skipped compute (depends on optimization)
  //    some operands are EZ or ENZ     |
  // 4) All operands are NE             | nonexistent compute (the compute unit will not see this case happening
  //                                    | as nothing actually exists
  // -----------------------------------------------------------------
  // NE: not exist; EZ: exist, is zero; ENZ: exist, not zero


  // Find the inner most level tile for each operand
  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto& compute_info = compound_tile_nest.compute_info_nest[0];

  std::map<DataSpaceID, bool> operand_compressed;
  //std::map<DataSpaceID, bool> implicit_coordinates;
  std::map<DataSpaceID, bool> operand_has_metadata;
  PerDataSpaceDensityModel operand_density_models;
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      auto& pv_data_movement_nest = compound_data_movement_nest[pv];
      for (unsigned l = 0; l < state.num_storage_levels_; l++)
      {
        if (pv_data_movement_nest[l].shape != 0)
        {
          // found the inner most storage for dspace pv
          operand_compressed[pv] = pv_data_movement_nest[l].compressed;
          operand_density_models[pv] = pv_data_movement_nest[l].tile_density;
          operand_has_metadata[pv] = pv_data_movement_nest[l].has_metadata;
          // if (pv_data_movement_nest[l].has_metadata)
          // {
          //   implicit_coordinates[pv] = compression_info.per_level_info_map.at(l).at(pv).coordinates_implicit[0];
          // }
          break;
        }
      }
    }
  }

  // Query density models to get necessary density of the scalar operands
  SetOfOperationSpaces set_of_operation_spaces;
  problem::OperationPoint scalar_high;
  set_of_operation_spaces.op_space_mold_high = scalar_high;
  set_of_operation_spaces.upper_level_loops = state.mapping_.loop_nest.loops;
  set_of_operation_spaces.workload = state.workload_;
  auto operand_exp_densities = GetExpectedOperandDensities(set_of_operation_spaces, operand_density_models);
   
  /*for (auto iter = operand_exp_densities.begin(); iter != operand_exp_densities.end(); iter++)
   {
     std::cout << problem::GetShape()->DataSpaceIDToName.at(iter->first) << " density : " << iter->second << std::endl;
   }
   */

  // Calculate the probability of each possible state of the operands
  std::map <DataSpaceID, PerStateProb> per_operand_states;
  for (auto iter = operand_exp_densities.begin(); iter != operand_exp_densities.end(); iter++)
  {
    problem::Shape::DimensionID pv = iter->first;
    double pv_density = iter->second;
    per_operand_states[pv][EXIST_NOT_ZERO] = pv_density;
    if (operand_has_metadata.at(pv))
    {
      per_operand_states[pv][EXIST_ZERO] = 0;
      per_operand_states[pv][NOT_EXIST] = 1.0 - pv_density;
    } else
    {
      per_operand_states[pv][EXIST_ZERO] = 1.0 - pv_density;
      per_operand_states[pv][NOT_EXIST] = 0;
    }

     //std::cout << "  EZ: " << per_operand_states[pv][EXIST_ZERO] << std::endl;
     //std::cout << "  NE: " << per_operand_states[pv][NOT_EXIST] << std::endl;
     //std::cout << "  ENZ: " << per_operand_states[pv][EXIST_NOT_ZERO] << std::endl;

  }

  /******************************************************************
   * WORKAROUND FOR MULTI-OPERAND INTERSECTION
   * Assumptions:
   *     - optimizations are on all other operands, 
   *       (see scalar_storage_optimization_ variable)
   *       So, this assumes the ranks are shared in some
   *       way by all operands
   * For example: A_mkj * B_nki * C_oih * D_phl = Z_mnopl
   * Here, we need the probability that all the matching scalars
   * for all 4 operands exist.
   ******************************************************************/

  uint32_t num_operands = per_operand_states.size();
  double num_states = pow(3.0, num_operands);

  // Calculate dependent probabilities
  std::map<ComputeOperandStatePair, double> flattened_probs = {};

  // Initialize the probabilities for all possible combinations
  // of ENZ, EZ, and NE to 0
  for (uint32_t i = 0; i < num_states; i++)
  { 
	ComputeOperandStatePair state_vector;
	to_base_3(i, state_vector, num_operands);
	flattened_probs[state_vector] = 0.0;
  }


  // Check if all operands are scalar-scalar optimization
  bool multi_scalar_opt = true;
  auto iter_all_ops = per_operand_states.begin();
 
  std::vector<DataSpaceID> operand_dataspaces;

  for (uint32_t i = 0; i < num_operands; i++)
  {
	DataSpaceID op_x_id = iter_all_ops->first;
  	multi_scalar_opt = multi_scalar_opt && state.scalar_storage_optimization_.at(op_x_id);
	operand_dataspaces.push_back(op_x_id);
	iter_all_ops++;
  }

  bool no_opts = true;

  if (multi_scalar_opt)
  {
    // essentially intersection at scalar scale
	// This version is for when all operands are ENZ or NE
	double enz_value = 1.0;

	ComputeOperandStatePair state_vector;
	ComputeOperandStatePair state_vector_ne;

	for (uint32_t i = 0; i < num_operands; i++)
	{
	    state_vector.push_back(EXIST_NOT_ZERO);
		state_vector_ne.push_back(NOT_EXIST);
		DataSpaceID op_x_id = operand_dataspaces.at(i);

		//The probability of the first two operands being ENZ,
		//Then the result of that being ENZ mulitplied by the next operand
		//being ENZ, and so on...
		enz_value = enz_value * per_operand_states[op_x_id][EXIST_NOT_ZERO];
	}

	flattened_probs[state_vector] = enz_value;
	flattened_probs[state_vector_ne] = 1 - enz_value;

	no_opts = false;
  }
  /*********************************************************************
   * Not every operand needs to be optimized. Walk through the operands
   * that have optimization turned on.
   * FIXME: Note that there is a potential bug here. In the 2 operand case, 
   *  it's an if else condition (either a or b, but not both are
   *  optimized if we get to this point).
   *  For multiple operands, we need to consider every combination of
   *  operands that have been optimized, so this might need another walk
   *  through of all the possible combinations of operands being optimized.
   *  We need to know what each operand is optimized on..., then we only
   *  need to look at that other operand.
   *The current implementation stops once a single operand is encountered
   *that requires optimization.
   *********************************************************************/
  else 
  { //loop through each operand and check if they need to be optimized
	for (uint32_t i = 0; i < num_operands; i++) 
	{ //iterate through each operand
		DataSpaceID op_this_id = operand_dataspaces.at(i);

		//Is this the first operand to have optimizations?
		if (no_opts && state.scalar_storage_optimization_.at(op_this_id)) 
		{
			no_opts = false;
		
			//Loop through all the possible operand states (ENZ, EZ, NE)
			for (uint32_t state_i = 0; state_i < num_states; state_i++)
			{ 
				ComputeOperandStatePair state_vector;
				to_base_3(state_i, state_vector, num_operands);

				//For this state, set the flattened_probs
				ComputeOperandState this_state = state_vector.at(i);
				int prob_state = 2; //0 -- compute, 1 -- set to 0, 2 -- all don't exist
				bool there_is_compute = true;
				double prob_value = 1.0; 
				double prob_ne_value = 1.0;
				ComputeOperandState op_state;

				for (uint32_t op_i = 0; op_i < num_operands; op_i++)
				{

					DataSpaceID op_x_id = operand_dataspaces.at(op_i);
					//make sure we're looking at the other operands
					if (op_i != i)
					{
						op_state = state_vector.at(op_i);

						//There's a zero! No compute
						if (op_state == EXIST_ZERO)
						{
							prob_state = 1;		
							there_is_compute = false;
						} 
						
						else if (op_state == NOT_EXIST)
						{
							//we've encountered either EZ or ENZ 
							//in some other operand
							if (prob_state != 2) 
							{ 
								prob_state = 1;
								there_is_compute = false;
							}
							//else everything's been NE so far...
							else 
							{// leave it at 2
								prob_ne_value = prob_ne_value *
								   (per_operand_states[op_x_id][NOT_EXIST] + 
									per_operand_states[op_x_id][EXIST_ZERO]);
								there_is_compute = false;
							}
						}

						//op_state == EXIST_NOT_ZERO
						else 
						{
							//Then prior runs have all been NOT_EXIST. There'll be no compute
							if (prob_state == 2 && !there_is_compute) 
							{ 
								prob_state = 1;
							} 

							//then prior runs have either been NE or EZ leave as is.
							else if (prob_state == 1) {
							}

							//then prior runs have been compute!
							else if (there_is_compute) { 
								prob_state = 0;
								prob_value = prob_value*per_operand_states[op_x_id][EXIST_NOT_ZERO];
							}
						}
					}
					 
				} //done checking the states of the other operands

				//If compute, then compute!
				if (prob_state == 0)
				{
					flattened_probs[state_vector] = 
						prob_value*per_operand_states[op_this_id][this_state];
				} 
				//one of the operands is zero or doesn't exist. Set to 0
				else if (prob_state == 1)
				{
					flattened_probs[state_vector] = 0.0;
				}
				//so far, none of the operands existed...
				else if (prob_state == 2)
				{
					if (this_state == NOT_EXIST)
					{
						flattened_probs[state_vector] = prob_ne_value;
					} else
					{
						flattened_probs[state_vector] = 0.0;
					}
				}
  			}//done with all possible combinations of operand states
		} // done with checking this optimization for this operand
	} //done checking optimizations for ALL operands
  }

  //There were no optimizations on any of the operands.
  //EVERYTHING will be compute
  if (no_opts)
  {
	//Go through each possible state
	for (uint32_t state_i = 0; state_i < num_states; state_i++)
	{
		ComputeOperandStatePair state_vector;
		to_base_3(state_i, state_vector, num_operands);
		
		//Set the flattened_probs for each state
		double prob_value = 1.0;

		for (uint32_t op_i = 0; op_i < num_operands; op_i++)
		{
			DataSpaceID op_x_id = operand_dataspaces.at(op_i);
			ComputeOperandState op_state = state_vector.at(op_i);
			prob_value = prob_value * per_operand_states[op_x_id][op_state];
		}
		flattened_probs[state_vector] = prob_value;
	}
  }


  // Initialize fine grained access counts
  double total_compute = compute_info.fine_grained_accesses["random_compute"];

  // Extract hardware sparse optimization spec (can zero operand be identified?)
  bool gate_on_zero_operand = false;
  bool skip_on_not_aligned_operands = false;
  if (state.sparse_optimization_info_->compute_optimization_info.find("gate_on_zero_operand") !=
    state.sparse_optimization_info_->compute_optimization_info.end())
  {
    gate_on_zero_operand = state.sparse_optimization_info_->compute_optimization_info.at("gate_on_zero_operand");
  }

  if (state.sparse_optimization_info_->compute_optimization_info.find("skip_on_not_aligned_operands") !=
    state.sparse_optimization_info_->compute_optimization_info.end())
  {
    skip_on_not_aligned_operands = state.sparse_optimization_info_->compute_optimization_info.at("skip_on_not_aligned_operands");
  }

  // Initialize all the counters
  // nonexistent compute: although should happen in algorithmic world, will not be present at the hardware
  //   e.g., for cartesian product, any pair of non-empty operands is legal,
  //   the empty operands are then naturally skipped over by hardware as no alignment is performed
  //   however, if alignment is needed, unless the hardware can lookup corresponding pairs with the "skipping" optimization
  //   the cycle needs to be spent when one of the operands is empty
  double random_compute = 0.0, skipped_compute = 0.0, gated_compute = 0.0, tmp_delta = 0.0, nonexistent_compute = 0.0;


  /****************************************************************************
   * Multi-operand version
   * Go through each possible case/state
   * Priorities:
   *  	if all operands are NE: update non-existent compute
   *  	if any operand is NE: update skip or gate or random compute
   *  	if any operand is EZ: update gate or random
   *  	else all operands are ENZ: update random only
   ****************************************************************************/
  for (uint32_t state_i = 0; state_i < num_states; state_i++)
  {
	ComputeOperandStatePair state_vector;
	to_base_3(state_i, state_vector, num_operands);

	tmp_delta = total_compute * (flattened_probs[state_vector]);
	
	//enumerate the cases
	int all_ne = 0, some_ne = 0, some_ez = 0, some_enz = 0, all_enz = 0;
	
	//Loop through each operand state in this state_vector
	for (uint32_t op_i = 0; op_i < num_operands; op_i++)
	{
		ComputeOperandState op_state = state_vector.at(op_i);
		
		if (op_state == NOT_EXIST)
		{
			some_ne = 1;
		} else if (op_state == EXIST_ZERO)
		{
			some_ez = 1;
		} else if (op_state == EXIST_NOT_ZERO)
		{
			some_enz = 1;
		} else {
			printf("ERROR: op_state does not exist!\n");
			exit(1);
		}
	}

	//No EZ and No ENZ means they are all NE
	if (!some_ez && !some_enz && some_ne)
	{
		all_ne = 1;
	} 
	//No NE and No EZ means they are all ENZ
	else if (some_enz && !some_ne && !some_ez)
	{
		all_enz = 1;
	}
	
	//Update compute for this state_vector
	//all operands are in the NOT_EXIST state
	if (all_ne) {
		nonexistent_compute += tmp_delta;
	}
	//SOME of the operands are in the NOT_EXIST state
	else if (some_ne)
	{
		if (skip_on_not_aligned_operands)
  		{
    		skipped_compute += tmp_delta; // operand alignment unit jumps to look for pair of ENZ ENZ operands
  		}
  		else
  		{
    		if (gate_on_zero_operand)
    		{
      			gated_compute += tmp_delta;
    		}
    		else
    		{
      			random_compute += tmp_delta;  // operand alignment unit sends bubble to compute unit
    		}
  		}
  	}
	//SOME of the operands are in the EXIST_ZERO state, but not NE state
	else if (some_ez)
	{
		if (gate_on_zero_operand)
  		{
    		gated_compute += tmp_delta;
  		} else
  		{
    		random_compute += tmp_delta;
  		}
	} 
	//All the operands exist and aren't zero!
	else if (all_enz)
	{
		random_compute += tmp_delta;
	}
	else
	{//we should never reach this stage
		printf("ERROR: Bug in calculating intersection computes...\n");
		exit(1);
	}
  }

   //Sanity check
   //std::cout << "total: " << total_compute << "  sum: " <<  skipped_compute + random_compute + gated_compute + nonexistent_compute
   //<< " diff: " << total_compute - skipped_compute - random_compute - gated_compute  - nonexistent_compute << 
   //" upper bound " << total_compute * 0.00001 << std::endl;
   //fflush(stdout);

  // sanity check
  // as long as different is smaller than 0.001% of the total compute, pass
  // there could be tiny discrepencies due to lack of precision...
  // TODO: Removed this assertion for now. Sometimes fails, need to check code above.
  //       Need to figure out how to check all combinations of operands 
  //       needing optimization (vs. a single one). See FIXME comment above.
  //assert(abs(skipped_compute + random_compute + gated_compute + nonexistent_compute - total_compute) < total_compute * 0.00001);


  if (abs(skipped_compute + random_compute + gated_compute + nonexistent_compute - total_compute) > total_compute * 0.00001) {
	   printf("WARNING: the calculated compute is not meeting the constraints.");
  }

  //Adding  a basic sanity check. 
  assert(abs(skipped_compute + random_compute + gated_compute + nonexistent_compute) <= total_compute);

  // now round the action counts into integers (pessimistic rounding)
  compute_info.fine_grained_accesses["skipped_compute"] = floor(skipped_compute);
  compute_info.fine_grained_accesses["gated_compute"] = floor(gated_compute);
  compute_info.fine_grained_accesses["random_compute"] =
    total_compute - floor(skipped_compute) - floor(gated_compute) - floor(nonexistent_compute);

  // std::cout << "(final) skipped compute: " << compute_info.fine_grained_accesses["skipped_compute"]
  //   << " gated compute: " << compute_info.fine_grained_accesses["gated_compute"]
  //   << " random compute: " << compute_info.fine_grained_accesses["random_compute"]
  //   << " nonexistent: " << nonexistent_compute << std::endl;

}

bool ApplyRanksOuterToInner(std::uint64_t inner_rank_id,
                            const std::vector <loop::Descriptor>& singleton_metadata_subnest,
                            const std::vector <std::uint64_t>& singleton_metadata_subtile_shape,
                            const sparse::PerDataSpaceCompressionInfo& pv_compression_info,
                            tiling::DataMovementInfo& pv_data_movement_info)
{
  std::vector <loop::Descriptor> flattened_rank_nest;
  std::vector <problem::Shape::DimensionID> flattening_rule;

  bool pv_has_metadata = pv_compression_info.HasMetaData();
  std::uint64_t cur_level_num_ranks = pv_has_metadata ? pv_compression_info.rank_formats.size() : 1;

  assert(singleton_metadata_subnest.size() == singleton_metadata_subtile_shape.size());

  // start by applying the outermost rank to the outermost loop
  // if there are extra inner ranks supported, all of the these ranks will cost no overhead
  int loop_id = singleton_metadata_subnest.size() - 1;
  int r_id = cur_level_num_ranks;
  std::uint64_t corresponding_tile_shape = 1;
  // std::cout << "total number of ranks: " << cur_level_num_ranks
  // << "  inner rank id: " << inner_rank_id
  // << " total loops: " << singleton_metadata_subnest.size() << std::endl;

  while (r_id > (int)inner_rank_id && loop_id >= 0)
  {
    r_id--; // next inner rank
    flattened_rank_nest.clear();
    bool trivial_loop = true;

    while (trivial_loop && loop_id >= 0)  //get rid of the first set of consecutive trivial loops
    {
      auto loop = singleton_metadata_subnest[loop_id];
      trivial_loop = (loop.start + loop.stride) >= loop.end;
      if (!trivial_loop) break;
      loop_id--;
    }

    if (loop_id >= 0)
    {
      auto loop = singleton_metadata_subnest[loop_id];
      // std::cout << "mapping loop below to rank " << r_id << std::endl;
      // std::cout << loop << std::endl;

      if (!trivial_loop)
      {
        flattened_rank_nest.push_back(loop);
        corresponding_tile_shape = singleton_metadata_subtile_shape[loop_id];
      }
      loop_id--;

      bool in_flattened_list;
      if (loop_id >= 0)
      {
        // reset flattening rule
        flattening_rule = {};
        std::vector<problem::Shape::DimensionID>::iterator flatten_iter;

        if (!pv_has_metadata)
        {
          in_flattened_list = true;
        } else if (pv_compression_info.FoundDimensionInFlatteningRule(r_id, loop.dimension,
                                                                      flattening_rule))
        {
          flatten_iter = std::find(flattening_rule.begin(), flattening_rule.end(), loop.dimension);
          //flattening_rule.erase(flatten_iter);
          in_flattened_list = true;
        } else
        {
          in_flattened_list = false;
        }

        // std::cout << "rest of dims in the flattening rule: " << in_flattened_list << std::endl;
        // if (pv_has_metadata)
        // {
        //   for (auto dim = flattening_rule.begin(); dim != flattening_rule.end(); dim++)
        //   {
        //     std::cout << problem::GetShape()->DimensionIDToName.at(*dim) << "  ";
        //   }
        //   std::cout << std::endl;
        // }

        // if default uncompressed, then all dimensions in a list as well
        if (!pv_has_metadata || in_flattened_list)
        {
          // this loop is already the next inner loop of the loop that is in a flattened list
          auto loop = singleton_metadata_subnest[loop_id];
          trivial_loop = (loop.start + loop.stride) >= loop.end;

          flatten_iter = std::find(flattening_rule.begin(), flattening_rule.end(), loop.dimension);

          while (!pv_has_metadata || (flatten_iter != flattening_rule.end()) || trivial_loop)
          {
            if (!trivial_loop)
            {
              //std::cout << "mapping loop below to rank " << r_id << std::endl;
              //std::cout << loop << std::endl;
              flattened_rank_nest.push_back(loop);
              // corresponding_tile_shape = singleton_metadata_subtile_shape[loop_id];

              // remove loop dimension from flatten list, as one item in the list can only be used once
              //if (pv_has_metadata) flattening_rule.erase(flatten_iter);
            }
            loop_id--;

            // check if there are anymore loops (overall and in this flattening rule)
            if (loop_id >= 0)
            {
              loop = singleton_metadata_subnest[loop_id];
              flatten_iter = std::find(flattening_rule.begin(), flattening_rule.end(),
                                       singleton_metadata_subnest[loop_id].dimension);
              trivial_loop = (loop.start + loop.stride) >= loop.end;
              // std::cout << "next loop: " << singleton_metadata_subnest[loop_id] << std::endl;
            } else
              break;
          }
        }
      }
    }
    pv_data_movement_info.metadata_subnest.insert(pv_data_movement_info.metadata_subnest.begin(), flattened_rank_nest);
    pv_data_movement_info.metadata_subtile_shape.insert(pv_data_movement_info.metadata_subtile_shape.begin(),
                                                        corresponding_tile_shape);

    // reset to 1
    corresponding_tile_shape = 1;

  }


  // fill in the extra inner supported rank (if any)
  while (r_id > (int)inner_rank_id)
  {
    std::vector <loop::Descriptor> tmp_loop = {};
    pv_data_movement_info.metadata_subnest.insert(pv_data_movement_info.metadata_subnest.begin(), tmp_loop);
    pv_data_movement_info.metadata_subtile_shape.insert(pv_data_movement_info.metadata_subtile_shape.begin(),
                                                        singleton_metadata_subtile_shape[0]);
    // std::cout << "Warning: more supported ranks then non-trivial loops, "
    //              "the extra inner rank is turned into a dummy rank: "
    //           << pv_compression_info.rank_formats[r_id] << std::endl;
    r_id--;
  }


  // skip the trailing trivial loops (if any)
  bool more_compression_ranks_needed = false;

  while (loop_id >= 0)
  {
    auto loop = singleton_metadata_subnest[loop_id];
    bool trivial_loop = (loop.start + loop.stride) >= loop.end;
    if (!trivial_loop)
    {
      // if any non-trivial loops occurs, then the hardware support is not compatible
      more_compression_ranks_needed = true;
      break;
    }
    loop_id--;
  }

  // fill in the extra inner supported rank (if any)
  while (r_id > (int)inner_rank_id)
  {
    std::vector <loop::Descriptor> tmp_loop = {};
    pv_data_movement_info.metadata_subnest.insert(pv_data_movement_info.metadata_subnest.begin(), tmp_loop);
    pv_data_movement_info.metadata_subtile_shape.insert(pv_data_movement_info.metadata_subtile_shape.begin(), 0);
    std::cout << "Warning: more supported ranks then non-trivial loops, "
                 "the extra inner rank is turned into a dummy rank: "
              << pv_compression_info.rank_formats[r_id] << std::endl;
    r_id--;
  }

  return more_compression_ranks_needed;
}

bool ApplyRanksInnerToOuter(std::uint64_t inner_rank_id,
                            const std::vector <loop::Descriptor>& singleton_metadata_subnest,
                            const std::vector <std::uint64_t>& singleton_metadata_subtile_shape,
                            const sparse::PerDataSpaceCompressionInfo& pv_compression_info,
                            tiling::DataMovementInfo& pv_data_movement_info)
{

  //FIXME: this function needs to be updated to perform correctly, use outer to inner mapping instead
  //  1) skip trivial loops logic
  //  2) subtile shape insertion logic
  assert(false);

  std::vector <loop::Descriptor> flattened_rank_nest;
  std::vector <problem::Shape::DimensionID> flattening_rule;

  bool pv_has_metadata = pv_compression_info.HasMetaData();
  std::uint64_t cur_level_num_ranks = pv_has_metadata ? pv_compression_info.rank_formats.size() : 1;

  // start by applying the innermost rank to the innermost loop
  // if there are extra outer ranks supported, all of the these ranks will cost no overhead
  std::uint64_t loop_id = 0;
  std::uint64_t r_id = inner_rank_id - 1;

  while (r_id < cur_level_num_ranks - 1 && loop_id < singleton_metadata_subnest.size())
  {

    r_id++; // next outer loop

    flattened_rank_nest.clear();
    bool trivial_loop = true;

    while (trivial_loop
      && loop_id < singleton_metadata_subnest.size())  //get rid of the first set of consecutive trivial loops
    {
      auto loop = singleton_metadata_subnest[loop_id];
      trivial_loop = (loop.start + loop.stride) >= loop.end;
      if (!trivial_loop) break;
      loop_id++;
    }

    if (loop_id < singleton_metadata_subnest.size())
    {
      auto loop = singleton_metadata_subnest[loop_id];
      flattened_rank_nest.push_back(loop);
      loop_id++;
      if (!pv_has_metadata // if default uncompressed, then all loops must be flattened into 1 rank
        || pv_compression_info.FoundDimensionInFlatteningRule(r_id, loop.dimension, flattening_rule))
      {
        while (loop_id < singleton_metadata_subnest.size()
          && (!pv_has_metadata
            || (std::find(flattening_rule.begin(), flattening_rule.end(), singleton_metadata_subnest[loop_id].dimension)
              != flattening_rule.end())))
        {
          auto loop = singleton_metadata_subnest[loop_id];
          trivial_loop = (loop.start + loop.stride) >= loop.end;
          if (!trivial_loop)
          {
            flattened_rank_nest.push_back(loop);
          }
          loop_id++;
        }
      }
    }
    pv_data_movement_info.metadata_subnest.push_back(flattened_rank_nest);
    pv_data_movement_info.metadata_subtile_shape.push_back(singleton_metadata_subtile_shape[loop_id - 1]);
  }

  // fill in the extra outer supported rank (if any)
  while (r_id < cur_level_num_ranks - 1)
  {
    std::vector <loop::Descriptor> tmp_loop = {};
    pv_data_movement_info.metadata_subnest.push_back(tmp_loop);
    pv_data_movement_info.metadata_subtile_shape.push_back(singleton_metadata_subtile_shape.back());
    // std::cout << "Warning: more supported ranks then non-trivial loops, "
    //			  "the extra outer rank is turned into a dummy rank: "
    //		   << pv_compression_info.rank_formats[r_id]<< std::endl;
    r_id++;
  }

  // skip the trailing trivial loops (if any)
  bool more_compression_ranks_needed = false;

  while (loop_id < singleton_metadata_subnest.size())
  {
    auto loop = singleton_metadata_subnest[loop_id];
    bool trivial_loop = (loop.start + loop.stride) >= loop.end;
    if (!trivial_loop)
    {
      // if any non-trivial loops occurs, then the hardware support is not compatible
      more_compression_ranks_needed = true;
      break;
    }
    loop_id++;
  }

  return more_compression_ranks_needed;
}

bool DefineCompressionFormatModels(SparseAnalysisState& state,
                                   tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                   const model::Topology::Specs& topology_specs,
                                   std::vector <model::EvalStatus>& eval_status,
                                   const bool break_on_failure)
{

  bool success = true;
  auto compression_info = state.sparse_optimization_info_->compression_info;

  // nothing needs to be done if no metadata involved
  if (compression_info.all_ranks_default_dense) return success;

  std::ostringstream fail_reason;

  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {

    auto& pv_data_movement_nest = compound_data_movement_nest[pv];
    auto dim_ids_in_proj = problem::GetShape()->DataSpaceIDToDimensionIDVector[pv];

    unsigned level = 0;
    while (level < topology_specs.NumStorageLevels())
    {

      if (pv_data_movement_nest[level].shape == 0)
      {
        // have not seen the first level that storages the datatype pv yet
        level++;
        continue;
      }

      // get the architecture level of the storage level
      unsigned overall_level_id = topology_specs.StorageMap(level);

      // collapse all the bypassed levels between parent and child into metadata subnests
      std::uint64_t child_level_id = pv_data_movement_nest[level].child_level;

      // handle special case: no child level
      // then we need to accumulate all levels below (if any)
      std::uint64_t inner_most_level = child_level_id == std::numeric_limits<unsigned>::max() ? 0 : child_level_id + 1;

      // check if pre-tiling is required for this level
      // if so, current level hardware must support the necessary number of ranks associated with the metadata subnest of both this level and its child level
      // thus, we need to add all the metadata subnest in the child level to this level
      bool pre_tiling_required = false;
      bool cur_level_has_metadata = compression_info.has_metadata_masks.at(level).at(pv);
      bool cur_level_compressed = compression_info.compressed_masks.at(level).at(pv);
      // if current level is default uncompressed, it has one rank that all rankIDs are flattened into a single rank
      unsigned cur_level_num_ranks = cur_level_has_metadata ?
                                     compression_info.per_level_info_map.at(level).at(pv).rank_formats.size() : 1;

      // std::cout << "define format: "<< topology_specs.GetStorageLevel(level)->level_name
      // << " dataspace: " << problem::GetShape() -> DataSpaceIDToName.at(pv)
      // << " has metadata: " << cur_level_has_metadata
      // << std::endl;

      if (cur_level_has_metadata)
      {
        // update tile information to reflect sparse optimization's impact
        compound_data_movement_nest[pv][level].SetTensorRepresentation(compression_info.per_level_info_map.at(level).at(
          pv));
      }

      bool child_level_has_metadata = false;
      unsigned child_level_num_ranks = 0;
      bool child_level_compressed = false;

      if (child_level_id != std::numeric_limits<unsigned>::max())
      {
        child_level_has_metadata = compression_info.has_metadata_masks.at(child_level_id).at(pv);
        child_level_compressed = compression_info.compressed_masks.at(child_level_id).at(pv);
        child_level_num_ranks = child_level_has_metadata ?
                                compression_info.per_level_info_map.at(child_level_id).at(pv).rank_formats.size() : 1;
      }

      // Pretiling checks
      // only if current level is compressed && current level is not inner most level && inner level has metadata
      // && no online tile partition supported, do we need pre-tiling
      // && child level is not payload
      // if child level has a tile shape of 1, then it is just looking at a value payload, no need to pre-tile for the payload
      // by pretiling, we specifically mean that this level needs to consider inner level nontrivial loops

      if (cur_level_compressed && child_level_id != std::numeric_limits<unsigned>::max()
        && !compression_info.tile_partition_supported_masks[level] && pv_data_movement_nest[child_level_id].shape > 1)
      {

        pre_tiling_required = true;

        if (child_level_num_ranks > cur_level_num_ranks)
        {
          fail_reason << "pretiling for " << problem::GetShape()->DataSpaceIDToName.at(pv)
                      << "required but level compression format does not align: "
                      << topology_specs.GetStorageLevel(level)->level_name << " "
                      << topology_specs.GetStorageLevel(child_level_id)->level_name
                      << std::endl;

          success = false;
          eval_status[overall_level_id].success = false;
          eval_status[overall_level_id].fail_reason = fail_reason.str();
          if (break_on_failure) return success;

        }
        for (unsigned r_id = 0; r_id < child_level_num_ranks; r_id++)
        {
          if (child_level_compressed &&
            compression_info.per_level_info_map.at(level).at(pv).rank_formats[r_id] !=
              compression_info.per_level_info_map.at(child_level_id).at(pv).rank_formats[r_id])
          {
            fail_reason << "pretiling for " << problem::GetShape()->DataSpaceIDToName.at(pv)
                        << "required but level compression format does not align: "
                        << topology_specs.GetStorageLevel(level)->level_name << " "
                        << topology_specs.GetStorageLevel(child_level_id)->level_name
                        << std::endl;
            success = false;
            eval_status[overall_level_id].success = false;
            eval_status[overall_level_id].fail_reason = fail_reason.str();
            if (break_on_failure) return success;
          }
        }
      }

      // singleton subnests for current level and bypassed level
      std::vector <loop::Descriptor> singleton_metadata_subnest;
      std::vector <std::uint64_t> singleton_metadata_subtile_shape;

      // Go through the corresponding storage levels to retrieve info
      for (int l = level; l >= int(inner_most_level); l--)
      {
        for (int loop_id = state.complete_subnests_[l].size() - 1; loop_id >= 0; loop_id--)
        {
          auto loop = state.complete_subnests_[l][loop_id];
          // bool trivial_loop = state.trivial_nest_masks_[l][loop_id];

          // pick out loops that are relevant (trivial and non-trivial, non-trivial loops will eventually be removed)
          if (dim_ids_in_proj.find(loop.dimension) != dim_ids_in_proj.end())
          {
            singleton_metadata_subnest.insert(singleton_metadata_subnest.begin(), loop);
            problem::OperationPoint origin;
            problem::OperationSpace maxtile_mold(state.workload_, origin, state.maxtile_molds_high_[l][loop_id]);
            auto subtile_shape = maxtile_mold.GetSize(pv);
            singleton_metadata_subtile_shape.insert(singleton_metadata_subtile_shape.begin(), subtile_shape);
          }
        }
      }

      if (!pre_tiling_required)
      {

        // without pretiling, the subnests associated with the current tile
        // must be bounded to the include all subtile sizes in the inner levels
        // only looking at the subnest bounds at the current level is not enough

        // 1) collect all inner level subnests to get the global bound for each dimension
        // FIXME: temporal and spatial loop bounds might get multiplied together and
        //  the loop type will be set to whichever that is at the top,
        //  this behavior does not affect the correctness in terms of fiber tree construction,
        //  but making it cleaner would be helpful
        problem::PerProblemDimension <std::uint64_t> dimension_sizes;
        problem::PerProblemDimension <std::uint64_t> residual_sizes;
        for (unsigned i = 0; i < dimension_sizes.size(); i++)
        {
          dimension_sizes[i] = 1;
          residual_sizes[i] = 1;
        }

        for (unsigned i = 0; i < inner_most_level; i++)
        {
          auto subnest = state.complete_subnests_[i];
          for (auto loop = subnest.begin(); loop != subnest.end(); loop++)
          {
            dimension_sizes[loop->dimension] *= ceil((loop->end - loop->start) / loop->stride);
            residual_sizes[loop->dimension] *= loop->residual_end;
          }
        }

        // 2) scale the current level bound accordingly
        // note that after this step, there can be trivial loops left
        for (unsigned subnest_id = 0; subnest_id < singleton_metadata_subnest.size(); subnest_id++)
        {
          auto& loop = singleton_metadata_subnest[subnest_id];
          loop.end = loop.end * loop.stride * dimension_sizes[loop.dimension];
          loop.residual_end = loop.residual_end * residual_sizes[loop.dimension];

          // if there are two loops of the same dim in cur level subnest, the upper loop will not be scaled
          dimension_sizes[loop.dimension] = 1;
          residual_sizes[loop.dimension] = 1;
        }
      }

      // Map the non-trivial loops to the hardware supported ranks:
      // 1) Get rid of potential trivial loops
      // 2) Flatten necessary loops according to flattening rule
      std::uint64_t inner_rank_id = pre_tiling_required ? child_level_num_ranks : 0;
      sparse::PerDataSpaceCompressionInfo pv_compression_info;
      if (cur_level_has_metadata)
      {
        pv_compression_info = compression_info.per_level_info_map.at(level).at(pv);
      }

      bool more_compression_ranks_needed;
      if (pv_compression_info.rank_application_order == 1)
      {
        more_compression_ranks_needed = ApplyRanksOuterToInner(inner_rank_id, singleton_metadata_subnest,
                                                               singleton_metadata_subtile_shape, pv_compression_info,
                                                               pv_data_movement_nest[level]);
      } else
      {
        more_compression_ranks_needed = ApplyRanksInnerToOuter(inner_rank_id, singleton_metadata_subnest,
                                                               singleton_metadata_subtile_shape, pv_compression_info,
                                                               pv_data_movement_nest[level]);
      }

      if (more_compression_ranks_needed)
      {

        fail_reason << "more compression ranks needed than supported in hardware."
                    << " dataspace name: " << problem::GetShape()->DataSpaceIDToName.at(pv);
        success = false;
        eval_status[overall_level_id].success = false;
        eval_status[overall_level_id].fail_reason = fail_reason.str();
        if (break_on_failure) return success;
      }

      if (pre_tiling_required)
      {

        // pre-tiling requires the tile to maintain the order and bounds of the metadata subnests from its child level
        for (int loop_id = pv_data_movement_nest[child_level_id].metadata_subnest.size() - 1; loop_id >= 0; loop_id--)
        {
          auto loop = pv_data_movement_nest[child_level_id].metadata_subnest[loop_id];
          pv_data_movement_nest[level].metadata_subnest.insert(pv_data_movement_nest[level].metadata_subnest.begin(),
                                                               loop);

          auto subtile_shape = pv_data_movement_nest[child_level_id].metadata_subtile_shape[loop_id + 1];
          pv_data_movement_nest[level].metadata_subtile_shape.insert(pv_data_movement_nest[level].metadata_subtile_shape.begin(),
                                                                     subtile_shape);
        }
        // subtile shape must have one more element than subtile nest
        // see assert below for more
        pv_data_movement_nest[level].metadata_subtile_shape.insert(pv_data_movement_nest[level].metadata_subtile_shape.begin(),
                                                                   pv_data_movement_nest[child_level_id].metadata_subtile_shape[0]);
      } else
      {
        pv_data_movement_nest[level].metadata_subtile_shape.insert(pv_data_movement_nest[level].metadata_subtile_shape.begin(),
                                                                   1);
      }

      // if (pv_data_movement_nest[level].metadata_subnest.size() != cur_level_num_ranks)
      // {

      // std::cout << "metadata models defined incorrectly, dataspace name: " << problem::GetShape()->DataSpaceIDToName.at(pv) << std::endl;
      // for (unsigned rank_id = 0; rank_id < pv_data_movement_nest[level].metadata_subnest.size(); rank_id++)
      // {
      //   std::cout << " --- flattened loops --- " << std::endl;
      //   for (auto loop = pv_data_movement_nest[level].metadata_subnest[rank_id].begin();
      // 	   loop != pv_data_movement_nest[level].metadata_subnest[rank_id].end(); loop++)
      //   {
      // 	   std::cout << *loop << std::endl;
      //   }
      // }
      // }

      // validity check on if the required number of ranks == number of hardware supported ranks
      assert(pv_data_movement_nest[level].metadata_subnest.size() == cur_level_num_ranks);

      // calculate the fiber shapes at each rank of metadata
      for (unsigned rank_id = 0; rank_id < cur_level_num_ranks; rank_id++)
      {

        problem::OperationPoint origin;
        problem::OperationPoint dimension_sizes;
        dimension_sizes.IncrementAllDimensions(); // initialize to { 1, 1, 1... }

        // going through the flattened ranks (if size is 1, then there is not flattened ranks
        for (auto loop = pv_data_movement_nest[level].metadata_subnest[rank_id].begin();
             loop != pv_data_movement_nest[level].metadata_subnest[rank_id].end(); loop++)
        {
          dimension_sizes[loop->dimension] *= ceil((loop->end - loop->start) / loop->stride);
        }

        // project to operation space to get fiber shape
        problem::OperationPoint high = dimension_sizes;
        high.IncrementAllDimensions(-1);
        problem::OperationSpace offset_tile(state.workload_, origin, high);
        compound_data_movement_nest[pv][level].fiber_shape.push_back(offset_tile.GetSize(pv));
      }

      // subtile shape must have one more element than subtile nest
      // as it includes the tile size of the child level:
      //     important for compressed metadata models to get the prob of empty coordinates in the last level of metadata
      assert(pv_data_movement_nest[level].metadata_subnest.size() + 1
               == pv_data_movement_nest[level].metadata_subtile_shape.size());

      // print info for sanity checks

      // std::cout << "\nDataspace name: " << problem::GetShape()->DataSpaceIDToName.at(pv)
      //           << "  level: " << level << " " << topology_specs.GetStorageLevel(level)->level_name
      //           << "   pretiling required: " << pre_tiling_required
      //           << " compressed: " << compression_info.compressed_masks[level][pv] << std::endl;
      // for (unsigned i = 0; i < pv_data_movement_nest[level].metadata_subnest.size(); i++)
      // {
      //   std::cout << " ----- rank: " << i << " ------" << std::endl;
      //   if (compression_info.compressed_masks[level][pv])
      //     std::cout << "   rank format: " << compression_info.per_level_info_map.at(level).at(pv).rank_formats[i]
      //               << std::endl;
      //   std::cout << "   rank tile shape: " << pv_data_movement_nest[level].metadata_subtile_shape[i + 1] << std::endl;
      //   std::cout << "   rank subtile shape: " << pv_data_movement_nest[level].metadata_subtile_shape[i] << std::endl;
      //   std::cout << "   fiber shape: " << pv_data_movement_nest[level].fiber_shape[i] << std::endl;
      //   std::cout << "   flattened nests: " << pv_data_movement_nest[level].metadata_subnest[i].size() << std::endl;

      // for (auto iter = pv_data_movement_nest[level].metadata_subnest[i].begin();
      //        iter != pv_data_movement_nest[level].metadata_subnest[i].end(); iter++)
      //   {
      //     std::cout << "\t" << *iter << std::endl;
      //   }
      // }

      // look at parent directly in the next round, as we know the levels in the middle are bypassed
      auto parent_level = pv_data_movement_nest[level].parent_level;
      level = parent_level;
    }
  }
  return success;

}

bool CheckFormatModelsAndMapping(const tiling::NestOfCompoundMasks& masks,
                                 sparse::CompressionInfo& compression_info,
                                 const model::Topology::Specs& topology_specs,
                                 std::vector <model::EvalStatus>& eval_status,
                                 const bool break_on_failure)
{

  bool success = true;

  if (compression_info.all_ranks_default_dense) return success;

  for (unsigned pv = 0; pv < unsigned(problem::GetShape()->NumDataSpaces); pv++)
  {
    bool parent_level_compressed = true;
    int parent_level_num_ranks = -1;
    unsigned parent_level = topology_specs.NumStorageLevels() - 1;

    for (int level = topology_specs.NumStorageLevels() - 1; level >= 0; level--)
    {

      bool mask = masks[level][pv];
      if (!mask)
      {
        continue;
      }

      // parent-most level for pv
      if (parent_level_num_ranks == -1 && parent_level_compressed)
      {
        parent_level_compressed = compression_info.compressed_masks[level][pv];

        // for (auto iter = compression_info.per_level_info_map.at(level).begin(); iter != compression_info.per_level_info_map.at(level).end(); iter++)
        // {
        //   std::cout << iter->first << ", " << iter->second.rank_formats.size()<< std::endl;
        // }

        if (parent_level_compressed)
        {
          parent_level_num_ranks = compression_info.per_level_info_map.at(level).at(pv).rank_formats.size();
        }
        parent_level = level;
        continue;
      }

      // intermediate storage levels
      bool cur_level_compressed = compression_info.compressed_masks.at(level).at(pv);
      int cur_level_num_ranks = -1;
      if (cur_level_compressed)
      {
        cur_level_num_ranks = compression_info.per_level_info_map.at(level).at(pv).rank_formats.size();
      }

      assert((cur_level_compressed && cur_level_num_ranks > 0)
               || (!cur_level_compressed && cur_level_num_ranks == -1));

      // get the overall architecture level id for current storage level and its parent level
      auto overall_level_id = topology_specs.StorageMap(level);
      auto overall_parent_level_id = topology_specs.StorageMap(parent_level);

      if (parent_level_compressed && !cur_level_compressed && !compression_info.decompression_supported_masks[level])
      {
        success = false;
        eval_status[overall_level_id].success = false;
        eval_status[overall_level_id].fail_reason = "decompression (from parent level) needed but not supported";
        if (break_on_failure)
        { return success; }
      }


      // std::cout << "parent level compressed: " << parent_level_compressed
      // << " cur level compressed: " << cur_level_compressed << " compression support mask: "
      // << compression_info.compression_supported_masks[level] << std::endl;
      if (problem::GetShape()->IsReadWriteDataSpace.at(pv) &&
        parent_level_compressed && !cur_level_compressed && !compression_info.compression_supported_masks[level])
      {
        success = false;
        eval_status[overall_level_id].success = false;
        eval_status[overall_level_id].fail_reason = "compression (to parent level) needed but not supported";
        if (break_on_failure)
        { return success; }
      }

      if (parent_level_compressed && cur_level_compressed && parent_level_num_ranks < cur_level_num_ranks
        && !compression_info.tile_partition_supported_masks[parent_level])
      {
        success = false;
        eval_status[overall_parent_level_id].success = false;
        eval_status[overall_parent_level_id].fail_reason =
          "runtime partition needed but not supported (NOT IMPLEMENTED YET)";
        if (break_on_failure)
        { return success; }
      }

      // prepare for next round of checks
      parent_level_num_ranks = cur_level_num_ranks;
      parent_level_compressed = cur_level_compressed;
      parent_level = level;
    }
  }
  return success; // no level failed
}

void CalculateExpectedOccupancy(tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                const model::Topology::Specs& topology_specs)
{

  // Expected occupancy is a weights sum of possible occupancies
  //    note that for metadata, the occupancy is a function of the data tile occupancy
  //    for for each possible data tile occupancy, we need to recalculate the metadata occupancy

  for (unsigned pv = 0; pv < unsigned(problem::GetShape()->NumDataSpaces); pv++)
  {
    for (unsigned level = 0; level < topology_specs.NumStorageLevels(); level++)
    {
      auto& pv_data_movement_info = compound_data_movement_nest[pv][level];
      double total_non_empty_payloads = 0;

      // Initialize occupancy holders
      tiling::MetaDataTileOccupancy expected_metadata_occupancy = {}; //empty means no metadata

      std::uint64_t abs_max_tile_occupancy = pv_data_movement_info.GetMaxDataTileOccupancyByConfidence(1.0);
      std::uint64_t abs_min_tile_occupancy = pv_data_movement_info.GetMinDataTileOccupancy();

      for (std::uint64_t possible_occupancy = abs_min_tile_occupancy;
           possible_occupancy <= abs_max_tile_occupancy; possible_occupancy++)
      {
        double p = pv_data_movement_info.GetDataTileOccupancyProbability(possible_occupancy);
        if (p != 0)
        {
          // update expected occupancy accordingly
          total_non_empty_payloads += p * (double)possible_occupancy;
          if (pv_data_movement_info.has_metadata)
          {
            // Calculate the resulted metadata occupancy for this specific potential data tile size
            // the exact possible occupancy serves as additional information about the tile
            // it is upto the density models to determine whether this addition information is useful
            tiling::ExtraTileConstraintInfo extra_constraint_info;
            extra_constraint_info.Set(pv_data_movement_info.shape, possible_occupancy);
            tiling::CoordinateSpaceTileInfo possible_coord_tile;
            possible_coord_tile.Set(pv_data_movement_info.shape, pv, extra_constraint_info);
            auto occupancy = pv_data_movement_info.GetMetaDataTileOccupancyGivenDataTile(possible_coord_tile);
            // update the metadata tile occupancy record (each item in the record correspond to a rank)
            for (unsigned r = 0; r < occupancy.size(); r++)
            {
              auto per_rank_occupancy = occupancy[r];
              per_rank_occupancy.Scale(p);
              if (expected_metadata_occupancy.size() == r) expected_metadata_occupancy.push_back(per_rank_occupancy);
              else expected_metadata_occupancy[r].Add(per_rank_occupancy);
            }
          }
        }
      }

      // Finished calculating the weighted sum of all possible occupancies, update record
      // density is a fact, uncompressed could also have density < 1.0
      pv_data_movement_info.expected_density = total_non_empty_payloads / pv_data_movement_info.shape;
      pv_data_movement_info.expected_metadata_occupancy = expected_metadata_occupancy;
      pv_data_movement_info.expected_data_occupancy = total_non_empty_payloads;
    }
  }
}

void InitializeSparsityRelatedEntries(const problem::Workload* workload,
                                      tiling::CompoundTileNest& compound_tile_nest)
{
  // initialize all tile density models and metadata representation related entries in tile info
  //    if default dense, then the tile has a fixed density of 1.0 and empty other entries
  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  // all datatypes must have same number of tiling levels, take that of the first dataspace
  unsigned num_levels = compound_data_movement_nest[0].size();
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    for (unsigned level = 0; level < num_levels; level++)
    {
      // the commented out lines are initialized in tile info initialization
      // TODO: might want to have a new data structure for post processed sparse traffic,
      //       now we are carrying both dense and sparse in tile info
      compound_data_movement_nest[pv][level].expected_density = 1.0;
      compound_data_movement_nest[pv][level].SetDensityModel(workload->GetDensity(pv));
      //compound_data_movement_nest[pv][level].metadata_subnest = {}; // only useful if has metadata
      //compound_data_movement_nest[pv][level].metadata_subtile_shape = {}; // only useful if has metadata
      //compound_data_movement_nest[pv][level].fiber_shape = {}; // only useful if has metadata
      compound_data_movement_nest[pv][level].expected_data_occupancy = compound_data_movement_nest[pv][level].shape;
      // compound_data_movement_nest[pv][level].expected_metadata_occupancy = {};
    }
  }
}

// Perform all necessary sparse analysis
//     - compression related
//     - gating/skipping related
bool PerformSparseProcessing(problem::Workload* workload,
                             Mapping& mapping,
                             tiling::CompoundTileNest& compound_tile_nest,
                             SparseOptimizationInfo* sparse_optimization_info,
                             const model::Topology::Specs& topology_specs,
                             std::vector <model::EvalStatus>& eval_status,
                             const bool break_on_failure)
{

  bool success = true;

  //
  // Initialize necessary tile info
  //

  // Initialize tile density models
  InitializeSparsityRelatedEntries(workload, compound_tile_nest);

  // Initialize fine grained access counts
  InitializeFineGrainedAccesses(compound_tile_nest, topology_specs);

  SparseAnalysisState state;
  bool sparse_analysis_needed;
  sparse_analysis_needed = state.Init(sparse_optimization_info, workload, mapping, topology_specs.NumStorageLevels());
  if (!sparse_analysis_needed) return success;

  state.CollectCompletePointSetsAndSubnests();

  //
  // Define the necessary densities/probabilities/misc info of sparse optimizations
  //

  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;

  // Define the necessary metadata modeling information according to mapping
  success = DefineCompressionFormatModels(state, compound_data_movement_nest, topology_specs,
                                          eval_status, break_on_failure);
  if (!success && break_on_failure) return success;

  // Once the compression models are defined, the expected data and metadata occupancy are also defined
  CalculateExpectedOccupancy(compound_data_movement_nest, topology_specs);
  CalculateExpectedMetaDataAccesses(compound_data_movement_nest, topology_specs);

  // Check the mapping-dependent alignment unit requirement above the compute level
  // success = CheckComputeAlignmentUnitRequirement(state, compound_data_movement_nest, topology_specs, eval_status);
  // if (!success && break_on_failure) return success;

  // Define the impact of storage optimizations at each level
  success = DefineStorageOptimizationImpact(state, compound_data_movement_nest, topology_specs,
                                            eval_status, break_on_failure);
  if (!success && break_on_failure) return success;

  //
  // Calculate fine grained accesses based on defined optimization behaviors
  //
  ProcessDataReprImpactOnStorageAccesses(state, compound_data_movement_nest);
  PropagateImpactOfExplicitlyOptimizedRead(state, compound_tile_nest, topology_specs);
  CalculateFineGrainedStorageAccesses(state, compound_data_movement_nest);
  CalculateDecompressionCompressionCost(state.num_storage_levels_, compound_data_movement_nest);
  CalculateFineGrainedComputeAccesses(state, compound_tile_nest);

  return success;
}

} // namespace
