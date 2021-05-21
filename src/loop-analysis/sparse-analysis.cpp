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

#include "sparse-analysis.hpp"
#include "coordinate-space-tile-info.hpp"

namespace sparse
{

//
// SparseAnalysisState Function Implementations
//
bool SparseAnalysisState::Init(sparse::SparseOptimizationInfo *sparse_optimization_info,
                               problem::Workload *workload,
                               Mapping mapping,
                               std::uint64_t num_storage_levels)
{

  bool sparse_analysis_needed = false;

  if (sparse_optimization_info->compression_info.all_ranks_default_dense) return sparse_analysis_needed;
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
  c_operand_prop_impact_ = {};

  // by default, no propagation impact
  for (DataSpaceID pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      c_operand_prop_impact_[pv] = 0.0;
    }
  }

  // by default, no explicit optimization applied
  dspace_optimization_masks_ = {{"gate", {}}, {"skip", {}}};
  for (unsigned l = 0; l < num_storage_levels_; l++)
  {
    dspace_optimization_masks_["gate"].push_back({});
    dspace_optimization_masks_["skip"].push_back({});
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      dspace_optimization_masks_["gate"][l][pv] = false;
      dspace_optimization_masks_["skip"][l][pv] = false;
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
  auto &loops = mapping_.complete_loop_nest.loops;
  for (unsigned loop_level = 0; loop_level < loops.size(); loop_level++)
  {
    auto &loop = loops[loop_level];
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
  std::vector<DataSpaceID> condition_on_dspace_ids;
  unsigned target_dspace_level;
  double optimization_prob;
  double expected_target_tile_occupancy;
};


bool CheckComputeAlignmentUnitRequirement(SparseAnalysisState &state,
                                          tiling::CompoundDataMovementNest &compound_data_movement_nest,
                                          const model::Topology::Specs &topology_specs,
                                          std::vector <model::EvalStatus> &eval_status)
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
        auto &loop = state.complete_subnests_[l][loop_id];
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

void InitializeFineGrainedAccesses(tiling::CompoundTileNest &compound_tile_nest,
                                   const model::Topology::Specs &topology_specs)
{

  auto &compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto &compute_info_nest = compound_tile_nest.compute_info_nest;

  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    for (unsigned l = 0; l < topology_specs.NumStorageLevels(); l++)
    {
      auto &fine_grained_accesses = compound_data_movement_nest[pv][l].fine_grained_accesses;

      for (unsigned op_id = 0; op_id < tiling::storageOperationTypes.size(); op_id++){
        auto op_name = tiling::storageOperationTypes[op_id];
        fine_grained_accesses[op_name] = 0;
      }

      // default to uncompressed without metadata
      fine_grained_accesses["random_read"] = compound_data_movement_nest[pv][l].reads;
      fine_grained_accesses["random_fill"] = compound_data_movement_nest[pv][l].fills;
      fine_grained_accesses["random_update"] = compound_data_movement_nest[pv][l].updates;
    }
  }

  auto &compute_info = compute_info_nest[0];

  for (unsigned op_id = 0; op_id < tiling::arithmeticOperationTypes.size(); op_id++){
    auto op_name = tiling::arithmeticOperationTypes[op_id];
    compute_info.fine_grained_accesses[op_name] = 0;
  }
  compute_info.fine_grained_accesses["random_compute"] = compute_info.accesses * compute_info.replication_factor;
}

bool ComputeIneffectualReadImpact(const SparseAnalysisState &state,
                                  tiling::CompoundDataMovementNest &compound_data_movement_nest,
                                  const unsigned storage_level_id,
                                  const model::Topology::Specs &topology_specs,
                                  ExplicitReadOptimizationImpact &resulted_impact,
                                  std::vector <model::EvalStatus> &eval_status)
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
  // 2) go through loop nests above its child level in a bottom up fashion, locate the first loop that projects to target, named target-loop
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

  for (unsigned l = child_level + 1; l <= storage_level_id && !found_target_dspace_loop; l++)
  {
    for (unsigned loop_id = 0; loop_id < state.complete_subnests_[l].size() && !found_target_dspace_loop; loop_id++)
    {
      if (!state.trivial_nest_masks_[l][loop_id])
      {
        target_loop_dim = state.complete_subnests_[l][loop_id].dimension;
        if (target_dspace_dimensions.find(target_loop_dim) != target_dspace_dimensions.end())
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
          }
            // innermost loop of level l: next level top most operation point
          else if (loop_id == 0)
          {
            mold_high = state.maxtile_molds_high_[l - 1].back();
          }
            // intermediate loop of a level l: next loop in level l
          else
          {
            mold_high = state.maxtile_molds_high_[l][loop_id - 1];
          }
        }
      }
    }
  }

  // target-loop is trivial and topmost
  if (!found_target_dspace_loop)
  {
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
  //   else, it makes no different on the condition on dataspace tile shape
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
  double prob_target_dspace_effectual = 1.0;
  for (unsigned i = 0; i < resulted_impact.condition_on_dspace_ids.size(); i++)
  {
    DataSpaceID condition_on_dspace_id = resulted_impact.condition_on_dspace_ids[i];
    // construct the corresponding coordinate space tile for dataspace b and calculate the prob of the tile being empty
    tiling::CoordinateSpaceTileInfo cspace_tile;
    cspace_tile.Set(operation_space_mold.GetSize(condition_on_dspace_id), condition_on_dspace_id);

    double prob_condition_on_dspace_empty = state.workload_->GetDensity(condition_on_dspace_id)
      ->GetTileOccupancyProbability(cspace_tile, 0);
    prob_target_dspace_effectual *= (1 - prob_condition_on_dspace_empty);

    //std::cout << " target dspace: " << problem::GetShape()->DataSpaceIDToName.at(resulted_impact.target_dspace_id)
    //          << " condition on dspace: " << problem::GetShape()->DataSpaceIDToName.at(condition_on_dspace_id)
    //          << "\n operation space mold derived dataspace sizes (target, conditioned): "
    //          << operation_space_mold.GetSize(target_dspace_id) << "  "
    //          << operation_space_mold.GetSize(condition_on_dspace_id)
    //          << "\n condtion on tile empty probability(target skip prob): " << prob_condition_on_dspace_empty
    //          << std::endl;
  }
  double prob_target_dspace_ineffectual = 1 - prob_target_dspace_effectual;
  // std::cout << " (after aggregation) target skip prob: " << prob_target_dspace_ineffectual << std::endl;

  //
  // Occupancy calculations
  //
  double expected_target_tile_occupancy = compound_data_movement_nest[target_dspace_id][target_dspace_level].expected_data_occupancy;

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
  std::vector<ExplicitReadOptimizationImpact> possible_impact(group_of_action_optimization.size());
  for (unsigned  i = 0; i < group_of_action_optimization.size(); i++)
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

  for (unsigned i = 0; i < possible_impact.size(); i++)
  {
    auto &impact = possible_impact[i];

    access_cost = topology_specs.GetStorageLevel(impact.target_dspace_level)->op_energy_map.at("random_read");
    auto block_size = topology_specs.GetStorageLevel(impact.target_dspace_level)->block_size.Get();
    double savings = (access_cost / block_size) * impact.expected_target_tile_occupancy * impact.optimization_prob;
    max_savings = savings >= max_savings ? savings : max_savings;
    option_id = savings >= max_savings ? i : option_id;
  }

  //
  // record the impact of the better choice
  //
  auto target_dspace_level = possible_impact[option_id].target_dspace_level;
  auto optimization_prob = possible_impact[option_id].optimization_prob;
  auto target_dspace_id = possible_impact[option_id].target_dspace_id;

   // std::cout << "\t=== Final: target dspace: " << problem::GetShape()->DataSpaceIDToName.at(target_dspace_id)
   // << "  optimization prob: " << optimization_prob
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
    }
  } else
  {
    state.dspace_optimization_masks_[optimization_type][target_dspace_level][target_dspace_id] = true;
    state.prob_explicitly_optimized_read_[target_dspace_level][target_dspace_id] = optimization_prob;
  }

  return success;
}

bool DefineStorageOptimizationImpact(SparseAnalysisState &state,
                                     tiling::CompoundDataMovementNest &compound_data_movement_nest,
                                     const model::Topology::Specs &topology_specs,
                                     std::vector <model::EvalStatus> &eval_status,
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

void CalculateExpectedMetaDataAccesses(tiling::CompoundDataMovementNest &compound_data_movement_nest,
                                       const model::Topology::Specs &topology_specs)
{
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    for (int l = topology_specs.NumStorageLevels() - 1; l >= 0; l--)
    {

      auto &data_movement_info = compound_data_movement_nest[pv][l];
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
  // auto& compute_info_nest =  compound_tile_nest.compute_info_nest;

  std::vector <problem::PerDataSpace<double>> max_reads = {};
  std::vector <problem::PerDataSpace<double>> max_updates = {};
  std::vector <problem::PerDataSpace<double>> max_fills = {};

  std::vector <problem::PerDataSpace<double>> max_metadata_reads = {};
  std::vector <problem::PerDataSpace<double>> max_metadata_updates = {};
  std::vector <problem::PerDataSpace<double>> max_metadata_fills = {};

  // std::vector<double> max_computes = {};

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
      max_reads[l][pv] = compound_data_movement_nest[pv][l].reads;
      max_fills[l][pv] = compound_data_movement_nest[pv][l].fills;
      max_updates[l][pv] = compound_data_movement_nest[pv][l].updates;

      max_metadata_reads[l][pv] = compound_data_movement_nest[pv][l].metadata_reads;
      max_metadata_fills[l][pv] = compound_data_movement_nest[pv][l].metadata_fills;
      max_metadata_updates[l][pv] = compound_data_movement_nest[pv][l].metadata_updates;
    }
  }

  // propagate the impact of explicitly applied read optimization
  // for reads and fills of lower levels in a top down fashion

  for (int l = topology_specs.NumStorageLevels() - 1; l >= 0; l--)
  {
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
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
        std::string type = state.dspace_optimization_masks_.at("skip").at(l).at(pv) ? "skipped" : "gated";

        unsigned impacted_level_id = compound_data_movement_nest[pv][l].child_level;
        while (impacted_level_id != std::numeric_limits<unsigned>::max())
        {
          // std::cout << "\timpacted level: " << topology_specs.GetStorageLevel(impacted_level_id)->level_name << std::endl;
          auto &data_movement_record = compound_data_movement_nest[pv][impacted_level_id];
          data_movement_record.fine_grained_accesses[type + "_read"] += floor(max_reads[impacted_level_id][pv] * p);
          data_movement_record.fine_grained_accesses[type + "_fill"] += floor(max_fills[impacted_level_id][pv] * p);
          data_movement_record.fine_grained_accesses[type + "_metadata_read"] += floor(max_metadata_reads[impacted_level_id][pv] * p);
          data_movement_record.fine_grained_accesses[type + "_metadata_fill"] += floor(max_metadata_fills[impacted_level_id][pv] * p);

          if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
          {
            data_movement_record.fine_grained_accesses[type + "_update"] += floor(max_updates[impacted_level_id][pv] * p);
            data_movement_record.fine_grained_accesses[type + "_metadata_update"] += floor(max_metadata_updates[impacted_level_id][pv] * p);
          }

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

        if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
        {
          state.c_operand_prop_impact_[pv] += (1 - state.c_operand_prop_impact_[pv]) * p;
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
}

void CalculateFineGrainedStorageAccesses(const SparseAnalysisState &state,
                                         tiling::CompoundDataMovementNest &compound_data_movement_nest)
{
  for (int l = state.num_storage_levels_ - 1; l >= 0; l--)
  {
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      if (compound_data_movement_nest[pv][l].compressed ||
        state.dspace_optimization_masks_.at("gate").at(l).at(pv)
        || state.dspace_optimization_masks_.at("skip").at(l).at(pv))
      {
        auto &data_movement_record = compound_data_movement_nest[pv][l];
        auto max_reads = data_movement_record.fine_grained_accesses["random_read"];
        auto max_fills = data_movement_record.fine_grained_accesses["random_fill"];
        auto max_updates = data_movement_record.fine_grained_accesses["random_update"];
        auto max_metadata_reads = data_movement_record.fine_grained_accesses["random_metadata_read"];
        // auto max_metadata_fills = data_movement_record.fine_grained_accesses["random_metadata_fill"];
        auto max_metadata_updates = data_movement_record.fine_grained_accesses["random_metadata_update"];

        //std::cout << "\t original counts: "<< std::endl;
        //for (auto iter=compound_data_movement_nest[pv][l].fine_grained_accesses.begin();
        //	 iter!=compound_data_movement_nest[pv][l].fine_grained_accesses.end(); iter++)
        //{
        //  std::cout << "\t" << iter->first << ": " << iter->second << std::endl;
        //}
        // apply compression impact (compression impact on metadata already applied)
        if (compound_data_movement_nest[pv][l].compressed)
        {
          double expected_sparsity = (1 - compound_data_movement_nest[pv][l].GetExpectedTileDensity());
          data_movement_record.fine_grained_accesses["skipped_read"] += floor(max_reads * expected_sparsity);
          data_movement_record.fine_grained_accesses["skipped_fill"] += floor(max_fills * expected_sparsity);
          max_reads -= floor(max_reads * expected_sparsity);
          max_fills -= floor(max_fills * expected_sparsity);
          if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
          {
            data_movement_record.fine_grained_accesses["skipped_update"] += floor(max_updates * expected_sparsity);
            max_updates -= floor(max_updates * expected_sparsity);
          }
        }

        // apply per level explicit read optimization impact
        if (state.dspace_optimization_masks_.at("gate").at(l).at(pv))
        {
          //std::cout << "\t gated read..." << std::endl;
          data_movement_record.fine_grained_accesses["gated_read"] += floor(max_reads * state.prob_explicitly_optimized_read_.at(l).at(pv));
          // we can only gate the portion of the metadata transferred to child level
          data_movement_record.fine_grained_accesses["gated_metadata_read"]
            += floor(max_metadata_reads * data_movement_record.child_level_metadata_occupancy_ratio * state.prob_explicitly_optimized_read_.at(l).at(pv));

          // the optimized away reads will not lead to any updates to this level
          data_movement_record.fine_grained_accesses["gated_update"] +=  floor(max_updates * state.prob_explicitly_optimized_read_.at(l).at(pv));
          data_movement_record.fine_grained_accesses["gated_metadata_update"]
            += floor(max_metadata_updates * data_movement_record.child_level_metadata_occupancy_ratio * state.prob_explicitly_optimized_read_.at(l).at(pv));
        }

        if (state.dspace_optimization_masks_.at("skip").at(l).at(pv))
        {
          //std::cout << "\t skipped read..." << std::endl;
          data_movement_record.fine_grained_accesses["skipped_read"] +=
            floor(max_reads * state.prob_explicitly_optimized_read_.at(l).at(pv));
          // we can only skip the portion of the metadata transferred to child level
          data_movement_record.fine_grained_accesses["skipped_metadata_read"] += floor(
            max_metadata_reads * data_movement_record.child_level_metadata_occupancy_ratio *
            state.prob_explicitly_optimized_read_.at(l).at(pv));
          // the optimized away reads will not lead to any updates to this level
          data_movement_record.fine_grained_accesses["skipped_update"] +=  floor(max_updates * state.prob_explicitly_optimized_read_.at(l).at(pv));
          data_movement_record.fine_grained_accesses["skipped_metadata_update"]
            += floor(max_metadata_updates * data_movement_record.child_level_metadata_occupancy_ratio * state.prob_explicitly_optimized_read_.at(l).at(pv));
        }

        data_movement_record.fine_grained_accesses["random_read"] = data_movement_record.reads
          - data_movement_record.fine_grained_accesses["gated_read"]
          - data_movement_record.fine_grained_accesses["skipped_read"];
        data_movement_record.fine_grained_accesses["random_fill"] = data_movement_record.fills
          - data_movement_record.fine_grained_accesses["gated_fill"]
          - data_movement_record.fine_grained_accesses["skipped_fill"];
        data_movement_record.fine_grained_accesses["random_update"] = data_movement_record.updates
          - data_movement_record.fine_grained_accesses["gated_update"]
          - data_movement_record.fine_grained_accesses["skipped_update"];
        data_movement_record.fine_grained_accesses["random_metadata_fill"] = data_movement_record.metadata_fills
          - data_movement_record.fine_grained_accesses["gated_metadata_fill"]
          - data_movement_record.fine_grained_accesses["skipped_metadata_fill"];
        data_movement_record.fine_grained_accesses["random_metadata_read"] = data_movement_record.metadata_reads
          - data_movement_record.fine_grained_accesses["gated_metadata_read"]
          - data_movement_record.fine_grained_accesses["skipped_metadata_read"];
        data_movement_record.fine_grained_accesses["random_metadata_update"] = data_movement_record.metadata_updates
          - data_movement_record.fine_grained_accesses["gated_metadata_update"]
          - data_movement_record.fine_grained_accesses["skipped_metadata_update"];
      }

      // for (auto iter = compound_data_movement_nest[pv][l].fine_grained_accesses.begin();
      //           iter != compound_data_movement_nest[pv][l].fine_grained_accesses.end(); iter++)
      // {
      //    std::cout << iter->first << ": " << iter->second << std::endl;
      // }

    }
  }

}

void CalculateDecompressionCompressionCost(const std::uint64_t num_storage_levels,
                                           tiling::CompoundDataMovementNest &compound_data_movement_nest)
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
    auto &mold_high = set_of_operation_spaces.op_space_mold_high;
    problem::OperationPoint origin;
    tiling::CoordinateSpaceTileInfo ctile_info;
    problem::OperationSpace mold_op_space(set_of_operation_spaces.workload, origin, mold_high);
    ctile_info.Set(mold_op_space.GetSize(iter->first), iter->first);
    expected_operand_densities[iter->first] = iter->second->GetExpectedTileOccupancy(ctile_info)/ctile_info.GetShape();
  }
  return expected_operand_densities;
}

void CalculateFineGrainedComputeAccesses(const SparseAnalysisState &state,
                                         tiling::CompoundTileNest& compound_tile_nest)
{

  // scenarios of operands states       | resulted compute actions
  // -----------------------------------------------------------------
  // 1) A: ENZ, B: ENZ                  | random compute
  // 2) A: ENZ, B: EZ, 4) A: EZ, B: ENZ | random/gated compute (w/o vs. w/ zero operand detection)
  // 3) A: ENZ, B: NE, 7) A: NE, B: ENZ | skipped/gated compute (w/o vs. w/ intersection)
  // 5) A: EZ, B: EZ                    | random/gated compute (w/o vs. w/ zero operand detection)
  // 6) A: EZ, B: NE,  8) A: NE, B: EZ  | skipped/gated compute (w/o vs. w/ intersection)
  // 9) A: NE, B: NE                    | skipped compute
  // -----------------------------------------------------------------
  // NE: not exist; EZ: exist, is zero; ENZ: exist, not zero


  // Find the inner most level tile for each operand
  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto& compute_info = compound_tile_nest.compute_info_nest[0];

  std::map<DataSpaceID, bool> operand_compressed;
  PerDataSpaceDensityModel operand_density_models;
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      auto &pv_data_movement_nest = compound_data_movement_nest[pv];
      for (unsigned l = 0; l < state.num_storage_levels_; l++)
      {
        if (pv_data_movement_nest[l].shape != 0)
        {
          // found the inner most storage for dspace pv
          operand_compressed[pv] = pv_data_movement_nest[l].compressed;
          operand_density_models[pv] = pv_data_movement_nest[l].tile_density;
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
    per_operand_states[pv][EXIST_NOT_ZERO] = pv_density * (1 - state.c_operand_prop_impact_.at(pv));
    if (operand_compressed.at(pv))
    {
      per_operand_states[pv][EXIST_ZERO] = 0;
      per_operand_states[pv][NOT_EXIST] = 1.0 - per_operand_states[pv][EXIST_NOT_ZERO];
    } else
    {
      per_operand_states[pv][EXIST_ZERO] =  (1 - pv_density) * (1 - state.c_operand_prop_impact_.at(pv));
      per_operand_states[pv][NOT_EXIST] = pv_density * state.c_operand_prop_impact_.at(pv) + (1 - pv_density) * state.c_operand_prop_impact_.at(pv);
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

  // Initialize fine grained access counts
  double total_compute = compute_info.replication_factor * (double)compute_info.accesses;

  // Extract hardware sparse optimization spec (can zero operand be identified?)
  bool compute_recognize_zero_operands = false;
  if (state.sparse_optimization_info_->compute_optimization_info.find("zero_gating") !=
    state.sparse_optimization_info_->compute_optimization_info.end())
  {
    compute_recognize_zero_operands = state.sparse_optimization_info_->compute_optimization_info.at("zero_gating");
  }

  // Initialize all the counters
  double random_compute = 0.0, skipped_compute = 0.0, gated_compute = 0.0, tmp_delta = 0.0;
  double prob_a, prob_b;

  // std::cout << "impact: (" << problem::GetShape()->DataSpaceIDToName.at(op_a_id)
  // << "  " << problem::GetShape()->DataSpaceIDToName.at(op_b_id) << ") "
  // << state.c_operand_prop_impact_.at(op_a_id) << "  "
  // << state.c_operand_prop_impact_.at(op_b_id) << std::endl;
  // Analyze case by case
  // 1) A: ENZ, B: ENZ                  | random compute
  prob_a = per_operand_states.at(op_a_id).at(EXIST_NOT_ZERO);
  prob_b = per_operand_states.at(op_b_id).at(EXIST_NOT_ZERO);
  random_compute += prob_a * prob_b * total_compute;
  //std::cout << "(1) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute << std::endl;

  // 2) A: ENZ, B: EZ, 4) A: EZ, B: ENZ | random/gated compute (w/o vs. w/ zero operand detection)
  prob_a = per_operand_states.at(op_a_id).at(EXIST_NOT_ZERO);
  prob_b = per_operand_states.at(op_b_id).at(EXIST_ZERO);

  tmp_delta = total_compute * prob_a * prob_b;

  if (compute_recognize_zero_operands)
  {
    gated_compute += tmp_delta;
  }
  else
  {
    random_compute += tmp_delta;
  }

  prob_a = per_operand_states.at(op_a_id).at(EXIST_ZERO);
  prob_b = per_operand_states.at(op_b_id).at(EXIST_NOT_ZERO);

  tmp_delta = total_compute * prob_a * prob_b;

  if (compute_recognize_zero_operands)
  {
    gated_compute += tmp_delta;
  }
  else
  {
    random_compute += tmp_delta;
  }

  // std::cout << "(2)(4) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute <<
  //          std::endl;

  // 3) A: ENZ, B: NE, 7) A: NE, B: ENZ | skipped/gated compute (w/o vs. w/ intersection)
  prob_a = per_operand_states.at(op_a_id).at(EXIST_NOT_ZERO);
  prob_b = per_operand_states.at(op_b_id).at(NOT_EXIST);
  tmp_delta = prob_a * prob_b * total_compute;
  if (state.c_intersection_dims_.size() > 0)
  {
    gated_compute += tmp_delta;
  }
  else
  {
    skipped_compute += tmp_delta;
  }

  prob_a = per_operand_states.at(op_a_id).at(NOT_EXIST);
  prob_b = per_operand_states.at(op_b_id).at(EXIST_NOT_ZERO);
  tmp_delta = prob_a * prob_b * total_compute;
  if (state.c_intersection_dims_.size() > 0)
  {
    gated_compute += tmp_delta;
  }
  else
  {
    skipped_compute += tmp_delta;
  }

  // std::cout << "(3)(7) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute <<
  //          std::endl;


  // 5) A: EZ, B: EZ                    | random/gated compute (w/o vs. w/ zero operand detection)
  prob_a = per_operand_states.at(op_a_id).at(EXIST_ZERO);
  prob_b = per_operand_states.at(op_b_id).at(EXIST_ZERO);
  tmp_delta = total_compute * prob_a * prob_b;

  if (compute_recognize_zero_operands)
  {
    gated_compute += tmp_delta;
  }
  else
  {
    random_compute += tmp_delta;
  }

  // std::cout << "(5) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute << std::endl;


  // 6) A: EZ, B: NE,  8) A: NE, B: EZ  | skipped/gated compute (w/o vs. w/ intersection)
  prob_a = per_operand_states.at(op_a_id).at(EXIST_ZERO);
  prob_b = per_operand_states.at(op_b_id).at(NOT_EXIST);
  tmp_delta = prob_a * prob_b * total_compute;
  if (state.c_intersection_dims_.size() > 0)
  {
    gated_compute += tmp_delta;
  }
  else
  {
    skipped_compute += tmp_delta;
  }

  prob_a = per_operand_states.at(op_a_id).at(NOT_EXIST);
  prob_b = per_operand_states.at(op_b_id).at(EXIST_ZERO);
  tmp_delta = prob_a * prob_b * total_compute;
  if (state.c_intersection_dims_.size() > 0)
  {
    gated_compute += tmp_delta;
  }
  else
  {
    skipped_compute += tmp_delta;
  }

  // std::cout << "(6)(8) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute <<
  //          std::endl;


  // 9) A: NE, B: NE                    | skipped compute
  prob_a = per_operand_states.at(op_a_id).at(NOT_EXIST);
  prob_b = per_operand_states.at(op_b_id).at(NOT_EXIST);
  skipped_compute += prob_a * prob_b * total_compute;

  // std::cout << "(9) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute << std::endl;

  // std::cout << "total: " << total_compute << "  sum: " <<  skipped_compute + random_compute + gated_compute
  // << " diff: " << total_compute - skipped_compute - random_compute - gated_compute  << std::endl;

  // sanity check
  // as long as different is smaller than 0.001% of the total compute, pass
  // there could be tiny discrepencies due to lack of precision...
  assert(abs(skipped_compute + random_compute + gated_compute - total_compute) < total_compute * 0.00001);


  // now round the action counts into integers (pessimistic rounding)
  compute_info.fine_grained_accesses["skipped_compute"] = floor(skipped_compute);
  compute_info.fine_grained_accesses["gated_compute"] = floor(gated_compute);
  compute_info.fine_grained_accesses["random_compute"] = total_compute - floor(skipped_compute) - floor(gated_compute);

  // std::cout << "(final) skipped compute: " << compute_info.fine_grained_accesses["skipped_compute"]
  //   << " gated compute: " << compute_info.fine_grained_accesses["gated_compute"]
  //   << " random compute: " << compute_info.fine_grained_accesses["random_compute"]
  //   << std::endl;

}

bool ApplyRanksOuterToInner(std::uint64_t inner_rank_id,
                            const std::vector <loop::Descriptor> &singleton_metadata_subnest,
                            const std::vector <std::uint64_t> &singleton_metadata_subtile_shape,
                            const sparse::PerDataSpaceCompressionInfo &pv_compression_info,
                            tiling::DataMovementInfo &pv_data_movement_info)
{
  std::vector <loop::Descriptor> flattened_rank_nest;
  std::set <problem::Shape::DimensionID> flattening_rule;

  bool pv_has_metadata = pv_compression_info.HasMetaData();
  std::uint64_t cur_level_num_ranks = pv_has_metadata ? pv_compression_info.rank_formats.size() : 1;

  // start by applying the outermost rank to the outermost loop
  // if there are extra inner ranks supported, all of the these ranks will cost no overhead
  int loop_id = singleton_metadata_subnest.size() - 1;
  int r_id = cur_level_num_ranks;
  std::uint64_t corresponding_tile_shape = 1;
  // std::cout << "total number of ranks: " << cur_level_num_ranks
  // << "  inner rank id: " << inner_rank_id
  // << " total loops: " << singleton_metadata_subnest.size() << std::endl;

  while(r_id > (int)inner_rank_id && loop_id >= 0)
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
      //std::cout << "mapping loop below to rank " << r_id << std::endl;
      //std::cout << loop << std::endl;

      if (!trivial_loop)
      {
        flattened_rank_nest.push_back(loop);
        corresponding_tile_shape = singleton_metadata_subtile_shape[loop_id + 1];
      }
      loop_id--;

      bool in_flattened_list;
      if (loop_id >= 0)
      {
        in_flattened_list = !pv_has_metadata ?
                            true : pv_compression_info.FoundDimensionInFlatteningRule(r_id, loop.dimension,
                                                                                      flattening_rule);

        // std::cout << "in flattening list: " << in_flattened_list << std::endl;
        //  if (pv_has_metadata)
        //  {
        //    for (auto dim = flattening_rule.begin(); dim != flattening_rule.end(); dim++)
        //    {
        //      std::cout << problem::GetShape()->DimensionIDToName.at(*dim) << "  ";
        //    }
        //    std::cout << std::endl;
        //  }


        if (!pv_has_metadata // if default uncompressed, then all loops must be flattened into 1 rank
            || in_flattened_list)
        {

          auto loop = singleton_metadata_subnest[loop_id];
          trivial_loop = (loop.start + loop.stride) >= loop.end;

          while (loop_id >= 0 && (!pv_has_metadata
                                  || (flattening_rule.find(singleton_metadata_subnest[loop_id].dimension) !=
                                      flattening_rule.end())
                                  || trivial_loop))
          {
            auto loop = singleton_metadata_subnest[loop_id];
            trivial_loop = (loop.start + loop.stride) >= loop.end;
            if (!trivial_loop)
            {
              //std::cout << "mapping loop below to rank " << r_id << std::endl;
              //std::cout << loop << std::endl;
              flattened_rank_nest.push_back(loop);
              corresponding_tile_shape = singleton_metadata_subtile_shape[loop_id + 1];
            }
            loop_id--;
          }
        }
      }
    }
    pv_data_movement_info.metadata_subnest.insert(pv_data_movement_info.metadata_subnest.begin(), flattened_rank_nest);
    pv_data_movement_info.metadata_subtile_shape.insert(pv_data_movement_info.metadata_subtile_shape.begin(), corresponding_tile_shape);

    // reset to 1
    corresponding_tile_shape = 1;

  }


  // fill in the extra inner supported rank (if any)
  while (r_id > (int)inner_rank_id)
  {
    std::vector<loop::Descriptor> tmp_loop = {};
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

  while (loop_id >= 0 )
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
    std::vector<loop::Descriptor> tmp_loop = {};
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
                            const std::vector <loop::Descriptor> &singleton_metadata_subnest,
                            const std::vector <std::uint64_t> &singleton_metadata_subtile_shape,
                            const sparse::PerDataSpaceCompressionInfo &pv_compression_info,
                            tiling::DataMovementInfo &pv_data_movement_info)
{

  //FIXME: this function needs to be updated to perfor correctly, use outer to inner mapping instead
  //  1) skip trivial loops logic
  //  2) subtile shape insertion logic
  assert(false);

  std::vector <loop::Descriptor> flattened_rank_nest;
  std::set <problem::Shape::DimensionID> flattening_rule;

  bool pv_has_metadata = pv_compression_info.HasMetaData();
  std::uint64_t cur_level_num_ranks = pv_has_metadata ? pv_compression_info.rank_formats.size() : 1;

  // start by applying the innermost rank to the innermost loop
  // if there are extra outer ranks supported, all of the these ranks will cost no overhead
  std::uint64_t loop_id = 0;
  std::uint64_t r_id = inner_rank_id - 1;

  while(r_id < cur_level_num_ranks - 1 && loop_id < singleton_metadata_subnest.size())
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
            || (flattening_rule.find(singleton_metadata_subnest[loop_id].dimension)
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

bool DefineCompressionFormatModels(SparseAnalysisState &state,
                                   tiling::CompoundDataMovementNest &compound_data_movement_nest,
                                   const model::Topology::Specs &topology_specs,
                                   std::vector <model::EvalStatus> &eval_status,
                                   const bool break_on_failure)
{

  bool success = true;
  auto compression_info = state.sparse_optimization_info_->compression_info;

  // nothing needs to be done if no metadata involved
  if (compression_info.all_ranks_default_dense) return success;

  std::ostringstream fail_reason;

  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {

    auto &pv_data_movement_nest = compound_data_movement_nest[pv];
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
      // by pretiling, we specifically mean that this level needs to consider inner level nontrivial loops

      if (cur_level_compressed && child_level_id != std::numeric_limits<unsigned>::max()
        && !compression_info.tile_partition_supported_masks[level])
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
          auto &loop = singleton_metadata_subnest[subnest_id];
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
      }
      else
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
        pv_data_movement_nest[level].metadata_subtile_shape.insert(pv_data_movement_nest[level].metadata_subtile_shape.begin(), 1);
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

      //   for (auto iter = pv_data_movement_nest[level].metadata_subnest[i].begin();
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

bool CheckFormatModelsAndMapping(const tiling::NestOfCompoundMasks &masks,
                                 sparse::CompressionInfo &compression_info,
                                 const model::Topology::Specs &topology_specs,
                                 std::vector <model::EvalStatus> &eval_status,
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
      pv_data_movement_info.expected_data_occupancy = pv_data_movement_info.shape; // default
      pv_data_movement_info.expected_metadata_occupancy = {};    // default

      if (pv_data_movement_info.has_metadata)
      {
        // Initialize occupancy holders
        tiling::MetaDataTileOccupancy expected_metadata_occupancy = {}; //empty means no metadata
        double expected_data_occupancy = 0;

        std::uint64_t abs_max_tile_occupancy = pv_data_movement_info.GetMaxDataTileOccupancyByConfidence(1.0);
        for (std::uint64_t possible_occupancy = 0; possible_occupancy <= abs_max_tile_occupancy; possible_occupancy++)
        {
          double p = pv_data_movement_info.GetDataTileOccupancyProbability(possible_occupancy);
          if (p != 0)
          {
            // update expected occupancy accordingly
            if (pv_data_movement_info.compressed) expected_data_occupancy += p * (double) possible_occupancy;

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
        // Finished calculating the weight sum of all possible occupancies, update record
        pv_data_movement_info.expected_metadata_occupancy = expected_metadata_occupancy;
        pv_data_movement_info.expected_data_occupancy = expected_data_occupancy;
      }
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
  success = CheckComputeAlignmentUnitRequirement(state, compound_data_movement_nest, topology_specs, eval_status);
  if (!success && break_on_failure) return success;

  // Define the impact of storage optimizations at each level
  success = DefineStorageOptimizationImpact(state, compound_data_movement_nest, topology_specs,
                                            eval_status, break_on_failure);
  if (!success && break_on_failure) return success;

  //
  // Calculate fine grained accesses based on define optimization behaviors
  //
  PropagateImpactOfExplicitlyOptimizedRead(state, compound_tile_nest, topology_specs);
  CalculateFineGrainedStorageAccesses(state, compound_data_movement_nest);
  CalculateDecompressionCompressionCost(state.num_storage_levels_, compound_data_movement_nest);
  CalculateFineGrainedComputeAccesses(state, compound_tile_nest);

  return success;
}

} // namespace