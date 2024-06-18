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
#include "sparse-analysis/state.hpp"
#include "mapping/loop.hpp"

namespace sparse
{

// Necessary data structures
struct ExplicitReadOptimizationImpact
{
  DataSpaceID target_dspace_id;
  std::vector <DataSpaceID> condition_on_dspace_ids;
  unsigned target_dspace_level;
  double optimization_prob;
  double expected_target_tile_occupancy;
  std::uint64_t spatial_instances;
};


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

  // 1) find the closest storage level >= storage level id that stores target dataspace, named target_dspace_level)
  // 2)
  // if this is not the last level that stores target dataspace (i.e., reuse possible in next level)
  //    go through loop nests above its child level in a bottom up fashion, locate the first loop that projects to target, named target-loop
  //    if the located loop is a temporal loop, proceed
  //    if the located loop is a spatial loop
  //       go up the spatial stack until you find a temporal loop that projects to the target dataspace
  //       along the process, if there is any co-iterated loop below a loop that iterates on conditioned on dataspace only, invalid optimization
  //       as it is not legal anymore to perform a fully concordant "conditioned on" operation

  // if this is the last level that stores target dataspace
  //    will just be the bottom most loop (i.e., no reuse no matter how the loops are ordered)

  // 3) find the operation space that defines the dataspace tiles that is conditioned on
  //    note that the block size of the storage that target dataspace is stored in affects the dataspace b tile that we look at

  //    Specifically, if target_dspace_level has a block size > 1, then finest skipping/gating granularity is  block-size-a
  //    if target-loop is a coiteration loop (dimension of this loop projects to both target dataspace and conditioned on dataspace),
  //    then in order to find the corresponding conditioned on dspace tile,
  //    we should look at the operation space defined by *block-size-a iterations* of target-loop

  DataSpaceID target_dspace_id = resulted_impact.target_dspace_id;
  auto target_dspace_dimensions = state.workload_->GetShape()->DataSpaceIDToDimensionIDVector.at(target_dspace_id);

  // for (auto iter = target_dspace_dimensions.begin(); iter != target_dspace_dimensions.end(); iter++)
  // {
  //  std::cout << state.workload_->GetShape()->FlattenedDimensionIDToName.at(*iter) << std::endl;
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
      fail_reason << "skipping/gating for " << state.workload_->GetShape()->DataSpaceIDToName.at(target_dspace_id)
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
  // step 2) and 3)
  //

  int child_level = compound_data_movement_nest[target_dspace_id][target_dspace_level].child_level
    == std::numeric_limits<unsigned>::max() ?
    -1 : compound_data_movement_nest[target_dspace_id][target_dspace_level].child_level; // -1 is compute level...

  bool found_target_dspace_loop = false;
  problem::Shape::FlattenedDimensionID target_loop_dim = std::numeric_limits<unsigned>::max();
  problem::OperationPoint mold_high;
  loop::Descriptor target_loop; // target loop itself
  unsigned target_loop_storage_level, target_loop_id; // together defines the exact location of the target loop

  problem::OperationPoint origin;
 
  bool construct_scalar_mold = false;
  for (unsigned l = child_level + 1; l <= target_dspace_level && !found_target_dspace_loop; l++)
  {
    for (unsigned loop_id = 0; loop_id < state.complete_subnests_[l].size() && !found_target_dspace_loop; loop_id++)
    {
      if (child_level != -1)
      {
        if(!state.trivial_nest_masks_[l][loop_id]
            && target_dspace_dimensions.find(state.complete_subnests_[l][loop_id].dimension) != target_dspace_dimensions.end())
        {
          found_target_dspace_loop = true;
        }
      }
      else // child_level == -1:
      {
        // we want to look at the loop below the inner most temp loop (could just be a scalar if there is no temporal loops)
        //   or any inner spatial loop that projects to target dspace
        //   prepare for the case where there are spatial loops that projects to cond-on dspace but not target dataspace
        //   in this case, this logic makes sure the spatial tile is accounted for when calculating optimization ratio
        //   (optimizatin no longer based on a single cond-on scalar)
        if (!loop::IsSpatial(state.complete_subnests_[l][loop_id + 1].spacetime_dimension))
        {
          found_target_dspace_loop = true;
          if (!loop::IsSpatial(state.complete_subnests_[l][loop_id].spacetime_dimension))
          {
            // std::cout << "inner most loop is temporal, explicitly construct scalar mold, later we should not consider co-iteration factor: "
            //   << state.complete_subnests_[l][loop_id] << std::endl;
            construct_scalar_mold = true;
          }
        }
        else if (target_dspace_dimensions.find(state.complete_subnests_[l][loop_id].dimension) != target_dspace_dimensions.end()
                 && !state.trivial_nest_masks_[l][loop_id])
        {
          found_target_dspace_loop = true;
        }
      }
 
      if (found_target_dspace_loop)
      {
        // found loop related to target dataspace
        target_loop = state.complete_subnests_[l][loop_id];
        target_loop_dim = target_loop.dimension;
        target_loop_storage_level = l;
        target_loop_id = loop_id;
        // record the mold high for the operation space associated with the *target loop*
        // (note: not *an iteration of the target loop*)
        mold_high = construct_scalar_mold ? origin : state.maxtile_molds_high_[l][loop_id];
      }
    }
  }
 
  if (!found_target_dspace_loop)
  {
    // assign the topmost (tirvial) loop as the target loop that's related to target dataspace
    target_loop = state.complete_subnests_[target_dspace_level].back();
    target_loop_dim = target_loop.dimension;
    target_loop_storage_level = target_dspace_level;
    target_loop_id = state.complete_subnests_[target_dspace_level].size()-1;
    // record the mold high for the operation space associated with the *target loop*
    // (note: not *an iteration of the target loop*)
    mold_high = construct_scalar_mold ? origin : state.maxtile_molds_high_[target_dspace_level][target_loop_id];
    found_target_dspace_loop = true;
  }
 
  // sanity check: target_loop_dim must be assigned, i.e., a loop that describes the operation space must be found
  assert(target_loop_dim != std::numeric_limits<unsigned>::max());
  assert(found_target_dspace_loop);
 
  // factor in the block size of the target dataspace storage
  //   if the conditioned on dataspace co-iteration on this dimension, the corresponding tile shape will be impacted
  //   else, this scaling will make no difference on the condition on dataspace tile shape
  // auto target_dspace_level_block_size = topology_specs.GetStorageLevel(target_dspace_level)->block_size.Get();
  auto default_target_dspace_mold_high = mold_high;

  // construct default  operation space mold for target dataspace
  // we will always be optimizing on the target dataspace tile described by this operation space
 
  problem::OperationSpace default_target_operation_space_mold(state.workload_, origin, default_target_dspace_mold_high);
 
  // Prepare the default mold high for conditioned on dataspace
  // Go through all the loops between child level and target daspace level:
  // 1. collect all non-trivial spatial loops above target loop
  // 2. integrate the loops bounds the spatial loops to base mold high
  //
  // We need to consider spatial loops above the target loop in dependent of what type of loop the target loops is (i.e., spatial or temporal)
  // because the spatial tiles will stay stationary as we go through each iteration of the temporal target loop
  // Thus, if any spatial loop projects to the conditioned on data space, we should be looking at a larger tile
  //
  // Furthermore,
  //   if we are looking at a spatial target loop
  //   1) since there is actually no order between spatial loops, we need to look at the largest spatial tile (which is described by the topmost spatial loop)
  //   2) we also need to look at all the temporal loops until we find a temporal loop that projects to target dataspace, because the spatial target
  //      tile needs to stay stationary across the iterations of these temporal loops as well 
  //   (if innermost level, then the inner most temp loop as there is no order between loops)
  //   Thus, the aggregated operation space is what we will be using for cond on dataspaces

  bool is_spatial = target_loop.spacetime_dimension != spacetime::Dimension::Time;
  std::vector<loop::Descriptor> relevant_loops = {target_loop};
  problem::OperationPoint default_cond_on_mold_high;

  default_cond_on_mold_high = mold_high;

  default_cond_on_mold_high.IncrementAllDimensions();
  bool locate_immediate_upper_target_tmp_loop = is_spatial ? false : true;
  for (unsigned l = target_loop_storage_level; l <= target_dspace_level; l++)
  {
    unsigned innermost_idx = l == target_loop_storage_level ? target_loop_id + 1 : 0;
    for (unsigned loop_id = innermost_idx; loop_id < state.complete_subnests_[l].size(); loop_id++)
    {
      if (state.trivial_nest_masks_[l][loop_id]) continue;
      auto uloop = state.complete_subnests_[l][loop_id];
      bool uloop_spatial = loop::IsSpatial(uloop.spacetime_dimension);
      bool non_target_loop_above_spatial_target_loop = !locate_immediate_upper_target_tmp_loop && 
            !uloop_spatial && (child_level != -1 && target_dspace_dimensions.find(uloop.dimension) == target_dspace_dimensions.end());
      if (uloop_spatial || non_target_loop_above_spatial_target_loop)
      {
        // std::cout << "integrate spatial loop into default condition on dataspace: " << state.complete_subnests_[l][loop_id] << std::endl;
        default_cond_on_mold_high[uloop.dimension] *= ((uloop.end - uloop.start)/uloop.stride);
        relevant_loops.push_back(uloop);
      }
      
      if (is_spatial && !loop::IsSpatial(uloop.spacetime_dimension)
          && (child_level == -1 || target_dspace_dimensions.find(uloop.dimension) != target_dspace_dimensions.end()))
      {
        // std::cout << "fond nearest target temporal loop: " << uloop << std::endl;
        locate_immediate_upper_target_tmp_loop = true;
      }
    }
  }
  default_cond_on_mold_high.IncrementAllDimensions(-1);
  
  //
  // Ineffectual read probability calculations
  //

  // go through each conditioned on dataspace id to get the probability of optimized away reads
  double prob_target_dspace_effectual = 1.0;
  for (unsigned i = 0; i < resulted_impact.condition_on_dspace_ids.size(); i++)
  {
    DataSpaceID condition_on_dspace_id = resulted_impact.condition_on_dspace_ids[i];
    auto co_iterated_dimensions = state.workload_->GetShape()->GetCoIteratedDimensions({target_dspace_id, condition_on_dspace_id});
    
   
    // Prepare the relevant point sets 
    std::uint64_t condition_on_granularity, target_optimization_granularity;  // the shape of cond-on and target tile 
    auto target_dspace_mold_high = default_target_dspace_mold_high;
    auto cond_on_mold_high = default_cond_on_mold_high;
    
    // Generate per dimension coiteration factors 
    bool ineffective_optimization = false;
    bool is_co_iterated_dim = false;
    std::map<problem::Shape::FlattenedDimensionID, double> co_iteration_factors;
    double aggregated_co_iteration_factor = 1.0;
    for (auto rloop = relevant_loops.begin(); !construct_scalar_mold && rloop != relevant_loops.end(); rloop++) 
    {
      problem::Shape::FlattenedDimensionID dim = rloop->dimension;
      if(is_spatial && is_co_iterated_dim && 
         state.workload_->GetShape()->DataSpaceIDToDimensionIDVector.at(condition_on_dspace_id).find(dim) 
         != state.workload_->GetShape()->DataSpaceIDToDimensionIDVector.at(condition_on_dspace_id).end())
      {
        // if there are loops above the co-iterated spatial loop that projects to "conditioned on" dataspace,
        // then the optimization is invalid as discordant traversal on conditioned on dataspace is required to perform
        // the desired optimization
        ineffective_optimization = true;
        break;
      }
      if (co_iterated_dimensions.find(dim) != co_iterated_dimensions.end())
      {
        // an intermediate co-iterated dimension found 
        // -> update co-iteration factor, so that a smaller conditioned on tile is examined
        // -> update record, so that any upper loop projecting to cond-on dataspace will result in ineffective optimizations
        double co_iteration_factor = (rloop->end - rloop->start)/rloop->stride;
        co_iteration_factors[dim] = co_iteration_factors.find(dim) != co_iteration_factors.end() ?
                                    co_iteration_factors[dim] * co_iteration_factor : co_iteration_factor;
        aggregated_co_iteration_factor *= co_iteration_factor;
        is_co_iterated_dim = true;
      }
    }
    
    if (ineffective_optimization) 
    {
      // FIXME: effectuive if the ranks that matter are flattened into one
      //  std::cout << "target dspace: " << state.workload_->GetShape()->DataSpaceIDToName.at(resulted_impact.target_dspace_id)
      //     << " condition on dspace: " << state.workload_->GetShape()->DataSpaceIDToName.at(condition_on_dspace_id)
      //     << " \ntarget loop: " << target_loop << std::endl;
      //  std::cout << "!!!found spatial dimension projecting to conditioned on dataspace above the co-iterated spatial dimension," << 
      //     "ineffectual optimization" << std::endl;
      continue;
    }
   
    // Parepare to scale
    target_dspace_mold_high.IncrementAllDimensions();
    cond_on_mold_high.IncrementAllDimensions();
    
    // Without considering the co-iteration factos, compare the default granularity to block size
    target_optimization_granularity = default_target_operation_space_mold.GetSize(target_dspace_id);
    auto target_dspace_level_block_size = topology_specs.GetStorageLevel(target_dspace_level)->block_size.Get();
   
    double scaling_needed_to_fit_block_size = ceil((double)target_dspace_level_block_size/
        (target_optimization_granularity/aggregated_co_iteration_factor)); 
    
    // Apply the co-iteration factors (offsetting block size if necessary)
    for (auto iter = co_iteration_factors.begin(); iter != co_iteration_factors.end(); iter++)
    {
      double effective_scaling = iter->second/scaling_needed_to_fit_block_size;
      target_dspace_mold_high[iter->first] = ceil(target_dspace_mold_high[iter->first]/effective_scaling);
      cond_on_mold_high[iter->first] = ceil(cond_on_mold_high[iter->first]/effective_scaling);
      scaling_needed_to_fit_block_size = scaling_needed_to_fit_block_size/iter->second;
    }
    
    // Scale target dim loop if the block size is still not met
    if (scaling_needed_to_fit_block_size > 1)
    {
      cond_on_mold_high[target_loop_dim] *= scaling_needed_to_fit_block_size;
      target_dspace_mold_high[target_loop_dim] *= scaling_needed_to_fit_block_size;
    }   

    // Properly adjust the point set bounds
    target_dspace_mold_high.IncrementAllDimensions(-1);
    cond_on_mold_high.IncrementAllDimensions(-1);

    // Construct the appropriate operation spaces for taget and conditioned on dataspaces
    problem::OperationSpace target_operation_space_mold(state.workload_, origin, target_dspace_mold_high);
    problem::OperationSpace cond_on_operation_space_mold(state.workload_, origin, cond_on_mold_high);

    // Construct the corresponding coordinate space tile for conditioned on and calculate the prob of the tile being empty
    target_optimization_granularity = target_operation_space_mold.GetSize(target_dspace_id);
    condition_on_granularity = cond_on_operation_space_mold.GetSize(condition_on_dspace_id);
    
    tiling::CoordinateSpaceTileInfo cspace_tile;
    cspace_tile.Set(condition_on_granularity, condition_on_dspace_id);
    cspace_tile.SetMold(cond_on_operation_space_mold.GetDataSpace(condition_on_dspace_id));
    
    // Compute the probability of the conditioned on tile being empty
    double prob_condition_on_dspace_empty = state.workload_->GetDensity(condition_on_dspace_id)
      ->GetTileOccupancyProbability(cspace_tile, 0);
    prob_target_dspace_effectual *= (1 - prob_condition_on_dspace_empty);

    // std::cout << " \n\n target dspace: " << state.workload_->GetShape()->DataSpaceIDToName.at(resulted_impact.target_dspace_id)
    //          << " condition on dspace: " << state.workload_->GetShape()->DataSpaceIDToName.at(condition_on_dspace_id)
    //          << " \n target loop: " << target_loop
    //          << " co-iterated: " << is_co_iterated_dim
    //          << " child level: " << child_level
    //          << "\n operation space mold derived dataspace sizes (target, conditioned): "
    //          << target_optimization_granularity << "  "
    //          << condition_on_granularity
    //          << "  condition on mold: " << cspace_tile.GetPointSetRepr()
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
  auto ratio = level_specs->default_md_word_bits.Get()/level_specs->word_bits.Get();
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
                                           topology_specs, possible_impact[i], eval_status);
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
    auto& impact = possible_impact[i];

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

  // std::cout << "\t=== Final: target dspace: " << state.workload_->GetShape()->DataSpaceIDToName.at(target_dspace_id)
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

void InitializeSpatialInstances(SparseAnalysisState& state,
                                tiling::CompoundTileNest& compound_tile_nest,
                                const model::Topology::Specs& topology_specs)
{
  (void) topology_specs;

  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto& compute_info = compound_tile_nest.compute_info_nest[0];

  // Initialize spatial instances
  problem::PerDataSpace<std::vector<double>> avg_effective_expansion_ratio;
  problem::PerDataSpace<std::vector<SpatialExpansion>> max_spatial_expansion;
  problem::PerDataSpace<std::vector<bool>> spatial_reduction_masks;
  
  std::uint16_t cur_architecture_level;
  
  for (unsigned pv = 0; pv < state.workload_->GetShape()->NumDataSpaces; pv++)
  {
    avg_effective_expansion_ratio[pv].emplace_back(1.0);

    SpatialExpansion max_compute_expansion;
    max_compute_expansion.X = compute_info.max_x_expansion;
    max_compute_expansion.Y = compute_info.max_y_expansion;
    max_compute_expansion.XY = max_compute_expansion.X * max_compute_expansion.Y;
    max_spatial_expansion[pv].emplace_back(max_compute_expansion);
    spatial_reduction_masks[pv].emplace_back(false);
  
    cur_architecture_level = 0; // compute is always the inner most level in the architecture 
    
    // each level's number of required instances is impacted by the set of par-fors from the level above,
    // as a result, we first look at the inner most storage level to determine the number of compute instances needed,
    // and move upward
    for (unsigned upper_storage_id = 0; upper_storage_id < state.num_storage_levels_; upper_storage_id++)
    {
      
      auto per_level_loop_nests = state.complete_subnests_[upper_storage_id];
      unsigned loop_id = -1;
      unsigned level_dspace_in = upper_storage_id;

      // find the first upper level that stores the target dataspace
      while(compound_data_movement_nest[pv][level_dspace_in].shape == 0 && level_dspace_in < state.num_storage_levels_-1)
      {
        level_dspace_in++; 
      }
      
      // if we have some way to identify the zeros in this upper storage, we can skip some spatial instances
      if (compound_data_movement_nest[pv][level_dspace_in].has_metadata)
      {
        bool inner_most_level = true; 
        double effective_ratio = 1.0;
        
        for (auto loop = per_level_loop_nests.begin(); loop != per_level_loop_nests.end(); loop++)
        {
          loop_id++;
          if (loop->spacetime_dimension == spacetime::Dimension::SpaceX 
            || loop->spacetime_dimension == spacetime::Dimension::SpaceY)
          {
            std::uint64_t fiber_shape = (loop->end - loop->start)/loop->stride;
            if (fiber_shape > 1)
            {              
              problem::PerDataSpace<double> required_spatial_instances;
              auto& cond_on_dims = state.workload_->GetShape()->DataSpaceIDToDimensionIDVector.at(pv);
              if (cond_on_dims.find(loop->dimension) != cond_on_dims.end())
              {
                problem::OperationPoint mold_high;
                if (upper_storage_id == 0 && loop_id == 0)
                {
                  problem::OperationPoint scalar_high;
                  mold_high = scalar_high;
                }    
                else if (loop_id == 0)
                {
                  mold_high = state.maxtile_molds_high_[upper_storage_id-1].back();
                }
                else
                {
                  mold_high = state.maxtile_molds_high_[upper_storage_id][loop_id - 1];
                }

                problem::OperationPoint origin;
                problem::OperationSpace subtile_mold(state.workload_, origin, mold_high);
                tiling::CoordinateSpaceTileInfo ctile_info;
                ctile_info.Set(subtile_mold.GetDataSpace(pv), pv);
                
                if (inner_most_level)
                {
                  double prob_empty_coord = compound_data_movement_nest[pv][level_dspace_in].tile_density->GetTileOccupancyProbability(ctile_info, 0);
                  effective_ratio = (1.0 - prob_empty_coord);
                  inner_most_level = false;
                  // std::cout <<"loop: " << *loop <<  " shape: " << ctile_info.GetShape() << " opt prob: " << prob_empty_coord << "  effective ratio: " << effective_ratio << std::endl;
                }
                
                problem::OperationPoint fiber_mold_high = mold_high;
                fiber_mold_high.IncrementAllDimensions();
                fiber_mold_high[loop->dimension] *= fiber_shape;
                fiber_mold_high.IncrementAllDimensions(-1);
                problem::OperationSpace fiber_tile_mold(state.workload_, origin, fiber_mold_high);
                tiling::CoordinateSpaceTileInfo fiber_ctile_info;
                fiber_ctile_info.Set(fiber_tile_mold.GetDataSpace(pv), pv);

                // get max number of elements in this fiber
                std::uint64_t num_elements = compound_data_movement_nest[pv][level_dspace_in].tile_density
                  ->GetMaxNumElementByConfidence(fiber_ctile_info, ctile_info);
                double ratio = (double)num_elements/fiber_shape;

                if (loop->spacetime_dimension == spacetime::Dimension::SpaceX)
                  max_spatial_expansion[pv][cur_architecture_level].X *= ratio;
                else
                  max_spatial_expansion[pv][cur_architecture_level].Y *= ratio;
                
                // std::cout << "-->loop: " << *loop << " max num elements: " <<  num_elements << "  ratio: " << ratio
                //   << " max X: " << max_spatial_expansion[pv][cur_architecture_level].X 
                //   << " max Y: " << max_spatial_expansion[pv][cur_architecture_level].Y
                //   << " arch level: " << cur_architecture_level
                //   << std::endl;
              }
            } // if fiber non-trivial
          } // if spatial loop
        } // for each loop
        avg_effective_expansion_ratio[pv][cur_architecture_level] = effective_ratio;
        if (effective_ratio != 1.0)
          spatial_reduction_masks[pv][cur_architecture_level] = true;
      } // if needs more processing
       
      // prepare for the next level: increment arch level, and push in the original number of instances of current upper storage level
      // note that the top most storage level never gets spatial skip saf
      cur_architecture_level++;
      avg_effective_expansion_ratio[pv].emplace_back(1.0);
    
      SpatialExpansion max_storage_expansion;
      max_storage_expansion.X = compound_data_movement_nest[pv][upper_storage_id].max_x_expansion;
      max_storage_expansion.Y = compound_data_movement_nest[pv][upper_storage_id].max_y_expansion;
      max_storage_expansion.XY = max_storage_expansion.X * max_storage_expansion.Y;
      max_spatial_expansion[pv].emplace_back(max_storage_expansion);
      spatial_reduction_masks[pv].emplace_back(false);
    }
  }
  state.avg_effective_expansion_ratio_ = avg_effective_expansion_ratio;
  state.max_spatial_expansion_ = max_spatial_expansion;
  
  // propagte expansion and avg ratio 
  for (unsigned pv = 0; pv < state.workload_->GetShape()->NumDataSpaces; pv++)
  {
    for (int architecture_level = state.num_storage_levels_; architecture_level >= 0; architecture_level-- )
    {
      if (architecture_level != 0)
      {
        double x_reduction_ratio = (double)state.max_spatial_expansion_[pv][architecture_level].X/compound_data_movement_nest[pv][architecture_level-1].max_x_expansion;
        double y_reduction_ratio = (double)state.max_spatial_expansion_[pv][architecture_level].Y/compound_data_movement_nest[pv][architecture_level-1].max_y_expansion;
        if (x_reduction_ratio != 1.0 || y_reduction_ratio != 1.0)
        {
          compound_data_movement_nest[pv][architecture_level-1].max_x_expansion = state.max_spatial_expansion_[pv][architecture_level].X ;
          compound_data_movement_nest[pv][architecture_level-1].max_y_expansion = state.max_spatial_expansion_[pv][architecture_level].Y ;
          compound_data_movement_nest[pv][architecture_level-1].avg_replication_factor 
            = compound_data_movement_nest[pv][architecture_level-1].replication_factor * state.avg_effective_expansion_ratio_[pv][architecture_level-1]; 
          bool cross_boundary = false;
          for (int inner_level = architecture_level - 1; inner_level >= 0; inner_level--)
          {
            state.max_spatial_expansion_[pv][inner_level].X *= x_reduction_ratio;
            state.max_spatial_expansion_[pv][inner_level].Y *= y_reduction_ratio;
            if (spatial_reduction_masks[pv][inner_level]) cross_boundary = true; 
            if (!cross_boundary)
            {
              //std::cout << "inner level: " << inner_level << std::endl;
              state.avg_effective_expansion_ratio_[pv][inner_level] *= state.avg_effective_expansion_ratio_[pv][architecture_level];
              //std::cout << state.avg_effective_expansion_ratio_[pv][inner_level] << std::endl;
            }
          }
        }
      }
      else
      {
        compute_info.max_x_expansion = state.max_spatial_expansion_[pv][architecture_level].X;
        compute_info.max_y_expansion = state.max_spatial_expansion_[pv][architecture_level].Y;
        compute_info.avg_replication_factor = state.avg_effective_expansion_ratio_[pv][architecture_level];
      }
      
      // Sanity Check
      // if (architecture_level == int(state.num_storage_levels_))
      //    std::cout << "Dataspace: " << state.workload_->GetShape()->DataSpaceIDToName.at(pv) << std::endl;
      // std::cout << "  Initialize: " 
      // << topology_specs.GetLevel(architecture_level)->level_name  
      // << "  avg effective ratio: " <<  avg_effective_expansion_ratio[pv][architecture_level]
      // << "  max x expansion: " << state.max_spatial_expansion_[pv][architecture_level].X 
      // << "  max y expansion:"  <<  state.max_spatial_expansion_[pv][architecture_level].Y
      // << std::endl;
    }
  }
}



void CalculateSpatialOptimizationImpact(SparseAnalysisState& state,
                                        const tiling::CompoundTileNest& compound_tile_nest,
                                        ExplicitReadOptimizationImpact& resulted_impact,
                                        const std::uint64_t upper_storage_level,
                                        const model::Topology::Specs& topology_specs)
{
   
  (void) topology_specs;

  auto compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto compute_info = compound_tile_nest.compute_info_nest[0];

  DataSpaceID target_dspace_id = resulted_impact.target_dspace_id;
  auto condition_on_dspace_ids = resulted_impact.condition_on_dspace_ids;
  
  // find the child level that stores the target dataspace
  // if child level is compute, then there is no stationarity of the spatial tiles
  // else spatial target tile stay stationary in child level until a temporal upper target loop is met
  int child_level;
  for (child_level = int(upper_storage_level) - 1; child_level >= 0; child_level--)
    if (compound_data_movement_nest[target_dspace_id][child_level].shape > 0) break;
 
  // go through the loops in this storage level
  // note that spatial skipping is only applied to the spatial instances below this level, it is irrelevant to 
  // whether this storage level stores the target dataspace or not
  auto per_level_loop_nests = state.complete_subnests_[upper_storage_level];
  auto& target_dims = state.workload_->GetShape()->DataSpaceIDToDimensionIDVector.at(target_dspace_id);
  unsigned loop_id = -1;

  std::map<problem::Shape::DataSpaceID, double> per_cond_on_effective_ratio;
  auto architecture_level = upper_storage_level; 
  
  for (unsigned pvi = 0; pvi < condition_on_dspace_ids.size(); pvi++)
  {
    auto condition_on_dspace_id = condition_on_dspace_ids[pvi];
    auto& cond_on_dims = state.workload_->GetShape()->DataSpaceIDToDimensionIDVector.at(condition_on_dspace_id);
    
    // find the level that stores the conditioned on dataspace
    unsigned level_cond_on_dspace_in = upper_storage_level;
    while(compound_data_movement_nest[condition_on_dspace_id][level_cond_on_dspace_in].shape == 0
        && level_cond_on_dspace_in < state.num_storage_levels_)
    {
      level_cond_on_dspace_in++; 
    }   
    
    if (level_cond_on_dspace_in == state.num_storage_levels_ 
        || !compound_data_movement_nest[condition_on_dspace_id][level_cond_on_dspace_in].has_metadata)
    {
      // if there is no upper level that stores the conditioned on tile, 
      // or if there is no metadata information for us to identify the sparsity of the cond on dspace
      // then the optimization is ineffective
      per_cond_on_effective_ratio[condition_on_dspace_id] = 1.0;
      continue;
    }   
    
    bool innermost_captured = false;
    
    for (auto loop = per_level_loop_nests.begin(); loop != per_level_loop_nests.end(); loop++)
    {
      loop_id++;
      
      if (loop::IsSpatial(loop->spacetime_dimension))
      {
        std::uint64_t fiber_shape = (loop->end - loop->start)/loop->stride;
        if (fiber_shape > 1)
        {
          // only if the dimension is relevant to the conditioned on dataspace do we proceed to analyze the impact
          // otherwise the optimization is ineffective
          if (cond_on_dims.find(loop->dimension) != cond_on_dims.end())
          {
            
            problem::OperationPoint mold_high;
            if (upper_storage_level == 0 && loop_id == 0)
            {
              problem::OperationPoint scalar_high;
              mold_high = scalar_high;
            }    
            else if (loop_id == 0)
            {
              mold_high = state.maxtile_molds_high_[upper_storage_level-1].back();
            }
            else
            {
              mold_high = state.maxtile_molds_high_[upper_storage_level][loop_id - 1];
            }
            
            // only if the loop is a co-iterated loop, do we consider the stationarity of spatial loops relative to the temporal loops above it
            // the target dataspace tile described by this spatial loop needs to be stationary 
            // until next upper temp loop that projects to target dataspace is encountered
            // go through all the temp loops above the current loop to find the appropriate operation space 
            // (note that inner most level (child level = -1) is never stationary, so skip this analysis for innet most level) 
            auto co_iterated_dimensions = state.workload_->GetShape()->GetCoIteratedDimensions({target_dspace_id, condition_on_dspace_id});
            if (child_level != -1 && co_iterated_dimensions.find(loop->dimension) != co_iterated_dimensions.end())
            {
              unsigned target_dspace_storage_level = upper_storage_level;
              bool target_dspace_in_level = compound_data_movement_nest[target_dspace_id][target_dspace_storage_level].shape == 0 ? false: true;
              while(!target_dspace_in_level && target_dspace_storage_level < state.num_storage_levels_ - 1)
              {
                target_dspace_storage_level++;
                target_dspace_in_level = compound_data_movement_nest[target_dspace_id][target_dspace_storage_level].shape == 0 ? false: true;
              }
              
              bool found_target_temp_loop = false;
              for(unsigned l = upper_storage_level; l <= target_dspace_storage_level && !found_target_temp_loop; l++)
              {
                for (auto uloop = state.complete_subnests_[l].begin(); 
                     uloop != state.complete_subnests_[l].end() && !found_target_temp_loop; uloop++)
                {
                  std::uint64_t uloop_bound = (uloop->end - uloop->start)/uloop->stride; 
                  if (uloop->spacetime_dimension == spacetime::Dimension::Time && uloop_bound > 1)
                  {
                    if (target_dims.find(uloop->dimension) != target_dims.end()) found_target_temp_loop = true;
                    else if (cond_on_dims.find(loop->dimension) != cond_on_dims.end())
                    {
                      // the same spatial target tile is stationary while iterating through the temporal loops not projecting to target dataspace
                      // take this loop into account for mold high representation as it might impact the cond on dspace tile we will be looking at
                      // for this optimization
                      mold_high.IncrementAllDimensions();
                      unsigned scaled_dim_bound = mold_high[uloop->dimension] * uloop_bound ;
                      mold_high[uloop->dimension] = scaled_dim_bound;
                      mold_high.IncrementAllDimensions(-1);
                      // std::cout << " found temp conditioned on loop above spatial reduction loop: " << *uloop << " scale space: " << uloop_bound << std::endl;
                    }
                    else
                    { 
                      // irrelevant loop, pass 
                    }
                  }
                }
              }
            } 
            // construct the tile processed by each spatial instance
            problem::OperationPoint origin;
            problem::OperationSpace subtile_mold(state.workload_, origin, mold_high);
            tiling::CoordinateSpaceTileInfo ctile_info;
            ctile_info.Set(subtile_mold.GetDataSpace(condition_on_dspace_id), condition_on_dspace_id);
              
            // get the probability of the tile being empty, i.e., the probability of not needing a spatial instance to process the tile
            if (!innermost_captured)
            {
              double prob_empty_coord = compound_data_movement_nest[condition_on_dspace_id][level_cond_on_dspace_in].tile_density->GetTileOccupancyProbability(ctile_info, 0);

              per_cond_on_effective_ratio[condition_on_dspace_id] = 1 - prob_empty_coord;
              
              // std::cout << "conditioned on dataspace: " << state.workload_->GetShape()->DataSpaceIDToName.at(condition_on_dspace_id)
              // << "  innermost relevant spatial loop: " << pstate.workload_->GetShape()->DimensionIDToName.at(loop->dimension) 
              // << "  effective ratio:  " << 1 - prob_empty_coord << std::endl;
              
              // the innermost spatial loop detemrines the needed spatial instances, break out of the looping  
              innermost_captured = true;
            }
            
            // construct a fiber tile
            problem::OperationPoint fiber_mold_high = mold_high;
            fiber_mold_high.IncrementAllDimensions();
            fiber_mold_high[loop->dimension] *= fiber_shape;
            fiber_mold_high.IncrementAllDimensions(-1);
            problem::OperationSpace fiber_tile_mold(state.workload_, origin, fiber_mold_high);
            tiling::CoordinateSpaceTileInfo fiber_ctile_info;
            fiber_ctile_info.Set(fiber_tile_mold.GetDataSpace(condition_on_dspace_id), condition_on_dspace_id);
            
            // get max number of elements in this fiber
            std::uint64_t num_elements = compound_data_movement_nest[condition_on_dspace_id][level_cond_on_dspace_in].tile_density
              ->GetMaxNumElementByConfidence(fiber_ctile_info, ctile_info);
            
            double ratio = (double)num_elements/fiber_shape;
            if (loop->spacetime_dimension == spacetime::Dimension::SpaceX)
              state.max_spatial_expansion_[target_dspace_id][architecture_level].X *= ratio;
            else
              state.max_spatial_expansion_[target_dspace_id][architecture_level].Y *= ratio;

          } // if dimension does map to cond on dataspace 
        } //if fiber non-trivial
      } // if spatial loop
    } // for each loop
  } // for each conditioned on dpsace
  
  // mulitplicative effect for all conditioned on dataspaces
  double aggregated_effective_ratio = 1.0;
  for (auto iter = per_cond_on_effective_ratio.begin(); iter != per_cond_on_effective_ratio.end(); iter++)
  {
    aggregated_effective_ratio *= iter->second;
  }
  
  state.avg_effective_expansion_ratio_[target_dspace_id][architecture_level] = aggregated_effective_ratio;
  resulted_impact.optimization_prob =  1 - aggregated_effective_ratio;
  state.max_spatial_expansion_[target_dspace_id][architecture_level].XY = state.max_spatial_expansion_[target_dspace_id][architecture_level].X *
                                                                          state.max_spatial_expansion_[target_dspace_id][architecture_level].Y;
  std::string arch_level_name = architecture_level != 0 ? topology_specs.GetStorageLevel(architecture_level- 1)->level_name : "compute";

  // std::cout << "architecture level: " << arch_level_name << "(" << architecture_level << ")"
  // << "  aggregated spatial ratio: " << aggregated_effective_ratio 
  // << "  XY total instances: " << state.max_spatial_expansion_[target_dspace_id][architecture_level].X 
  //                                * state.max_spatial_expansion_[target_dspace_id][architecture_level].Y
  // << "  avg effective ratio: " <<  state.avg_effective_expansion_ratio_[target_dspace_id][architecture_level]<< std::endl;

}

void DefineSpatialOptimizationImpact(SparseAnalysisState& state,
                                     const tiling::CompoundTileNest& compound_tile_nest,
                                     const unsigned storage_level_id,
                                     const GroupOfActionOptimization& group_of_action_optimization,
                                     const model::Topology::Specs& topology_specs,
                                     std::string optimization_type)
{


  // FIXME: consider options
  assert(group_of_action_optimization.size() == 1);

  std::vector <ExplicitReadOptimizationImpact> possible_impact(group_of_action_optimization.size());
  for (unsigned i = 0; i < group_of_action_optimization.size(); i++)
  {

    //
    // compute the optimization impact of the various choices
    //
    assert(group_of_action_optimization[i].type == CONDITIONED_ON);
    possible_impact[i].target_dspace_id = group_of_action_optimization[i].cond_on_opt.target_dspace_id;
    possible_impact[i].condition_on_dspace_ids = group_of_action_optimization[i].cond_on_opt.condition_on_dspace_ids;
    CalculateSpatialOptimizationImpact(state, compound_tile_nest, possible_impact[i], storage_level_id, topology_specs);
  }

  //double optimization_prob = possible_impact[0].optimization_prob;
  auto target_dspace_id = possible_impact[0].target_dspace_id;
  state.dspace_optimization_masks_[optimization_type][storage_level_id][target_dspace_id] = true;
}


bool SummarizeAndPropagateSpatialCapacityReduction(SparseAnalysisState& state,
                                                   tiling::CompoundTileNest& compound_tile_nest,
                                                   const model::Topology::Specs& topology_specs,
                                                   std::vector <model::EvalStatus>& eval_status)
{

  bool success = true;

  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto& compute_info = compound_tile_nest.compute_info_nest[0];
  
  // Summarize storage effective instances and propagate to inner levels
  //    level 0: compute   level 1: inner most storage etc.  
  // as a result, we are summarizing the spatial instances of the storage levels in this loop
  for (unsigned pv = 0; pv < state.workload_->GetShape()->NumDataSpaces; pv++)
  { 
    
    std::vector<SpatialExpansion> orig_spatial_expansion = {};
    for (unsigned storage_level = 0; storage_level < state.num_storage_levels_ ; storage_level++)
    {
      SpatialExpansion se;
      se.X = compound_data_movement_nest[pv][storage_level].max_x_expansion;
      se.Y = compound_data_movement_nest[pv][storage_level].max_y_expansion;
      se.XY = se.X * se.Y;
      orig_spatial_expansion.emplace_back(se);
    }
      
    for (int architecture_level = state.num_storage_levels_; architecture_level >= 0; architecture_level--)
    {
      auto pv_max_x = state.max_spatial_expansion_[pv][architecture_level].X;
      auto pv_max_y = state.max_spatial_expansion_[pv][architecture_level].Y;
      auto pv_avg_ratio = state.avg_effective_expansion_ratio_[pv][architecture_level];
      double y_reduction_ratio = 1.0;
      double x_reduction_ratio = 1.0;
      double pv_avg_replication_factor = 0; 
      
      if (architecture_level != 0)
      {
        // impact of spatial skip is manifested as the reduced number of spatial instances in the lower level
        auto storage_level = architecture_level - 1;
        auto& data_movement_info = compound_data_movement_nest[pv][storage_level];
        pv_avg_replication_factor = data_movement_info.replication_factor * pv_avg_ratio;

        data_movement_info.max_x_expansion = pv_max_x;
        data_movement_info.max_y_expansion = pv_max_y;
        data_movement_info.max_replication_factor = pv_max_x * pv_max_y;
        data_movement_info.avg_replication_factor = (double)data_movement_info.replication_factor * pv_avg_ratio;

        y_reduction_ratio = (double)pv_max_y/orig_spatial_expansion[storage_level].Y;
        x_reduction_ratio = (double)pv_max_x/orig_spatial_expansion[storage_level].X;
      } 
      else
      {
        compute_info.max_x_expansion = pv == 0 ? 
          pv_max_x : (compute_info.max_x_expansion < pv_max_x ? pv_max_x : compute_info.max_x_expansion); 
        compute_info.max_y_expansion = pv == 0 ? 
          pv_max_y : (compute_info.max_y_expansion < pv_max_y ? pv_max_y : compute_info.max_y_expansion); 
        pv_avg_replication_factor =  compute_info.replication_factor * pv_avg_ratio;
        compute_info.avg_replication_factor = pv == 0 ? pv_avg_replication_factor 
           : (compute_info.avg_replication_factor < pv_avg_replication_factor ? pv_avg_replication_factor : compute_info.avg_replication_factor); 
      }
      
      // The optimized away ratio is recorded as an attribute of the level above
      if ((unsigned)architecture_level < state.num_storage_levels_)
        state.prob_explicitly_spatially_optimized_read_[architecture_level][pv] =  1- pv_avg_ratio; 
      
      // Sanity check 
      // if (architecture_level == int(state.num_storage_levels_))
      //   std::cout << "Dataspace: " << state.workload_->GetShape()->DataSpaceIDToName.at(pv) << std::endl;
      // std::cout << "  Final: " << topology_specs.GetLevel(architecture_level)->level_name 
      //   << " avg effective rep factor: " << pv_avg_replication_factor 
      //   << " max X expansion: " << pv_max_x
      //   << " max Y expansion: " << pv_max_y
      //   << " max rep factor : " << pv_max_x * pv_max_y
      //   << " opt prob: " << 1 - pv_avg_ratio
      //   << " architecture level: " << architecture_level 
      //   << " y-reduction ratio: " << y_reduction_ratio 
      //   << " x-reduction ratio: " << x_reduction_ratio
      //   << std::endl;

      // We should not propagate to levels below next spatial skipping
      // since the probability of being optimized we get is the absolute probability for a certain tile
      // irrelevant to whether the upper level has any optimization
      if (x_reduction_ratio != 1 || y_reduction_ratio != 1)
      {
        bool cross_boundary = false;
        for (int inner_level = architecture_level - 1; inner_level >= 0; inner_level--)
        {
          // offset exactly by 1, e.g., arch level 0 (compute)'s upper storage level is storage level 0
          int upper_storage_level = inner_level; 
          if (state.dspace_optimization_masks_["spatial_skip"][upper_storage_level][pv])
            cross_boundary = true;
          
          if (!cross_boundary)
          {
            state.avg_effective_expansion_ratio_[pv][inner_level] = state.avg_effective_expansion_ratio_[pv][inner_level] * pv_avg_ratio;
          } 
          // std::cout << "      propagate expansion reduction to inner arch level: " << inner_level << std::endl;
          state.max_spatial_expansion_[pv][inner_level].X = (double)state.max_spatial_expansion_[pv][inner_level].X * x_reduction_ratio;
          state.max_spatial_expansion_[pv][inner_level].Y = (double)state.max_spatial_expansion_[pv][inner_level].Y * y_reduction_ratio;
        } // for each inner level
      }
    } // for each level
  } // for each datspace
  
  std::ostringstream fail_reason;
  

  for (unsigned pv = 0; pv < state.workload_->GetShape()->NumDataSpaces; pv++)
  {
  
    // after propagation, summarize the final amount of skipping at each level
    for (int architecture_level = state.num_storage_levels_; architecture_level >= 0; architecture_level--)
    {
      bool topmost_level = architecture_level == int(topology_specs.NumLevels() - 1); 
      
      // Check Fanout
      if (!topmost_level)
      {
        // std::uint64_t fanoutX =  state.max_spatial_expansion_[pv][architecture_level].X/topology_specs.GetStorageLevel(architecture_level)->meshX.Get();
        // std::uint64_t fanoutY =  state.max_spatial_expansion_[pv][architecture_level].Y/topology_specs.GetStorageLevel(architecture_level)->meshY.Get();       
        
        std::uint64_t fanoutX =  state.max_spatial_expansion_[pv][architecture_level].X/state.max_spatial_expansion_[pv][architecture_level + 1].X;
        std::uint64_t fanoutY =  state.max_spatial_expansion_[pv][architecture_level].Y/state.max_spatial_expansion_[pv][architecture_level + 1].Y;
        if (fanoutX > state.sparse_optimization_info_-> max_fanoutX.at(architecture_level))
        {
          fail_reason << "Required fanoutX " << fanoutX 
            << " does not meet hardware constraint " 
            << state.sparse_optimization_info_-> max_fanoutX.at(architecture_level)         
            << std::endl;

          success = false;
          auto overall_level_id = architecture_level;
          eval_status[overall_level_id].success = false;
          eval_status[overall_level_id].fail_reason = fail_reason.str();   
        }
        
        if (fanoutY > state.sparse_optimization_info_-> max_fanoutY.at(architecture_level))
        {
          fail_reason << "Required fanoutY " << fanoutY 
            << " does not meet hardware constraint " 
            << state.sparse_optimization_info_-> max_fanoutY.at(architecture_level)   
            << std::endl;

          success = false;
          auto overall_level_id = architecture_level;
          eval_status[overall_level_id].success = false;
          eval_status[overall_level_id].fail_reason = fail_reason.str();   
        }
        // std::cout << "  required FanoutX: " << fanoutX << "   FanoutY: " << fanoutY << std::endl;
      }     
    }
  }
  
  return success;
}

bool DefineStorageOptimizationImpact(SparseAnalysisState& state,
                                     tiling::CompoundTileNest& compound_tile_nest,
                                     const model::Topology::Specs& topology_specs,
                                     std::vector <model::EvalStatus>& eval_status,
                                     const bool break_on_failure)
{

  bool success = true;
  auto action_gating_info = state.sparse_optimization_info_->action_gating_info;
  auto action_skipping_info = state.sparse_optimization_info_->action_skipping_info;
  auto action_spatial_skipping_info = state.sparse_optimization_info_->action_spatial_skipping_info;
  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;

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

  if ((success || !break_on_failure) && action_spatial_skipping_info.size() > 0)
  {
    InitializeSpatialInstances(state, compound_tile_nest, topology_specs);
    // Define how much spatial capacity can be optimized away
    for (auto level_iter = action_spatial_skipping_info.begin();
         level_iter != action_spatial_skipping_info.end(); level_iter++)
    {
      auto storage_level_id = level_iter->first;
      auto optimization_list = level_iter->second;

      for (unsigned group_id = 0; group_id < optimization_list.size(); group_id++)
      {
        GroupOfActionOptimization group = optimization_list[group_id];
        DefineSpatialOptimizationImpact(state, compound_tile_nest, storage_level_id, group, topology_specs, "spatial_skip");
      }
    }
    success = SummarizeAndPropagateSpatialCapacityReduction(state, compound_tile_nest, topology_specs, eval_status);
  }
  
  return success;
}

} // namespace
