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
#include "sparse-analysis/sparse-analysis.hpp"
#include "mapping/loop.hpp"

namespace sparse
{

void SetPointSetTileRepresentations(const SparseAnalysisState& state,
                                    tiling::CompoundDataMovementNest& compound_data_movement_nest)
{
  // empty operation space for bypassed tiles
  problem::OperationPoint origin;
  problem::OperationSpace empty_mold(state.workload_);
 
  unsigned tiling_level = 0;
  unsigned loop_offset = 0;
  auto& loops = state.mapping_.complete_loop_nest.loops;

  for (unsigned loop_level = 0; loop_level < loops.size(); loop_level++)
  {
    if (loop_level == state.mapping_.complete_loop_nest.storage_tiling_boundaries.at(tiling_level))
    {
      problem::OperationSpace operation_space_mold(state.workload_, origin,
                                                   state.maxtile_molds_high_.at(tiling_level).at(loop_level-loop_offset));
      for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
      {
        auto& tile_point_set_mold = compound_data_movement_nest.at(pv).at(tiling_level).shape != 0 ? operation_space_mold.GetDataSpace(pv) : empty_mold.GetDataSpace(pv);
        compound_data_movement_nest.at(pv).at(tiling_level).coord_space_info.SetMold(tile_point_set_mold);
        // std::cout << " point set representation of tile " << problem::GetShape()->DataSpaceIDToName.at(pv)
        //   << compound_data_movement_nest.at(pv).at(tiling_level).coord_space_info.tile_point_set_mold_ << std::endl;
      }
      tiling_level++;
      loop_offset = loop_level + 1;
    }
  }
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
      auto& fine_grained_data_accesses = compound_data_movement_nest[pv][l].fine_grained_data_accesses;
      auto& fine_grained_format_accesses = compound_data_movement_nest[pv][l].fine_grained_format_accesses;

      fine_grained_format_accesses = {};

      for (unsigned op_id = 0; op_id < tiling::storageOperationTypes.size(); op_id++)
      {
        auto op_name = tiling::storageOperationTypes[op_id];
        fine_grained_data_accesses[op_name] = 0;
        if (op_name.find("metadata") != std::string::npos)
        {
          fine_grained_format_accesses[op_name] = {};
        }
        else
        {
          fine_grained_data_accesses[op_name] = 0;
        }
      }

      // default to uncompressed without metadata
      fine_grained_data_accesses["random_read"] = compound_data_movement_nest[pv][l].reads;
      fine_grained_data_accesses["random_fill"] = compound_data_movement_nest[pv][l].fills;
      fine_grained_data_accesses["random_update"] = compound_data_movement_nest[pv][l].updates;
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

void InitializeMaxRequiredSpatialExpansion(tiling::CompoundTileNest& compound_tile_nest,
                                           const model::Topology::Specs& topology_specs,
                                           Mapping mapping)
{
  
  int tiling_level = topology_specs.NumStorageLevels() - 1;
 
  auto& loops = mapping.loop_nest.loops;
  std::uint64_t max_x_expansion = 1;
  std::uint64_t max_y_expansion = 1;
  int boundary_loop_id = mapping.loop_nest.storage_tiling_boundaries.at(tiling_level);

  // top down fashion for expansion calacultion at each storage level
  for (int loop_level = loops.size()-1; loop_level >= 0; loop_level--)
  {
    auto& loop = loops[loop_level];
    // std::cout << loop << std::endl;
 
    // boundary is the top most loop at a specific storage level,
    if (loop_level == boundary_loop_id)
    {
      // std::cout <<"tiling level: " << tiling_level << "  name: "
      //  << topology_specs.GetStorageLevel(tiling_level)->level_name << std::endl;
      std::uint64_t storage_level = tiling_level;
      for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
      {
        // By default, we require dense number of instances for sparse worloads
        auto& datamovement_record = compound_tile_nest.compound_data_movement_info_nest[pv][storage_level];
        datamovement_record.max_x_expansion = max_x_expansion;
        datamovement_record.max_y_expansion = max_y_expansion;
      }

      // Update tiling level and the boundary loop to next level
      // if this is the innermost storage level, the boundary is outside all the loops, i.e., -1
      tiling_level--;
      boundary_loop_id = tiling_level >= 0 ? mapping.loop_nest.storage_tiling_boundaries.at(tiling_level) : -1;
    }

    if (loop.spacetime_dimension != spacetime::Dimension::Time)
    {
      auto factor = ceil((loop.end - loop.start) / loop.stride);
      if (loop.spacetime_dimension == spacetime::Dimension::SpaceX)
      {
        max_x_expansion *= factor;
      }
      else
      {
        max_y_expansion *= factor;
      }
    }
  }

  // inner most tiling level is compute
  compound_tile_nest.compute_info_nest[0].max_x_expansion = max_x_expansion;
  compound_tile_nest.compute_info_nest[0].max_y_expansion = max_y_expansion;
  // std::cout << "tiling level: " << tiling_level << "   name: compute"  << std::endl;
  // std::cout << "hw mesh x: " << topology_specs.GetArithmeticLevel()->meshX
  //   << "  mesh y: " << topology_specs.GetArithmeticLevel()->meshY << std::endl;
  // std::cout <<"max x expansion: " << max_x_expansion << "  max y expansion: " << max_y_expansion << std::endl;
}
  
  
void InitializeSparsityRelatedEntries(const problem::Workload* workload,
                                      tiling::CompoundTileNest& compound_tile_nest)
{
  // Initialize all tile density models and metadata representation related entries in tile info
  //    if default dense, then the tile has a fixed density of 1.0 and empty other entries
  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  // all datatypes must have same number of tiling levels, take that of the first dataspace
  unsigned num_levels = compound_data_movement_nest[0].size();
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    for (unsigned level = 0; level < num_levels; level++)
    {
      // TODO: might want to have a new data structure for post processed sparse traffic,
      //       now we are carrying both dense and sparse in tile info
      compound_data_movement_nest[pv][level].expected_density = 1.0;
      compound_data_movement_nest[pv][level].SetDensityModel(workload->GetDensity(pv));
      compound_data_movement_nest[pv][level].expected_data_occupancy = compound_data_movement_nest[pv][level].shape;
      compound_data_movement_nest[pv][level].avg_replication_factor = compound_data_movement_nest[pv][level].replication_factor;
    }
  }

  auto& compute_info_nest = compound_tile_nest.compute_info_nest;
  auto& compute_info = compute_info_nest[0];
  compute_info.avg_replication_factor = compute_info.replication_factor;
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

  // Initialize All Entries to Default Dense 
  InitializeSparsityRelatedEntries(workload, compound_tile_nest);
  InitializeFineGrainedAccesses(compound_tile_nest, topology_specs);
  InitializeMaxRequiredSpatialExpansion(compound_tile_nest, topology_specs, mapping);
  
  // Perform quick check on if sparse anlaysis is needed
  SparseAnalysisState state(workload);
  bool sparse_analysis_needed = state.Init(sparse_optimization_info, workload, mapping, topology_specs.NumStorageLevels());
  if (!sparse_analysis_needed) return success;
 
  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  state.CollectCompletePointSetsAndSubnests();

  // Populate the point set representation for the data tiles
  SetPointSetTileRepresentations(state, compound_data_movement_nest);

  //
  // Define the necessary densities/probabilities/misc info of sparse optimizations
  //

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
  success = DefineStorageOptimizationImpact(state, compound_tile_nest, topology_specs,
                                            eval_status, break_on_failure);
  if (!success && break_on_failure) return success;
  
  // Combine the impact of storage and representation related optimizations 
  CombineStorageOptimizationImpact(state, compound_tile_nest, topology_specs);
  
  // Finalize the compute related statistics
#ifdef USE_MULTI_OPERAND  //under debug
  CalculateFineGrainedComputeAccesses(state, compound_tile_nest);
#else
  CalculateFineGrainedComputeAccesses2Operand(state, compound_tile_nest);
#endif

  return success;
}

} // namespace
