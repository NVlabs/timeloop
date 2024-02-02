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

void ApplyLocalStorageSAFImpact(tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                const double p,
                                const unsigned pv,
                                const unsigned l,
                                const std::string type)

{

  auto& data_movement_record = compound_data_movement_nest[pv][l];
  
  auto max_reads = data_movement_record.fine_grained_data_accesses["random_read"];
  auto max_updates = data_movement_record.fine_grained_data_accesses["random_update"];

  auto max_format_reads = data_movement_record.fine_grained_format_accesses["random_metadata_read"];
  auto max_format_updates = data_movement_record.fine_grained_format_accesses["random_metadata_update"];

  std::uint64_t delta_reads;
  if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    delta_reads = ceil(max_reads * p); // use ceil to account for the potential rounding differences due to the RMW setup
  else
    delta_reads = floor(max_reads * p);

  data_movement_record.fine_grained_data_accesses[type + "_read"] += delta_reads;
  max_reads -= delta_reads;

  // the optimized away reads will not lead to any updates to this level
  auto delta_updates = floor(max_updates * p);
  data_movement_record.fine_grained_data_accesses[type + "_update"] += delta_updates;
  max_updates -= delta_updates;

  // we can only optimize the portion of format data sent to child level
  std::uint64_t num_child_format_ranks;
  if (data_movement_record.child_level != std::numeric_limits<unsigned>::max())
  {
    // upon a read, only ranks associated with the child level will be sent out
    num_child_format_ranks =
      compound_data_movement_nest[pv][data_movement_record.child_level].GetNumMetaDataRanks();
  } else
  {
    // upon a read, last level storage sends all metadata
    num_child_format_ranks = data_movement_record.GetNumMetaDataRanks();
  }
  
  for (unsigned r_id = 0; r_id < num_child_format_ranks; r_id++) // for each rank
  {
    for (unsigned type_id = 0; type_id < max_format_reads[r_id].size(); type_id++) // for metadata and payload
    {
      auto delta_reads = floor(max_format_reads[r_id][type_id] * p);
      max_format_reads[r_id][type_id] -= delta_reads;
      data_movement_record.fine_grained_format_accesses[type + "_metadata_read"][r_id][type_id] += delta_reads;
      auto delta_updates = floor(max_format_updates[r_id][type_id] * p);
      max_format_updates[r_id][type_id]-= delta_updates;
      data_movement_record.fine_grained_format_accesses[type + "_metadata_update"][r_id][type_id] += delta_updates;
    }
  }
  

  // Finalize random counts -> which is just the left over max number of each type of action
  data_movement_record.fine_grained_data_accesses["random_read"] = max_reads;
  data_movement_record.fine_grained_data_accesses["random_update"] = max_updates;

  data_movement_record.fine_grained_format_accesses["random_metadata_read"] = max_format_reads;
  data_movement_record.fine_grained_format_accesses["random_metadata_update"] = max_format_updates;
  
  //
  // Temporal Reduction
  //

  // only the updates that actually happened lead to actual temporal reductions
  if (data_movement_record.size != 0 && problem::GetShape()->IsReadWriteDataSpace.at(pv))
  {
    data_movement_record.temporal_reductions = ceil(
      data_movement_record.temporal_reductions * (double)data_movement_record.fine_grained_data_accesses["random_update"]
        / data_movement_record.updates);
  }

  // Sanity print
  // std::cout << "----- after post processing " << std::endl;
  // for (auto iter = data_movement_record.fine_grained_data_accesses.begin(); iter != data_movement_record.fine_grained_data_accesses.end(); iter++)
  // {
  //   std::cout << iter->first << ": " << iter->second << std::endl;
  // }
}

void ScalePerTileFormatAccesses(tiling::PerTileFormatAccesses& per_tile_accesses, double ratio, 
                                unsigned lower_rank_id, unsigned upper_rank_id )
{
  for (unsigned id = lower_rank_id; id <= upper_rank_id; id++)
  {
    per_tile_accesses[id][0] -= floor(per_tile_accesses[id][0] * ratio);
    per_tile_accesses[id][1] -= floor(per_tile_accesses[id][1] * ratio);
  }
}

void PropagateImpactOfExplicitlyOptimizedRead(SparseAnalysisState& state,
                                              tiling::CompoundTileNest& compound_tile_nest,
                                              const model::Topology::Specs& topology_specs)
{

  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto& compute_info_nest = compound_tile_nest.compute_info_nest;
  
  std::vector<problem::PerDataSpace<double>> max_reads = {};
  std::vector<problem::PerDataSpace<double>> max_updates = {};
  std::vector<problem::PerDataSpace<double>> max_fills = {};
 
  std::vector<problem::PerDataSpace<tiling::PerTileFormatAccesses>> max_format_reads = {};
  std::vector<problem::PerDataSpace<tiling::PerTileFormatAccesses>> max_format_updates = {};
  std::vector<problem::PerDataSpace<tiling::PerTileFormatAccesses>> max_format_fills = {};

  std::vector<problem::PerDataSpace<bool>> fine_grained_action_finalized = {};
  std::vector<double> max_computes = {};

  // initialize vectors to record the maximum possible number of each type of accesses
  // storage levels
  for (unsigned l = 0; l < topology_specs.NumStorageLevels(); l++)
  {
    max_reads.push_back({});
    max_updates.push_back({});
    max_fills.push_back({});
    max_format_reads.push_back({});
    max_format_fills.push_back({});
    max_format_updates.push_back({});
    fine_grained_action_finalized.push_back({});    
    
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      // data representation impact already reflected in the fine grained access counts
      //    if the fibertree element is not even there due to compression, propagation impact is meaningless
      max_reads[l][pv] = compound_data_movement_nest[pv][l].fine_grained_data_accesses["random_read"];
      max_fills[l][pv] = compound_data_movement_nest[pv][l].fine_grained_data_accesses["random_fill"];
      max_updates[l][pv] = compound_data_movement_nest[pv][l].fine_grained_data_accesses["random_update"];

      max_format_reads[l][pv] = compound_data_movement_nest[pv][l].format_reads;
      max_format_fills[l][pv] = compound_data_movement_nest[pv][l].format_fills;
      max_format_updates[l][pv] = compound_data_movement_nest[pv][l].format_updates;

      fine_grained_action_finalized[l][pv] = false;
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
          && state.dspace_optimization_masks_.at("skip").at(l).at(pv)
          && state.dspace_optimization_masks_.at("spatial_skip").at(l).at(pv))
      {
        // not allowed to have gating and skipping applied to the same tile
        assert(false);
      }
      
      // perform processing only if this level has some saf applied
      // (levels w/o any safs are passively processed during propagation analysis)
      if (state.dspace_optimization_masks_.at("gate").at(l).at(pv)
        || state.dspace_optimization_masks_.at("skip").at(l).at(pv)
        || state.dspace_optimization_masks_.at("spatial_skip").at(l).at(pv))
      {
        
        double p = 0.0;
        if (state.dspace_optimization_masks_.at("gate").at(l).at(pv)
        || state.dspace_optimization_masks_.at("skip").at(l).at(pv))
        {
          p = state.prob_explicitly_optimized_read_.at(l).at(pv);
        }
        
        if (state.dspace_optimization_masks_.at("spatial_skip").at(l).at(pv))
        {
          // FIXME: handle the impact of applying spatial skip and temporal skip at the same time
          if (p != 0) 
          {
            std::cout << "Not implemented error: we do not support applying spatial " 
              << "and temporal skip at the same time yet"
              << std::endl;
            assert(false); 
          }
          p = state.prob_explicitly_spatially_optimized_read_.at(l).at(pv);
        }
        
        // std::cout << topology_specs.GetStorageLevel(l)->level_name << ": dspace: " << problem::GetShape()->DataSpaceIDToName.at(pv) 
        //   << " abs opt ratio (after prop): " << p 
        //   << " has metadata: " << compound_data_movement_nest[pv][l].has_metadata << std::endl;

        int impacted_level_id = l - 1;
        while (impacted_level_id >= 0) 
        {
          // upper level propagation essentially chops off a subtree
          //  child level will not see the subtree, so the accesses are nonexistent
          //  do no need to increment the skipped and gated counts

          // std::cout << " impacted level: " << topology_specs.GetStorageLevel(impacted_level_id)->level_name << std::endl;
          // apply the popagational impact first to all action types 
          max_fills[impacted_level_id][pv] -= floor(max_fills[impacted_level_id][pv] * p);
          max_reads[impacted_level_id][pv] -= floor(max_reads[impacted_level_id][pv] * p);
          max_updates[impacted_level_id][pv] -= floor(max_updates[impacted_level_id][pv] * p);
            
          // if impacted level has the same number of ranks as this level, then only need to go up to size - 1
          // happens when inner storage storage a tile of shape 1 with one rank of format data
          if (compound_data_movement_nest[pv][impacted_level_id].has_metadata)
          {          
            ScalePerTileFormatAccesses(max_format_fills[impacted_level_id][pv], p, std::min(max_format_fills[impacted_level_id][pv].size(), max_format_reads[l][pv].size()-1), 0);
            ScalePerTileFormatAccesses(max_format_reads[impacted_level_id][pv], p, std::min(max_format_reads[impacted_level_id][pv].size(), max_format_reads[l][pv].size()-1), 0);
            ScalePerTileFormatAccesses(max_format_updates[impacted_level_id][pv], p, max_format_updates[impacted_level_id][pv].size(), 0);
          }
          
          
          // now check if this level has its own SAF specified
          // if so, the fine-grained reads and updates can now be finalized
          // note that all the ops eliminated by upper levels are nonexistent, so they do not count towards skipped/gated ops incurred at this specific level
          bool local_saf_detected = state.dspace_optimization_masks_.at("gate").at(impacted_level_id).at(pv)
                                    || state.dspace_optimization_masks_.at("skip").at(impacted_level_id).at(pv)
                                    || state.dspace_optimization_masks_.at("spatial_skip").at(impacted_level_id).at(pv);
          
          if (local_saf_detected)
          {
            // std::cout <<  "   this is the next inner level with sparse optimization specified, finalize fine-grained action counts at " 
            //  << topology_specs.GetStorageLevel(impacted_level_id)->level_name << std::endl;
            std::string saf_type = state.dspace_optimization_masks_.at("spatial_skip").at(impacted_level_id).at(pv) ? "spatial_skip" :
              state.dspace_optimization_masks_.at("gate").at(impacted_level_id).at(pv) ? "gated" : "skipped";
            
            double local_saf_p = saf_type != "spatial_skip" ? state.prob_explicitly_optimized_read_.at(impacted_level_id).at(pv) :
                                                              state.prob_explicitly_spatially_optimized_read_.at(impacted_level_id).at(pv);
            double effective_p = 1 - ((1-local_saf_p)/(1-p));
            compound_data_movement_nest[pv][impacted_level_id].fine_grained_data_accesses["random_fill"] = max_fills[impacted_level_id][pv];
            compound_data_movement_nest[pv][impacted_level_id].fine_grained_format_accesses["random_metadata_fill"] = max_format_fills[impacted_level_id][pv];
            compound_data_movement_nest[pv][impacted_level_id].fine_grained_data_accesses["random_read"] = max_reads[impacted_level_id][pv];
            compound_data_movement_nest[pv][impacted_level_id].fine_grained_format_accesses["random_metadata_read"] = max_format_updates[impacted_level_id][pv];
            compound_data_movement_nest[pv][impacted_level_id].fine_grained_data_accesses["random_update"] = max_updates[impacted_level_id][pv];
            compound_data_movement_nest[pv][impacted_level_id].fine_grained_format_accesses["random_metadata_update"] = max_format_updates[impacted_level_id][pv];           
            
            if (state.dspace_optimization_masks_.at("spatial_skip").at(impacted_level_id).at(pv))
            {
              // std::cout <<  "   spatial sparse optimization specified, saf has no fine grained actin impact to current level: " 
              // << topology_specs.GetStorageLevel(impacted_level_id)->level_name << std::endl;
            }
            else
            {
              ApplyLocalStorageSAFImpact(compound_data_movement_nest, effective_p, pv, impacted_level_id, saf_type);
            }
            // fine grained access at this level is determined by its local saf
            fine_grained_action_finalized[impacted_level_id][pv] = true; 
            // all of the levels below will be impacted by this level's optimized away ops, 
            // stop propating upper level's saf's impact
            break;
          }
          else
          {
            impacted_level_id--;
          }
        }
        
        // if this level has SAF and is still not finalized, it must be the upper most level with SAF for this dataspace
        if (!fine_grained_action_finalized[l][pv])
        {
          // std::cout << " first level with SAF for this dataspce, apply impact directly" << std::endl;
          fine_grained_action_finalized[l][pv] = true;
          
          compound_data_movement_nest[pv][l].fine_grained_data_accesses["random_fill"] = max_fills[l][pv];
          compound_data_movement_nest[pv][l].fine_grained_format_accesses["random_metadata_fill"] = max_format_fills[l][pv];
          compound_data_movement_nest[pv][l].fine_grained_data_accesses["random_read"] = max_reads[l][pv];
          compound_data_movement_nest[pv][l].fine_grained_format_accesses["random_metadata_read"] = max_format_reads[l][pv];
          compound_data_movement_nest[pv][l].fine_grained_data_accesses["random_update"] = max_updates[l][pv];
          compound_data_movement_nest[pv][l].fine_grained_format_accesses["random_metadata_update"] = max_format_updates[l][pv];
  
          // spatial skip does not need local SAF impact since it does not change the number of accesses to local storages 
          if (!state.dspace_optimization_masks_.at("spatial_skip").at(l).at(pv))
          {
            std::string saf_type = state.dspace_optimization_masks_.at("gate").at(l).at(pv) ? "gated" : "skipped";
            ApplyLocalStorageSAFImpact(compound_data_movement_nest, p,  pv, l, saf_type);
          }
        }
       
        // only the innermost level for the target gives the final impact on compute units, propagated the impact to compute
        if (impacted_level_id < 0 && !problem::GetShape()->IsReadWriteDataSpace.at(pv))
        {
          
          max_computes[0] -= floor(max_computes[0] * p);
          state.storage_gs_saf_[pv] = true;
          state.innermost_empty_cond_on_prob_[pv] = round(p*1000000)/1000000;
        }
      } 
      else
        continue;
    }
  }

  // finalize the levels without gating or skipping SAFs
  for (unsigned l = 0; l < topology_specs.NumStorageLevels(); l++)
  {
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      if (!state.dspace_optimization_masks_.at("gate").at(l).at(pv) && !state.dspace_optimization_masks_.at("skip").at(l).at(pv))
      {
        compound_data_movement_nest[pv][l].fine_grained_data_accesses["random_fill"] = max_fills[l][pv];
        compound_data_movement_nest[pv][l].fine_grained_format_accesses["random_metadata_fill"] = max_format_fills[l][pv];
        compound_data_movement_nest[pv][l].fine_grained_data_accesses["random_read"] = max_reads[l][pv];
        compound_data_movement_nest[pv][l].fine_grained_data_accesses["random_update"] = max_updates[l][pv];
        compound_data_movement_nest[pv][l].fine_grained_format_accesses["random_metadata_read"] = max_format_reads[l][pv];
        compound_data_movement_nest[pv][l].fine_grained_format_accesses["random_metadata_update"] = max_format_updates[l][pv];
        compound_data_movement_nest[pv][l].temporal_reductions =  compound_data_movement_nest[pv][l].updates == 0 ? 0 : 
                       ceil(compound_data_movement_nest[pv][l].temporal_reductions 
                            * (double)compound_data_movement_nest[pv][l].fine_grained_data_accesses["random_update"]
                            / compound_data_movement_nest[pv][l].updates);
      }
      else
      {
        if (!fine_grained_action_finalized[l][pv])
        {
          // std::cout << "!!! fine grained action counts not finalized "
          // << topology_specs.GetStorageLevel(l)->level_name << ": dspace: " 
          // << problem::GetShape()->DataSpaceIDToName.at(pv) << std::endl;
          assert(fine_grained_action_finalized[l][pv]);
        }
      }
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

        access_record.fine_grained_data_accesses["random_read"] =
          access_record.reads - floor(access_record.reads * expected_sparsity);
        access_record.fine_grained_data_accesses["random_fill"] =
          access_record.fills - floor(access_record.fills * expected_sparsity);

        if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
        {
          access_record.fine_grained_data_accesses["random_update"] =
            access_record.updates - floor(access_record.updates * expected_sparsity);
        }
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
            compound_data_movement_nest[pv][l].fine_grained_data_accesses["compression_count"] +=
              compound_data_movement_nest[pv][parent_level].fine_grained_data_accesses["random_update"];
          }
          // compressed data from parent and decompress at the current level
          compound_data_movement_nest[pv][l].fine_grained_data_accesses["decompression_count"] +=
            compound_data_movement_nest[pv][l].fine_grained_data_accesses["random_fill"];
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

void CombineStorageOptimizationImpact(SparseAnalysisState& state,
                                      tiling::CompoundTileNest& compound_tile_nest,
                                      const model::Topology::Specs& topology_specs)
{
 
  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  
  ProcessDataReprImpactOnStorageAccesses(state, compound_data_movement_nest);
  PropagateImpactOfExplicitlyOptimizedRead(state, compound_tile_nest, topology_specs);
  CalculateDecompressionCompressionCost(state.num_storage_levels_, compound_data_movement_nest);
}

} // namespace
