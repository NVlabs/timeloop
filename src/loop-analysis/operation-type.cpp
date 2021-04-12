/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <string>

#include "operation-type.hpp"
#include "mapping/loop.hpp"

#include <random>


namespace tiling
{

int GetNumOpTypes()
{
  // default placeholder: assuming one op type
  return 1;
}

int GetNumOpTypes(std::string component_type){
  if (component_type == "arithmetic"){
        return sizeof(arithmeticOperationTypes) / sizeof(arithmeticOperationTypes[0]);

  } else if (component_type == "storage"){
    return sizeof(storageOperationTypes) / sizeof(storageOperationTypes[0]);
        
  } else if (component_type == "network") {
    return sizeof(networkOperationTypes) / sizeof(networkOperationTypes[0]);
 
  } else {
    assert(false);
  }
}

double GetAvgNonEmptyIntersectionProbByActionName(const std::string dataspace_name,
                                                  const sparse::PerDataSpaceActionOptimizationInfo& data_space_optimization_info,
                                                  const std::string action_name,
                                                  const tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                                  const unsigned level)
{

  (void) level;
  (void) dataspace_name;

  double avg_non_empty_intersection_prob = 1.0;
  unsigned storage_level_id, data_space_id;


  // if there is optimization specified for the action
  if (data_space_optimization_info.find(action_name)!= data_space_optimization_info.end()){
      auto optimization_condition = data_space_optimization_info.at(action_name);

      // we are always looking at the "OR", "AND", and "XOR" of the sub-tiles of the listed dataspace-storage pairs
      // e.g., for paris A-spad and B-spad and a "OR" type intersection
      //       an empty intersection is resulted if either A subtile from spad or B subtile from spad is empty

      for (auto c_iter = optimization_condition.conditions.begin();
           c_iter != optimization_condition.conditions.end(); c_iter++){
        data_space_id = problem::GetShape()->DataSpaceNameToID.at(c_iter->first);
        storage_level_id = c_iter->second;

      double empty_prob;
      if (action_name.find("fill") == std::string::npos)
      {

        empty_prob = nest_of_compound_tiles[storage_level_id].data_movement_info[data_space_id].GetChildLevelDataTileOccupancyProbability(0);
        //std::cout << "empty prob: " << empty_prob << std::endl;
        //std::cout << " child level tile shape: "
        //<< nest_of_compound_tiles[storage_level_id].data_movement_info[data_space_id].GetChildTileCoordinateSpaceInfo().GetShape()
        //<< std::endl;
      }
      else
      {
        // fill optimization is performed based on intersection of the current level tile
        empty_prob = nest_of_compound_tiles[storage_level_id].data_movement_info[data_space_id].GetDataTileOccupancyProbability(0);
		//std::cout << "empty prob: " << empty_prob << std::endl;
		//std::cout << " current level tile shape: " << nest_of_compound_tiles[storage_level_id].data_movement_info[data_space_id].shape << std::endl;
      }

      if (optimization_condition.type == "OR") avg_non_empty_intersection_prob*= (1-empty_prob);
      else assert(false); //TODO: more intersection types: AND and XOR

      } // for each condition
  }

  return avg_non_empty_intersection_prob;
}



//
// Storage
//

void ComputeFineGrainDataMovementAccesses(tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                          const unsigned level,
                                          const sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_gating,
                                          const sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_skipping){

  tiling::CompoundDataMovementInfo& compound_data_movement = nest_of_compound_tiles[level].data_movement_info;

  for (unsigned pv =0; pv < problem::GetShape() -> NumDataSpaces; pv++){
    
    std::string data_space_name = problem::GetShape()->DataSpaceIDToName.at(pv);

    //
    // process the user-defined skipping and gating optimizations
    //
    std::map<std::string, double> avg_non_empty_intersection_probs = {{"gated_read", 1.0}, {"gated_fill", 1.0}, {"gated_update", 1.0},
                                                                      {"skipped_read", 1.0}, {"skipped_fill", 1.0}, {"skipped_update", 1.0}};

    bool needs_sparse_processing = false;
    // process skipping
    if (per_level_sparse_skipping.find(data_space_name) != per_level_sparse_skipping.end())
    {


      sparse::PerDataSpaceActionOptimizationInfo data_space_skipping_info = per_level_sparse_skipping.at(data_space_name);
      avg_non_empty_intersection_probs["skipped_read"] = GetAvgNonEmptyIntersectionProbByActionName
                                                                    (data_space_name, data_space_skipping_info,
                                                                    "read", nest_of_compound_tiles, level);
      avg_non_empty_intersection_probs["skipped_fill"] = GetAvgNonEmptyIntersectionProbByActionName
                                                                    (data_space_name, data_space_skipping_info,
                                                                     "fill", nest_of_compound_tiles, level);
      avg_non_empty_intersection_probs["skipped_update"] = GetAvgNonEmptyIntersectionProbByActionName
                                                                    (data_space_name, data_space_skipping_info,
                                                                     "update", nest_of_compound_tiles, level);
      needs_sparse_processing = true;
    }

    // process gating
    if (per_level_sparse_gating.find(data_space_name) != per_level_sparse_gating.end())
    {

      sparse::PerDataSpaceActionOptimizationInfo data_space_gating_info = per_level_sparse_gating.at(data_space_name);
      avg_non_empty_intersection_probs["gated_read"] = GetAvgNonEmptyIntersectionProbByActionName
                                                                  (data_space_name, data_space_gating_info,
                                                                   "read", nest_of_compound_tiles, level);
      avg_non_empty_intersection_probs["gated_fill"] = GetAvgNonEmptyIntersectionProbByActionName
                                                                  (data_space_name, data_space_gating_info,
                                                                   "fill", nest_of_compound_tiles, level);
      avg_non_empty_intersection_probs["gated_update"] = GetAvgNonEmptyIntersectionProbByActionName
                                                                    (data_space_name, data_space_gating_info,
                                                                     "update", nest_of_compound_tiles, level);
      needs_sparse_processing = true;
    }

    // initialize the gated and skipped action counts
   compound_data_movement[pv].fine_grained_accesses["skipped_read"] = 0;
   compound_data_movement[pv].fine_grained_accesses["skipped_fill"] = 0;
   compound_data_movement[pv].fine_grained_accesses["skipped_update"] = 0;
   compound_data_movement[pv].fine_grained_accesses["gated_read"] = 0;
   compound_data_movement[pv].fine_grained_accesses["gated_fill"] = 0;
   compound_data_movement[pv].fine_grained_accesses["gated_update"] = 0;
   compound_data_movement[pv].fine_grained_accesses["random_read"] = compound_data_movement[pv].reads;
   compound_data_movement[pv].fine_grained_accesses["random_fill"] = compound_data_movement[pv].fills;
   compound_data_movement[pv].fine_grained_accesses["random_update"] = compound_data_movement[pv].updates;

   if (compound_data_movement[pv].compressed || needs_sparse_processing)
   {
     compound_data_movement[pv].ComputeExpectedDataAccesses(avg_non_empty_intersection_probs);
   }
  }
}

//
// MetaData
//
void ComputeFineGrainMetaDataAccesses(tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                      const unsigned level,
                                      const sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_gating){

  double metadata_read_avg_density = 1.0;
  double metadata_write_avg_density = 1.0;
  double metadata_update_avg_density = 1.0;

  for (unsigned pv =0; pv < problem::GetShape() -> NumDataSpaces; pv++) {

    std::string data_space_name = problem::GetShape()->DataSpaceIDToName.at(pv);
    tiling::CompoundDataMovementInfo &compound_data_movement = nest_of_compound_tiles.at(level).data_movement_info;


    std::map<std::string, double> avg_non_empty_intersection_probs = {{"gated_read", 1.0}, {"gated_fill", 1.0}, {"gated_update", 1.0},
                                                                      {"skipped_read", 1.0}, {"skipped_fill", 1.0}, {"skipped_update", 1.0}};

    if (per_level_sparse_gating.find(data_space_name) != per_level_sparse_gating.end()) {
      sparse::PerDataSpaceActionOptimizationInfo data_space_gating_info = per_level_sparse_gating.at(data_space_name);
      metadata_read_avg_density = GetAvgNonEmptyIntersectionProbByActionName(data_space_name, data_space_gating_info,
                                                                      "metadata_read", nest_of_compound_tiles, level);
      metadata_write_avg_density = GetAvgNonEmptyIntersectionProbByActionName(data_space_name, data_space_gating_info,
                                                                       "metadata_write", nest_of_compound_tiles, level);
      metadata_update_avg_density = GetAvgNonEmptyIntersectionProbByActionName(data_space_name, data_space_gating_info,
                                                                        "metadata_update", nest_of_compound_tiles,
                                                                        level);
      avg_non_empty_intersection_probs["gated_read"] = metadata_read_avg_density;
      avg_non_empty_intersection_probs["gated_fill"] = metadata_write_avg_density;
      avg_non_empty_intersection_probs["gated_update"] = metadata_update_avg_density;

    }

    compound_data_movement[pv].ComputeExpectedMetaDataAccesses(avg_non_empty_intersection_probs);
    //
    // infer the counts for compression and decompression
    //
    compound_data_movement[pv].fine_grained_accesses["decompression_count"] = 0;
    compound_data_movement[pv].fine_grained_accesses["compression_count"] = 0;

    // auto compression/decompression is always performed at the level that's not compressed
    if (compound_data_movement[pv].compressed == false)
    {
      // check parent and child level to determine whether decompression/compression logic is needed
      if (compound_data_movement[pv].parent_level != std::numeric_limits<unsigned>::max()
          && compound_data_movement[pv].parent_level_compressed) {
        // parent compressed, this level uncompressed
        compound_data_movement[pv].fine_grained_accesses["decompression_count"] = compound_data_movement[pv].fills;

        if (problem::GetShape()->IsReadWriteDataSpace.at(pv)) {
          compound_data_movement[pv].fine_grained_accesses["compression_count"] = compound_data_movement[pv].updates;
        }
      }

      if (compound_data_movement[pv].child_level != std::numeric_limits<unsigned>::max()
          && compound_data_movement[pv].child_level_compressed) {
        // this level uncompressed, child compressed
        assert(false); // we do not allow on-chip compression for now
        compound_data_movement[pv].fine_grained_accesses["compression_count"] += compound_data_movement[pv].reads;
      }
    }
  }
}


//
// Arithmetic
//

void ComputeFineGrainComputeAccesses(tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                     const unsigned level,
                                     const sparse::ComputeActionOptimizationInfo& compute_gating_info,
                                     const sparse::ComputeActionOptimizationInfo& compute_skipping_info){

  uint64_t total_accesses;
  tiling::ComputeInfo& compute_info = nest_of_compound_tiles[level].compute_info;
  total_accesses = compute_info.replication_factor * compute_info.accesses;

  double compute_avg_density = 1.0;

  std::string dataspace_name = "compute";
  compute_avg_density = GetAvgNonEmptyIntersectionProbByActionName(dataspace_name, compute_skipping_info,
                                                                   "compute", nest_of_compound_tiles, level);

  compute_info.fine_grained_accesses["skipped_compute"] = ceil(total_accesses * (1-compute_avg_density));

  compute_avg_density = GetAvgNonEmptyIntersectionProbByActionName(dataspace_name, compute_gating_info,
                                                            "compute", nest_of_compound_tiles, level);

  compute_info.fine_grained_accesses["gated_compute"] = ceil(total_accesses * (1-compute_avg_density));

  compute_info.fine_grained_accesses["random_compute"] = total_accesses
                                                         - compute_info.fine_grained_accesses["gated_compute"]
                                                         - compute_info.fine_grained_accesses["skipped_compute"];
}


}// namespace problem