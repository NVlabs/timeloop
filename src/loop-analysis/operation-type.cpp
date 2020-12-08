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

double GetDensityByActionOptimizationNames(std::string dataspace_name,
                                           sparse::PerDataSpaceActionOptimizationInfo data_space_optimization_info,
                                           std::string action_name,
                                           tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                           unsigned level) {
  
  double density = 1.0;
  unsigned storage_level_id, data_space_id;

  // if there is optimization specified for the action
  if (data_space_optimization_info.find(action_name)!= data_space_optimization_info.end()){
      sparse::Conditions optimization_conditions = data_space_optimization_info.at(action_name);

      for (sparse::Conditions::iterator c_iter = optimization_conditions.begin();
           c_iter != optimization_conditions.end(); c_iter++){

        data_space_id = problem::GetShape()->DataSpaceNameToID.at(c_iter->first);
        storage_level_id = c_iter->second;
        std::uint32_t gating_granularity;

        // determine the granularity of "zero-block"
        //  1) scalar zero values: granularity = 1 as we are looking at each value
        //  2) entire tile of zeros: granularity = tile shape, we are looking at the fiber
        if (dataspace_name == "compute" || (c_iter->first == dataspace_name && storage_level_id == level)){
          gating_granularity = 1;
        } else {
          gating_granularity = nest_of_compound_tiles[storage_level_id].data_movement_info[data_space_id].child_level_tile_size;
        }

        tiling::CompoundDataMovementInfo& compound_data_movement = nest_of_compound_tiles[storage_level_id].data_movement_info;
        density *= (1-compound_data_movement[data_space_id].tile_density->GetProbability(gating_granularity, 0));

      } // for each condition
  }
  return density;
}



//
// Storage
//

void ComputeFineGrainDataMovementAccesses(tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                          unsigned level,
                                          sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_gating,
                                          sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_skipping){

  tiling::CompoundDataMovementInfo& compound_data_movement = nest_of_compound_tiles[level].data_movement_info;

  for (unsigned pv =0; pv < problem::GetShape() -> NumDataSpaces; pv++){
    
    std::string data_space_name = problem::GetShape()->DataSpaceIDToName.at(pv);

    // initialize all actions
    std::uint64_t total_reads = compound_data_movement[pv].reads;
    std::uint64_t total_fills = compound_data_movement[pv].fills;
    std::uint64_t total_updates = compound_data_movement[pv].updates;
    std::uint64_t num_random_reads = 0;
    std::uint64_t num_random_fills = 0;
    std::uint64_t num_random_updates = 0;
    std::uint64_t num_gated_reads = 0;
    std::uint64_t num_gated_fills = 0;
    std::uint64_t num_gated_updates = 0;
    std::uint64_t num_skipped_reads = 0;
    std::uint64_t num_skipped_fills = 0;
    std::uint64_t num_skipped_updates = 0;

    double read_avg_density = 1.0;
    double write_avg_density = 1.0;
    double update_avg_density = 1.0;

    // initialize the gated and skipped action counts, if optimizations applied, these values will be updated
    compound_data_movement[pv].fine_grained_accesses["skipped_read"] = num_skipped_reads;
    compound_data_movement[pv].fine_grained_accesses["skipped_fill"] = num_skipped_fills;
    compound_data_movement[pv].fine_grained_accesses["skipped_update"] = num_skipped_updates;
    compound_data_movement[pv].fine_grained_accesses["gated_read"] = num_gated_reads;
    compound_data_movement[pv].fine_grained_accesses["gated_fill"] = num_gated_fills;
    compound_data_movement[pv].fine_grained_accesses["gated_update"] = num_gated_updates;

    //
    // process the impact of sparse optimizations on memory accesses
    //

    // process skipping first
    if (per_level_sparse_skipping.find(data_space_name) != per_level_sparse_skipping.end()){

      sparse::PerDataSpaceActionOptimizationInfo data_space_skipping_info = per_level_sparse_skipping.at(data_space_name);
      read_avg_density *= GetDensityByActionOptimizationNames(data_space_name, data_space_skipping_info,
                                                              "read", nest_of_compound_tiles, level);
      write_avg_density *= GetDensityByActionOptimizationNames(data_space_name, data_space_skipping_info,
                                                               "write", nest_of_compound_tiles, level);
      update_avg_density *= GetDensityByActionOptimizationNames(data_space_name, data_space_skipping_info,
                                                                "update", nest_of_compound_tiles, level);

      // skipped actions
      num_skipped_reads = ceil((1-read_avg_density)* total_reads);
      num_skipped_fills = ceil((1-write_avg_density) * total_fills);
      num_skipped_updates = ceil((1-update_avg_density) * total_updates);
      compound_data_movement[pv].fine_grained_accesses["skipped_read"] = num_skipped_reads;
      compound_data_movement[pv].fine_grained_accesses["skipped_fill"] = num_skipped_fills;
      compound_data_movement[pv].fine_grained_accesses["skipped_update"] = num_skipped_updates;

    }

    // gating should always happen after data are skipped -> coupled density calculation
    if (per_level_sparse_gating.find(data_space_name) != per_level_sparse_gating.end()){

      sparse::PerDataSpaceActionOptimizationInfo data_space_gating_info = per_level_sparse_gating.at(data_space_name);
      read_avg_density *= GetDensityByActionOptimizationNames(data_space_name, data_space_gating_info,
                                                              "read", nest_of_compound_tiles, level);
      write_avg_density *= GetDensityByActionOptimizationNames(data_space_name, data_space_gating_info,
                                                               "write", nest_of_compound_tiles, level);
      update_avg_density *= GetDensityByActionOptimizationNames(data_space_name, data_space_gating_info,
                                                                "update", nest_of_compound_tiles, level);

      // gated actions
      num_gated_reads = ceil((1-read_avg_density)* total_reads);
      num_gated_fills = ceil((1-write_avg_density) * total_fills);
      num_gated_updates = ceil((1-update_avg_density) * total_updates);
      compound_data_movement[pv].fine_grained_accesses["gated_read"] = num_gated_reads;
      compound_data_movement[pv].fine_grained_accesses["gated_fill"] = num_gated_fills;
      compound_data_movement[pv].fine_grained_accesses["gated_update"] = num_gated_updates;
    }

    num_random_reads = total_reads - num_gated_reads - num_skipped_reads;
    num_random_fills = total_fills - num_gated_fills - num_skipped_fills;
    num_random_updates = total_updates - num_skipped_updates - num_gated_updates;
    compound_data_movement[pv].fine_grained_accesses["random_read"] = num_random_reads;
    compound_data_movement[pv].fine_grained_accesses["random_fill"] = num_random_fills;
    compound_data_movement[pv].fine_grained_accesses["random_update"] = num_random_updates;

  }
}

//
// MetaData
//
void ComputeFineGrainMetaDataAccesses(tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                      unsigned level,
                                      sparse::PerStorageLevelCompressionInfo& per_level_compression_info,
                                      sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_gating){

  double metadata_read_avg_density = 1.0;
  double metadata_write_avg_density = 1.0;
  double metadata_update_avg_density = 1.0;

  for (unsigned pv =0; pv < problem::GetShape() -> NumDataSpaces; pv++){

    std::string data_space_name = problem::GetShape()->DataSpaceIDToName.at(pv);
    tiling::CompoundDataMovementInfo& compound_data_movement = nest_of_compound_tiles.at(level).data_movement_info;


    // initialize all fine-grained metadata related accesses to 0
    compound_data_movement[pv].fine_grained_accesses["metadata_read"] = 0;
    compound_data_movement[pv].fine_grained_accesses["gated_metadata_read"] = 0;
    compound_data_movement[pv].fine_grained_accesses["metadata_fill"] = 0;
    compound_data_movement[pv].fine_grained_accesses["gated_metadata_fill"] = 0;
    compound_data_movement[pv].fine_grained_accesses["metadata_update"] = 0;
    compound_data_movement[pv].fine_grained_accesses["gated_metadata_update"] =  0;

    if (per_level_compression_info.find(data_space_name) != per_level_compression_info.end()){
      // compression info found, calculate the number of metadata reads/fills according to
      // (1) metadata format
      // (2) metadata gating specifications

      std::uint64_t dense_memory_reads = compound_data_movement[pv].reads;
      std::uint64_t dense_memory_fills = compound_data_movement[pv].fills;
      std::uint64_t dense_memory_updates = compound_data_movement[pv].updates;

      std::string metadata_format = per_level_compression_info.at(data_space_name).metadata_format;
      double data_space_density = compound_data_movement[pv].tile_density->GetTileExpectedDensity(compound_data_movement[pv].size);
      double compression_rate = per_level_compression_info.at(data_space_name).compression_rate;

      // calculate the number of metadata reads/fills according to metadata format
      if (metadata_format == "bitmask"){
         compound_data_movement[pv].metadata_reads = dense_memory_reads;
         compound_data_movement[pv].metadata_fills = dense_memory_fills;
         compound_data_movement[pv].metadata_updates = dense_memory_updates;

      } else if (metadata_format == "RLE"){
         // assume memory layout is concordant with mapping under eval
         compound_data_movement[pv].metadata_reads = dense_memory_reads * data_space_density / compression_rate;
         compound_data_movement[pv].metadata_fills = dense_memory_fills * data_space_density / compression_rate;
         compound_data_movement[pv].metadata_updates = dense_memory_updates * data_space_density / compression_rate;

      } else if (metadata_format == "CSR"){

        // CSR == rank2 fiber:
        //    rank1 == rows (uncompressed format)
        //    rank0 == cols (compressed coordinate payload)

        std::vector<problem::Shape::DimensionID> rank1_index = compound_data_movement[pv].rank1_list;
        std::vector<problem::Shape::DimensionID> rank0_index = compound_data_movement[pv].rank0_list;

        loop::Descriptor descriptor;
        std::uint64_t total_num_rank1_elements = 1;
        std::uint64_t total_num_rank0_elements = 1;

        // tile includes all inner levels
        std::uint16_t rank1_detected = std::numeric_limits<uint16_t>::max();

        std::uint64_t rank0_reads = 0;
        std::uint64_t rank1_reads = 0;

        std::vector<std::string> nest_ranks; // for recording whether the loop is for rank0, rank1 or none

       // collect the total number of rows and cols for this tile
       for (unsigned inner = 0; inner <= level; inner++){
          std::vector<loop::Descriptor> subnest = nest_of_compound_tiles.at(inner).data_movement_info[pv].subnest;
          for (unsigned loop = 0; loop < subnest.size(); loop++){
          // go through the loops in the subnest to intialize the subnest tile size and the total tile size
            problem::Shape::DimensionID id = subnest[loop].dimension;
            if (std::find(rank0_index.begin(), rank0_index.end(), id) != rank0_index.end()){
              uint64_t num_rank0_elements =  (subnest[loop].end - subnest[loop].start)/subnest[loop].stride;
              total_num_rank0_elements *= num_rank0_elements;
              nest_ranks.push_back("rank0");
            } else if (std::find(rank1_index.begin(), rank1_index.end(), id) != rank1_index.end()){
              uint64_t num_rank1_elements = (subnest[loop].end - subnest[loop].start)/subnest[loop].stride;
              total_num_rank1_elements *= num_rank1_elements;
              nest_ranks.push_back("rank1");
            } else {
              nest_ranks.push_back("none");
            }
          }
       }

      // number of binary searches needed to find the location in a compressed fiber
      // # of accesses = log2(occupancy)
      double raw_num_binary_searches = log2(total_num_rank0_elements * data_space_density);
      if (raw_num_binary_searches < 0){
        raw_num_binary_searches = 0;
      }
      std::uint64_t num_binary_searches = ceil(raw_num_binary_searches);

      // look at all the subnests for all the levels that are below this storage level to extract the access patterns
      // note: since we are using the read patterns of the levels below to characterize the read pattern of this storage,
      // this can result in discordant fills/reads to the lower level storage, but the pipeline write/read accesses are guarenteed in the levels below
      // the overhead of discordant fills are not charged, but the the overhead of discordant reads are charged
      // FIXME: we can consider more cases, e.g., concordant fills to the lower level storage. fills/reads will not be pipelinable in this case as a trade-off

      // the logic below is only good for dataspaces whose dimensions are not a sum of more than one dimensions
      // if the dataspace's dimension needs to be projected from other dimensions, this will not work, e.g., Inputs H=R+P W=S+Q
      // FIXME: add the more accurate support for CSR/CSC for dataytpes whose dimensions are described in SoC form in the prob spec, inputs in conv calculations
      // note:the major difference for Inputs is that we need to look at the inner loops and calculate halos until a loop for a different rank shows up.
      // From that on, we will not have any halos

      unsigned index = -1;
      for (unsigned inner = 0; inner <= level; inner++){
        std::vector<loop::Descriptor> subnest = nest_of_compound_tiles.at(inner).data_movement_info[pv].subnest;

        for (unsigned loop = 0; loop < subnest.size(); loop++){
        // go through the loops in the subnest to intialize the subnest tile size and the total tile size
          index += 1; // keeping track of the loop index

          if (nest_ranks[index] == "rank0"){

            // if the loop is describing a dimension that belongs to rank0 (compressed format)
            uint64_t num_rank0_elements =  (subnest[loop].end - subnest[loop].start)/subnest[loop].stride;

            if (rank1_detected == std::numeric_limits<uint16_t>::max()){ //inner most loop
              rank1_reads += 1;
              rank0_reads += num_rank0_elements*data_space_density + num_binary_searches+1;
              rank1_detected = 0;
            } else if (rank1_detected == 0){
              rank0_reads *= num_rank0_elements;
            } else {
              rank0_reads *= num_rank0_elements;
              rank1_reads *= num_rank0_elements;
            }

          } else if (nest_ranks[index] == "rank1"){

            // if the loop is describing a dimension that belongs to rows/rank1 (uncompressed format)

            uint64_t num_rank1_elements = (subnest[loop].end - subnest[loop].start)/subnest[loop].stride;
            if (rank1_detected == std::numeric_limits<uint16_t>::max()){ //inner most loop
              rank1_reads += num_rank1_elements;
              rank0_reads += num_binary_searches * num_rank1_elements;
            } else {
              rank0_reads *= num_rank1_elements;
              rank1_reads *= num_rank1_elements;
            }
            rank1_detected = 1;
          }
        }
      }

      // assume uniform sparsity per row/rank1
      // FIXME: consider more complicated sparsity distribution
      int num_rank1_index_fills = total_num_rank1_elements + 1;
      int num_rank0_index_fills = total_num_rank1_elements * total_num_rank0_elements * data_space_density;

       compound_data_movement[pv].dense_rank1_fills = total_num_rank1_elements + 1;
       compound_data_movement[pv].dense_rank0_fills = total_num_rank1_elements * total_num_rank0_elements;

      if (compound_data_movement[pv].fills == 0){
        // for DRAM cases, there is no fills needed, so no metadata fills needed
        compound_data_movement[pv].metadata_fills = 0;
      } else {
        compound_data_movement[pv].metadata_fills = num_rank1_index_fills + num_rank0_index_fills;
      }
      compound_data_movement[pv].metadata_reads = rank1_reads + rank0_reads;
      // FIXME: distinguish updates from fills
      // only useful when output is in compressed format
      compound_data_movement[pv].metadata_updates = 0;

      } else {
        std::cout << "cannot recognize metadata format specified: " << metadata_format << std::endl;
        exit(1);
      }

      // process gating optimization on metadata
      std::uint64_t gated_metadata_reads = 0;
      std::uint64_t gated_metadata_fills = 0;
      std::uint64_t gated_metadata_updates = 0;
      if (per_level_sparse_gating.find(data_space_name) != per_level_sparse_gating.end()){
        sparse::PerDataSpaceActionOptimizationInfo data_space_gating_info = per_level_sparse_gating.at(data_space_name);
        metadata_read_avg_density = GetDensityByActionOptimizationNames(data_space_name, data_space_gating_info,
                                                                        "metadata_read", nest_of_compound_tiles, level);
        metadata_write_avg_density = GetDensityByActionOptimizationNames(data_space_name, data_space_gating_info,
                                                                         "metadata_write", nest_of_compound_tiles, level);
        metadata_update_avg_density = GetDensityByActionOptimizationNames(data_space_name, data_space_gating_info,
                                                                          "metadata_update", nest_of_compound_tiles, level);
        gated_metadata_reads = ceil(compound_data_movement[pv].metadata_reads * (1-metadata_read_avg_density));
        gated_metadata_fills = ceil(compound_data_movement[pv].metadata_fills * (1-metadata_write_avg_density));
        gated_metadata_updates = ceil(compound_data_movement[pv].metadata_updates * (1-metadata_update_avg_density));
      }

      compound_data_movement[pv].fine_grained_accesses["metadata_read"] = compound_data_movement[pv].metadata_reads - gated_metadata_reads;
      compound_data_movement[pv].fine_grained_accesses["gated_metadata_read"] = gated_metadata_reads;
      compound_data_movement[pv].fine_grained_accesses["metadata_fill"] = compound_data_movement[pv].metadata_fills - gated_metadata_fills;
      compound_data_movement[pv].fine_grained_accesses["gated_metadata_fill"] = gated_metadata_fills;
      compound_data_movement[pv].fine_grained_accesses["metadata_update"] = compound_data_movement[pv].metadata_updates - gated_metadata_updates;
      compound_data_movement[pv].fine_grained_accesses["gated_metadata_update"] =  gated_metadata_updates;
    }

    else {
      // no compression info found, default to no compression and no metadata
      compound_data_movement[pv].metadata_reads = 0;
      compound_data_movement[pv].metadata_fills = 0;
      compound_data_movement[pv].metadata_updates = 0;
      compound_data_movement[pv].fine_grained_accesses["metadata_read"] = 0;
      compound_data_movement[pv].fine_grained_accesses["gated_metadata_read"] = 0;
      compound_data_movement[pv].fine_grained_accesses["metadata_fill"] = 0;
      compound_data_movement[pv].fine_grained_accesses["gated_metadata_fill"] = 0;
      compound_data_movement[pv].fine_grained_accesses["metadata_update"] = 0;
      compound_data_movement[pv].fine_grained_accesses["gated_metadata_update"] =  0;

    }

    //
    // infer the counts for compression and decompression
    //
     compound_data_movement[pv].fine_grained_accesses["decompression_count"] = 0;
     compound_data_movement[pv].fine_grained_accesses["compression_count"]  = 0;

    // auto compression/decompression is always performed at the level that's compressed
    if (compound_data_movement[pv].compressed == false){

      // check parent and child level to determine whether decompression/compression logic is needed
      if(compound_data_movement[pv].parent_level!=std::numeric_limits<unsigned>::max()
         && compound_data_movement[pv].parent_level_compressed){
         // parent compressed, this level uncompressed
        compound_data_movement[pv].fine_grained_accesses["decompression_count"] = compound_data_movement[pv].fills;

        if (problem::GetShape()->IsReadWriteDataSpace.at(pv)){
          compound_data_movement[pv].fine_grained_accesses["compression_count"] = compound_data_movement[pv].updates;
        }
      }

      if (compound_data_movement[pv].child_level!=std::numeric_limits<unsigned>::max()
         && compound_data_movement[pv].child_level_compressed){
         // this level uncompressed, child compressed
        compound_data_movement[pv].fine_grained_accesses["compression_count"] += compound_data_movement[pv].reads;
      }
    }
  }
}


//
// Arithmetic
//

void ComputeFineGrainComputeAccesses(tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                     unsigned level,
                                    sparse::ComputeActionOptimizationInfo& compute_gating_info,
                                    sparse::ComputeActionOptimizationInfo& compute_skipping_info){

  uint64_t total_accesses;
  tiling::ComputeInfo& compute_info = nest_of_compound_tiles[level].compute_info;
  total_accesses = compute_info.replication_factor * compute_info.accesses;

  double compute_avg_density = 1.0;

  std::string dataspace_name = "compute";
  compute_avg_density = GetDensityByActionOptimizationNames(dataspace_name, compute_skipping_info,
                                                            "compute", nest_of_compound_tiles, level);

  compute_info.fine_grained_accesses["skipped_compute"] = ceil(total_accesses * (1-compute_avg_density));
  compute_avg_density = GetDensityByActionOptimizationNames(dataspace_name, compute_gating_info,
                                                            "compute", nest_of_compound_tiles, level);
  compute_info.fine_grained_accesses["gated_compute"] = ceil(total_accesses * (1-compute_avg_density));

  compute_info.fine_grained_accesses["random_compute"] = total_accesses
                                                         - compute_info.fine_grained_accesses["gated_compute"]
                                                         - compute_info.fine_grained_accesses["skipped_compute"];
}


}// namespace problem