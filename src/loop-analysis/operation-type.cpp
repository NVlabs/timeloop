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

double GetDensityByActionOptimizationNames(sparse::PerDataSpaceActionOptimizationInfo data_space_gating_info,
                                       std::string action_name,
                                       tiling::CompoundDataMovementInfo& compound_data_movement){
  
  double density = 1.0;
  unsigned id;
  if (data_space_gating_info.find(action_name)!= data_space_gating_info.end()){
    std::vector<std::string> gated_data_space_names = data_space_gating_info.at(action_name);
      for (unsigned i = 0; i < gated_data_space_names.size(); i++){
        id = problem::GetShape()->DataSpaceNameToID.at(gated_data_space_names[i]);
        density *= compound_data_movement[id].tile_density.GetTileExpectedDensity(compound_data_movement[id].size);
//        std::cout << "id: " << gated_data_space_names[i] << " tile size: " << compound_data_movement[id].size << " density: "
//<< compound_data_movement[id].tile_density.GetTileExpectedDensity(compound_data_movement[id].size) << std::endl;
      }
  }

  return density;
}


//double GetVarianceByActionOptimizationNames(sparse::PerDataSpaceActionOptimizationInfo data_space_gating_info,
//                                            std::string action_name,
//                                            tiling::CompoundDataMovementInfo& compound_data_movement){
//
//  double variance = 0.0;
//  unsigned id;
//
//  if (data_space_gating_info.find(action_name)!= data_space_gating_info.end()){
//    std::vector<std::string> gated_data_space_names = data_space_gating_info.at(action_name);
//    for (unsigned i = 0; i < gated_data_space_names.size(); i++){
//      id = problem::GetShape()->DataSpaceNameToID.at(gated_data_space_names[i]);
//      //  VAR(A+B) = VAR(A) + VAR(B) + COV(A,B), assume independent random variables COV(A,B)=0
//      variance += compound_data_movement[id].tile_density.GetVariance();
//    }
//  }
//
//  return variance;
//}


//
// Storage
//

void ComputeFineGrainDataMovementAccesses(tiling::CompoundDataMovementInfo& compound_data_movement,
                                          sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_gating,
                                          sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_skipping){

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

    // process skipping first
    if (per_level_sparse_skipping.find(data_space_name) != per_level_sparse_skipping.end()){
//      std::cout << "   skipping for dataspace: " << data_space_name << std::endl;

      sparse::PerDataSpaceActionOptimizationInfo data_space_skipping_info = per_level_sparse_skipping.at(data_space_name);
      read_avg_density *= GetDensityByActionOptimizationNames(data_space_skipping_info, "read", compound_data_movement);
      write_avg_density *= GetDensityByActionOptimizationNames(data_space_skipping_info, "write", compound_data_movement);

      // skipped actions
      num_skipped_reads = ceil((1-read_avg_density)* total_reads);
      num_skipped_fills = ceil((1-write_avg_density) * total_fills);
      num_skipped_updates = ceil((1-write_avg_density) * total_updates);
      compound_data_movement[pv].fine_grained_accesses["skipped_read"] = num_skipped_reads;
      compound_data_movement[pv].fine_grained_accesses["skipped_fill"] = num_skipped_fills;
      compound_data_movement[pv].fine_grained_accesses["skipped_update"] = num_skipped_updates;

    }

    // gating should always happen after data are skipped -> coupled density calculation
    if (per_level_sparse_gating.find(data_space_name) != per_level_sparse_gating.end()){

//       std::cout << "   gating for dataspace: " << data_space_name << std::endl;

      sparse::PerDataSpaceActionOptimizationInfo data_space_gating_info = per_level_sparse_gating.at(data_space_name);
      read_avg_density *= GetDensityByActionOptimizationNames(data_space_gating_info, "read", compound_data_movement);
      write_avg_density *= GetDensityByActionOptimizationNames(data_space_gating_info, "write", compound_data_movement);

      // gated actions
      num_gated_reads = ceil((1-read_avg_density)* total_reads);
      num_gated_fills = ceil((1-write_avg_density) * total_fills);
      num_gated_updates = ceil((1-write_avg_density) * total_updates);
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
void ComputeFineGrainMetaDataAccesses(sparse::PerStorageLevelCompressionInfo& per_level_compression_info,
                                      tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                      sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_gating,
                                      unsigned level){

  double metadata_read_avg_density;
  double metadata_write_avg_density;

//  std::cout << "level:  " << level << " --------------------" << std::endl;

  for (unsigned pv =0; pv < problem::GetShape() -> NumDataSpaces; pv++){

    std::string data_space_name = problem::GetShape()->DataSpaceIDToName.at(pv);
    tiling::CompoundDataMovementInfo& compound_data_movement = nest_of_compound_tiles.at(level).data_movement_info;

    if (per_level_compression_info.find(data_space_name) != per_level_compression_info.end()){
      // compression info found, calculate the number of metadata reads/fills according to
      // (1) metadata format
      // (2) metadata gating specifications

      std::uint64_t dense_memory_reads = compound_data_movement[pv].reads;
      std::uint64_t dense_memory_fills = compound_data_movement[pv].fills;

      std::string metadata_format = per_level_compression_info.at(data_space_name).metadata_format;
      double data_space_density = compound_data_movement[pv].tile_density.GetTileExpectedDensity(compound_data_movement[pv].size);
      double compression_rate = per_level_compression_info.at(data_space_name).compression_rate;

      // calculate the number of metadata reads/fills according to metadata format
      if (metadata_format == "bitmask"){
         compound_data_movement[pv].metadata_reads = dense_memory_reads;
         compound_data_movement[pv].metadata_fills = dense_memory_fills;
         //compound_data_movement[pv].metadata_tile_size = compound_data_movement[pv].size;

      } else if (metadata_format == "RLE"){
         // assume memory layout is concordant with mapping under eval
         compound_data_movement[pv].metadata_reads = dense_memory_reads * data_space_density / compression_rate;
         compound_data_movement[pv].metadata_fills = dense_memory_fills * data_space_density / compression_rate;
         //compound_data_movement[pv].metadata_tile_size = compound_data_movement[pv].compressed_size;

      } else if (metadata_format == "CSR"){

        // CSR == rank2 fiber:
        //    rank1 == rows (uncompressed format)
        //    rank0 == cols (compressed coordinate payload)

        std::vector<problem::Shape::DimensionID> rank1_index = compound_data_movement[pv].rank1_list;
        std::vector<problem::Shape::DimensionID> rank0_index = compound_data_movement[pv].rank0_list;

        loop::Descriptor descriptor;
        std::uint64_t total_num_rank1_elements = 1;
        std::uint64_t total_num_rank0_elements = 1;

//        std::uint64_t num_reads = 1;

        // tile includes all inner levels
        std::uint16_t rank1_detected = std::numeric_limits<uint16_t>::max();
//        std::uint16_t current_rank = std::numeric_limits<uint16_t>::max();

        std::uint64_t rank0_reads = 0;
        std::uint64_t rank1_reads = 0;

        std::vector<std::string> nest_ranks; // for recording whether the loop is for rank0, rank1 or none

       // collect the total number of rows and cols for this tile
       for (unsigned inner = 0; inner <= level; inner++){
          std::vector<loop::Descriptor> subnest = nest_of_compound_tiles.at(inner).data_movement_info[pv].subnest;
          for (unsigned loop = 0; loop < subnest.size(); loop++){
          // go through the loops in the subnest to intialize the subnest tile size and the total tile size
            // subnest[loop].Print(std::cout);
            // std::cout<<std::endl;
            problem::Shape::DimensionID id = subnest[loop].dimension;
            if (std::find(rank0_index.begin(), rank0_index.end(), id) != rank0_index.end()){
              uint64_t num_rank0_elements =  (subnest[loop].end - subnest[loop].start)/subnest[loop].stride;
              total_num_rank0_elements *= num_rank0_elements;
              // std::cout << "num_rank0_elements: " << num_rank0_elements << " total_num_rank0_elements: " << total_num_rank0_elements << std::endl;
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
      std::uint64_t num_binary_searches = std::uint64_t(raw_num_binary_searches);
      // std::cout << "number of binary searches: " << num_binary_searches << std::endl;

      // look at all the subnests for all the levels that are below this storage level to extract the access patterns
      // note: since we are using the read patterns of the levels below to characterize the read pattern of this storage,
      // this can result in discordant fills/reads to the lower level storage, but the pipeline write/read accesses are guarenteed in the levels below
      // the overhead of discordant fills are not charged, but the the overhead of discordant reads are charged
      // FIXME: we can consider more cases, e.g., concordant fills to the lower level storage. fills/reads will not be pipelinable in this case as a trade-off

      // the logic below is only good for dataspaces whose dimensions are not a sum of more than one dimensions
      // if the dataspace's dimension needs to be projected from other dimensions, this will not work, e.g., Inputs H=R+P W=S+Q
      // FIXME: add the support for CSR/CSC for Inputs
      // note:the major difference for Inputs is that we need to look at the inner loops and calculate halos until a loop for a different rank shows up.
      // From that on, we will not have any halos

      unsigned index = -1;
      for (unsigned inner = 0; inner <= level; inner++){
        std::vector<loop::Descriptor> subnest = nest_of_compound_tiles.at(inner).data_movement_info[pv].subnest;
//        std::cout << "=== inner level: " << inner << std::endl;

        for (unsigned loop = 0; loop < subnest.size(); loop++){
        // go through the loops in the subnest to intialize the subnest tile size and the total tile size
//          subnest[loop].Print(std::cout);
//          std::cout<<std::endl;

          index += 1; // keeping track of the loop index

          if (nest_ranks[index] == "rank0"){

            // if the loop is describing a dimension that belongs to rank0 (compressed format)
            uint64_t num_rank0_elements =  (subnest[loop].end - subnest[loop].start)/subnest[loop].stride;
//            std::cout << "num cols: " << num_cols << std::endl;
//            current_rank = 0;

            if (rank1_detected == std::numeric_limits<uint16_t>::max()){ //inner most loop
              rank1_reads += 1;
              rank0_reads += num_rank0_elements + num_binary_searches;
              rank1_detected = 0;
            } else if (rank1_detected == 0){
              rank0_reads *= num_rank0_elements;
            } else {
              rank0_reads *= num_rank0_elements;
              rank1_reads *= num_rank0_elements;
            }

//            std::cout << "current_rank: " << current_rank << " rank1_detected: " << rank1_detected << std::endl;
//            std::cout << "rank0_reads: " << rank0_reads << " rank1_reads: " << rank1_reads << std::endl << std::endl;

          } else if (nest_ranks[index] == "rank1"){

            // if the loop is describing a dimension that belongs to rows/rank1 (uncompressed format)

            uint64_t num_rank1_elements = (subnest[loop].end - subnest[loop].start)/subnest[loop].stride;
//            std::cout << "num rows: " << num_rows << std::endl;
//            current_rank = 1;
            if (rank1_detected == std::numeric_limits<uint16_t>::max()){ //inner most loop
              rank1_reads += num_rank1_elements;
              rank0_reads += num_rank1_elements + num_binary_searches * num_rank1_elements;
            } else {
              rank0_reads *= num_rank1_elements;
              rank1_reads *= num_rank1_elements;
            }
            rank1_detected = 1;
//            std::cout << "current_rank: " << current_rank << " rank1_detected: " << rank1_detected << std::endl;
//            std::cout << "rank0_reads: " << rank0_reads << " rank1_reads: " << rank1_reads << std::endl << std::endl;
          }
        }
      }

      // assume uniform sparsity per row/rank1
      // FIXME: consider more complicated sparsity distribution
      int num_rank1_index_fills = total_num_rank1_elements + 1;
      int num_rank0_index_fills = total_num_rank1_elements * total_num_rank0_elements * data_space_density;

       compound_data_movement[pv].dense_rank1_fills = total_num_rank1_elements + 1;
       compound_data_movement[pv].dense_rank0_fills = total_num_rank1_elements * total_num_rank0_elements;

      // total fills == total metadata tile size needed
      // compound_data_movement[pv].metadata_tile_size = num_rank1_index_fills + num_rank0_index_fills;
      // std::cout << "metadata tile size: " << compound_data_movement[pv].metadata_tile_size << std::endl;

      compound_data_movement[pv].metadata_fills = num_rank1_index_fills + num_rank0_index_fills;
      compound_data_movement[pv].metadata_reads = rank1_reads + rank0_reads * data_space_density;
//      std::cout << "data density: " << data_space_density << std::endl;
//      int num_data_fills = total_num_rows * total_num_cols * data_space_density;
//      std::cout << "metadata reads: " << compound_data_movement[pv].metadata_reads << std::endl;
//      std::cout << "num_data_fills: " << num_data_fills << std::endl;
//      std::cout << "num_rank1_index_fills: " << num_rank1_index_fills << std::endl;
//      std::cout << "num_rank0_index_fills: " << num_rank0_index_fills << std::endl;

      } else {
        std::cout << "cannot recognize metadata format specified: " << metadata_format << std::endl;
        exit(1);
      }

      // process gating optimization on metadata
      std::uint64_t gated_metadata_reads = 0;
      std::uint64_t gated_metadata_fills = 0;
      if (per_level_sparse_gating.find(data_space_name) != per_level_sparse_gating.end()){
        sparse::PerDataSpaceActionOptimizationInfo data_space_gating_info = per_level_sparse_gating.at(data_space_name);
        metadata_read_avg_density = GetDensityByActionOptimizationNames(data_space_gating_info, "metadata_read", compound_data_movement);
        metadata_write_avg_density = GetDensityByActionOptimizationNames(data_space_gating_info, "metadata_write", compound_data_movement);
        gated_metadata_reads = ceil(compound_data_movement[pv].metadata_reads * (1-metadata_read_avg_density));
        gated_metadata_fills = ceil(compound_data_movement[pv].metadata_fills * (1-metadata_write_avg_density));
      }

      compound_data_movement[pv].fine_grained_accesses["metadata_read"] = compound_data_movement[pv].metadata_reads - gated_metadata_reads;
      compound_data_movement[pv].fine_grained_accesses["gated_metadata_read"] = gated_metadata_reads;
      compound_data_movement[pv].fine_grained_accesses["metadata_fill"] = compound_data_movement[pv].metadata_fills - gated_metadata_fills;
      compound_data_movement[pv].fine_grained_accesses["gated_metadata_fill"] = gated_metadata_fills;

    }

    else {
      // no compression info found, default to no compression and no metadata
      compound_data_movement[pv].metadata_reads = 0;
      compound_data_movement[pv].metadata_fills = 0;
      compound_data_movement[pv].fine_grained_accesses["metadata_read"] = 0;
      compound_data_movement[pv].fine_grained_accesses["gated_metadata_read"] = 0;
      compound_data_movement[pv].fine_grained_accesses["metadata_fill"] = 0;
      compound_data_movement[pv].fine_grained_accesses["gated_metadata_fill"] = 0;

    }

    //
    // infer the counts for compression and decompression
    //

    // auto compression/decompression is always performed at the level that's compressed
    if (compound_data_movement[pv].compressed == false){

      // check parent and child level to determine whether decompression/compression logic is needed
      if(compound_data_movement[pv].parent_level!=std::numeric_limits<unsigned>::max()
         && compound_data_movement[pv].parent_level_compressed){
//         std::cout << "has compressed parent: " << data_space_name << std::endl;
         // parent compressed, this level uncompressed
        compound_data_movement[pv].fine_grained_accesses["decompression_count"] = compound_data_movement[pv].fills;
//        std::cout << "count: " << compound_data_movement[pv].fine_grained_accesses["decompression_count"] << std::endl;

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

void ComputeFineGrainComputeAccesses(tiling::ComputeInfo& compute_info,
	                                  tiling::CompoundDataMovementInfo& compound_data_movement,
                                    sparse::ComputeActionOptimizationInfo& compute_gating_info,
                                    sparse::ComputeActionOptimizationInfo& compute_skipping_info){

  uint64_t total_accesses;
  total_accesses = compute_info.replication_factor * compute_info.accesses;

  double compute_avg_density = 1.0;

  compute_avg_density = GetDensityByActionOptimizationNames(compute_skipping_info, "compute", compound_data_movement);

  compute_info.fine_grained_accesses["skipped_compute"] = ceil(total_accesses * (1-compute_avg_density));
//  std::cout << "---------------------" << std::endl;
  compute_avg_density = GetDensityByActionOptimizationNames(compute_gating_info, "compute", compound_data_movement);
//  std::cout << "compute avg density: " << compute_avg_density << std::endl;
  compute_info.fine_grained_accesses["gated_compute"] = ceil(total_accesses * (1-compute_avg_density));

  compute_info.fine_grained_accesses["random_compute"] = total_accesses
                                                         - compute_info.fine_grained_accesses["gated_compute"]
                                                         - compute_info.fine_grained_accesses["skipped_compute"];
}


//void ComputeFineGrainComputeAccesses(tiling::ComputeInfo& compute_info,
//	                                  tiling::CompoundDataMovementInfo& compound_data_movement,
//                                    sparse::ComputeActionOptimizationInfo& compute_gating_info,
//                                    sparse::ComputeActionOptimizationInfo& compute_skipping_info){
//
//  // total number of dense accesses
//  uint64_t total_accesses = compute_info.replication_factor * compute_info.accesses;
//  //  std::cout << "dense compute cycles: " << compute_info.accesses << std::endl;
//
//  // the number of cycles needed for the slowest PE
//  uint64_t compute_cycles;
//
//  // gating related stats
//  double gating_avg_density = GetDensityByActionOptimizationNames(compute_gating_info, "compute", compound_data_movement);
//  double gating_variance = GetVarianceByActionOptimizationNames(compute_gating_info, "compute", compound_data_movement);
//
//  // skipping related stats
//  double skipping_avg_density = GetDensityByActionOptimizationNames(compute_skipping_info, "compute", compound_data_movement);
//  double skipping_variance = GetVarianceByActionOptimizationNames(compute_skipping_info, "compute", compound_data_movement);
//
//  // only allow gated compute or skipped compute or none, but seems to be sufficient for many architectures
//  assert(((gating_avg_density!=1.0 || gating_variance!=0.0) &&(skipping_avg_density ==1.0 && skipping_variance==0.0)) ||
//         ((skipping_avg_density!=1.0 || skipping_variance!=0.0) &&(gating_avg_density ==1.0 && gating_variance==0.0)) ||
//         ((skipping_avg_density==1.0 && skipping_variance==0.0) &&(gating_avg_density ==1.0 && gating_variance==0.0))
//         );
//
//  // set a flag for what optimization is applied
//  std::string type;
//  if (gating_avg_density!=1.0 || gating_variance!=0.0){
//     type = "gated";
//  } else if (skipping_avg_density!=1.0 || skipping_variance!=0.0){
//     type = "skipped";
//  } else {
//     type = "none";
//  }
//
//  //std::cout << "type: " << type << std::endl;
//
//  if (type == "none"){
//
//    // dense data or no optimization at all, use the dense formulation
//    // all of the compute instances have the same performance
//    compute_cycles = compute_info.accesses;
//    compute_info.fine_grained_accesses["skipped_compute"] = 0;
//    compute_info.fine_grained_accesses["gated_compute"] = 0;
//    compute_info.fine_grained_accesses["random_compute"] = total_accesses;
//
//  } else {
//
//    double mean = type == "gated" ? gating_avg_density : skipping_avg_density;
//    double var = type == "gated" ? gating_variance : skipping_variance;
//    double std_dev = std::sqrt(var);
//    // std::cout << "mean: " << mean << " var: " << var << " std_dev: " << std_dev << std::endl;
//
//    double max_instance_density = 0.0; // for cases with skipping -- capture the slowest compute instance
//
////    if ((type == "gated" && gating_variance==0.0) || (type== "skipped" && skipping_variance==0.0)){
//
//    if (type == "gated"){
//      compute_info.fine_grained_accesses["gated_compute"] = ceil(total_accesses * (1-gating_avg_density));
//    } else {
//      compute_info.fine_grained_accesses["skipped_compute"] = ceil(total_accesses * (1-skipping_avg_density));
//    }

//       // phase 2 skipping cycles
//       if(type!="gated"){
//         max_instance_density = skipping_avg_density;
//       }

//    } else {
//
//      // phase 2 nonuniform distribution (normal distribution here) and skipping cycles
//      // create a random engine for sampling
//
//      std::default_random_engine generator;
//      std::normal_distribution<double> distribution(mean,std_dev);
//
//      for (std::uint64_t utilized_instance_id = 0; utilized_instance_id < compute_info.replication_factor; utilized_instance_id++){
//
//         double instance_density = distribution(generator);
//         if (instance_density < 0) { // we don't allow density less than zero
//           instance_density = 0;
//         }
//         // std::cout << "generated density: " << instance_density << std::endl;
//
//         // update on the slowest instance if involves skipping
//         if(type!="gating" && instance_density > max_instance_density){
//            max_instance_density = instance_density;
//         }
//
//         // std::cout << "PE compute cycles: " << ceil(compute_info.accesses * instance_density) << std::endl;
//
//         if (type == "gated"){
//           compute_info.fine_grained_accesses["gated_compute"] += ceil(compute_info.accesses * (1-instance_density));
//         } else {
//           compute_info.fine_grained_accesses["skipped_compute"] += ceil(compute_info.accesses * (1-instance_density));
//         }
//      }
//    }

//    // generate # of random computes according to gated computes and skipped computes
//    compute_info.fine_grained_accesses["random_compute"] = total_accesses
//                                                           - compute_info.fine_grained_accesses["gated_compute"]
//                                                           - compute_info.fine_grained_accesses["skipped_compute"];
//  }
//
//
//
//  compute_info.compute_cycles = compute_cycles;
//  // std::cout << "slowest compute instance cycles: " << compute_cycles << std::endl;
//
//}

}// namespace problem