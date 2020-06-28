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
		return sizeof(networkOperationTypes) / sizeof(networkOperationTypes[0]     );
 
  } else {
  	assert(false);
  }
}


double GetDensityByGatedDataSpaceNames(sparse::PerDataSpaceGatingInfo data_space_gating_info, 
                                       std::string action_name,
                                       tiling::CompoundDataMovementInfo& compound_data_movement){
  
  double density = 1.0;
  unsigned id;

  if (data_space_gating_info.find(action_name)!= data_space_gating_info.end()){
    std::vector<std::string> gated_data_space_names = data_space_gating_info.at(action_name);
    for (unsigned i = 0; i < gated_data_space_names.size(); i++){
      id = problem::GetShape()->DataSpaceNameToID.at(gated_data_space_names[i]);
      density *= compound_data_movement[id].tile_density.GetAverageDensity();
    }
  }

  return density;
}

//
// Storage
//

void ComputeFineGrainDataMovementAcesses(tiling::CompoundDataMovementInfo& compound_data_movement, 
                                         sparse::PerStorageLevelGatingInfo& per_level_sparse_gating){

  double read_avg_density;
  double write_avg_density;
  
  for (unsigned pv =0; pv < problem::GetShape() -> NumDataSpaces; pv++){  
    
    std::string data_space_name = problem::GetShape()->DataSpaceIDToName.at(pv);

    if (per_level_sparse_gating.find(data_space_name) != per_level_sparse_gating.end()){

      // std::cout << "   gating for dataspace: " << data_space_name << std::endl;

      sparse::PerDataSpaceGatingInfo data_space_gating_info = per_level_sparse_gating.at(data_space_name);

      read_avg_density = GetDensityByGatedDataSpaceNames(data_space_gating_info, "read", compound_data_movement);
      write_avg_density = GetDensityByGatedDataSpaceNames(data_space_gating_info, "write", compound_data_movement);

    } else {
      // no gating for this datatype at all
      read_avg_density = 1.0; 
      write_avg_density = 1.0;
    }

    // fine-grained read actions
    std::uint64_t total_reads = compound_data_movement[pv].reads; 
    std::uint64_t num_random_reads = ceil(read_avg_density * total_reads);
    compound_data_movement[pv].fine_grained_accesses["random_read"] = num_random_reads;
    compound_data_movement[pv].fine_grained_accesses["gated_read"] = total_reads - num_random_reads;

    // fine-grained calculations for fill actions
    std::uint64_t total_fills = compound_data_movement[pv].fills; 
    std::uint64_t num_random_fills = ceil(write_avg_density * total_fills);
    compound_data_movement[pv].fine_grained_accesses["random_fill"] = num_random_fills;
    compound_data_movement[pv].fine_grained_accesses["gated_fill"] = total_fills - num_random_fills;

    // fine-grained calculations for update actions
    std::uint64_t total_updates = compound_data_movement[pv].updates; 
    std::uint64_t num_random_updates = ceil(write_avg_density * total_updates);
    compound_data_movement[pv].fine_grained_accesses["random_update"] = num_random_updates;
    compound_data_movement[pv].fine_grained_accesses["gated_update"] = total_updates - num_random_updates;
    // std::cout << "level " << level << " reads: " << total_reads << " random reads: " << num_random_reads << std::endl;
  }
}


//
// Arithmetic
//

void ComputeFineGrainComputeAcesses(tiling::ComputeInfo& compute_info, 
	                                  tiling::CompoundDataMovementInfo& compound_data_movement,
                                    sparse::ComputeGatingInfo compute_gating_info){

 
  int total_accesses;  
  
  double compute_avg_density = GetDensityByGatedDataSpaceNames(compute_gating_info, "compute", compound_data_movement);
  total_accesses = compute_info.replication_factor * compute_info.accesses; 

  // generate the necessary fine-grained action counts
  compute_info.fine_grained_accesses["random_compute"] = ceil(total_accesses * compute_avg_density);
  compute_info.fine_grained_accesses["gated_compute"] = total_accesses - compute_info.fine_grained_accesses.at("random_compute");
}


}// namespace problem