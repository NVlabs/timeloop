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

//
// Storage
//

void ComputeFineGrainDataMovementAcesses(tiling::CompoundDataMovementInfo& compound_data_movement, int level){

  double avg_density=1.0;
  std::uint64_t total_reads;
  std::uint64_t num_random_reads;

  // get the average density across various data types  #FIXME: this calculation should be dependent on arch spec
  for (int pv = 0; pv < int(problem::GetShape()->NumDataSpaces); pv++){
    if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    avg_density *= compound_data_movement[pv].tile_density.GetAverageDensity();
  }

  for (int pv =0; pv < int(problem::GetShape() -> NumDataSpaces); pv++){  
    // given the current access info, compute the gated/random/sequential access types    

    total_reads = compound_data_movement[pv].reads;
    
    if (level == 0){  // we are only considering gating for the inner most level for now
      num_random_reads = ceil(avg_density * total_reads);
    } else {
      num_random_reads = total_reads;
    }
    
    compound_data_movement[pv].fine_grained_accesses["random_read"] = num_random_reads;
    compound_data_movement[pv].fine_grained_accesses["gated_read"] = total_reads - num_random_reads;

    // fine-grained calculations for fill actions
    compound_data_movement[pv].fine_grained_accesses["random_fill"] = compound_data_movement[pv].fills;
    
    // fine-grained calculations for update actions
    compound_data_movement[pv].fine_grained_accesses["random_update"] = compound_data_movement[pv].updates;

    // std::cout << "level " << level << " reads: " << total_reads << " random reads: " << num_random_reads << std::endl;
  }
}


//
// Arithmetic
//

void ComputeFineGrainComputeAcesses(tiling::ComputeInfo& compute_info, 
	                                  tiling::CompoundDataMovementInfo& compound_data_movement){

 
  int total_accesses;  
  double avg_density = 1.0;
  std::string op_name;
  
  // get the average density across various data types #FIXME: this calculation should be dependent on arch spec
  for (int pv = 0; pv < int(problem::GetShape()->NumDataSpaces); pv++){
    if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    avg_density *= compound_data_movement[pv].tile_density.GetAverageDensity();
  }

  total_accesses = compute_info.replication_factor * compute_info.accesses; 

  // generate the necessary fine-grained action counts
  compute_info.fine_grained_accesses["random_compute"] = ceil(total_accesses * avg_density);
  compute_info.fine_grained_accesses["gated_compute"] = total_accesses - compute_info.fine_grained_accesses.at("random_compute");
}


}// namespace problem