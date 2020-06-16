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
		return sizeof(networkOperationTypes) / sizeof(networkOperationTypes[0]);
 
  } else {
  	assert(false);
  }
}


int CalculateNumGatedCompute(tiling::ComputeInfo compute_info, tiling::CompoundDataMovementInfo compound_data_movement){

	int total_accesses = compute_info.replication_factor * compute_info.accesses;
	double avg_density = 1;
  for (int pv = 0; pv < int(problem::GetShape()->NumDataSpaces); pv++)
  	if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    	avg_density *= compound_data_movement[pv].tile_density.GetAverageDensity();
	return int(total_accesses * (1-avg_density));
}

int CalculateNumRandomCompute(tiling::ComputeInfo compute_info,tiling::CompoundDataMovementInfo compound_data_movement){

	int total_accesses = compute_info.replication_factor * compute_info.accesses;
	double avg_density = 1;
  for (int pv = 0; pv < int(problem::GetShape()->NumDataSpaces); pv++)
  	if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    	avg_density *= compound_data_movement[pv].tile_density.GetAverageDensity();
	return int(total_accesses * avg_density);
}

int CalculateNumArithmeticOps(tiling::ComputeInfo compute_info, tiling::CompoundDataMovementInfo compound_data_movement, std::string op_name){
	if (op_name == "random_compute")
		return CalculateNumRandomCompute(compute_info, compound_data_movement);
	else if (op_name == "gated_compute")
		return CalculateNumGatedCompute(compute_info, compound_data_movement);
	else
		assert(false);
}



}// namespace problem