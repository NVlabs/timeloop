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

#include "sparse.hpp"
#include "sparse-factory.hpp"
#include "mapping/arch-properties.hpp"

namespace sparse
{

	ArchGatingInfo arch_gating_info_;
	ArchProperties arch_props_;

	// forward declaration
  unsigned FindTargetStorageLevel(std::string storage_level_name);


	ArchGatingInfo Parse(config::CompoundConfigNode sparse_config, model::Engine::Specs& arch_specs){
		
		
		arch_props_.Construct(arch_specs);
		config::CompoundConfigNode opt_target_list;
		
		if (sparse_config.exists("targets")){
      opt_target_list = sparse_config.lookup("targets");
      assert(opt_target_list.isList());

      for (int i = 0; i < opt_target_list.getLength(); i ++){
      	// each element in the list represent a storage level's information
      	auto directive = opt_target_list[i];
      	// std::string optimization_type;
      	// assert(directive.lookupValue("type", optimization_type));

        std::string level_name;
        assert(directive.exists("name"));
        directive.lookupValue("name", level_name);

        // if (optimization_type == "data-gating"){
        if (directive.exists("action-gating")){

        	auto action_gating_directive = directive.lookup("action-gating");

          if (arch_props_.Specs().topology.GetArithmeticLevel()->name.Get() == level_name){
	      	   // compute level gating optimization
	           auto action_list = action_gating_directive.lookup("actions");
	           assert(action_list.isList());
	           PerDataSpaceGatingInfo compute_gating_info;
	           
	          for (int action_id = 0; action_id < action_list.getLength(); action_id ++){

	           	std::string action_name;
	           	action_list[action_id].lookupValue("name", action_name);
	           	assert(action_name == "compute"); // we only recognize compute for MACs

	            std::vector<std::string> action_data_space_list; 
	              
	            action_list[action_id].lookupArrayValue("criteria", action_data_space_list);

	            for (unsigned action_pv = 0; action_pv < unsigned(action_data_space_list.size()); action_pv++){
	            	// go thorugh the data spaces that the action should gate on
	            	std::string action_pv_name = action_data_space_list[action_pv];
	            	compute_gating_info[action_name].push_back(action_pv_name);
	            	// std::cout << "action " << action_name << " gated on " << action_pv_name << std::endl;
		          }
            }

            arch_gating_info_.compute_info = compute_gating_info;

          } else {
      	    // stroage level gating optimization
            unsigned storage_level_id = FindTargetStorageLevel(level_name);

	        	// parse for data-gating optimization
	        	if (action_gating_directive.exists("data-spaces")){
	        		
	        		PerStorageLevelGatingInfo per_storage_level_gating_info;
	        		auto data_space_list = action_gating_directive.lookup("data-spaces");
	        		assert(data_space_list.isList());
	        		       		
	        		for(unsigned pv = 0; pv < unsigned(data_space_list.getLength()); pv++){
	            // go through the data spaces that have gated actions specified 
	              if(data_space_list[pv].exists("actions")){
	              	PerDataSpaceGatingInfo data_space_gating_info;
	        			  std::string data_space_name;
	                assert(data_space_list[pv].lookupValue("name", data_space_name));
	                
	                auto action_list = data_space_list[pv].lookup("actions");
	                assert(action_list.isList());
	                
	                for(unsigned action_id = 0; action_id < unsigned(action_list.getLength()); action_id++){
	                  // go through the gated actions specified for that specific data type
	                  std::string action_name;
	                  action_list[action_id].lookupValue("name", action_name);

	                  // we only recognize read and write actions for buffers and metadata buffers
	                  assert(action_name == "read" ||action_name == "write" || action_name == "metadata_write" || action_name == "metadata_read"); 
	                  
	                  std::vector<std::string> action_data_space_list; 
	                  action_list[action_id].lookupArrayValue("criteria", action_data_space_list);

	                  for (unsigned action_pv = 0; action_pv < unsigned(action_data_space_list.size()); action_pv++){
	                  	// go thorugh the data spaces that the action should gate on
	                  	std::string action_pv_name = action_data_space_list[action_pv];
	                  	data_space_gating_info[action_name].push_back(action_pv_name);
	                  	// std::cout << "action " << action_name << " gated on " << action_pv_name << std::endl;
	                  }
	                } // go through action list
	                
	                per_storage_level_gating_info[data_space_name] = data_space_gating_info;
	              } // if exists gated-actions 
	        		} // go through data-space list
	        		
	        		arch_gating_info_.storage_info[storage_level_id] = per_storage_level_gating_info;
	        	}
	        }
	          
        } else {
        	std::cout << "Cannot recognize sparse optimization type" << std::endl;
        	exit(1);
        }
      }
    }

    return arch_gating_info_;
	}

//
// FindTargetTilingLevel()
//
unsigned FindTargetStorageLevel(std::string storage_level_name){
 
  auto num_storage_levels = arch_props_.StorageLevels();
    
  //
  // Find the target storage level using its name
  //
  
   unsigned storage_level_id;
  // Find this name within the storage hierarchy in the arch specs.
  for (storage_level_id = 0; storage_level_id < num_storage_levels; storage_level_id++)
  {
    if (arch_props_.Specs().topology.GetStorageLevel(storage_level_id)->level_name == storage_level_name)
      break;
  }
  
  if (storage_level_id == num_storage_levels)
  {
    std::cerr << "ERROR: target storage level not found: " << storage_level_name << std::endl;
    exit(1);
  }

  // std::cout << "locate sparse optimization directive at: " << storage_level_name << std::endl;
  return storage_level_id;
}

} // namespace