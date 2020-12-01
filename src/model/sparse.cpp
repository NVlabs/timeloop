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

SparseOptimizationInfo sparse_optimization_info_;

ActionGatingInfo action_gating_info_;
ActionSkippingInfo action_skipping_info_;
CompressionInfo compression_info_;

ArchProperties arch_props_;

// forward declaration
unsigned FindTargetStorageLevel(std::string storage_level_name);
void ParseActionOptimizationInfo(config::CompoundConfigNode directive, std::string optimization_type);
void ParseCompressionInfo(config::CompoundConfigNode directive);

// highest level parse function
SparseOptimizationInfo Parse(config::CompoundConfigNode sparse_config, model::Engine::Specs& arch_specs){

  arch_props_.Construct(arch_specs);
  config::CompoundConfigNode opt_target_list;

  if (sparse_config.exists("targets")){
    opt_target_list = sparse_config.lookup("targets");
    assert(opt_target_list.isList());

    for (int i = 0; i < opt_target_list.getLength(); i ++){
      // each element in the list represent a storage level's information
      auto directive = opt_target_list[i];

      // populate the gating and skipping info first, they will be used for later sanity check when parsing for compression
      if (directive.exists("action-gating")){
//        std::cout << "target: " << i << " parsing for action-gating directive" << std::endl;
        ParseActionOptimizationInfo(directive, "action-gating");
      }
      if (directive.exists("action-skipping")){
//        std::cout << "target: " << i << " parsing for action-skipping directive" << std::endl;
        ParseActionOptimizationInfo(directive, "action-skipping");
      }

      // parse for compression
      if (directive.exists("compression")){
//        std::cout << "target: " << i << " parsing for compression directive" << std::endl;
        ParseCompressionInfo(directive);
      }
    }

    std::cout << "Sparse optimization configuration complete." << std::endl;
  }

  sparse_optimization_info_.action_gating_info = action_gating_info_;
  sparse_optimization_info_.action_skipping_info = action_skipping_info_;
  sparse_optimization_info_.compression_info = compression_info_;
  return sparse_optimization_info_;
}

// parse for compression info (storage only) of one directive
void ParseCompressionInfo(config::CompoundConfigNode directive){

  std::string level_name;
  assert(directive.exists("name"));
  directive.lookupValue("name", level_name);

  auto compression_directive = directive.lookup("compression");
  unsigned storage_level_id = FindTargetStorageLevel(level_name);

  if (compression_directive.exists("data-spaces")){
     auto data_space_list = compression_directive.lookup("data-spaces");
     assert(data_space_list.isList());
     PerStorageLevelCompressionInfo per_storage_level_compression_info;

     // action optimization specifications for later checking
     PerStorageLevelActionOptimizationInfo per_storage_level_gating_info = {};
     if (action_gating_info_.storage_info.find(storage_level_id) != action_gating_info_.storage_info.end()){
       per_storage_level_gating_info = action_gating_info_.storage_info[storage_level_id];
     }

     PerStorageLevelActionOptimizationInfo per_storage_level_skipping_info = {};
     if (action_skipping_info_.storage_info.find(storage_level_id) != action_skipping_info_.storage_info.end()){
       per_storage_level_skipping_info = action_skipping_info_.storage_info[storage_level_id];
     }

     // check if there is any compressed datatypes
     for(unsigned pv = 0; pv < unsigned(data_space_list.getLength()); pv++){
       std::string data_space_name;
       data_space_list[pv].lookupValue("name", data_space_name);

       // tensor representation format must be specified (for compressed data or uncompressed data with special metadata)
       std::vector<std::string> format_list;
       assert(data_space_list[pv].exists("format"));
       data_space_list[pv].lookupArrayValue("format", format_list);

       if ((format_list.size() == 1 && format_list[0] == "uncompressed")
            || ((format_list.size() == 2 && format_list[0] == "uncompressed" && format_list[1] == "uncompressed"))){
            // this is either a 1D compressed tensor or a 2D compressed tensor, nothing special needs to be done
            continue;
       }

       std::string metadata_format;
       bool compressed = false;
       double compression_rate;

       if (format_list.size() == 1 && (format_list[0] == "uncompressed-bitmask" || format_list[0] == "UB")) {
         // rank-1 bitmask uncompressed
         metadata_format = "bitmask";
         compression_rate = 0.0;
         compressed = false;
       } else if (format_list.size() == 1 && (format_list[0] == "compressed-bitmask" || format_list[0] == "B")) {
         // rank-1 bitmask compressed
         metadata_format = "bitmask";
         compression_rate = 1.0;
         compressed = true;
       } else if (format_list.size() == 1 && (format_list[0] == "compressed-RLE" || format_list[0] == "R")){
         // rank-1 RLE compressed
         metadata_format = "RLE";
         compression_rate = 1.0;
         compressed = true;
       } else if (format_list.size() == 2 && ( (format_list[0] == "uncompressed" || format_list[0] == "U")
                                             && (format_list[1] == "compressed-CP" || format_list[1] == "C"))){
         // rank-2 uncompressed + coordinate list
         metadata_format = "CSR";
         compression_rate = 1.0;
         compressed = true;
       } else {
         std::cout << "ERROR: cannot recognize compression format" << std::endl;
       }

       //check for compression rate, default to fully compressed if the data is supposed to be compressed (according to metadata format)

       if (compressed){
         if (data_space_list[pv].exists("compression_rate")){
           data_space_list[pv].lookupValue("compression_rate", compression_rate);
         }

         if (metadata_format == "CSR"){

             if (!data_space_list[pv].exists("rank0") || !data_space_list[pv].exists("rank1")){
               std::cout << " must specify the rank order for the rank2 tensor representation" << std::endl;
               exit(1);
             }

             std::vector<std::string> rank0_list, rank1_list;
             data_space_list[pv].lookupArrayValue("rank0", rank0_list);
             data_space_list[pv].lookupArrayValue("rank1", rank1_list);

             for(unsigned i = 0 ; i < rank0_list.size(); i++){
               problem::Shape::DimensionID id = problem::GetShape()->DimensionNameToID.at(rank0_list[i]);
               per_storage_level_compression_info[data_space_name].rank0_list.push_back(id);
             }

             for(unsigned i = 0 ; i < rank1_list.size(); i++){
               problem::Shape::DimensionID id = problem::GetShape()->DimensionNameToID.at(rank1_list[i]);
               per_storage_level_compression_info[data_space_name].rank1_list.push_back(id);
             }
         }

       } else { // uncompressed
         if (data_space_list[pv].exists("compression_rate")){
           std::cout << "ERROR: cannot have compression rate for uncompressed data" << std::endl;
           exit(1);
         }
       }

       // populate compression info for data space
       // (1) uncompressed with some metadata format
       // (2) compressed with some compression rate and metadata format
       per_storage_level_compression_info[data_space_name].compressed = compressed;
       per_storage_level_compression_info[data_space_name].compression_rate = compression_rate;
       per_storage_level_compression_info[data_space_name].metadata_format = metadata_format;

       //
       // check if any architecture optimization is specified for the compressed data
       //
       if (metadata_format == "CSR" || metadata_format=="RLE"){

         if (per_storage_level_gating_info.find(data_space_name) == per_storage_level_gating_info.end() &&
             per_storage_level_skipping_info.find(data_space_name) == per_storage_level_skipping_info.end()){
           std::cout << "ERROR: " << level_name << ": please specify architecture optimization for processing compressed data" << std::endl;
           exit(1);
         } else {
            // both read and write optimizations need to be specified
            bool read_specified = false;
            bool fill_specified = false;
            std::map<ActionName, Conditions> action_info;

            // read and write optimizations can be specified in different optimization directives
            // check gating optimization directive
            if (per_storage_level_gating_info.find(data_space_name) != per_storage_level_gating_info.end()){
                action_info = per_storage_level_gating_info.at(data_space_name);
                for(auto const& action: action_info){
                  if (action.first == "read")
                    read_specified = true;
                  if (action.first == "write")
                    fill_specified = true;
                }
            }

            // check skipping optimization directive
            if (per_storage_level_skipping_info.find(data_space_name) != per_storage_level_skipping_info.end()){
                action_info = per_storage_level_skipping_info.at(data_space_name);
                for(auto const& action: action_info){
                  if (action.first == "read")
                    read_specified = true;
                  if (action.first == "write")
                    fill_specified = true;
                }
            }
            assert(read_specified && fill_specified);
         }
       }
     } // loop through the data spaces

     compression_info_[storage_level_id] = per_storage_level_compression_info;
  } // if there is compression specification in terms of data spaces
}


// parse for action gating info (storage and compute) of one directive
void ParseActionOptimizationInfo(config::CompoundConfigNode directive, std::string optimization_type){

  std::string level_name;
  assert(directive.exists("name"));
  directive.lookupValue("name", level_name);

  config::CompoundConfigNode action_optimization_directive;
  if (optimization_type == "action-gating"){
     action_optimization_directive = directive.lookup("action-gating");
  } else {
     action_optimization_directive = directive.lookup("action-skipping");
  }

  if (arch_props_.Specs().topology.GetArithmeticLevel()->name.Get() == level_name){
     // compute level gating optimization
     auto action_list = action_optimization_directive.lookup("actions");
     assert(action_list.isList());
     PerDataSpaceActionOptimizationInfo compute_optimization_info;

    for (int action_id = 0; action_id < action_list.getLength(); action_id ++){

      std::string action_name;
      action_list[action_id].lookupValue("name", action_name);
      assert(action_name == "compute"); // we only recognize compute for MACs

      auto conditions_list = action_list[action_id].lookup("conditions");
      Conditions conditions;
      for (unsigned pv_storage_pair_id = 0; pv_storage_pair_id < unsigned(conditions_list.getLength()); pv_storage_pair_id++){
        // go through the dataspace-storage pair that the action should gate/skip on
        std::string pv_name;
        std::string storage_name;
        unsigned storage_level_id;
        conditions_list[pv_storage_pair_id].lookupValue("data-space", pv_name);
        conditions_list[pv_storage_pair_id].lookupValue("storage", storage_name);
        storage_level_id = FindTargetStorageLevel(storage_name);
        conditions[pv_name] = storage_level_id;
        // std::cout << "action " << action_name << " gated on " << action_pv_name << std::endl;
      }
      compute_optimization_info[action_name] = conditions;
    }
    if (optimization_type == "action-gating"){
      action_gating_info_.compute_info = compute_optimization_info;
    } else {
      action_skipping_info_.compute_info = compute_optimization_info;
    }


  } else {
    // storage level action optimization (gating/skipping)
    unsigned storage_level_id = FindTargetStorageLevel(level_name);

    // parse for action optimization (gating/skipping)
    if (action_optimization_directive.exists("data-spaces")){

      PerStorageLevelActionOptimizationInfo per_storage_level_optimization_info;
      auto data_space_list = action_optimization_directive.lookup("data-spaces");
      assert(data_space_list.isList());

      for(unsigned pv = 0; pv < unsigned(data_space_list.getLength()); pv++){
      // go through the data spaces that have action optimizations specified
        if(data_space_list[pv].exists("actions")){
          PerDataSpaceActionOptimizationInfo data_space_optimization_info;
          std::string data_space_name;
          assert(data_space_list[pv].lookupValue("name", data_space_name));

          auto action_list = data_space_list[pv].lookup("actions");
          assert(action_list.isList());

          for(unsigned action_id = 0; action_id < unsigned(action_list.getLength()); action_id++)
          {
            // go through the  action optimizations specified for that specific data type
            std::string action_name;
            action_list[action_id].lookupValue("name", action_name);

            auto conditions_list = action_list[action_id].lookup("conditions");
            Conditions conditions;
            for (unsigned pv_storage_pair_id = 0; pv_storage_pair_id < unsigned(conditions_list.getLength()); pv_storage_pair_id++)
            {
              // go through the dataspace-storage pair that the action should gate/skip on
              std::string pv_name;
              std::string storage_name;
              unsigned storage_level_id;
              conditions_list[pv_storage_pair_id].lookupValue("data-space", pv_name);
              conditions_list[pv_storage_pair_id].lookupValue("storage", storage_name);
              storage_level_id = FindTargetStorageLevel(storage_name);
              conditions[pv_name] = storage_level_id;
              // std::cout << "action " << action_name << " gated on " << action_pv_name << std::endl;
            }

            data_space_optimization_info[action_name] = conditions;

          } // go through action list
          per_storage_level_optimization_info[data_space_name] = data_space_optimization_info;
        } // if exists action optimizations
      } // go through data-space list
      if (optimization_type == "action-gating"){
         action_gating_info_.storage_info[storage_level_id] = per_storage_level_optimization_info;
      } else { // action-skipping
         action_skipping_info_.storage_info[storage_level_id] = per_storage_level_optimization_info;
      }

    }
  }
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