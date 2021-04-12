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

#include <cctype>
#include <stdlib.h>
#include "sparse-optimization-parser.hpp"
#include "mapping/arch-properties.hpp"

namespace sparse {

//
// Shared state.
//
ArchProperties arch_props_;

//
// Forward declarations
//
void Parse(config::CompoundConfigNode sparse_config, SparseOptimizationInfo& sparse_optimization_info);
void InitializeCompressionInfo(SparseOptimizationInfo& sparse_optimization_info);
void InitializeActionOptimizationInfo(SparseOptimizationInfo& sparse_optimization_info);
void ParseCompressionInfo(SparseOptimizationInfo& sparse_optimization_info,
						  const config::CompoundConfigNode &directive);
void ParsePerRankSpec(const config::CompoundConfigNode rank_specs,
					  PerDataSpaceCompressionInfo& per_data_space_compression_info);
unsigned FindTargetStorageLevel(std::string storage_level_name);
void ParseActionOptimizationInfo(SparseOptimizationInfo& sparse_optimization_info,
								 const config::CompoundConfigNode directive,
								 const std::string optimization_type);

//
// Parsing
//
SparseOptimizationInfo ParseAndConstruct(config::CompoundConfigNode sparse_config,
										  model::Engine::Specs& arch_specs) {
  arch_props_.Construct(arch_specs);
  SparseOptimizationInfo sparse_optimization_info;

  InitializeCompressionInfo(sparse_optimization_info);
  InitializeActionOptimizationInfo(sparse_optimization_info);
  Parse(sparse_config, sparse_optimization_info);
  return sparse_optimization_info;
}

// highest level parse function
void Parse(config::CompoundConfigNode sparse_config,
		   SparseOptimizationInfo& sparse_optimization_info) {

  config::CompoundConfigNode opt_target_list;

  if (sparse_config.exists("targets")) {
	opt_target_list = sparse_config.lookup("targets");
	assert(opt_target_list.isList());
	for (int i = 0; i < opt_target_list.getLength(); i++) {
	  // each element in the list represent a storage level's information
	  auto directive = opt_target_list[i];
	  if (directive.exists("action-gating")) {
		ParseActionOptimizationInfo(sparse_optimization_info, directive, "action-gating");
	  }
	  if (directive.exists("action-skipping")) {
		ParseActionOptimizationInfo(sparse_optimization_info, directive, "action-skipping");
	  }
	  // parse for compression
	  if (directive.exists("compression")) {
		ParseCompressionInfo(sparse_optimization_info, directive);
		sparse_optimization_info.compression_info.all_ranks_default_dense = false;
	  }
	}
  }

  std::cout << "Sparse optimization configuration complete." << std::endl;
}

void InitializeCompressionInfo(SparseOptimizationInfo& sparse_optimization_info) {

  CompressionInfo compression_info;

  compression_info.all_ranks_default_dense = true;

  // initialize all mask structures to empty
  compression_info.has_metadata_masks = {};
  compression_info.compressed_masks = {};
  compression_info.decompression_supported_masks = {};
  compression_info.compression_supported_masks = {};
  compression_info.tile_partition_supported_masks = {};

  // set all storage levels to uncompressed
  for (unsigned storage_level_id = 0; storage_level_id < arch_props_.StorageLevels(); storage_level_id++) {
	auto storage_level_specs = arch_props_.Specs().topology.GetStorageLevel(storage_level_id);

	// get arch related info
	bool tile_partition_supported = storage_level_specs->tile_partition_supported.Get();
	assert(tile_partition_supported == false); //TODO: implement online tile partition related logic
	bool decompression_supported = storage_level_specs->decompression_supported.Get();
	bool compression_supported = storage_level_specs->compression_supported.Get();

	// initial/populate masks
	problem::PerDataSpace<bool> temp;
	compression_info.has_metadata_masks.push_back(temp);
	compression_info.compressed_masks.push_back(temp);
	compression_info.decompression_supported_masks.push_back(decompression_supported);
	compression_info.compression_supported_masks.push_back(compression_supported);
	compression_info.tile_partition_supported_masks.push_back(tile_partition_supported);
	for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++) {
	  compression_info.has_metadata_masks[storage_level_id][pv] = false;
	  compression_info.compressed_masks[storage_level_id][pv] = false;
	}
  }
  compression_info.per_level_info_map = {};
  sparse_optimization_info.compression_info = compression_info;
}

void InitializeActionOptimizationInfo(SparseOptimizationInfo& sparse_optimization_info) {

  sparse_optimization_info.action_skipping_info.compute_info = {};
  sparse_optimization_info.action_skipping_info.storage_info = {};

  sparse_optimization_info.action_gating_info.compute_info = {};
  sparse_optimization_info.action_gating_info.storage_info = {};
}

void ParsePerRankSpec(const config::CompoundConfigNode rank_specs,
					  PerDataSpaceCompressionInfo& per_data_space_compression_info) {

  // 0) keyword check
  if (!rank_specs.exists("format")) {
	std::cout << "compression specification: keyword \"format\" missing in compression spec" << std::endl;
	exit(1);
  }

  // 1) construct metadata model objects
  auto metadata_specs = problem::MetaDataFormatFactory::ParseSpecs(rank_specs);
  auto metadata_model = problem::MetaDataFormatFactory::Construct(metadata_specs);
  per_data_space_compression_info.metadata_models.push_back(metadata_model);

  // 2) record the per-rank formats, whether each dimension is compressed, and user-defined flattening rules
  std::string lc_format = metadata_specs->Name();
  bool rank_compressed = metadata_specs->RankCompressed();

  std::vector <std::set<problem::Shape::DimensionID>> per_level_flattened_rankIDs = {};
  if (rank_specs.exists("flattened_rankIDs")) {
	auto list_of_rankID_list = rank_specs.lookup("flattened_rankIDs");
	for (auto id = 0; id < list_of_rankID_list.getLength(); id++) {
	  std::vector <std::string> dim_name_list;
	  list_of_rankID_list[id].getArrayValue(dim_name_list);
	  std::set <problem::Shape::DimensionID> dim_id_set;

	  for (auto iter = dim_name_list.begin(); iter != dim_name_list.end(); iter++) {
		auto id = problem::GetShape()->DimensionNameToID.at(*iter);
		dim_id_set.insert(id);
	  }
	  per_level_flattened_rankIDs.push_back(dim_id_set);
	}
  }

  per_data_space_compression_info.flattened_rankIDs.push_back(per_level_flattened_rankIDs);
  per_data_space_compression_info.rank_formats.push_back(lc_format);
  per_data_space_compression_info.rank_compressed.push_back(rank_compressed);
  if (rank_compressed) { per_data_space_compression_info.tensor_compressed = true; }

}

// parse for compression info (storage only) of one directive
void ParseCompressionInfo(SparseOptimizationInfo& sparse_optimization_info,
						  const config::CompoundConfigNode &directive) {

  std::string level_name;
  assert(directive.exists("name"));
  directive.lookupValue("name", level_name);

  auto& compression_info = sparse_optimization_info.compression_info;

  auto compression_directive = directive.lookup("compression");
  unsigned storage_level_id = FindTargetStorageLevel(level_name);

  if (compression_directive.exists("data-spaces")) {
	auto data_space_list = compression_directive.lookup("data-spaces");
	assert(data_space_list.isList());
	PerStorageLevelCompressionInfo per_storage_level_compression_info;

	// check if there is any compressed data types
	for (unsigned pv = 0; pv < unsigned(data_space_list.getLength()); pv++) {
	  std::string data_space_name;
	  data_space_list[pv].lookupValue("name", data_space_name);
	  auto data_space_id = problem::GetShape()->DataSpaceNameToID.at(data_space_name);

	  // for compressed data or uncompressed data with special metadata
	  // for each rank: format and dimensions must be specified
	  PerDataSpaceCompressionInfo per_data_space_compression_info;
	  if (!data_space_list[pv].exists("ranks")) {
		std::cout << "keyword \"ranks\" missing in compression specification: " << level_name << std::endl;
		exit(1);
	  }

	  // parse accordingly
	  auto rank_spec_list = data_space_list[pv].lookup("ranks");
	  for (auto r_id = 0; r_id < rank_spec_list.getLength(); r_id++) {
		ParsePerRankSpec(rank_spec_list[r_id], per_data_space_compression_info);
	  }

	  // sanity check: all the info are pushed correctly
	  assert(per_data_space_compression_info.rank_formats.size() ==
		  per_data_space_compression_info.rank_compressed.size() &&
		  per_data_space_compression_info.rank_formats.size() ==
			  per_data_space_compression_info.metadata_models.size());

	  //check for compression rate, default to fully compressed if the data is supposed to be compressed (according to metadata format)
	  if (per_data_space_compression_info.tensor_compressed) {
		if (data_space_list[pv].exists("compression_rate")) {
		  data_space_list[pv].lookupValue("compression_rate", per_data_space_compression_info.compression_rate);
		}
	  } else { // uncompressed
		if (data_space_list[pv].exists("compression_rate")) {
		  std::cout << " cannot have compression rate for uncompressed data" << std::endl;
		  exit(1);
		}
	  }
	  // update masks
	  compression_info.has_metadata_masks[storage_level_id][data_space_id] =
		  per_data_space_compression_info.HasMetaData();
	  compression_info.compressed_masks[storage_level_id][data_space_id] =
		  per_data_space_compression_info.tensor_compressed;

	  per_storage_level_compression_info[data_space_id] = per_data_space_compression_info;
	}
	compression_info.per_level_info_map[storage_level_id] = per_storage_level_compression_info;
  } // if there is compression specification in terms of data spaces
}


// parse for action gating info (storage and compute) of one directive
void ParseActionOptimizationInfo(SparseOptimizationInfo& sparse_optimization_info,
	                             const config::CompoundConfigNode directive,
								 const std::string optimization_type) {

  std::string level_name;
  assert(directive.exists("name"));
  directive.lookupValue("name", level_name);

  config::CompoundConfigNode action_optimization_directive;
  if (optimization_type == "action-gating") {
	action_optimization_directive = directive.lookup("action-gating");
  } else {
	action_optimization_directive = directive.lookup("action-skipping");
  }

  if (arch_props_.Specs().topology.GetArithmeticLevel()->name.Get() == level_name) {
	// compute level gating optimization
	auto action_list = action_optimization_directive.lookup("actions");
	assert(action_list.isList());
	PerDataSpaceActionOptimizationInfo compute_optimization_info;

	for (int action_id = 0; action_id < action_list.getLength(); action_id++) {

	  std::string action_name;
	  action_list[action_id].lookupValue("name", action_name);
	  assert(action_name == "compute"); // we only recognize compute for MACs

	  Condition condition;
	  if (action_list[action_id].exists("type")) { action_list[action_id].lookupValue("type", condition.type); }
	  else { condition.type = "OR"; }
	  auto conditions_list = action_list[action_id].lookup("conditions");
	  for (unsigned pv_storage_pair_id = 0; pv_storage_pair_id < unsigned(conditions_list.getLength());
		   pv_storage_pair_id++) {
		// go through the dataspace-storage pair that the action should gate/skip on
		std::string pv_name;
		std::string storage_name;
		unsigned storage_id;
		if (!conditions_list[pv_storage_pair_id].lookupValue("data-space", pv_name)) exit(1);
		if (!conditions_list[pv_storage_pair_id].lookupValue("storage", storage_name)) exit(1);
		storage_id = FindTargetStorageLevel(storage_name);
		condition.conditions[pv_name] = storage_id;
	  }
	  compute_optimization_info[action_name] = condition;
	}
	if (optimization_type == "action-gating") {
	  sparse_optimization_info.action_gating_info.compute_info = compute_optimization_info;
	} else {
	  sparse_optimization_info.action_skipping_info.compute_info = compute_optimization_info;
	}

  } else {
	// storage level action optimization (gating/skipping)
	unsigned cur_storage_level_id = FindTargetStorageLevel(level_name);

	// parse for action optimization (gating/skipping)
	if (action_optimization_directive.exists("data-spaces")) {

	  PerStorageLevelActionOptimizationInfo per_storage_level_optimization_info;
	  auto data_space_list = action_optimization_directive.lookup("data-spaces");
	  assert(data_space_list.isList());

	  for (unsigned pv = 0; pv < unsigned(data_space_list.getLength()); pv++) {
		// go through the data spaces that have action optimizations specified
		if (data_space_list[pv].exists("actions")) {
		  PerDataSpaceActionOptimizationInfo data_space_optimization_info;
		  std::string cur_data_space_name;
		  assert(data_space_list[pv].lookupValue("name", cur_data_space_name));

		  auto action_list = data_space_list[pv].lookup("actions");
		  assert(action_list.isList());

		  for (unsigned action_id = 0; action_id < unsigned(action_list.getLength()); action_id++) {
			// go through the  action optimizations specified for that specific data type
			std::string action_name;
			action_list[action_id].lookupValue("name", action_name);

			Condition condition;
			if (action_list[action_id].exists("type")) { action_list[action_id].lookupValue("type", condition.type); }
			else { condition.type = "OR"; }
			auto conditions_list = action_list[action_id].lookup("conditions");
			for (unsigned pv_storage_pair_id = 0; pv_storage_pair_id < unsigned(conditions_list.getLength());
				 pv_storage_pair_id++) {
			  // go through the dataspace-storage pair that the action should gate/skip on
			  std::string pv_name;
			  std::string storage_name;
			  unsigned storage_id;
			  if (!conditions_list[pv_storage_pair_id].lookupValue("data-space", pv_name)) {
				std::cout << " missing \"data-space\" spec for skipping/gating optimization specification" << std::endl;
				exit(1);
			  }
			  if (!conditions_list[pv_storage_pair_id].lookupValue("storage", storage_name)) {
				std::cout << "  missing \"storage\" spec for skipping/gating optimization specification" << std::endl;
				exit(1);
			  }
			  storage_id = FindTargetStorageLevel(storage_name);
			  condition.conditions[pv_name] = storage_id;

			}
			data_space_optimization_info[action_name] = condition;
		  } // go through action list
		  per_storage_level_optimization_info[cur_data_space_name] = data_space_optimization_info;
		} // if exists action optimizations
	  } // go through data-space list
	  if (optimization_type == "action-gating") {
		sparse_optimization_info.action_gating_info.storage_info[cur_storage_level_id] =
			                                                              per_storage_level_optimization_info;
	  } else { // action-skipping
		sparse_optimization_info.action_skipping_info.storage_info[cur_storage_level_id] =
			                                                                per_storage_level_optimization_info;
	  }
	}
  }
}

//
// FindTargetTilingLevel()
//
unsigned FindTargetStorageLevel(std::string storage_level_name) {

  auto num_storage_levels = arch_props_.StorageLevels();

  //
  // Find the target storage level using its name
  //

  unsigned storage_level_id;
  // Find this name within the storage hierarchy in the arch specs.
  for (storage_level_id = 0; storage_level_id < num_storage_levels; storage_level_id++) {
	if (arch_props_.Specs().topology.GetStorageLevel(storage_level_id)->level_name == storage_level_name)
	  break;
  }

  if (storage_level_id == num_storage_levels) {
	std::cerr << "ERROR: target storage level not found: " << storage_level_name << std::endl;
	exit(1);
  }

  return storage_level_id;
}

} // namespace