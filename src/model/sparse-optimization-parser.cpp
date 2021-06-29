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

namespace sparse
{

//
// Shared state.
//
ArchProperties arch_props_;

//
// Forward declarations
//
void Parse(config::CompoundConfigNode sparse_config, SparseOptimizationInfo &sparse_optimization_info);
void InitializeCompressionInfo(SparseOptimizationInfo &sparse_optimization_info);
void InitializeActionOptimizationInfo(SparseOptimizationInfo &sparse_optimization_info);
void ParseCompressionInfo(SparseOptimizationInfo &sparse_optimization_info,
                          const config::CompoundConfigNode &directive);
void ParsePerRankSpec(const config::CompoundConfigNode rank_specs,
                      PerDataSpaceCompressionInfo &per_data_space_compression_info);
unsigned FindTargetStorageLevel(std::string storage_level_name);
void ParseActionOptimizationInfo(SparseOptimizationInfo &sparse_optimization_info,
                                 const config::CompoundConfigNode& directive);
void ParseComputeOptimizationInfo(SparseOptimizationInfo &sparse_optimization_info,
                                  const config::CompoundConfigNode& directive);


//
// Parsing
//
SparseOptimizationInfo ParseAndConstruct(config::CompoundConfigNode sparse_config,
                                         model::Engine::Specs &arch_specs)
{
  arch_props_.Construct(arch_specs);
  SparseOptimizationInfo sparse_optimization_info;
  sparse_optimization_info.no_optimization_applied = true;

  InitializeCompressionInfo(sparse_optimization_info);
  InitializeActionOptimizationInfo(sparse_optimization_info);
  Parse(sparse_config, sparse_optimization_info);
  return sparse_optimization_info;
}

// highest level parse function
void Parse(config::CompoundConfigNode sparse_config,
           SparseOptimizationInfo &sparse_optimization_info)
{

  config::CompoundConfigNode opt_target_list;

  if (sparse_config.exists("targets"))
  {
    opt_target_list = sparse_config.lookup("targets");
    assert(opt_target_list.isList());
    for (int i = 0; i < opt_target_list.getLength(); i++)
    {
      // each element in the list represent a storage level's information
      auto directive = opt_target_list[i];
      if (directive.exists("action-optimization")
          || directive.exists("representation-format")
          || directive.exists("compute-optimization"))
      {
        if (directive.exists("action-optimization"))
        {
          ParseActionOptimizationInfo(sparse_optimization_info, directive);
          sparse_optimization_info.no_optimization_applied = false;
        }

        // parse for representation format
        if (directive.exists("representation-format"))
        {
          ParseCompressionInfo(sparse_optimization_info, directive);
          sparse_optimization_info.compression_info.all_ranks_default_dense = false;
          sparse_optimization_info.no_optimization_applied = false;
        }

        if (directive.exists("compute-optimization"))
        {
          ParseComputeOptimizationInfo(sparse_optimization_info, directive);
          sparse_optimization_info.no_optimization_applied = false;
        }
      }
      else
      {
        std::string level_name;
        directive.lookupValue("name", level_name);
        std::cout << "Warning: unrecognized sparse optimization optimization specification: " << level_name << std::endl;
      }
    }
  }
  std::cout << "Sparse optimization configuration complete." << std::endl;
}

void InitializeCompressionInfo(SparseOptimizationInfo &sparse_optimization_info)
{

  CompressionInfo compression_info;

  compression_info.all_ranks_default_dense = true;

  // initialize all mask structures to empty
  compression_info.has_metadata_masks = {};
  compression_info.compressed_masks = {};
  compression_info.decompression_supported_masks = {};
  compression_info.compression_supported_masks = {};
  compression_info.tile_partition_supported_masks = {};

  // set all storage levels to uncompressed
  for (unsigned storage_level_id = 0; storage_level_id < arch_props_.StorageLevels(); storage_level_id++)
  {
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
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      compression_info.has_metadata_masks[storage_level_id][pv] = false;
      compression_info.compressed_masks[storage_level_id][pv] = false;
    }
  }
  compression_info.per_level_info_map = {};
  sparse_optimization_info.compression_info = compression_info;
}

void InitializeActionOptimizationInfo(SparseOptimizationInfo &sparse_optimization_info)
{
  sparse_optimization_info.action_skipping_info = {};
  sparse_optimization_info.action_skipping_info = {};
  sparse_optimization_info.compute_optimization_info = {};
}

void ParsePerRankSpec(const config::CompoundConfigNode rank_specs,
                      PerDataSpaceCompressionInfo &per_data_space_compression_info)
{

  // 0) keyword check
  if (!rank_specs.exists("format"))
  {
    std::cout << "ERROR: data representation format specification: keyword \"format\" missing" << std::endl;
    exit(1);
  }

  // 1) construct metadata model objects
  auto metadata_specs = problem::MetaDataFormatFactory::ParseSpecs(rank_specs);
  auto metadata_model = problem::MetaDataFormatFactory::Construct(metadata_specs);
  per_data_space_compression_info.metadata_models.push_back(metadata_model);

  // 2) record the per-rank formats, whether each dimension is compressed, and user-defined flattening rules
  std::string lc_format = metadata_specs->Name();
  bool rank_compressed = metadata_model->RankCompressed();
  bool coordinates_implicit = metadata_model->CoordinatesImplicit();

  std::vector<std::vector<problem::Shape::FactorizedDimensionID>> per_level_flattened_rankIDs = {};

  if (rank_specs.exists("flattened-rankIDs"))
  {
    config::CompoundConfigNode list_of_rankID_list;
    list_of_rankID_list = rank_specs.lookup("flattened-rankIDs");

    for (auto id = 0; id < list_of_rankID_list.getLength(); id++)
    {
      std::vector <std::string> dim_name_list;
      list_of_rankID_list[id].getArrayValue(dim_name_list);
      std::vector<problem::Shape::FactorizedDimensionID> dim_id_list;

      for (auto iter = dim_name_list.begin(); iter != dim_name_list.end(); iter++)
      {
        auto id = problem::GetShape()->FactorizedDimensionNameToID.at(*iter);
        dim_id_list.push_back(id);
      }
      per_level_flattened_rankIDs.push_back(dim_id_list);
    }
  }

  per_data_space_compression_info.flattened_rankIDs.push_back(per_level_flattened_rankIDs);
  per_data_space_compression_info.rank_formats.push_back(lc_format);
  per_data_space_compression_info.rank_compressed.push_back(rank_compressed);
  per_data_space_compression_info.coordinates_implicit.push_back(coordinates_implicit);
  if (rank_compressed)
  {
    per_data_space_compression_info.tensor_compressed = true;
  }
}

// parse for compression info (storage only) of one directive
void ParseCompressionInfo(SparseOptimizationInfo &sparse_optimization_info,
                          const config::CompoundConfigNode& directive)
{

  std::string level_name;
  assert(directive.exists("name"));
  directive.lookupValue("name", level_name);

  auto &compression_info = sparse_optimization_info.compression_info;

  auto compression_directive = directive.lookup("representation-format");
  unsigned storage_level_id = FindTargetStorageLevel(level_name);

  if (compression_directive.exists("data-spaces"))
  {
    auto data_space_list = compression_directive.lookup("data-spaces");
    assert(data_space_list.isList());
    PerStorageLevelCompressionInfo per_storage_level_compression_info;

    // check if there is any compressed data types
    for (unsigned pv = 0; pv < unsigned(data_space_list.getLength()); pv++)
    {
      std::string data_space_name;
      data_space_list[pv].lookupValue("name", data_space_name);
      auto data_space_id = problem::GetShape()->DataSpaceNameToID.at(data_space_name);

      // for compressed data or uncompressed data with special metadata
      // for each rank: format and dimensions must be specified
      PerDataSpaceCompressionInfo per_data_space_compression_info;
      if (!data_space_list[pv].exists("ranks"))
      {
        std::cout << "keyword \"ranks\" missing in data representation specification: " << level_name << std::endl;
        exit(1);
      }

      // parse accordingly
      auto rank_spec_list = data_space_list[pv].lookup("ranks");
      // user spec has top rank on the top of the list (top rank has index 0 in list)
      // we reverse the order here to be little endian during parsing
      for (int r_id = rank_spec_list.getLength()-1; r_id >= 0; r_id--)
      {
        ParsePerRankSpec(rank_spec_list[r_id], per_data_space_compression_info);
      }

      // check whether data representation format supported
      // upper level compresssed lower level uncompressed will lead to partially compressed (sub) tensors
      // we currently just have a bool stating whether a tensor is fully compressed
      // TODO: per-rank bool to specify if a subtensor is compressed, or double in 0-1.0 stating how much is compressed
      bool lower_rank_uncompressed = per_data_space_compression_info.rank_compressed[0];
      for (int r_id = 1; r_id < rank_spec_list.getLength(); r_id++)
      {
        bool rank_compressed = per_data_space_compression_info.rank_compressed[r_id];
        if (lower_rank_uncompressed && rank_compressed)
        {
          std::cerr << "ERROR: data representation format not supported,"
                       "currently do not support lower rank uncompressed, upper rank compressed formats: "
                    << per_data_space_compression_info.rank_formats[r_id]
                    << std::endl;
          exit(1);
        }
      }

      // sanity check: all the info are pushed correctly
      assert(per_data_space_compression_info.rank_formats.size() ==
        per_data_space_compression_info.rank_compressed.size() &&
        per_data_space_compression_info.rank_formats.size() ==
          per_data_space_compression_info.metadata_models.size());

      //check for compression rate, default to fully compressed if the data is supposed to be compressed (according to metadata format)
      if (per_data_space_compression_info.tensor_compressed)
      {
        if (data_space_list[pv].exists("compression-rate"))
        {
          data_space_list[pv].lookupValue("compression-rate", per_data_space_compression_info.compression_rate);
        }
      } else
      { // uncompressed
        if (data_space_list[pv].exists("compression-rate"))
        {
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

void ParseComputeOptimizationInfo(SparseOptimizationInfo &sparse_optimization_info,
                                 const config::CompoundConfigNode& directive)
{
  std::string level_name;
  assert(directive.exists("name"));
  directive.lookupValue("name", level_name);
  assert(arch_props_.Specs().topology.GetArithmeticLevel()->name.Get() == level_name);
  ComputeOptimizationInfo compute_optimization_info = {};
  auto compute_opt_list = directive.lookup("compute-optimization");
  for (int i = 0; i < compute_opt_list.getLength(); i++)
  {
    std::string optimization_type;
    if (compute_opt_list[i].lookupValue("type", optimization_type))
    {
      if (optimization_type == "gate-on-zero-operand" || optimization_type == "gating")
      {
        compute_optimization_info["gate_on_zero_operand"] = true;
      } else if (optimization_type == "skip_on_not_aligned_operands" || optimization_type == "skipping")
      {
        compute_optimization_info["skip_on_not_aligned_operands"] = true;
      } else
      {
        std::cerr << "ERROR: compute optimization type not recognized: " << optimization_type << std::endl;
        assert(false);
      }
    }
  }
  sparse_optimization_info.compute_optimization_info = compute_optimization_info;
}


// parse for storage action optimization of one directive
void ParseActionOptimizationInfo(SparseOptimizationInfo& sparse_optimization_info,
                                 const config::CompoundConfigNode& directive)
{

  std::string level_name;
  assert(directive.exists("name"));
  directive.lookupValue("name", level_name);
  config::CompoundConfigNode optimization_list;
  optimization_list = directive.lookup("action-optimization");


  std::string optimization_type;
  PerStorageActionOptimization per_storage_action_optimization_skipping = {};
  PerStorageActionOptimization per_storage_action_optimization_gating = {};
  for (int id = 0; id < optimization_list.getLength(); id++)
  {
    optimization_list[id].lookupValue("type", optimization_type);

    GroupOfActionOptimization group = {};
    if (optimization_list[id].exists("options"))
    {
      // options are provided for sparse-analysis to choose from
      auto options_list = optimization_list[id].lookup("options");
      for (int choice = 0; choice < options_list.getLength(); choice++)
      {
        ActionOptimization opt;
        if (options_list[choice].exists("target") && options_list[choice].exists("condition-on"))
        {
          // optimize conditioned on type
          // parse for target dspace
          std::string target_dspace;
          options_list[choice].lookupValue("target", target_dspace);
          auto target_dspace_id = problem::GetShape()->DataSpaceNameToID.at(target_dspace);
          opt.cond_on_opt.target_dspace_id = target_dspace_id;

          // parse for condition on dspace
          auto condition_on_dspace_list = options_list[choice].lookup("condition-on");
          if (!condition_on_dspace_list.isArray())
          {
            std::cerr << "ERROR: sparse optimization spec invalid --"
                         " conditioned on dataspace(s) need to be specified as a list" << std::endl;
            exit(1);
          }
          std::vector <std::string> condition_on_dspaces;
          condition_on_dspace_list.getArrayValue(condition_on_dspaces);
          for (unsigned i = 0; i < condition_on_dspaces.size(); i++)
          {
            auto condition_dspace_id = problem::GetShape()->DataSpaceNameToID.at(condition_on_dspaces[i]);
            opt.cond_on_opt.condition_on_dspace_ids.push_back(condition_dspace_id);
          }

          opt.type = CONDITIONED_ON;
        } else
        {
          std::cerr << "ERROR: " << level_name << ": storage action optimization choice not recognized..." << std::endl;
          assert(false);
        }
        group.push_back(opt);
      } // end of group
    }
    else
    {
      // fixed optimization setup
      ActionOptimization opt;
      if (optimization_list[id].exists("target") && optimization_list[id].exists("condition-on"))
      {
        // optimize conditioned on type
        // parse for target dspace
        std::string target_dspace;
        optimization_list[id].lookupValue("target", target_dspace);
        auto target_dspace_id = problem::GetShape()->DataSpaceNameToID.at(target_dspace);
        opt.cond_on_opt.target_dspace_id = target_dspace_id;

        // parse for condition on dspace
        auto condition_on_dspace_list = optimization_list[id].lookup("condition-on");
        if (!condition_on_dspace_list.isArray())
        {
          std::cerr << "ERROR: sparse optimization spec invalid --"
                       " conditioned on dataspace(s) need to be specified as a list" << std::endl;
          exit(1);
        }
        std::vector <std::string> condition_on_dspaces;
        condition_on_dspace_list.getArrayValue(condition_on_dspaces);
        for (unsigned i = 0; i < condition_on_dspaces.size(); i++)
        {
          auto condition_dspace_id = problem::GetShape()->DataSpaceNameToID.at(condition_on_dspaces[i]);
          opt.cond_on_opt.condition_on_dspace_ids.push_back(condition_dspace_id);
        }

        opt.type = CONDITIONED_ON;
      } else
      {
        std::cerr << "ERROR: " << level_name << ": storage action optimization choice not recognized..." << std::endl;
        assert(false);
      }
      group.push_back(opt);
    }

    if (optimization_type == "gating")
    {
      per_storage_action_optimization_gating.push_back(group);
    } else if (optimization_type == "skipping")
    {
      per_storage_action_optimization_skipping.push_back(group);
    } else
    {
      std::cerr << "ERROR: " << level_name << ": storage action optimization type not recognized..." << std::endl;
      assert(false);
    } // end of type
  }

  unsigned cur_storage_level_id = FindTargetStorageLevel(level_name);
  sparse_optimization_info.action_skipping_info[cur_storage_level_id] = per_storage_action_optimization_skipping;
  sparse_optimization_info.action_gating_info[cur_storage_level_id] = per_storage_action_optimization_gating;
}

//
// FindTargetTilingLevel()
//
unsigned FindTargetStorageLevel(std::string storage_level_name)
{

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

  return storage_level_id;
}

} // namespace
