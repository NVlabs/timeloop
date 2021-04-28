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


#pragma once

#include "workload/util/per-data-space.hpp"
#include "workload/shape-models/problem-shape.hpp"
#include "workload/format-models/metadata-format-factory.hpp"
#include "workload/format-models/metadata-format.hpp"

namespace sparse{

  //
  // data structures shared by action-gating and action-skipping optimization info
  //
  typedef std::string ActionName;
  struct Condition
  {
    std::map<std::string, unsigned> conditions; //dataspace-name, storage level id pairs
    std::string type;                           //among OR, AND, XOR
  };

  typedef std::map<ActionName, Condition> PerDataSpaceActionOptimizationInfo;
  // dataspace_id, per-dataspace-action-optimization-info
  typedef std::map<std::string, PerDataSpaceActionOptimizationInfo> PerStorageLevelActionOptimizationInfo;

  // storage_level_id, per_storage_level_gating_info
  typedef std::map<unsigned, PerStorageLevelActionOptimizationInfo> StorageActionOptimizationInfo;

  typedef std::map<ActionName, Condition> ComputeActionOptimizationInfo;

  //TODO: unify the gating and sikipping data structures
  //
  // data structure for action gating information
  //
  struct ActionGatingInfo{
    StorageActionOptimizationInfo storage_info;
    ComputeActionOptimizationInfo compute_info;
  };

  //
  // data structure for action skipping information
  //
  struct ActionSkippingInfo{
    StorageActionOptimizationInfo storage_info;
    ComputeActionOptimizationInfo compute_info;
  };

  //
  // data structures for compression information
  //


  struct PerDataSpaceCompressionInfo{

    bool tensor_compressed = false; // whether this tensor is a compressed tensor

    // user-defined order for applying ranks to loops
	// 0: inner to outer, i.e., if fewer non-trivial loops than supported ranks, outer ranks get spared
	// 1: outer to inner, i.e., if more non-trivial loops than supported ranks, inner ranks get spared
    bool rank_application_order = 0;

    // all of the vectors below should have the same length... which is the fiber tree depth
    std::vector<bool> rank_compressed; // if each rank is compressed
    std::vector<std::string> rank_formats; // each rank of the tensor should have metadata format
	std::vector<std::vector<std::set<problem::Shape::DimensionID>>> flattened_rankIDs; // rankIDs that can be flattened together
    std::vector<std::shared_ptr<problem::MetaDataFormat>> metadata_models; // pointers to metadata format objs
    double compression_rate; // not very useful yet, placeholder

    bool HasMetaData() const
    {
      if (tensor_compressed) return true; // compressed tensor must have metadata
      if (metadata_models.size() > 1) return true; // intermediate levels must have payloads
      if (metadata_models.size() == 1 && !metadata_models[0]->ImplicitMetadata()) return true; // single level with metadata
      return false; // single level default dense
    }

    bool FoundDimensionInFlatteningRule(std::uint64_t rank_id, problem::Shape::DimensionID dim_id,
										std::set<problem::Shape::DimensionID>& rule_item) const
	{
      for (auto iter = flattened_rankIDs[rank_id].begin(); iter != flattened_rankIDs[rank_id].end(); iter++)
	  {
        if (iter->find(dim_id) != iter->end())
		{
          rule_item = *iter;
          return true;
		}
	  }
      return false;
	}

  };


  typedef std::map<unsigned, PerDataSpaceCompressionInfo> PerStorageLevelCompressionInfo;

  // each data space (if compressed or has metadata) will have a corresponding metadata tree

  // storage_level_id, per_storage_level_gating_info
  struct CompressionInfo
  {
    std::map<unsigned, PerStorageLevelCompressionInfo> per_level_info_map;
    // masks for important hardware properties
    std::vector<problem::PerDataSpace<bool>> has_metadata_masks;
    std::vector<problem::PerDataSpace<bool>> compressed_masks;
    std::vector<bool> tile_partition_supported_masks;
    std::vector<bool> decompression_supported_masks;
    std::vector<bool> compression_supported_masks;
    bool all_ranks_default_dense;

    bool GetDataSpaceCompressionInfo(unsigned level, unsigned pv, PerDataSpaceCompressionInfo& info)
    {
      if (has_metadata_masks[level][pv])
      {
        info = per_level_info_map[level][pv];
        return true;
      }
      return false;
    }

    bool GetStorageLevelCompressionInfo(unsigned level, PerStorageLevelCompressionInfo& info)
    {
      if (per_level_info_map.find(level) != per_level_info_map.end())
      {
        info = per_level_info_map[level];
        return true;
      }
        return false;
    }

    bool GetDataSpaceCompressed(unsigned level, unsigned pv)
    {
      return compressed_masks[level][pv];
    }
  };

  //
  // aggregation of all sparse optimization related information
  //

  struct SparseOptimizationInfo{
	// various types of sparse optimizations
    ActionGatingInfo action_gating_info;
	ActionSkippingInfo action_skipping_info;
	CompressionInfo compression_info;
  };

} // namespace