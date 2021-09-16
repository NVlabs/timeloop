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

namespace sparse
{

//
// data structures shared by action-gating and action-skipping optimization info
//

enum ActionOptimizationType
{
  CONDITIONED_ON
  // Prepare for more types, e.g.,INTERSECTION
};

struct ConditionedOnOptimization
{
  problem::Shape::DataSpaceID target_dspace_id;
  std::vector<problem::Shape::DataSpaceID> condition_on_dspace_ids;
};

struct ActionOptimization
{
  ActionOptimizationType type;
  ConditionedOnOptimization cond_on_opt;
};

// group of choices, select the best one depending on mapping
typedef std::vector<ActionOptimization> GroupOfActionOptimization;

// a list of groups
typedef std::vector<GroupOfActionOptimization> PerStorageActionOptimization;


// storage_level_id, per_storage_level_gating_info
typedef std::map<unsigned, PerStorageActionOptimization> StorageActionOptimizationInfo;


// compute hardware support name, whether supported
typedef std::map<std::string, bool> ComputeOptimizationInfo;

//
// data structures for compression information
//

struct PerDataSpaceCompressionInfo
{

  bool tensor_compressed = false; // whether this tensor is a compressed tensor

  // user-defined order for applying ranks to loops
  // ture: inner to outer, i.e., if fewer non-trivial loops than supported ranks, outer ranks get spared
  // false: outer to inner, i.e., if more non-trivial loops than supported ranks, inner ranks get spared
  bool apply_rank_inner_to_outer = false;

  // all of the vectors below should have the same length... which is the fiber tree depth
  std::vector<bool> rank_compressed; // if each rank is compressed
  std::vector<bool> coordinates_implicit;
  std::vector<std::string> rank_formats; // each rank of the tensor should have metadata format
  std::vector<std::vector<std::vector< problem::Shape::FlattenedDimensionID>>> flattened_rankIDs; // rankIDs that can be flattened together
  std::vector <std::shared_ptr<problem::MetaDataFormat>> metadata_models; // pointers to metadata format objs
  double compression_rate; // not very useful yet, placeholder

  bool HasMetaData() const;
  bool ExistFlatteningRule(std::uint64_t rank_id) const;
  bool FoundDimensionInFlatteningRule(std::uint64_t rank_id, problem::Shape::FlattenedDimensionID dim_id,
                                      std::vector<problem::Shape::FlattenedDimensionID> &rule_item) const;
  problem::Shape::FlattenedDimensionID GetFlatteningRule(std::uint64_t rank_id, std::uint64_t rule_idx = 0) const;
};

typedef std::map<unsigned, PerDataSpaceCompressionInfo> PerStorageLevelCompressionInfo;

// each data space (if compressed or has metadata) will have a corresponding metadata tree

// storage_level_id, per_storage_level_gating_info
struct CompressionInfo
{
  std::map<unsigned, PerStorageLevelCompressionInfo> per_level_info_map;
  // masks for important hardware properties
  std::vector <problem::PerDataSpace<bool>> has_metadata_masks;
  std::vector <problem::PerDataSpace<bool>> compressed_masks;
  std::vector<bool> tile_partition_supported_masks;
  std::vector<bool> decompression_supported_masks;
  std::vector<bool> compression_supported_masks;
  bool all_ranks_default_dense;

  bool GetDataSpaceCompressionInfo(unsigned level, unsigned pv, PerDataSpaceCompressionInfo &info);
  bool GetStorageLevelCompressionInfo(unsigned level, PerStorageLevelCompressionInfo &info);
  bool GetDataSpaceCompressed(unsigned level, unsigned pv);
};

//
// aggregation of all sparse optimization related information
//

struct SparseOptimizationInfo
{
  // various types of sparse optimizations
  StorageActionOptimizationInfo action_gating_info;
  StorageActionOptimizationInfo action_skipping_info;
  StorageActionOptimizationInfo action_spatial_skipping_info;
  ComputeOptimizationInfo compute_optimization_info;
  CompressionInfo compression_info;
  bool no_optimization_applied;
};

} // namespace
