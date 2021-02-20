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

#include "workload/problem-shape.hpp"



namespace sparse{

  //
  // data structures shared by action-gating and action-skipping optimization info
  //
  typedef std::string ActionName;
  typedef std::map<std::string, unsigned> Conditions; // <dataspace-name, storage-level-id> pairs
  typedef std::map<ActionName, Conditions> PerDataSpaceActionOptimizationInfo;
  typedef std::map<std::string, PerDataSpaceActionOptimizationInfo> PerStorageLevelActionOptimizationInfo;

  // storage_level_id, per_storage_level_gating_info
  typedef std::map<unsigned, PerStorageLevelActionOptimizationInfo> StorageActionOptimizationInfo;

  typedef std::map<ActionName, Conditions> ComputeActionOptimizationInfo;

  //
  // data structure for action gating information
  //
  struct ActionGatingInfo{
    StorageActionOptimizationInfo storage_info = {};
    ComputeActionOptimizationInfo compute_info = {};
  };

  //
  // data structure for action skipping information
  //
  struct ActionSkippingInfo{
    StorageActionOptimizationInfo storage_info = {};
    ComputeActionOptimizationInfo compute_info = {};
  };

  //
  // data structures for compression information
  //


  struct PerDataSpaceCompressionInfo{
    bool tensor_compressed; // whether this tensor is a compressed tensor
    // all of the vectors below should have the same length... which is the fiber tree depth
    std::vector<bool> rank_compressed={}; // if each rank is compressed
    std::vector<unsigned> rank_metadata={}; // each rank of the tensor should have metadata format, none for uncompressed
    std::vector<std::vector<problem::Shape::DimensionID>> rank_order={}; // rank order is only useful if the tensor is compressed, we allow random accesses to uncompressed data
    double compression_rate; // not very useful yet, placeholder

    // shortcut variables (for partial code integration)
    std::string metadata_format;
    bool compressed;
    std::vector<unsigned> rank0_list={};
    std::vector<unsigned> rank1_list={};



  };
  typedef std::map<std::string, PerDataSpaceCompressionInfo> PerStorageLevelCompressionInfo;
  // storage_level_id, per_storage_level_gating_info
  typedef std::map<unsigned, PerStorageLevelCompressionInfo> CompressionInfo;

  //
  // aggregation of all sparse optimization related information
  //

  struct SparseOptimizationInfo{
    ActionGatingInfo action_gating_info;
    ActionSkippingInfo action_skipping_info;
    CompressionInfo compression_info = {};
  };


} // namespace