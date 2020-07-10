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
  // data structures for action gating information
  //
	typedef std::string ActionName;
	typedef std::map<ActionName, std::vector<std::string>> PerDataSpaceActionGatingInfo;
	typedef std::map<std::string, PerDataSpaceActionGatingInfo> PerStorageLevelActionGatingInfo;

  // storage_level_id, per_storage_level_gating_info
	typedef std::map<unsigned, PerStorageLevelActionGatingInfo> StorageActionGatingInfo;

	typedef std::map<ActionName, std::vector<std::string>> ComputeActionGatingInfo;

	struct ActionGatingInfo{
    StorageActionGatingInfo storage_info = {};
    ComputeActionGatingInfo compute_info = {};
	};

  //
  // data structures for action skipping information
  //
  typedef std::map<ActionName, std::vector<std::string>> PerDataSpaceActionSkippingInfo;
  typedef std::map<std::string, PerDataSpaceActionSkippingInfo> PerStorageLevelActionSkippingInfo;
	typedef std::map<unsigned, PerStorageLevelActionSkippingInfo> StorageActionSkippingInfo;

	typedef std::map<ActionName, std::vector<std::string>> ComputeActionSkippingInfo;

	struct ActionSkippingInfo{
    StorageActionGatingInfo storage_info = {};
    ComputeActionGatingInfo compute_info = {};
	};

  //
  // data structures for compression information
  //
	typedef std::map<std::string, double> PerStorageLevelCompressionInfo;
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