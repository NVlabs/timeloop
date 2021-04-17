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

#include "model/level.hpp"
#include "mapping/mapping.hpp"
#include "tiling-tile-info.hpp"
#include "model/sparse-optimization-info.hpp"
#include "model/topology.hpp"

namespace sparse {

struct SparseAnalysisState {

  // statically defined information
  sparse::SparseOptimizationInfo *sparse_optimization_info_ = nullptr;
  problem::Workload *workload_ = nullptr;
  Mapping mapping_;

  // live state
  std::vector <std::vector<problem::OperationSpace>> maxtile_point_sets_;
  std::vector <std::vector<loop::Descriptor>> complete_subnests_;
  std::vector <std::vector<bool>> trivial_nest_masks_;

  SparseAnalysisState(){}

  void Init(sparse::SparseOptimizationInfo* sparse_optimization_info,
			problem::Workload* workload,
			Mapping mapping);
  void Reset();
  void CollectCompletePointSetsAndSubnests();

  // Serialization.
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive &ar, const unsigned int version = 0) {
	if (version == 0) {
	  ar & BOOST_SERIALIZATION_NVP(sparse_optimization_info_);
	}
  }

  friend std::ostream &operator<<(std::ostream &out, const SparseAnalysisState &n);
};

//
// Sparse Analysis API
//
  bool PerformSparseProcessing(problem::Workload *workload,
							   Mapping &mapping,
							   tiling::CompoundDataMovementNest &compound_data_movement_nest,
							   SparseOptimizationInfo* sparse_optimization_info,
							   const model::Topology::Specs &topology_specs,
							   std::vector <model::EvalStatus> &eval_status,
							   const bool break_on_failure);

  bool CheckFormatModelsAndMapping(const tiling::NestOfCompoundMasks &masks,
								   sparse::CompressionInfo& compression_info,
								   const model::Topology::Specs &topology_specs,
								   std::vector <model::EvalStatus> &eval_status,
								   const bool break_on_failure);
}