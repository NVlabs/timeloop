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

#include <cmath>
#include "sparse-analysis/state.hpp"

namespace sparse
{

//
// External-Facing Sparse Analysis API
//

// Fast format checking logic
bool CheckFormatModelsAndMapping(const tiling::NestOfCompoundMasks &masks,
                                 sparse::CompressionInfo &compression_info,
                                 const model::Topology::Specs &topology_specs,
                                 std::vector <model::EvalStatus> &eval_status,
                                 const bool break_on_failure);

 
// Heavy weight sparse analysis  
bool PerformSparseProcessing(problem::Workload *workload,
                             Mapping &mapping,
                             tiling::CompoundTileNest &compound_tile_nest,
                             SparseOptimizationInfo *sparse_optimization_info,
                             const model::Topology::Specs &topology_specs,
                             std::vector <model::EvalStatus> &eval_status,
                             const bool break_on_failure);


// 
// APIs For Various Sparse Analysis Analyzers
//

// Storage unit gating/skipping analyzer 
bool DefineStorageOptimizationImpact(SparseAnalysisState& state,
                                     tiling::CompoundTileNest& compound_tile_nest,
                                     const model::Topology::Specs& topology_specs,
                                     std::vector <model::EvalStatus>& eval_status,
                                     const bool break_on_failure);


// Representation analyzer
bool DefineCompressionFormatModels(SparseAnalysisState& state,
                                   tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                   const model::Topology::Specs& topology_specs,
                                   std::vector <model::EvalStatus>& eval_status,
                                   const bool break_on_failure);
void CalculateExpectedMetaDataAccesses(tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                       const model::Topology::Specs& topology_specs);
void CalculateExpectedOccupancy(tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                const model::Topology::Specs& topology_specs);

// Storage optimization and representation impact combiner
void CombineStorageOptimizationImpact(SparseAnalysisState& state,
                                      tiling::CompoundTileNest& compound_tile_nest,
                                      const model::Topology::Specs& topology_specs);

// Compute unit gating/skipping analyzer
void CalculateFineGrainedComputeAccesses2Operand(const SparseAnalysisState& state,
                                                 tiling::CompoundTileNest& compound_tile_nest);
void CalculateFineGrainedComputeAccesses(const SparseAnalysisState& state,
                                         tiling::CompoundTileNest& compound_tile_nest);


}
