/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "loop-analysis/coordinate-space-tile-info.hpp"
#include "sparse-analysis/state.hpp"
#include "mapping/loop.hpp"

namespace sparse
{

//
// SparseAnalysisState Function Implementations
//
SparseAnalysisState::SparseAnalysisState() :
    mapping_(nullptr)
{
}

bool SparseAnalysisState::Init(sparse::SparseOptimizationInfo* sparse_optimization_info,
                               problem::Workload* workload,
                               Mapping mapping,
                               std::uint64_t num_storage_levels)
{
  bool sparse_analysis_needed = false;

  if (sparse_optimization_info->no_optimization_applied) return sparse_analysis_needed;
  else sparse_analysis_needed = true;

  sparse_optimization_info_ = sparse_optimization_info;
  workload_ = workload;
  mapping_ = mapping;
  num_storage_levels_ = num_storage_levels;
  Reset();

  return sparse_analysis_needed;
}

void SparseAnalysisState::Reset()
{
  maxtile_molds_high_ = {};
  complete_subnests_ = {};
  trivial_nest_masks_ = {};
  prob_explicitly_optimized_read_ = {};
  prob_explicitly_spatially_optimized_read_ = {};
  c_operand_densities_ = {};
  c_intersection_dims_ = {};
  storage_gs_saf_ = {};
  cond_on_mold_highs_ = {};
 
  // by default, no propagation impact
  for (DataSpaceID pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      storage_gs_saf_[pv] = false;
    }
  }

  // by default, no explicit optimization applied
  dspace_optimization_masks_ = {{"gate", {}}, {"skip", {}}, {"spatial_skip", {}}};

  for (unsigned l = 0; l < num_storage_levels_; l++)
  {
    dspace_optimization_masks_["gate"].push_back({});
    dspace_optimization_masks_["skip"].push_back({});
    dspace_optimization_masks_["spatial_skip"].push_back({});
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      dspace_optimization_masks_["gate"][l][pv] = false;
      dspace_optimization_masks_["skip"][l][pv] = false;
      dspace_optimization_masks_["spatial_skip"][l][pv] = false;
    }
  }

}

void SparseAnalysisState::CollectCompletePointSetsAndSubnests()
{
  problem::OperationPoint origin;
  problem::OperationPoint dimension_sizes;
  dimension_sizes.IncrementAllDimensions(); // initialize to { 1, 1, 1... }

  maxtile_molds_high_.push_back({});
  complete_subnests_.push_back({});
  trivial_nest_masks_.push_back({});

  unsigned tiling_level = 0;
  auto& loops = mapping_.complete_loop_nest.loops;
  for (unsigned loop_level = 0; loop_level < loops.size(); loop_level++)
  {
    auto& loop = loops[loop_level];
    auto factor = ceil((loop.end - loop.start) / loop.stride);
    dimension_sizes[loop.dimension] *= factor;

    // origin gives us the low corner (inclusive) of the operation space.
    // dimension_sizes gives the high corner (exclusive) of the operation space.
    // We need the inclusive high corner to build the operation space. See
    // OperationSpace constructor for details.
    problem::OperationPoint high = dimension_sizes;
    high.IncrementAllDimensions(-1);
    maxtile_molds_high_[tiling_level].push_back(high);
    complete_subnests_[tiling_level].push_back(loop);
    trivial_nest_masks_[tiling_level].push_back(factor == 1);

    if (loop_level == mapping_.complete_loop_nest.storage_tiling_boundaries.at(tiling_level))
    {
      maxtile_molds_high_.push_back({});
      complete_subnests_.push_back({});
      trivial_nest_masks_.push_back({});
      tiling_level++;
    }
  }

  if (!workload_->IsWorkloadTensorSizesSet())
  {
    problem::OperationPoint high = dimension_sizes;
    high.IncrementAllDimensions(-1);
    problem::OperationSpace maxtile(workload_, origin, high);
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
      workload_->SetWorkloadTensorSize(problem::Shape::DataSpaceID(pvi), maxtile.GetDataSpace(pvi));
    workload_->AllTensorsSet();
  }
}

} // namespace
