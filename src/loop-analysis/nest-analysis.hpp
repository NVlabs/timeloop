/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <unordered_map>
#include <map>

#include "mapping/nest.hpp"
#include "workload/per-problem-dimension.hpp"
#include "nest-analysis-tile-info.hpp"

namespace analysis
{

class NestAnalysis
{
 private:
  // Cached copy of loop nest under evaluation (used for speedup).
  loop::Nest cached_nest;
  
  // Properties of the nest being analyzed (copied over during construction).
  std::vector<uint64_t> storage_tiling_boundaries_;

  // Live state.
  std::vector<analysis::LoopState> nest_state_;
  std::vector<int> indices_;
  std::uint64_t num_epochs_;
  
  // Identifies the spatial element
  // whose working set is currently being computed.
  // Dynamically updated by recursive calls.
  std::uint64_t spatial_id_;
  
  CompoundDataMovementNest working_sets_;
  ComputeInfo compute_info_;  
  CompoundComputeNest compute_info_sets_;

  // Memoization structures to accelerate IndexToOperationPoint()
  std::vector<problem::OperationPoint> vector_strides_;
  std::vector<problem::OperationPoint> mold_low_;
  std::vector<problem::OperationPoint> mold_high_;
  std::vector<problem::OperationPoint> mold_high_residual_;
  problem::OperationPoint cur_transform_;

  // per-level properties.
  std::vector<uint64_t> num_spatial_elems_;
  std::vector<uint64_t> spatial_fanouts_;

  // used to accelerate to IndexToOperationPoint computation
  // relevant only for master spatial levels.
  std::vector<uint64_t> horizontal_sizes_;
  std::vector<uint64_t> vertical_sizes_;

  // records if a level corresponds to the starting
  // point of a new storage tile.
  std::vector<bool> storage_boundary_level_;
  
  // architectural storage level corresponding to a given loop level.
  std::vector<unsigned> arch_storage_level_;

  // any level which is at the transition point from temporal to
  // spatial nests is a master spatial level.
  // there should be one such level between each set of
  // consecutive physical storage levels.
  std::vector<bool> master_spatial_level_;
  
  // true if the spatial elements at a given master spatial
  // level are connected by on-chip links.
  std::vector<bool> linked_spatial_level_;

  // The following data structures are used for skew calculation. We can
  // possibly optimize the implementation by holding the data in a few
  // OperationPoints instead of these maps. At each storage tiling
  // boundary, we initiate a new loop gist that captures the information
  // for all the loops in that block (i.e., before the next-inner
  // storage tiling boundary).
  struct LoopGist
  {
    int index = 0;
    int bound = 1;
  };
  // Hold the gists in a vector instead of a map. This is because trivial
  // unit-loops are omitted from the loop nest, which means the gist may
  // not be complete by the time we arrive at the innermost
  // FillSpatialDeltas in a loop block. Using a vector (along with the
  // default values in the struct above) allows us to pre-initialize all
  // loops. Just be careful to expand them in the Reset() call.
  std::vector<LoopGist> loop_gists_temporal_;
  std::vector<LoopGist> loop_gists_spatial_;
  // std::unordered_map<problem::Shape::FlattenedDimensionID, LoopGist> loop_gists_temporal_;
  // std::unordered_map<problem::Shape::FlattenedDimensionID, LoopGist> loop_gists_spatial_;

  // Storage level to fanout map.
  std::map<unsigned, std::uint64_t> fanoutX_map_; 
  std::map<unsigned, std::uint64_t> fanoutY_map_; 

  std::unordered_map<unsigned, loop::Nest::SkewDescriptor> skew_descriptors_;
  loop::Nest::SkewDescriptor* cur_skew_descriptor_ = nullptr;

  // Other state.

  bool working_sets_computed_ = false;
  bool imperfectly_factorized_ = false;

  problem::Workload* workload_ = nullptr;

  // Internal helper methods.
  void ComputeWorkingSets();

  void DetectImperfectFactorization();
  void InitializeNestProperties();
  void InitNumSpatialElems();
  void InitStorageBoundaries();
  void InitSpatialFanouts();
  void InitPerLevelDimScales();

  void InitializeLiveState();
  void CollectWorkingSets();

  problem::OperationPoint IndexToOperationPoint_(const std::vector<int>& indices) const;
  bool IsLastGlobalIteration_(int level, problem::Shape::FlattenedDimensionID dim) const;
  
  problem::OperationSpace ComputeDeltas(std::vector<analysis::LoopState>::reverse_iterator cur);

  void ComputeTemporalWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur,
                                 //problem::OperationSpace& point_set,
                                 analysis::ElementState& cur_state);
  void ComputeSpatialWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur);
  //problem::OperationSpace& point_set);

  void FillSpatialDeltas(std::vector<analysis::LoopState>::reverse_iterator cur,
                         std::unordered_map<std::uint64_t, problem::OperationSpace>& spatial_deltas,
                         //std::vector<problem::OperationSpace>& spatial_deltas,
                         //std::vector<bool>& valid_delta,
                         std::uint64_t base_index,
                         int depth = 0);

  std::uint64_t ApplySkew(std::uint64_t unskewed_index);

  void ComputeAccurateMulticastedAccesses(
      std::vector<analysis::LoopState>::reverse_iterator cur,
      const std::unordered_map<std::uint64_t, problem::OperationSpace>& spatial_deltas,
      std::set<std::pair<std::uint64_t, problem::Shape::DataSpaceID>>& unaccounted_delta,
      //const std::vector<problem::OperationSpace>& spatial_deltas,
      //std::vector<problem::PerDataSpace<bool>>& unaccounted_delta,
      problem::PerDataSpace<AccessStatMatrix>& access_stats);

  void ComputeNetworkLinkTransfers(
      std::vector<analysis::LoopState>::reverse_iterator cur,
      const std::unordered_map<std::uint64_t, problem::OperationSpace>& cur_spatial_deltas,
      std::set<std::pair<std::uint64_t, problem::Shape::DataSpaceID>>& unaccounted_delta,
      //const std::vector<problem::OperationSpace>& cur_spatial_deltas,
      //std::vector<problem::PerDataSpace<bool>>& unaccounted_delta,
      problem::PerDataSpace<std::uint64_t>& link_transfers);
 
 void ComputeDataDensity();


 public:  
  // API
  NestAnalysis();
  void Init(problem::Workload* wc, const loop::Nest* nest,
            std::map<unsigned, std::uint64_t> fanoutX_map,
            std::map<unsigned, std::uint64_t> fanoutY_map);
  void Reset();
 
  std::vector<problem::PerDataSpace<std::size_t>> GetWorkingSetSizes_LTW() const;

  CompoundDataMovementNest GetWorkingSets();
  CompoundComputeNest GetComputeInfo();
  problem::Workload* GetWorkload();
  

  // Serialization.
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0) 
  {
    if(version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(nest_state_);
      ar& boost::serialization::make_nvp("work_sets_",boost::serialization::make_array(working_sets_.data(),working_sets_.size()));
      ar& BOOST_SERIALIZATION_NVP(working_sets_computed_);
      // ar& BOOST_SERIALIZATION_NVP(compute_cycles_);
    }
  }

  friend std::ostream& operator << (std::ostream& out, const NestAnalysis& n);  
};

} // namespace analysis
