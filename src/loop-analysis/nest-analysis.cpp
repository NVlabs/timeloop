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

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <barvinok/isl.h>
#include <isl/aff.h>
#include <isl/cpp.h>
#include <isl/set.h>
#include <isl/space.h>

// FIXME: num_spatial_elems, spatial_fanouts, replication_factor etc. are
//        all maintained across datatypes. They should be per-datatype at
//        this analytical level of abstraction. It's only when we get to
//        the architecture level that hardware may map multiple datatypes
//        on to the same storage and network structures.

// FIXME: Spatial model is X/Y only. Fortunately, generalizing this isn't
//        too difficult (as far as this module is concerned) since it's
//        limited to the ComputeNetworkLinkTransfers() function.

#include "util/misc.hpp"
#include "isl-wrapper/ctx-manager.hpp"
#include "isl-wrapper/isl-functions.hpp"
#include "loop-analysis/isl-ir.hpp"
#include "loop-analysis/mapping-to-isl/mapping-to-isl.hpp"
#include "loop-analysis/nest-analysis.hpp"
#include "loop-analysis/spatial-analysis.hpp"
#include "loop-analysis/temporal-analysis.hpp"
#include "loop-analysis/isl-to-legacy-adaptor.hpp"
#include "mapping/fused-mapping.hpp"

bool gTerminateEval = false;

bool gEnableLinkTransfers =
  (getenv("TIMELOOP_DISABLE_LINK_TRANSFERS") == NULL) ||
  (strcmp(getenv("TIMELOOP_DISABLE_LINK_TRANSFERS"), "0") == 0);
bool gEnableToroidalLinks =
  (getenv("TIMELOOP_ENABLE_TOROIDAL_LINKS") != NULL) &&
  (strcmp(getenv("TIMELOOP_ENABLE_TOROIDAL_LINKS"), "0") != 0);
bool gExtrapolateUniformTemporal =
  (getenv("TIMELOOP_DISABLE_TEMPORAL_EXTRAPOLATION") == NULL) ||
  (strcmp(getenv("TIMELOOP_DISABLE_TEMPORAL_EXTRAPOLATION"), "0") == 0);
bool gExtrapolateUniformSpatial =
  (getenv("TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION") == NULL) ||
  (strcmp(getenv("TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION"), "0") == 0);
bool gDisableFirstElementOnlySpatialExtrapolation =
  (getenv("TIMELOOP_DISABLE_FIRST_ELEMENT_ONLY_SPATIAL_EXTRAPOLATION") != NULL) &&
  (strcmp(getenv("TIMELOOP_DISABLE_FIRST_ELEMENT_ONLY_SPATIAL_EXTRAPOLATION"), "0") != 0);
bool gEnableTracing =
  (getenv("TIMELOOP_ENABLE_TRACING") != NULL) &&
  (strcmp(getenv("TIMELOOP_ENABLE_TRACING"), "0") != 0);
bool gRunLastIteration =
  (getenv("TIMELOOP_RUN_LAST_ITERATION") != NULL) &&
  (strcmp(getenv("TIMELOOP_RUN_LAST_ITERATION"), "0") != 0);
bool gUseIslAnalysis =
  (getenv("TIMELOOP_USE_ISL") != NULL) &&
  (strcmp(getenv("TIMELOOP_USE_ISL"), "0") != 0);
bool gDumpNestAnalysisResult =
  (getenv("TIMELOOP_DUMP_NEST_ANALYSIS_RESULT") != NULL) &&
  (strcmp(getenv("TIMELOOP_DUMP_NEST_ANALYSIS_RESULT"), "0") != 0);

// Flattening => Multi-AAHRs
// => Can't use per-AAHR reset-on-stride-change logic
// => Have to run last temporal iteration (because tile residual that carries
//    over from iteration 1 back to iteration 0 is incorrect).
bool gResetOnStrideChange = false;

// Alternative cycle count computation hack for imperfect factorization.
bool gEnableImperfectCycleCount = false;

namespace analysis
{


NestAnalysis::NestAnalysis()
{
}

void NestAnalysis::Init(problem::Workload* wc, const loop::Nest* nest,
                        std::map<unsigned, std::uint64_t> fanoutX_map,
                        std::map<unsigned, std::uint64_t> fanoutY_map)
{
  ASSERT(nest != NULL);
  ASSERT(wc != NULL);

  ASSERT(fanoutX_map.size() == nest->storage_tiling_boundaries.size());
  ASSERT(fanoutY_map.size() == nest->storage_tiling_boundaries.size());

  workload_ = wc;

  if (working_sets_computed_ && cached_nest == *nest)
  {
    // We've already worked on an identical nest before.
  }
  else
  {
    Reset();
    cached_nest = *nest;

    // Copy over everything we need from the nest.
    storage_tiling_boundaries_ = nest->storage_tiling_boundaries;
    packed_skew_descriptors_ = nest->skew_descriptors;
    no_link_transfer_ = nest->no_link_transfer;
    no_multicast_ = nest->no_multicast;
    no_temporal_reuse_ = nest->no_temporal_reuse;

    physical_fanoutX_ = fanoutX_map;
    physical_fanoutY_ = fanoutY_map;

    // Construct nest_state_.
    for (auto descriptor: nest->loops)
    {
      analysis::LoopState cur;
      if (nest_state_.size() == 0)
      {
        cur.level = 0;
      }
      else
      {
        cur.level = nest_state_.back().level + 1;
      }
      cur.descriptor = descriptor;
      nest_state_.push_back(cur);    
    }
  }

  gResetOnStrideChange = !problem::GetShape()->UsesFlattening;
}

//
// Reset(): torpedo everything.
//
void NestAnalysis::Reset()
{
  storage_tiling_boundaries_.clear();
  
  nest_state_.clear();
  indices_.clear();
  num_epochs_ = 0;

  spatial_id_ = 0;
  for (auto& tile_nest: working_sets_)
  {
    tile_nest.clear();
  }

  vector_strides_.clear();
  cur_transform_ = problem::OperationPoint();

  num_spatial_elems_.clear();
  logical_fanouts_.clear();

  logical_fanoutX_.clear();
  logical_fanoutY_.clear();

  storage_boundary_level_.clear();
  master_spatial_level_.clear();
  linked_spatial_level_.clear();

  working_sets_computed_ = false;
  imperfectly_factorized_ = false;
  gEnableImperfectCycleCount = false;

  // compute_info_.Reset();
  compute_info_.clear();
  compute_info_sets_.clear();

  loop_gists_temporal_.clear();
  loop_gists_spatial_.clear();
  loop_gists_temporal_.resize(problem::GetShape()->NumFlattenedDimensions);
  loop_gists_spatial_.resize(problem::GetShape()->NumFlattenedDimensions);

  skew_descriptors_.clear();
  cur_skew_descriptor_ = nullptr;

  no_multicast_.clear();
  no_link_transfer_.clear();
  no_temporal_reuse_.clear();
}

// Ugly function for pre-checking capacity fits before running the heavyweight
// ComputeWorkingSets() algorithm. FIXME: Integrate with ComputeWorkingSets().
std::vector<problem::PerDataSpace<std::size_t>>
NestAnalysis::GetWorkingSetSizes_LTW() const
{
  std::vector<problem::PerDataSpace<std::size_t>> working_set_sizes;

  problem::OperationPoint origin;
  problem::OperationPoint dimension_sizes;
  dimension_sizes.IncrementAllDimensions(); // initialize to { 1, 1, 1... }

  unsigned tiling_level = 0;
  for (unsigned loop_level = 0; loop_level < nest_state_.size(); loop_level++)
  {
    auto & loop = nest_state_.at(loop_level).descriptor;
    ASSERT(loop.stride == 1);
    dimension_sizes[loop.dimension] *= loop.end;
        
    if (loop_level == storage_tiling_boundaries_.at(tiling_level))
    {
      // origin gives us the low corner (inclusive) of the operation space.
      // dimension_sizes gives the high corner (exclusive) of the operation space.
      // We need the inclusive high corner to build the operation space. See
      // OperationSpace constructor for details.
      problem::OperationPoint high = dimension_sizes;
      high.IncrementAllDimensions(-1);
      problem::OperationSpace maxtile(workload_, origin, high);
      working_set_sizes.push_back(maxtile.GetSizes());
      tiling_level++;
    }
  }

  ASSERT(working_set_sizes.size() == storage_tiling_boundaries_.size());
  
  // set the workload_tensor sizes that were not available during parsing stage
  // this step should only happen once to a workload
  if (! workload_->IsWorkloadTensorSizesSet()){
    problem::OperationPoint high = dimension_sizes;
    high.IncrementAllDimensions(-1);
    problem::OperationSpace maxtile(workload_, origin, high);
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
      workload_->SetWorkloadTensorSize(problem::Shape::DataSpaceID(pvi), maxtile.GetDataSpace(pvi));
    workload_->AllTensorsSet();
  }
  return working_set_sizes;
}

problem::PerDataSpace<std::vector<analysis::DataMovementInfo>>
NestAnalysis::GetWorkingSets()
{
  if (!working_sets_computed_)
  {
    ComputeWorkingSets();
  }
  ASSERT(working_sets_computed_);
  return working_sets_;
}

analysis::CompoundComputeNest NestAnalysis::GetComputeInfo()
{
  if (!working_sets_computed_)
  {
    ComputeWorkingSets();
  }
  ASSERT(working_sets_computed_);
  return compute_info_sets_;
}

problem::Workload* NestAnalysis::GetWorkload(){
  return workload_;
}

std::ostream& operator << (std::ostream& out, const NestAnalysis& n)
{
  for (auto cur = n.nest_state_.rbegin(); cur != n.nest_state_.rend(); cur++)
  {
    cur->descriptor.Print(out, false);
  }
  out << std::endl;
  return out;
}

void NestAnalysis::ComputeWorkingSets()
{
  if (nest_state_.size() != 0)
  {
    InitializeNestProperties();
    InitializeLiveState();
    DetectImperfectFactorization();
    if (!gUseIslAnalysis)
    {
      // Recursive call starting from the last element of the list.
      num_epochs_ = 1;
      ComputeDeltas(nest_state_.rbegin());
      CollectWorkingSets();
    }
  }

  if (gUseIslAnalysis)
  {
    auto occupancies =
      analysis::OccupanciesFromMapping(cached_nest, *workload_);

    auto reuse_analysis_input = ReuseAnalysisInput(occupancies);
    auto legacy_output =
      GenerateLegacyNestAnalysisOutput(
        ReuseAnalysis(reuse_analysis_input),
        nest_state_,
        storage_tiling_boundaries_,
        master_spatial_level_,
        storage_boundary_level_,
        num_spatial_elems_,
        logical_fanouts_,
        *workload_
      );

    compute_info_sets_ = legacy_output.first;
    working_sets_ = legacy_output.second;
  }

  if (gDumpNestAnalysisResult)
  {
    int dspace = 0;
    for (const auto& data_movement_nest : working_sets_)
    {
      std::cout << "Dspace: " << std::to_string(dspace) << std::endl;
      int idx = 0;
      for (const auto& data_movement : data_movement_nest)
      {
        std::cout << "  Idx: " << std::to_string(idx) << std::endl;
        std::cout << "    Dist. multicast: " << std::to_string(data_movement.distributed_multicast) << std::endl;
        std::cout << "    Size: " << std::to_string(data_movement.size) << std::endl;
        std::cout << "    Access stats: " << std::endl;
        for (const auto& [key, access_stat] : data_movement.access_stats.stats)
        {
          std::cout << "      key: (" << std::to_string(key.first) << "," << std::to_string(key.second) << ")\n";
          std::cout << "      accesses: " << std::to_string(access_stat.accesses) << std::endl;
          std::cout << "      hops: " << std::to_string(access_stat.hops) << std::endl;
        }
        std::cout << "    Link transfers: " << std::to_string(data_movement.link_transfers) << std::endl;
        std::cout << "    Replication fact.: " << std::to_string(data_movement.replication_factor) << std::endl;
        std::cout << "    Fanout: " << std::to_string(data_movement.fanout) << std::endl;
        std::cout << "    Dist. fanout: " << std::to_string(data_movement.distributed_fanout) << std::endl;
        std::cout << "    On boundary: " << std::to_string(data_movement.is_on_storage_boundary) << std::endl;
        std::cout << "    Is master spatial: " << std::to_string(data_movement.is_master_spatial) << std::endl;
        ++idx;
      }
      dspace++;
    }
  }

  // Done.
  working_sets_computed_ = true;
}

// Internal helper methods

void NestAnalysis::DetectImperfectFactorization()
{
  for (auto cur = nest_state_.rbegin(); cur != nest_state_.rend(); cur++)
  {
    if (cur->descriptor.end != cur->descriptor.residual_end)
    {
      imperfectly_factorized_ = true;
      gEnableImperfectCycleCount = true;
      break;
    }
  }

  if (imperfectly_factorized_ && gUseIslAnalysis)
  {
    throw std::runtime_error(
      "Imperfect factorization not supported in ISL analysis"
    );
  }
}

void NestAnalysis::InitializeNestProperties()
{
  InitNumSpatialElems();
  InitStorageBoundaries();
  InitSpatialFanouts();
  InitPerLevelDimScales();
}

void NestAnalysis::InitializeLiveState()
{
  indices_.resize(nest_state_.size());
  spatial_id_ = 0;
  
  // compute_info_.Reset();
  compute_info_.clear();
  compute_info_sets_.clear();

  for (auto loop = nest_state_.rbegin(); loop != nest_state_.rend(); loop++)
  {
    // if (!loop::IsSpatial(loop->descriptor.spacetime_dimension) ||
    //     master_spatial_level_[loop->level])
    // {
    //   // we don't need live state for non-master spatial levels
    //   loop->live_state.resize(num_spatial_elems_[loop->level]);
    // }

    // for (auto& it : loop->live_state)
    // {
    //   it.Reset();
    //   if (linked_spatial_level_[loop->level])
    //   {
    //     // Restore this line if MAX_TIME_LAPSE > 1. it.prev_spatial_deltas.resize(analysis::ElementState::MAX_TIME_LAPSE);
    //     // // for (auto& elem : it.prev_spatial_deltas)
    //     // // {
    //     // //   elem.resize(logical_fanouts_[loop->level]);
    //     // // }
    //   }
    // }
    loop->live_state.clear();
  }
}

// Helpers.
std::ostream& operator << (std::ostream& out, const std::vector<unsigned>& v)
{
  out << "/";
  for (auto& x: v) out << x << "/";
  return out;
}

void PrintStamp(const std::vector<unsigned>& v)
{
  std::cout << "/";
  for (auto it = v.begin(); it != v.end(); it++)
  {
    std::cout << *it << "/";
  }
}

void NestAnalysis::CollectWorkingSets()
{
  // Collect the data we want to return. Transpose the max_size_ and accesses_
  // matrix, pack them into an array of vectors and return.
  for (auto& cur : nest_state_)
  {
    // All spatial levels that are not a master-spatial level are not valid
    bool valid_level = !loop::IsSpatial(cur.descriptor.spacetime_dimension) || master_spatial_level_[cur.level];
    if (valid_level)
    {
      // Contains the collected state for this level.
      analysis::ElementState condensed_state(*workload_);
      for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
      {
        // Sanity check: All elements in a given level should
        // have similar working sets, accesses etc.
        // TODO Can we leverage this assertion to avoid redundant simulation
        // by only simulating one spatial element per level?
        if (!gExtrapolateUniformSpatial)
        {
          // FIXME: aggregate stats.
          // for (std::uint64_t i = 1; i < cur.live_state.size(); i++)
          // {
          //   ASSERT(cur.live_state[i].access_stats[pv] ==
          //          cur.live_state[i - 1].access_stats[pv]);
          //   ASSERT(cur.live_state[i].max_size[pv] ==
          //          cur.live_state[i - 1].max_size[pv]);
          //   ASSERT(cur.live_state[i].link_transfers[pv] ==
          //          cur.live_state[i - 1].link_transfers[pv]);
          // }
        }

        // // Since, all elements have the same properties, use the properties
        // // of the first element to build condensed_state
        // const uint64_t REPR_ELEM_ID = 0;  // representative element id.
        // condensed_state.access_stats[pv] =
        //     cur.live_state[REPR_ELEM_ID].access_stats[pv];
        // condensed_state.max_size[pv] =
        //     cur.live_state[REPR_ELEM_ID].max_size[pv];
        // condensed_state.link_transfers[pv] =
        //     cur.live_state[REPR_ELEM_ID].link_transfers[pv];
        // // condensed_state.data_densities[pv] =
        // //     cur.live_state[REPR_ELEM_ID].data_densities[pv];

        // We have 3 choices:
        // (1) Sample the stats from one spatial instance and report that as the
        //     per-instance stats.
        // (2) Aggregate stats from all spatial instances.
        // (3) Report all stats to the microarchitecture model.
        //
        // Per-instance stats will be amplified back into all-instances stats
        // later by the microarchitecture model. Each approach has a challenge.
        // (1) is simply inaccurate. Quantization artifacts from (2) due to
        // non-uniform stats across spatial instances will lead to under-
        // counting of overall stats when the model multiplies the stats
        // with the number of spatial instances. Using floating-point will
        // address this, but will require changes to the post-processing code.
        // (3) is probably the best approach but will require significant
        // reworking of the post-processing and microarchitecture code.

        // bool first = true;
        // for (auto& state: cur.live_state)
        // {
        //   if (first)
        //   {
        //     condensed_state.access_stats[pv] = state.second.access_stats[pv];
        //     condensed_state.max_size[pv] = state.second.max_size[pv];
        //     condensed_state.link_transfers[pv] = state.second.link_transfers[pv];
        //     first = false;
        //     // std::cout << "s";
        //     // PrintStamp(state.first);
        //     // std::cout << " store size " << condensed_state.max_size[pv] << std::endl;
        //     break;
        //   }
        // }

        for (auto& state: cur.live_state)
        {
          condensed_state.access_stats[pv].Accumulate(state.second.access_stats[pv]);
          condensed_state.max_size[pv] += state.second.max_size[pv];
          condensed_state.link_transfers[pv] += state.second.link_transfers[pv];
        }
        std::uint64_t num_sampled_instances = cur.live_state.size();
        condensed_state.access_stats[pv].Divide(num_sampled_instances);
        condensed_state.max_size[pv] /= num_sampled_instances;
        condensed_state.link_transfers[pv] /= num_sampled_instances;
      }

      // Build the subnest corresponding to this level.
      // We need a vector of nests because a master spatial level's
      // subnest should include the nests of the slave spatial levels.
      // This is very useful for debugging purposes.
      std::vector<loop::Descriptor> subnest;
      subnest.push_back(cur.descriptor);
      if (master_spatial_level_[cur.level])
      {
        int l = cur.level - 1;
        while (l >= 0 && loop::IsSpatial(nest_state_[l].descriptor.spacetime_dimension))
        {
          subnest.push_back(nest_state_[l].descriptor);
          l--;
        }
        std::reverse(subnest.begin(), subnest.end());
      }

      // Transfer data from condensed_state to working_sets_
      for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
      {
        DataMovementInfo tile;
        tile.size                   = condensed_state.max_size[pv];
        // tile.partition_size         = 0; // will be set later.
        tile.access_stats           = condensed_state.access_stats[pv];
        // tile.fills                  = 0; // will be set later
        // tile.content_accesses       = tile.GetTotalAccesses();
        tile.link_transfers         = condensed_state.link_transfers[pv];
        tile.subnest                = subnest;
        tile.replication_factor     = num_spatial_elems_[cur.level];
        tile.fanout                 = logical_fanouts_[cur.level];
        tile.is_on_storage_boundary = storage_boundary_level_[cur.level];
        tile.is_master_spatial      = master_spatial_level_[cur.level];
        // tile.tile_density           = condensed_state.data_densities[pv];
        working_sets_[pv].push_back(tile);
      }
    } // if (valid_level)
  } // for (nest)

  // Extract body data from innermost spatial level.
  bool innermost_level_compute_info_collected = false; 
  
  uint64_t max_temporal_iterations = 1;
  for (auto& cur : nest_state_)
  {
    if (!loop::IsSpatial(cur.descriptor.spacetime_dimension))
      max_temporal_iterations *= cur.descriptor.end;
  }

  for (auto& cur : nest_state_)
  {
    // All spatial levels that are not a master-spatial level are not valid
    bool valid_level = !loop::IsSpatial(cur.descriptor.spacetime_dimension) || master_spatial_level_[cur.level];
    if (valid_level)
    {
      if (!innermost_level_compute_info_collected)
      {
        analysis::ComputeInfo compute_info;
        compute_info.replication_factor = num_spatial_elems_[cur.level] * logical_fanouts_[cur.level];

        double avg_accesses = 0;
        for (auto& info: compute_info_)
        {
          avg_accesses += info.second.accesses;
        }
        avg_accesses /= compute_info_.size();

        compute_info.accesses = avg_accesses;
        compute_info.max_temporal_iterations = max_temporal_iterations;
        compute_info_sets_.push_back(compute_info);
        innermost_level_compute_info_collected = true;
      }
      else
      { // if not the inner most level
        analysis::ComputeInfo compute_info;
        compute_info.replication_factor = 0;
        compute_info.accesses = 0;
        compute_info_sets_.push_back(compute_info);
      } // inner most
    } // valid level
  }

}

// All but last vector.
std::vector<unsigned> AllButLast(const std::vector<unsigned>& v)
{
  ASSERT(v.size() >= 1);
  std::vector<unsigned> retval;
  for (auto it = v.begin(); it != v.end()-1; it++)
    retval.push_back(*it);
  return retval;
}

// Print space-time-stamp
void NestAnalysis::PrintSpaceTimeStamp()
{
  std::cout << "t/";
  for (auto time_it = time_stamp_.begin(); time_it != time_stamp_.end()-1; time_it++)
  {
    std::cout <<  *time_it << "/";
  }
  std::cout << " s/";
  for (auto space_it = space_stamp_.begin(); space_it != space_stamp_.end()-1; space_it++)
  {
    std::cout <<  *space_it << "/";
  }
}

// Delta computation (recursive call).
// Returns the delta between the working set of the
// previous iteration and the current iteration of the current level.
problem::OperationSpace NestAnalysis::ComputeDeltas(std::vector<analysis::LoopState>::reverse_iterator cur)
{
  ASSERT(cur != nest_state_.rend());
  //ASSERT(spatial_id_ < cur->live_state.size());

  if (gTerminateEval)
  {
    throw std::runtime_error("terminated");
  }

  int level = cur->level;

  // Before we begin -- if this is a storage tiling boundary, save the loop
  // gists and create a new gist set.
  std::vector<LoopGist> saved_loop_gists_temporal;
  std::vector<LoopGist> saved_loop_gists_spatial;

  loop::Nest::SkewDescriptor* saved_skew_descriptor = nullptr;

  if (storage_boundary_level_[level])
  {
    saved_loop_gists_temporal = loop_gists_temporal_;
    saved_loop_gists_spatial = loop_gists_spatial_;
    
    loop_gists_temporal_.clear();
    loop_gists_spatial_.clear();
    loop_gists_temporal_.resize(problem::GetShape()->NumFlattenedDimensions);
    loop_gists_spatial_.resize(problem::GetShape()->NumFlattenedDimensions);

    saved_skew_descriptor = cur_skew_descriptor_;
    cur_skew_descriptor_ = nullptr;
    auto skew_it = skew_descriptors_.find(level);
    if (skew_it != skew_descriptors_.end())
      cur_skew_descriptor_ = &skew_it->second;

    space_stamp_.push_back(0);
    time_stamp_.push_back(0);
  }

  // Get access to/allocate state. Be careful! We must do this after the
  // space_stamp has been potentially advanced.

  // std::cout << "CD level " << cur->level << " about to access live state with stamp: ";
  // PrintStamp(AllButLast(space_stamp_));
  // std::cout << std::endl;

  auto cur_state_it = cur->live_state.emplace(AllButLast(space_stamp_),
                                              ElementState(*workload_)).first;
  auto& cur_state = cur_state_it->second;

  // std::cout << "CD level " << cur->level << " potentially created live state entry. Full state:\n";
  // for (auto& state: cur->live_state)
  // {
  //   std::cout << "  ";
  //   PrintStamp(state.first);
  //   std::cout << "->" << state.second.max_size.at(0) << std::endl;
  // }

  //
  // Step I: Compute Accesses.
  //

  if (loop::IsSpatial(cur->descriptor.spacetime_dimension))
  {
    ComputeSpatialWorkingSet(cur);
  }
  else
  {
    ComputeTemporalWorkingSet(cur, cur_state);
  }

  //
  // Step II - Compute Working Set.
  //

  // The point set for this invocation. Note that we do *not* initialize this to
  // the last-seen state at the end of the prior invocation. Doing so causes the
  // state at this level to grow indefinitely, which isn't what we're trying to
  // model. The responsibility of this level is to supply all the deltas
  // demanded by the next-inner level for this invocation.
  problem::OperationSpace point_set = GetCurrentWorkingSet(cur);


  // Record the maximum point set size ever seen across all invocations
  // of this level.
  // Need to be done only for levels which will map to physical storage levels
  // after we run collapseTiles.
  // Also need to do this for master spatial levels in order to calculate
  // partition size later.
  if (storage_boundary_level_[level] || master_spatial_level_[level])
  {
    auto sizes = point_set.GetSizes();
    std::transform(sizes.begin(), sizes.end(), cur_state.max_size.begin(),
                   cur_state.max_size.begin(),
                   [](std::size_t x, std::size_t y) { return std::max(x, y); });
    // if (level == 1)
    // {
    //   std::cout << "sizes: " << sizes.at(0) << " " << sizes.at(1) << " " << sizes.at(2) << std::endl;
    //   std::cout << "max sizes: " << cur_state.max_size.at(0) << " " << cur_state.max_size.at(1) << " " << cur_state.max_size.at(2) << std::endl;
    // }
  }

  // Trace.
  if (gEnableTracing && storage_boundary_level_[level])
  {
    assert(time_stamp_.size() == space_stamp_.size());
    assert(storage_tiling_boundaries_.size() - arch_storage_level_.at(level) == time_stamp_.size());
    std::string indent = "";
    for (unsigned i = 0; i < storage_tiling_boundaries_.size() - arch_storage_level_.at(level); i++)
    {
      indent += "  ";
    }
    std::cout << indent;
    PrintSpaceTimeStamp();
    std::cout << " " << point_set << std::endl;
  }

  // Calculate delta to send up to caller.
#define NEW_RESET_ON_STRIDE_CHANGE_APPROACH
#ifdef NEW_RESET_ON_STRIDE_CHANGE_APPROACH
  // Hardware pattern generators may be unable to generate complicated patterns
  // arising from residuals left over from ancestor (grandparent-upwards) loop
  // iterations. With the specific exception of an entire tile staying resident
  // across ancestor iterations, we apply a simple heuristic to detect this
  // behavior and simply discard any residual state if the tile shape changes
  // the magnitude or direction of its stride.
  problem::PerDataSpace<bool> no_temporal_reuse;
  for(unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    no_temporal_reuse[pv] = false;
  }
  if(no_temporal_reuse_.find(arch_storage_level_[cur->level]) != no_temporal_reuse_.end())
  {
    no_temporal_reuse = no_temporal_reuse_[arch_storage_level_[cur->level]];
  }
  if (gResetOnStrideChange)
    point_set.SaveAndSubtractIfSameStride(cur_state.last_point_set, cur_state.last_translations, no_temporal_reuse);
  else
    point_set.SaveAndSubtract(cur_state.last_point_set, no_temporal_reuse);
  auto& delta = point_set;
#else
  problem::OperationSpace delta(workload_);
  delta = point_set - cur_state.last_point_set;
  cur_state.last_point_set = point_set;
#endif

  // Restore loop gist sets.
  if (storage_boundary_level_[level])
  {
    space_stamp_.pop_back();
    time_stamp_.pop_back();
    loop_gists_temporal_ = saved_loop_gists_temporal;
    loop_gists_spatial_ = saved_loop_gists_spatial;
    cur_skew_descriptor_ = saved_skew_descriptor;
  }

  return delta;
}

void NestAnalysis::ComputeTemporalWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur,
                                             analysis::ElementState& cur_state)
{
  // We do two things in this function: (a) calculate the size of the temporal
  // working set for this level, and (b) calculate the number of accesses to
  // this level from the inner level.
  //
  // We used to do both these tasks by accumulating the deltas returned by
  // recursive calls to inner nesting levels. That was problematic for task (a)
  // because inner levels can sometimes buffer data across iterations of *this*
  // level, which sometimes causes the union of the deltas propagated to this
  // level to form a fractured polyhedral space. Note that this fracturing is
  // fine in terms of calculating *accesses* to this level (b), since it
  // reflects filtering.
  //
  // To address this, we first attempted to restrict gradient direction changes
  // during delta computation. However, this only captures a subset of scenarios.
  // It also affects (b), but that is fine because most hardware pattern
  // generators are probably going to be unable to generate patterns that can
  // keep state alive through gradient direction changes.
  //
  // The solution we are now adopting is to use delta accumulation only for (b)
  // and to use an alternative tactic for (a). For certain problem shapes (such
  // as CNN's axis-aligned hyper-rectangles), we can trivially calculate working
  // sets by considering only the corner points in the problem sub-space walked
  // by the subnest starting from this level down. We assume that loops are
  // always ascending (FIXME: check for this during loop construction).

  int level = cur->level;

  bool dump = false; // (level >= 4);

  int end = IsLastGlobalIteration_(level+1, cur->descriptor.dimension) ?
    cur->descriptor.residual_end : cur->descriptor.end;

  // First, update loop gist. FIXME: handle base!=0, stride!=1.
  ASSERT(cur->descriptor.start == 0);
  ASSERT(cur->descriptor.stride == 1);
  loop_gists_temporal_[cur->descriptor.dimension] = { 0, end };
  
  //
  // Step II: Compute Accesses by accumulating deltas returned by inner levels.
  //

  std::uint64_t num_iterations = 1 + ((end - 1 - cur->descriptor.start) /
                                      cur->descriptor.stride);

  if (level == 0) // base
  {
    auto body_iterations = num_iterations * num_epochs_;

    // If all spatial indices were uniform then we can simply increment the
    // compute count at a representative spatial ID (e.g., 0) with the present
    // value of num_epochs_. That's what this commented-out code used to do.
    // However, we now handle non-uniformity due to several reasons, which is
    // why we need to update the compute count of a specific skewed spatial ID.

    // if (spatial_id_ == 0)
    // {
    //   compute_info_.accesses += body_iterations;
    // }
    compute_info_[AllButLast(space_stamp_)].accesses += body_iterations;

    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      // Set scatter factor (otherwise it will stay at 0 for temporal levels).
      std::uint64_t scatter_factor = 1;
      std::uint64_t multicast_factor = 1;

      auto& access_stats = cur_state.access_stats[pv](multicast_factor, scatter_factor);
      access_stats.accesses += body_iterations;

      // Set cumulative hops for temporal levels.
      access_stats.hops = 0.0;
    }
  }
  else // recurse
  {
    std::vector<problem::PerDataSpace<std::size_t>> temporal_delta_sizes;
    std::vector<std::uint64_t> temporal_delta_scale;

    bool run_last_iteration = imperfectly_factorized_ || problem::GetShape()->UsesFlattening || gRunLastIteration;
    bool run_second_last_iteration = imperfectly_factorized_ && run_last_iteration;

    if (gExtrapolateUniformTemporal && !disable_temporal_extrapolation_.at(level))
    {
      // What we would like to do is to *NOT* iterate through the entire loop
      // for this level, but instead fire iterations #0, #1 and #last, and
      // extrapolate the remainder based on the result of iteration #1.

      // Iteration #last is only required for accurate partition size tracking.
      // Otherwise, we reset the point set on any gradient change, and so
      // tracking the point set for the #last iteration is not needed.

      // Note that this entire approach will break if there is any irregularity
      // in working-set movement along the loop (e.g., a modulus in the index
      // expression).

      int dim = int(cur->descriptor.dimension);
      int scale = vector_strides_[level][dim];
      auto saved_transform = cur_transform_[dim];

      // Iteration #0.
      indices_[level] = cur->descriptor.start;
      loop_gists_temporal_.at(dim).index = indices_[level];
        
      if (num_iterations >= 1)
      {
        // Invoke next (inner) loop level.
        ++cur;
        auto temporal_delta = ComputeDeltas(cur);
        --cur;

        temporal_delta_sizes.push_back(temporal_delta.GetSizes());
        temporal_delta_scale.push_back(1);
        cur_transform_[dim] += scale;

        indices_[level] += cur->descriptor.stride;
        loop_gists_temporal_.at(dim).index = indices_[level];

        if (storage_boundary_level_[level-1] || master_spatial_level_[level-1])
          (time_stamp_.back())++;
      }

      // Iterations #1 through #last-1/last.
      if ((run_second_last_iteration && num_iterations >= 4) ||
          (run_last_iteration && !run_second_last_iteration && num_iterations >= 3) ||
          (!run_last_iteration && num_iterations >= 2))
      {
        // Invoke next (inner) loop level, scaling up the number of epochs
        // by the number of virtual iterations we want to simulate.
        std::uint64_t virtual_iterations =
          run_last_iteration ? num_iterations - 2 : num_iterations - 1;

        // Run one fewer iteration for imperfect factor support
        if (run_second_last_iteration)
            virtual_iterations = virtual_iterations - 1;

        auto saved_epochs = num_epochs_;
        num_epochs_ *= virtual_iterations;

        ++cur;
        auto temporal_delta = ComputeDeltas(cur);
        --cur;

        num_epochs_ = saved_epochs;

        temporal_delta_sizes.push_back(temporal_delta.GetSizes());
        temporal_delta_scale.push_back(virtual_iterations);

        cur_transform_[dim] += (scale * virtual_iterations);

        indices_[level] += (cur->descriptor.stride * virtual_iterations);
        loop_gists_temporal_.at(dim).index = indices_[level];

        if (storage_boundary_level_[level-1] || master_spatial_level_[level-1])
          time_stamp_.back() += virtual_iterations;
      }

      // Iteration # second last to find delta for imperfect factors
      if (run_second_last_iteration && num_iterations >= 3)
      {
        // Invoke next (inner) loop level.
        ++cur;
        auto temporal_delta = ComputeDeltas(cur);
        --cur;

        if (num_iterations >= 4)
        {
          temporal_delta_scale.back()++;
        } else {
          temporal_delta_sizes.push_back(temporal_delta.GetSizes());
          temporal_delta_scale.push_back(1);
        }

        cur_transform_[dim] += scale;

        indices_[level] += cur->descriptor.stride;
        loop_gists_temporal_.at(dim).index = indices_[level];

        if (storage_boundary_level_[level-1] || master_spatial_level_[level-1])
          (time_stamp_.back())++;
      }

      // Iteration #last.
      if (run_last_iteration && num_iterations >= 2)
      {
        // Invoke next (inner) loop level.
        ++cur;
        auto temporal_delta = ComputeDeltas(cur);
        --cur;

        // If we ran the virtual-iteration logic above, we shouldn't actually
        // use this returned delta, because we will receive the delta between
        // iteration #2 and #last. Instead, we just re-use the last delta by
        // increasing the #virtual iterations (scale) by 1.
        if (!run_second_last_iteration && num_iterations >= 3)
        {
          temporal_delta_scale.back()++;
        }
        else
        {
          temporal_delta_sizes.push_back(temporal_delta.GetSizes());
          temporal_delta_scale.push_back(1);
          cur_transform_[dim] += scale;
        }
      
        indices_[level] += cur->descriptor.stride;        
        loop_gists_temporal_.at(dim).index = indices_[level];

        if (storage_boundary_level_[level-1] || master_spatial_level_[level-1])
          (time_stamp_.back())++;
      }

      cur_transform_[dim] = saved_transform;
    }
    else // not gExtrapolateUniformTemporal
    {
      int dim = int(cur->descriptor.dimension);
      int scale = vector_strides_[level][dim];

      auto saved_transform = cur_transform_[dim];

      for (indices_[level] = cur->descriptor.start;
           indices_[level] < end;
           indices_[level] += cur->descriptor.stride)
      {
        loop_gists_temporal_.at(dim).index = indices_[level];
        
        // Invoke next (inner) loop level.
        ++cur;
        auto temporal_delta = ComputeDeltas(cur);
        --cur;

        temporal_delta_sizes.push_back(temporal_delta.GetSizes());
        temporal_delta_scale.push_back(1);

        cur_transform_[dim] += scale;

        if (storage_boundary_level_[level-1] || master_spatial_level_[level-1])
          (time_stamp_.back())++;
      }

      cur_transform_[dim] = saved_transform;
    } // gExtrapolateUniformTemporal
    
    if (dump)
    {
      std::cout << "-------\n";
      std::cout << "LEVEL " << level << std::endl;
      std::cout << "-------\n";
    }

    if (storage_boundary_level_[level - 1])
    {
      // Track accesses for only those levels that are relevant
      // in the final analysis after CollapseTiles.
      problem::PerDataSpace<std::size_t> final_delta_sizes;
      final_delta_sizes.fill(0);

      auto num_deltas = temporal_delta_sizes.size();
      for (unsigned i = 0; i < num_deltas; i++)
      {
        for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
        {
          final_delta_sizes[pv] += (temporal_delta_sizes[i][pv] * temporal_delta_scale[i]);
        }
      }

      for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
      {
        // Set scatter factor (otherwise it will stay at 0 for temporal levels).
        std::uint64_t scatter_factor = 1;
        std::uint64_t multicast_factor = 1;

        auto& access_stats = cur_state.access_stats[pv](multicast_factor, scatter_factor);
        access_stats.accesses += final_delta_sizes[pv] * num_epochs_;

        // Set cumulative hops for temporal levels.
        access_stats.hops = 0.0;

        // Update delta histogram. Hypothesis is we only need to do this for temporal levels.
        cur_state.delta_histograms[pv][final_delta_sizes[pv]] += num_epochs_;
        
      } // for (datatype)
    } // storage boundary
    
  } // level > 0

}

void NestAnalysis::ComputeSpatialWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur)
{
  int level = cur->level;
  ASSERT(master_spatial_level_[level]);

  //
  // Step II: Compute Spatial Deltas, etc.
  //

  std::uint64_t num_spatial_elems = logical_fanouts_[level];
  spatial_id_ *= num_spatial_elems;

  // Deltas needed by each of the spatial elements.
  // This array will be filled by recursive calls.
  // This used to be a dense array but is now a map
  // because spatial skews may end up filling it in a discontiguous manner.
  std::unordered_map<std::uint64_t, problem::OperationSpace> spatial_deltas;
  std::unordered_map<std::uint64_t, std::uint64_t> skew_table;

  FillSpatialDeltas(cur, spatial_deltas, skew_table,
                    0,   // base_index,
                    0,   // depth,
                    0,   // extrapolation_stride
                    cur);// extrapolation_level  
  
  // Check if the expected number of spatial_deltas was updated by
  // recursive calls.
  ASSERT(spatial_deltas.size() <= num_spatial_elems);
  // The above assertion used to be ==, but that condition may not hold if the
  // mapping uses imperfect factorization in a spatial loop.

  // Restore spatial_id_ to original value.
  spatial_id_ /= num_spatial_elems;

  // Records whether we have accounted for each delta
  // (in each problem dimension) either through
  // 1) Link transfers within current level
  // 2) Multicasted or non-multicasted transfers from previous level

  // Previously, we would first attempt to capture deltas via link
  // transfers. For all other residual deltas, we could compute
  // multicast access factors (1 = unicast). Unfortunately, that
  // led to awkward multicast patterns if deltas that *could* have
  // been multicast were captured via link-transfers.
  // New approach: First calculate multicasts. Then, if using link
  // transfers completely obliterates access to a producer level,
  // use those link transfers only.

  problem::PerDataSpace<std::unordered_set<std::uint64_t>> unaccounted_delta;
  for (auto& delta: spatial_deltas)
  {
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
      unaccounted_delta[pv].insert(delta.first);
  }

  // std::vector<problem::PerDataSpace<bool>> unaccounted_delta;
  // unaccounted_delta.resize(num_spatial_elems);
  // for (uint64_t i = 0; i < num_spatial_elems; i++)
  // {
  //   unaccounted_delta[i].fill(true);
  // }

  // auto& cur_state = cur->live_state[spatial_id_];
  //  auto& accesses = nest_state_[cur->level].live_state[spatial_id_].accesses;

  // std::cout << "CSWS level " << level << " about to access live state with space_stamp: ";
  // PrintStamp(AllButLast(space_stamp_));
  // std::cout << std::endl;

  auto cur_state_it =
      nest_state_[cur->level].live_state.emplace(AllButLast(space_stamp_),
                                                 ElementState(*workload_)).first;
  auto& cur_state = cur_state_it->second;
  //auto& cur_state = nest_state_[cur->level].live_state[spatial_id_];

  // std::cout << "CSWS level " << level << " potentially created live state entry. Full state:\n";
  // for (auto& state: nest_state_[cur->level].live_state)
  // {
  //   std::cout << "  ";
  //   PrintStamp(state.first);
  //   std::cout << "->" << state.second.max_size.at(0) << std::endl;
  // }

  problem::PerDataSpace<AccessStatMatrix> access_stats_without_link_transfers, access_stats_with_link_transfers;
  problem::PerDataSpace<AccessStatMatrix*> access_stats;

  for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
  {
    // Default: do not use link transfers.
    access_stats[pvi] = &access_stats_without_link_transfers[pvi];
  }
  
  ComputeAccurateMulticastedAccesses(cur, spatial_deltas, unaccounted_delta,
                                     access_stats_without_link_transfers);

  // *** FIXME *** for Read-Write data spaces, we must check if hardware
  // reduction is supported by the child level. If not, we cannot perform
  // link transfers.

  if (gEnableLinkTransfers && linked_spatial_level_[level])
  {
    // Reset unaccounted delta, and now count with link transfers.
    for (auto& delta: spatial_deltas)
    {
      for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
        unaccounted_delta[pv].insert(delta.first);
    }
    // for (uint64_t i = 0; i < num_spatial_elems; i++)
    // {
    //   unaccounted_delta[i].fill(true);
    // }

    problem::PerDataSpace<std::uint64_t> link_transfers;

    ComputeNetworkLinkTransfers(cur, spatial_deltas, unaccounted_delta, link_transfers);

    ComputeAccurateMulticastedAccesses(cur, spatial_deltas, unaccounted_delta,
                                       access_stats_with_link_transfers);

    // Compare.
    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      std::uint64_t total_without = access_stats_without_link_transfers[pvi].TotalAccesses();
      std::uint64_t total_with = access_stats_with_link_transfers[pvi].TotalAccesses();

      if (total_with < total_without)
      {
        cur_state.link_transfers[pvi] += link_transfers[pvi];        
        access_stats[pvi] = &access_stats_with_link_transfers[pvi];
      }
    }
  }

  for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
  {
    cur_state.access_stats[pvi].Accumulate(*access_stats[pvi]);
  }

  //  auto& accesses = nest_state_[cur->level].live_state[spatial_id_].accesses;

  // Check that all deltas were accounted for correctly.

  for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
  {
    ASSERT(unaccounted_delta[pvi].empty());
  }
  // ASSERT(unaccounted_delta.empty());
  // for (uint64_t i = 0; i < num_spatial_elems; i++)
  // {
  //   for (auto& it : unaccounted_delta[i])
  //   {
  //     ASSERT(!it);
  //   }
  // }

  // Consistency check.
  // for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
  // {
  //   std::uint64_t fanout = 0;
  //   for (unsigned i = 0; i < cur_state.accesses[pvi].size(); i++)
  //   {
  //     fanout += (i+1) * cur_state.scatter_factors[pvi][i];
  //   }
    
  //   if (fanout != logical_fanouts_[cur->level])
  //   {
  //     std::cerr << "FATAL: fanout mismatch, computed = " << fanout
  //               << " actual = " << logical_fanouts_[cur->level] << std::endl;
  //     exit(1);
  //   }
  // }    
}


// Apply skew (if required).
std::uint64_t NestAnalysis::ApplySkew(std::uint64_t unskewed_index)
{
  if (cur_skew_descriptor_ != nullptr)
  {
    // apply skew.
    std::int64_t numerator = 0;
    for (auto& term: cur_skew_descriptor_->terms)
    {
      std::int64_t prod = term.constant;
      if (term.variable.dimension != problem::GetShape()->NumFlattenedDimensions)
      {
        if (term.variable.is_spatial)
          prod *= loop_gists_spatial_.at(term.variable.dimension).index;
        else
          prod *= loop_gists_temporal_.at(term.variable.dimension).index;
      }
      if (term.bound.dimension != problem::GetShape()->NumFlattenedDimensions)
      {
        if (term.bound.is_spatial)
          prod *= loop_gists_spatial_.at(term.bound.dimension).index;
        else
          prod *= loop_gists_temporal_.at(term.bound.dimension).index;
      }
      numerator += prod;
    }
    
    std::int64_t skewed_index = numerator % cur_skew_descriptor_->modulo;
    if (skewed_index < 0)
      skewed_index += cur_skew_descriptor_->modulo;

    ASSERT(skewed_index >= 0);

    return static_cast<std::uint64_t>(skewed_index);
  }
  else
    return unskewed_index;
}


// Computes deltas needed by the spatial elements in the next level.
// Will update a subset of the elements of spatial_deltas
void NestAnalysis::FillSpatialDeltas(std::vector<analysis::LoopState>::reverse_iterator cur,
                                     std::unordered_map<std::uint64_t, problem::OperationSpace>& spatial_deltas,
                                     std::unordered_map<std::uint64_t, std::uint64_t>& skew_table,
                                     std::uint64_t base_index,
                                     int depth,
                                     int extrapolation_stride,
                                     std::vector<analysis::LoopState>::reverse_iterator extrapolation_level)
{
  int level = cur->level;
  auto dim = cur->descriptor.dimension;

  int end = IsLastGlobalIteration_(level+1, cur->descriptor.dimension) ?
    cur->descriptor.residual_end : cur->descriptor.end;

  unsigned num_iterations = 1 + ((end - 1 - cur->descriptor.start) /
                                 cur->descriptor.stride);

  // First, update loop gist. FIXME: handle base!=0, stride!=1.
  ASSERT(cur->descriptor.start == 0);
  ASSERT(cur->descriptor.stride == 1);
  loop_gists_spatial_[cur->descriptor.dimension] = { 0, end };
  
  // base_index determines which element of spatial_deltas
  // is going to be updated at the last recursive call to FillSpatialDeltas.
  // It's value is updated as we recursively call FillSpatialDeltas.
  // Very similar to how spatial_id_ is used to identify the spatial element
  // that we are currently computing the working set for.
  base_index *= end; // num_iterations?

  if (level == 0)
  {
    // Level 0 is a spatial loop.

    // Compute update.

    // If all spatial indices were uniform then we can simply increment the
    // compute count at a representative spatial ID (e.g., 0) with the present
    // value of num_epochs_. That's what this commented-out code used to do.
    // However, we now handle non-uniformity due to several reasons, which is
    // why we need to update the compute count of each spatial ID independently
    // after applying a skew.

    // if (base_index == 0 && spatial_id_ == 0)
    // {
    //   compute_info_.accesses += num_epochs_;
    // }

    // No more recursive calls, directly update spatial_deltas.
    for (indices_[level] = cur->descriptor.start;
         indices_[level] < end;
         indices_[level] += cur->descriptor.stride)
    {
      loop_gists_spatial_.at(dim).index = indices_[level];
      
      std::uint64_t spatial_delta_index = base_index + indices_[level];
      std::uint64_t skewed_delta_index = ApplySkew(spatial_delta_index);

      // If the following assertion fails, it means there's a collision in the
      // skew function.
      ASSERT(spatial_deltas.find(skewed_delta_index) == spatial_deltas.end());

      spatial_deltas.emplace(skewed_delta_index,
                             problem::OperationSpace(workload_));
      spatial_deltas.emplace(skewed_delta_index, problem::OperationSpace(workload_));
      spatial_deltas.at(skewed_delta_index) += IndexToOperationPoint_(indices_);

      space_stamp_.back() = skewed_delta_index;
      compute_info_[space_stamp_].accesses += num_epochs_;

      // FIXME: add log.
    }
  }
  else // level > 0
  {
    auto next = cur + 1;
    int dim = int(cur->descriptor.dimension);
    int scale = vector_strides_[level][dim];

    // Save state.
    auto orig_spatial_id = spatial_id_;
    auto saved_transform = cur_transform_[dim];

    unsigned iterations_run = 0;
    indices_[level] = cur->descriptor.start;
    loop_gists_spatial_.at(dim).index = indices_[level];

    unsigned iterations_to_run =
      (gExtrapolateUniformSpatial && !problem::GetShape()->UsesFlattening)
      ? (gDisableFirstElementOnlySpatialExtrapolation ? 3 : 1) : num_iterations;

    if (loop::IsSpatial(next->descriptor.spacetime_dimension))
    {
      // Next-inner loop level is spatial.
      // Make a recursive call for all iterations, but send a different
      // extrapolation_stride depending on the type of iteration:
      // "run" iteration => multiply current extrapolation_stride with my num_iterations.
      // "extrapolation" iteration => update extrapolation_stride with my stride.
      for (indices_[level] = cur->descriptor.start;
           indices_[level] < end;
           indices_[level] += cur->descriptor.stride, iterations_run++)
      {
        loop_gists_spatial_.at(dim).index = indices_[level];
        
        auto next_extrapolation_stride = extrapolation_stride * end; // * num_iterations?
        auto next_extrapolation_level = extrapolation_level;
        if (iterations_run >= iterations_to_run) // Extrapolate using this level
        {
          next_extrapolation_stride = cur->descriptor.stride;
          next_extrapolation_level = cur;
        }

        ++cur;

        FillSpatialDeltas(cur, spatial_deltas, skew_table,
                          base_index + indices_[level], depth+1,
                          next_extrapolation_stride,
                          next_extrapolation_level);

        --cur;
        cur_transform_[dim] += scale;
      }
    }
    else // Next-inner loop level is temporal.
    {
      // Logic is:
      // Parent stride == 0?
      // - 0:iterations_to_run-1 => recursive temporal ComputeDeltas().
      // - iterations_to_run:num_iterations-1 => extrapolate with cur.stride.
      // Parent stride != 0?
      // - 0:iterations_to_run-1 => extrapolate with extrapolation_stride*cur.end
      // - iterations_to_run:num_iterations-1 => extrapolate with cur.stride
      //                                         OR extrapolation_stride*cur.end.
      //                                         (we're going with the latter)
      if (extrapolation_stride == 0)
      {
        // If the final extrapolation_stride is 0, it means we have no data to
        // extrapolate from, so recurse into a full temporal ComputeDeltas() call.
        // This is the expensive bit.

        // Run iterations #0, #1, ... #iterations_to_run-1
        for (indices_[level] = cur->descriptor.start;
             indices_[level] < end && iterations_run < iterations_to_run;
             indices_[level] += cur->descriptor.stride, iterations_run++)
        {
          loop_gists_spatial_.at(dim).index = indices_[level];

          ++cur;

          std::uint64_t spatial_delta_index = base_index + indices_[level];
          std::uint64_t skewed_delta_index = ApplySkew(spatial_delta_index);
          skew_table[spatial_delta_index] = skewed_delta_index;

          // If the following assertion fails, it means there's a collision in the
          // skew function.
          ASSERT(spatial_deltas.find(skewed_delta_index) == spatial_deltas.end());

          // std::cout << indent + "  " << iterations_run << " sdi " << spatial_delta_index
          //           << " calling temporal " << std::endl;

          spatial_id_ = orig_spatial_id + spatial_delta_index; // note: unskewed.

          space_stamp_.back() = skewed_delta_index;

          // std::cout << "innermost FSD at level " << level
          //           << " spatial_id_ = " << spatial_id_ << " sdsize = " << spatial_deltas.size()
          //           << " unskewed = " << spatial_delta_index << " skewed = "
          //           << skewed_delta_index << std::endl;

          spatial_deltas.emplace(skewed_delta_index, ComputeDeltas(cur));

          --cur;
          cur_transform_[dim] += scale;
        }

        // Set up extrapolation stride for the remaining iterations.
        extrapolation_stride = cur->descriptor.stride;
        extrapolation_level = cur;
      }
      else
      {
        // std::cout << indent << "Parent stride is NZ, updating stride "
        //           << extrapolation_stride*end << std::endl;

        // Set up extrapolation stride for the remaining iterations.
        extrapolation_stride *= end; // num_iterations;
        indices_[level] = cur->descriptor.start;
        loop_gists_spatial_.at(dim).index = indices_[level];
      }

      //
      // Extrapolate all remaining iterations.
      //

      // Determine translation vector from #iterations_to_run-2 to #iterations_to_run-1.
      problem::PerDataSpace<Point> translation_vectors;
      if (indices_[level] < end)
      {
        if(!gDisableFirstElementOnlySpatialExtrapolation) 
        {
          translation_vectors = GetCurrentTranslationVectors(extrapolation_level);
        }
        else
        {
          auto last_skewed_index = skew_table.at(base_index + indices_[level] - extrapolation_stride);
          auto secondlast_skewed_index = skew_table.at(base_index + indices_[level] - 2*extrapolation_stride);

          auto& opspace_lastrun = spatial_deltas.at(last_skewed_index);
          auto& opspace_secondlastrun = spatial_deltas.at(secondlast_skewed_index);

          for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
          {
            translation_vectors[pv] =
              opspace_secondlastrun.GetDataSpace(pv).GetTranslation(opspace_lastrun.GetDataSpace(pv));
          }
        }
      }

      // Iterations #num_iterations_to_run through #last.
      for (;
           indices_[level] < end;
           indices_[level] += cur->descriptor.stride, iterations_run++)
      {
        loop_gists_spatial_.at(dim).index = indices_[level];

        std::uint64_t dst_delta_index = ApplySkew(base_index + indices_[level]);
        std::uint64_t src_delta_index = skew_table.at(base_index + indices_[level] - extrapolation_stride);
        skew_table[base_index + indices_[level]] = dst_delta_index;

        // If the following assertions fail, it means there's a collision in the
        // skew function.
        ASSERT(spatial_deltas.find(dst_delta_index) == spatial_deltas.end());
        ASSERT(spatial_deltas.find(src_delta_index) != spatial_deltas.end());

        spatial_id_ = orig_spatial_id + base_index + indices_[level]; // note: unskewed.

        auto dst_temporal_delta_it =
            spatial_deltas.emplace(dst_delta_index,
                                   problem::OperationSpace(workload_)).first;
        auto& dst_temporal_delta = dst_temporal_delta_it->second;
        auto& src_temporal_delta = spatial_deltas.at(src_delta_index);
        for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
        {
          dst_temporal_delta.GetDataSpace(pv) = src_temporal_delta.GetDataSpace(pv);
          dst_temporal_delta.GetDataSpace(pv).Translate(translation_vectors.at(pv));
        }
      } // extrapolated iterations

    } // next inner loop is temporal

    // Restore state.
    cur_transform_[dim] = saved_transform;
    spatial_id_ = orig_spatial_id;

  } // level > 0  
}

// Exhaustively compare all pairs of deltas and infer multicast opportunities.
void NestAnalysis::ComputeAccurateMulticastedAccesses(
    std::vector<analysis::LoopState>::reverse_iterator cur,
    const std::unordered_map<std::uint64_t, problem::OperationSpace>& spatial_deltas,
    problem::PerDataSpace<std::unordered_set<std::uint64_t>>& unaccounted_delta,
    //std::set<std::pair<std::uint64_t, problem::Shape::DataSpaceID>>& unaccounted_delta,
    //std::vector<problem::PerDataSpace<bool>>& unaccounted_delta,
    problem::PerDataSpace<AccessStatMatrix>& access_stats)
{
  //std::uint64_t num_deltas = spatial_deltas.size();

  // For each data type, records the number of unaccounted deltas
  // that the current delta matches with. This will be used
  // to infer the multicast factor for a specific delta.
  // reused across loop iterations to avoid initialization overheads.
  problem::PerDataSpace<uint64_t> num_matches;
  problem::PerDataSpace<bool> no_multicast;
  for(unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    no_multicast[pv] = false;
  }
  if(no_multicast_.find(arch_storage_level_[cur->level]) != no_multicast_.end())
  {
    no_multicast = no_multicast_[arch_storage_level_[cur->level]];
  }

  // Prepare a legacy-style multicast/scatter signature to collect the
  // delta data, then populate the new access stats.
  struct TempAccessStats
  {
    double accesses = 0;
    std::uint64_t scatter_factor = 0;
    double hops = 0.0;
  };
  problem::PerDataSpace<std::unordered_map<std::uint64_t, TempAccessStats>> temp_stats;

  // FIXME: we should only be looking at physical dimensions here. The problem
  // is that sparse mappings may appear to exceed the physical dimensions before
  // space-skipping is applied. The very notion of spatial skew and physical
  // location for space-skipping sparse mappings is something we need to figure
  // out.
  auto h_size = std::max(physical_fanoutX_.at(arch_storage_level_.at(cur->level)), logical_fanoutX_[cur->level]);
  auto v_size = std::max(physical_fanoutY_.at(arch_storage_level_.at(cur->level)), logical_fanoutY_[cur->level]);

  for (auto delta_it = spatial_deltas.begin(); delta_it != spatial_deltas.end(); delta_it++)
    //for (std::uint64_t i = 0; i < num_deltas; i++)
  {
    auto& skewed_spatial_index = delta_it->first;
    auto& delta = delta_it->second;

    num_matches.fill(0);
    
    problem::PerDataSpace<std::vector<std::uint64_t>> match_set;

    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      auto unaccounted_it = unaccounted_delta[pv].find(skewed_spatial_index);
      if (unaccounted_it == unaccounted_delta[pv].end())
        //if (!unaccounted_delta[i][pv])
      {
        // this delta was already accounted for,
        // skip the comparisons.
        continue;
      }

      unaccounted_delta[pv].erase(unaccounted_it);
      //unaccounted_delta[i][pv] = false;
      num_matches[pv] = 1;  // we match with ourselves.
      match_set[pv].push_back(skewed_spatial_index);

      if(!no_multicast[pv]) // If multicasting enabled, look for multicast opportunities
      {
        for (auto delta_other_it = std::next(delta_it); delta_other_it != spatial_deltas.end(); delta_other_it++)
          //for (std::uint64_t j = i + 1; j < num_deltas; j++)
        {
          auto& skewed_other_spatial_index = delta_other_it->first;
          auto& delta_other = delta_other_it->second;

          auto unaccounted_other_it = unaccounted_delta[pv].find(skewed_other_spatial_index);
          if (unaccounted_other_it != unaccounted_delta[pv].end())
            //if (unaccounted_delta[j][pv])
          {
            if (delta.CheckEquality(delta_other, pv))
              //if (spatial_deltas[i].CheckEquality(spatial_deltas[j], pv))
            {
              // We have a match, record it
              unaccounted_delta[pv].erase(unaccounted_other_it);
              //unaccounted_delta[j][pv] = false;
              num_matches[pv]++;
              match_set[pv].push_back(skewed_other_spatial_index);
            }
          }
        }
      }
    }

    // NOTE: multicast is # children sharing the same delta
    //       scatter factor is the # data spaces with the same multicast value

    // update the number of accesses at different multicast factors.
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      if (num_matches[pv] > 0 && delta.GetSize(pv) > 0)
      {
        auto& temp_struct = temp_stats[pv][num_matches[pv]];
        temp_struct.accesses += (delta.GetSize(pv) * num_epochs_);
        temp_struct.scatter_factor++;

        // Compute the average number of hops from the edge of the array
        // (at this level) to the nodes in the match set.
        // Assume injection point is at center of V-axis. Routing algorithm is
        // to go along H maximally, then drop vertical paths.

        ASSERT(num_matches[pv] == match_set[pv].size());
        
        double hops = 0;
        
        std::uint64_t h_max = 0;
        for (auto& linear_id : match_set[pv])
        {
          std::uint64_t h_id = linear_id % h_size;
          h_max = std::max(h_max, h_id);
        }
        hops += double(h_max);
        
        double v_center = double(v_size-1) / 2;
        for (auto& linear_id : match_set[pv])
        {
          std::uint64_t v_id = linear_id / h_size;
          hops += std::abs(double(v_id) - v_center);
        }

        // Accumulate this into the running hop count. We'll finally divide this
        // by the scatter factor to get average hop count.
        temp_struct.hops += hops;
      }
    }
  }

  // Populate the actual stats.
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    for (auto& x: temp_stats[pv])
    {
      auto multicast = x.first;
      auto scatter = x.second.scatter_factor;
      access_stats[pv](multicast, scatter) = { x.second.accesses, x.second.hops };
    }
  }
}

// Compares two deltas, and if they are equal, records the opportunity for
// inter-PE link transfers. Note that we only track transfers at each
// destination (i.e., recipient), we do not track who sent the data. This is
// because senders and receivers are at the same storage level, and we only
// track aggregate stats per level. 
void NestAnalysis::CompareSpatioTemporalDeltas(
    const std::unordered_map<std::uint64_t, problem::OperationSpace>& cur_spatial_deltas,
    const std::unordered_map<std::uint64_t, problem::OperationSpace>& prev_spatial_deltas,
    //const std::vector<problem::OperationSpace>& cur_spatial_deltas,
    //const std::vector<problem::OperationSpace>& prev_spatial_deltas,
    const std::uint64_t cur_spatial_index,
    const std::uint64_t prev_spatial_index,
    std::vector<problem::PerDataSpace<bool>>& inter_elem_reuse,
    const problem::PerDataSpace<bool>& ignore_dataspaces)
{
  //PrintSpaceTimeStamp();
  //std::cout << "comparing " << cur_spatial_index << " vs " << prev_spatial_index << std::endl;
  
  auto cur_delta_it = cur_spatial_deltas.find(cur_spatial_index);
  if (cur_delta_it == cur_spatial_deltas.end())
    return;

  auto prev_delta_it = prev_spatial_deltas.find(prev_spatial_index);
  if (prev_delta_it == prev_spatial_deltas.end())
    return;

  auto& cur_delta = cur_delta_it->second;
  auto& prev_delta = prev_delta_it->second;

  //std::cout << "  cur : " << cur_delta << std::endl;
  //std::cout << "  prev: " << prev_delta << std::endl;

  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    if (!ignore_dataspaces[pv] && !cur_delta.IsEmpty(pv))
    {
      if (cur_delta.CheckEquality(prev_delta, pv))
      {
        // ASSERT(!inter_elem_reuse[cur_spatial_index][pv]);
        inter_elem_reuse.at(cur_spatial_index)[pv] = true;
        //std::cout << "  match for pv " << pv << std::endl;
      }
    }
  }
}

void NestAnalysis::ComputeNetworkLinkTransfers(
    std::vector<analysis::LoopState>::reverse_iterator cur,
    const std::unordered_map<std::uint64_t, problem::OperationSpace>& cur_spatial_deltas,
    problem::PerDataSpace<std::unordered_set<std::uint64_t>>& unaccounted_delta,
    //std::set<std::pair<std::uint64_t, problem::Shape::DataSpaceID>>& unaccounted_delta,
    //std::vector<problem::PerDataSpace<bool>>& unaccounted_delta,
    problem::PerDataSpace<std::uint64_t>& link_transfers)
{
  // std::cout << "-----------------------------\n";
  // std::cout << "         LINK TRANSFERS      \n";
  // std::cout << "-----------------------------\n";
  
  // std::cout << "CUR BEFORE:" << std::endl;
  // for (std::uint64_t i = 0; i < cur_spatial_deltas.size(); i++)
  // {
  //   auto pv = problem::Shape::DataSpaceID::Weight;
  //   if (unaccounted_delta[i][int(pv)])
  //     std::cout << "  UNACCOUNTED: ";
  //   else
  //     std::cout << "    ACCOUNTED: ";
  //   cur_spatial_deltas[i].Print(pv);
  // }
  
  // FIXME: we should only be looking at physical dimensions here. The problem
  // is that sparse mappings may appear to exceed the physical dimensions before
  // space-skipping is applied. The very notion of spatial skew and physical
  // location for space-skipping sparse mappings is something we need to figure
  // out.
  auto h_size = std::max(physical_fanoutX_.at(arch_storage_level_.at(cur->level)), logical_fanoutX_[cur->level]);
  auto v_size = std::max(physical_fanoutY_.at(arch_storage_level_.at(cur->level)), logical_fanoutY_[cur->level]);

  // Imagine origin (0,0) at the top-left corner of a 2D spatial array.
  // Horizontal ids grow from left to right.
  // Vertical ids grow from top to bottom.
  auto GetLinearIndex = [&h_size, &v_size](std::uint64_t h_id, std::uint64_t v_id)
    {
      ASSERT(h_id < h_size && v_id < v_size);
      std::uint64_t linearIndex = v_id * h_size + h_id;  // row major layout
      return linearIndex;
    };

  // Note that spatial_id_ is in logical (i.e., unskewed) space. This is fine
  // because we are only using it to find the current live state at the parent.
  // The child nodes (over which we will compute link transfers) are in
  // physical (i.e., skewed) space.
  auto& cur_state = cur->live_state.at(AllButLast(space_stamp_));
  //auto& cur_state = cur->live_state[spatial_id_];
  auto& prev_spatial_deltas = cur_state.prev_spatial_deltas;
  //auto& prev_spatial_deltas = cur_state.prev_spatial_deltas[0];
  //ASSERT(cur_spatial_deltas.size() == prev_spatial_deltas.size());

  int num_spatial_elems = h_size * v_size;

  // std::cout << "PREV:" << std::endl;
  // for (std::uint64_t i = 0; i < prev_spatial_deltas.size(); i++)
  // {
  //   auto pv = problem::Shape::DataSpaceID::Weight;
  //   std::cout << "  "; prev_spatial_deltas[i].Print(pv);
  // }
  
  problem::PerDataSpace<bool> no_link_transfer;
  for(unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    no_link_transfer[pv] = false;
  }
  if(no_link_transfer_.find(arch_storage_level_[cur->level]) != no_link_transfer_.end())
  {
    no_link_transfer = no_link_transfer_[arch_storage_level_[cur->level]];
  }
  // Return if there's no transfers allowed at all
  if(std::all_of(no_link_transfer.begin(), no_link_transfer.end(), [](bool v) { return v; })) {return;}

  // for each spatial elements, this array records if the data
  // needed by the element can be obtained from any of the neighboring elements.
  std::vector<problem::PerDataSpace<bool>> inter_elem_reuse;
  inter_elem_reuse.resize(num_spatial_elems);
  for (int i = 0; i < num_spatial_elems; i++)
  {
    inter_elem_reuse.at(i).fill(false);
  }

  // FIXME: The loops below can be codified in some way to avoid redundant LOC.

  // Test for a few hard-coded transfer patterns in horizontal and vertical
  // dimensions.
  // FIXME: the connectivity graph should be derived from the arch spec.

  // downward vertical transfers in each column
  if (v_size > 1)
  {
    for (std::uint64_t h_id = 0; h_id < h_size; h_id++)
    {
      for (std::uint64_t v_id = (gEnableToroidalLinks ? 0 : 1); v_id < v_size; v_id++)
      {
        auto cur_skewed_spatial_index = GetLinearIndex(h_id, v_id);
        auto prev_skewed_spatial_index = GetLinearIndex(h_id, (v_id - 1 + v_size) % v_size);
        CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                    cur_skewed_spatial_index, prev_skewed_spatial_index,
                                    inter_elem_reuse,
                                    no_link_transfer);
      }
    }

    // upward vertical transfers in each column
    for (std::uint64_t h_id = 0; h_id < h_size; h_id++)
    {
      for (std::uint64_t v_id = 0; v_id < (gEnableToroidalLinks ? v_size : (v_size-1)); v_id++)
      {
        auto cur_skewed_spatial_index = GetLinearIndex(h_id, v_id);
        auto prev_skewed_spatial_index = GetLinearIndex(h_id, (v_id + 1) % v_size);
        CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                    cur_skewed_spatial_index, prev_skewed_spatial_index,
                                    inter_elem_reuse,
                                    no_link_transfer);
      }
    }
  }

  // horizontal transfers in each row from left to right
  if (h_size > 1)
  {
    for (std::uint64_t v_id = 0; v_id < v_size; v_id++)
    {
      for (std::uint64_t h_id = (gEnableToroidalLinks ? 0 : 1); h_id < h_size; h_id++)
      {
        auto cur_skewed_spatial_index = GetLinearIndex(h_id, v_id);
        auto prev_skewed_spatial_index = GetLinearIndex((h_id - 1 + h_size) % h_size, v_id);
        CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                    cur_skewed_spatial_index, prev_skewed_spatial_index,
                                    inter_elem_reuse,
                                    no_link_transfer);
      }
    }

    // horizontal transfers in each row from right to left
    for (std::uint64_t v_id = 0; v_id < v_size; v_id++)
    {
      for (std::uint64_t h_id = 0; h_id < (gEnableToroidalLinks ? h_size : (h_size-1)); h_id++)
      {
        auto cur_skewed_spatial_index = GetLinearIndex(h_id, v_id);
        auto prev_skewed_spatial_index = GetLinearIndex((h_id + 1) % h_size, v_id);
        CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                    cur_skewed_spatial_index, prev_skewed_spatial_index,
                                    inter_elem_reuse,
                                    no_link_transfer);
      }
    }
  }

  // Compute the total number of accesses that can be bypassed
  // by using link transfers
  for (auto& delta: cur_spatial_deltas)
//  for (int i = 0; i < num_spatial_elems; i++)
  {
    auto& cur_skewed_spatial_index = delta.first;
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      if (inter_elem_reuse.at(cur_skewed_spatial_index)[pv])
      {
        link_transfers[pv] += (delta.second.GetSize(pv) * num_epochs_);
        auto unaccounted_it = unaccounted_delta[pv].find(cur_skewed_spatial_index);
        ASSERT(unaccounted_it != unaccounted_delta[pv].end());
        unaccounted_delta[pv].erase(unaccounted_it);
      }
    }
  }

  // Time-shift the data in prev_spatial_deltas array
  cur_state.prev_spatial_deltas = cur_spatial_deltas;

  // for (std::uint64_t i = 1; i < analysis::ElementState::MAX_TIME_LAPSE; i++)
  // {
  //   cur_state.prev_spatial_deltas[i - 1] = cur_state.prev_spatial_deltas[i];
  // }

  // cur_state.prev_spatial_deltas[analysis::ElementState::MAX_TIME_LAPSE - 1] = cur_spatial_deltas;
}

// computes the number of spatial elements at each level
// and identifies master spatial levels.
void NestAnalysis::InitNumSpatialElems()
{
  num_spatial_elems_.resize(nest_state_.size());
  master_spatial_level_.resize(nest_state_.size());

  int cur_index = nest_state_.size() - 1;
  // cumulative product of spatial tiling factors.
  std::uint64_t product = 1;
  bool prev_loop_was_spatial = false;
  for (auto loop = nest_state_.rbegin(); loop != nest_state_.rend(); loop++)
  {
    ASSERT(cur_index >= 0);

    num_spatial_elems_[cur_index] = product;
    if (loop::IsSpatial(loop->descriptor.spacetime_dimension))
    {
      master_spatial_level_[cur_index] = !prev_loop_was_spatial;
      product *= loop->descriptor.end;
      prev_loop_was_spatial = true;
    }
    else
    {
      master_spatial_level_[cur_index] = false;
      prev_loop_was_spatial = false;
    }

    cur_index--;
  }

  linked_spatial_level_.resize(nest_state_.size(), false);
  for (std::uint64_t cur_level = 0; cur_level < nest_state_.size(); cur_level++)
  {
    if (master_spatial_level_[cur_level])
    {
      linked_spatial_level_[cur_level] = true;
    }
  }

}

void NestAnalysis::InitStorageBoundaries()
{
  storage_boundary_level_.resize(nest_state_.size(), false);
  arch_storage_level_.resize(nest_state_.size());
  disable_temporal_extrapolation_.resize(nest_state_.size(), false);

  unsigned storage_level = 0;
  unsigned loop_level = 0;
  for (auto& i : storage_tiling_boundaries_)
  {
    ASSERT(i < storage_boundary_level_.size());
    storage_boundary_level_[i] = true;

    auto skew_it = packed_skew_descriptors_.find(storage_level);
    if (skew_it != packed_skew_descriptors_.end())
    {
      skew_descriptors_[i] = skew_it->second;

      // Walk through the skew descriptor and poison all temporal loop
      // variables it touches.
      for (auto& term: skew_it->second.terms)
      {
        if (term.variable.dimension != problem::GetShape()->NumFlattenedDimensions && !term.variable.is_spatial)
        {
          auto dim = term.variable.dimension;
          // Walk through the loops in this loop block and poison the loop
          // corresponding to this problem dimension.
          for (unsigned level = loop_level; level <= i; level++)
          {
            if (nest_state_.at(level).descriptor.dimension == dim)
            {
              disable_temporal_extrapolation_.at(level) = true;
              // There can only be 1 such loop in each block.
              break;
            }
          }
        }
      }
    }

    // Establish loop level -> storage level map.
    for (; loop_level <= i; loop_level++)
    {
      arch_storage_level_[loop_level] = storage_level;
    }

    storage_level++;
  }

}

void NestAnalysis::InitSpatialFanouts()
{
  logical_fanouts_.resize(nest_state_.size(), 1);
  logical_fanoutX_.resize(nest_state_.size(), 1);
  logical_fanoutY_.resize(nest_state_.size(), 1);
  for (int cur_level = nest_state_.size() - 1; cur_level >= 0; cur_level--)
  {
    if (!loop::IsSpatial(nest_state_[cur_level].descriptor.spacetime_dimension))
    {
      logical_fanouts_[cur_level] = 1;
    }
    else if (!master_spatial_level_[cur_level])
    {
      logical_fanouts_[cur_level] = 0;
    }
    else
    {
      int next_temporal_level = cur_level;
      int scale_factor = 1;
      while (loop::IsSpatial(nest_state_[next_temporal_level].descriptor.spacetime_dimension))
      {
        if (loop::IsSpatialX(nest_state_[next_temporal_level].descriptor.spacetime_dimension))
        {
          logical_fanoutX_[cur_level] *=
              nest_state_[next_temporal_level].descriptor.end;
        }
        else
        {
          logical_fanoutY_[cur_level] *=
              nest_state_[next_temporal_level].descriptor.end;
        }

        if (next_temporal_level > 0)
        {
          next_temporal_level--;
        }
        else
        {
          scale_factor = nest_state_[0].descriptor.end;
          break;
        }
      }

      logical_fanouts_[cur_level] =
          num_spatial_elems_[next_temporal_level] / num_spatial_elems_[cur_level];
      logical_fanouts_[cur_level] *= scale_factor;

      ASSERT(logical_fanouts_[cur_level] ==
             logical_fanoutX_[cur_level] * logical_fanoutY_[cur_level]);
    }
  }

#if 0
  std::cout << "Spatial fanouts at each level" << std::endl;
  for (int i = num_spatial_elems_.size() - 1; i >= 0; i--)
  {
    std::cout << logical_fanouts_[i];
    std::cout << ", ";
  }
  std::cout << std::endl;
#endif
}

void NestAnalysis::InitPerLevelDimScales()
{
  for (unsigned dim = 0; dim < problem::GetShape()->NumFlattenedDimensions; dim++)
  {
    cur_transform_[dim] = 0;
  }

  std::uint64_t num_levels = nest_state_.size();

  vector_strides_.resize(num_levels);

  mold_low_.resize(num_levels);
  mold_high_.resize(num_levels);

  // Handle residual volumes created due to imperfect factorization.
  // The proper way to do this is to have a mold tree. We are trying
  // an approximation with the implemented approach.
  mold_high_residual_.resize(num_levels);

  // running scale maintained for each dimension.
  problem::PerFlattenedDimension<std::uint64_t> cur_scale;
  cur_scale.fill(1);

  problem::PerFlattenedDimension<std::uint64_t> cur_scale_residual;
  cur_scale_residual.fill(1);

  for (std::uint64_t level = 0; level < num_levels; level++)
  {
    auto desc = nest_state_[level].descriptor;
    int dim = int(desc.dimension);

    for (std::uint64_t dim = 0; dim < problem::GetShape()->NumFlattenedDimensions; dim++)
    {
      vector_strides_[level][dim] = cur_scale[dim];
    }

    cur_scale_residual[dim] += (cur_scale[dim]*(desc.residual_end - desc.start - 1)); // FIXME: assuming stride = 1
    cur_scale[dim] *= (desc.end - desc.start); // FIXME: assuming stride = 1
    
    for (std::uint64_t dim = 0; dim < problem::GetShape()->NumFlattenedDimensions; dim++)
    {
      //mold_low_[level][dim] = desc.start; Should be 0. FIXME: verify.
      mold_high_[level][dim] = cur_scale[dim] - 1;
      mold_high_residual_[level][dim] = cur_scale_residual[dim] - 1;
    }
  }
}

// Transform an index to a problem point.

// arm: This routine is called a lot of times (no. of MACs in CONV layer),
// But, it is totally unoptimized and does a lot of redundant computation.
// There is a not-so-complicated way to optimize this
// by exploiting the global loop nest information.
// instead of making naive local decisions.
problem::OperationPoint NestAnalysis::IndexToOperationPoint_(
  const std::vector<int>& indices) const
{
  problem::OperationPoint point;
  for (unsigned dim = 0; dim < problem::GetShape()->NumFlattenedDimensions; dim++)
  {
    point[dim] = 0;
  }

  for (unsigned int level = 0; level < indices.size(); level++)
  {
    auto desc = nest_state_[level].descriptor;
    int dim = int(desc.dimension);
    point[dim] += (vector_strides_[level][dim] * indices[level]);
  }

  return point;
}

// For a specific dimension, detemine if we are at the last iteration of a loop at
// every loop level from the root of the tree down to this level.
bool NestAnalysis::IsLastGlobalIteration_(int level, problem::Shape::FlattenedDimensionID dim) const
{
  // We need to look at all loops between root and the given level
  // and return true if they are all at their last iteration.
  // Note that we only need to look at the residual ends, because the
  // definition of global last iteration means that we are at the residuals
  // at each loop.

  // Note that this logic trivially works if "level" is the outermost
  // level of the nest, since we are effectively at the "last" iteration
  // of all theoretical outer loop nests. This means that the outermost
  // loop nest always uses its residual end and not its regular end.
  bool is_last = true;
  for (int l = level; l < int(nest_state_.size()); l++)
  {
    if (nest_state_[l].descriptor.dimension != dim)
      continue;

    if ((indices_[l] + nest_state_[l].descriptor.stride) < nest_state_[l].descriptor.residual_end)
    {
      is_last = false;
      break;
    }
  }
  return is_last;
}

// Calculate the translation vectors of the current nest level and transform.
// Finds the motion of the working sets between two iterations at the current level.
problem::PerDataSpace<Point> NestAnalysis::GetCurrentTranslationVectors(std::vector<analysis::LoopState>::reverse_iterator cur)
{
  int level = cur->level;
  auto dim = cur->descriptor.dimension;
  auto saved_transform = cur_transform_[dim];

  // Calculate the first working set
  auto firstrun = GetCurrentWorkingSet(cur);

  // Calcualte the second working set
  cur_transform_[dim] += vector_strides_[level][dim];
  auto secondrun = GetCurrentWorkingSet(cur);
  cur_transform_[dim] = saved_transform;
  
  // Calculate and return translation vectors
  problem::PerDataSpace<Point> translation_vectors;
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    translation_vectors[pv] = firstrun.GetDataSpace(pv).GetTranslation(secondrun.GetDataSpace(pv));
  }
  return translation_vectors;
}

// Calculate the working set of the current nest level & current transform
problem::OperationSpace NestAnalysis::GetCurrentWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur)
{
  int level = cur->level;
  // We use the pre-computed molds within this level range.
  // Above this level range, we use the transform problem-point to
  // translate, rotate or otherwise transform the mold.
  problem::OperationPoint low_problem_point;
  problem::OperationPoint high_problem_point;

  // Compute the polyhedron between the low and high problem
  // points (exclusive). Note that this special constructor
  // is only available for certain point-set implementations.
  // Note: we aren't using +=. This means we're ignoring subvolumes
  // returned to us by recursive FillSpatialDeltas calls.
  for (unsigned dim = 0; dim < unsigned(problem::GetShape()->NumFlattenedDimensions); dim++)
  {
    low_problem_point[dim] = cur_transform_[dim] + mold_low_[level][dim];
    high_problem_point[dim] = cur_transform_[dim] + (IsLastGlobalIteration_(level+1, dim) ?
                                                     mold_high_residual_[level][dim] :
                                                     mold_high_[level][dim]);
  }
  return problem::OperationSpace(workload_, low_problem_point, high_problem_point);
}

} // namespace analysis
