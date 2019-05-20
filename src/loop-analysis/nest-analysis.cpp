/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

// FIXME: num_spatial_elems, spatial_fanouts, replication_factor etc. are
//        all maintained across datatypes. They should be per-datatype at
//        this analytical level of abstraction. It's only when we get to
//        the architecture level that hardware may map multiple datatypes
//        on to the same storage and network structures.

// FIXME: Spatial model is X/Y only. Fortunately, generalizing this isn't
//        too difficult (as far as this module is concerned) since it's
//        limited to the ComputeNetworkLinkTransfers() function.

#include "util/misc.hpp"

#include "nest-analysis.hpp"

namespace analysis
{

NestAnalysis::NestAnalysis()
{
}

void NestAnalysis::Init(problem::WorkloadConfig* wc, const loop::Nest* nest)
{
  assert(nest != NULL);
  assert(wc != NULL);

  workload_config_ = wc;

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

  per_level_dim_scales_.clear();
  cur_transform_ = problem::OperationPoint();
  mold_low_.clear();
  mold_high_.clear();

  num_spatial_elems_.clear();
  spatial_fanouts_.clear();

  horizontal_sizes_.clear();
  vertical_sizes_.clear();

  storage_boundary_level_.clear();
  master_spatial_level_.clear();
  linked_spatial_level_.clear();

  working_sets_computed_ = false;
  
  body_info_.Reset();
}

// Ugly function for pre-checking capacity fits before running the heavyweight
// ComputeWorkingSets() algorithm. FIXME: Integrate with ComputeWorkingSets().
std::vector<problem::PerDataSpace<std::size_t>>
NestAnalysis::GetWorkingSetSizes_LTW() const
{
  std::vector<problem::PerDataSpace<std::size_t>> working_set_sizes;

  problem::PerProblemDimension<int> dimension_sizes;
  dimension_sizes.fill(1);

  unsigned tiling_level = 0;
  for (unsigned loop_level = 0; loop_level < nest_state_.size(); loop_level++)
  {
    auto & loop = nest_state_.at(loop_level).descriptor;
    ASSERT(loop.stride == 1);
    dimension_sizes[int(loop.dimension)] *= loop.end;
        
    if (loop_level == storage_tiling_boundaries_.at(tiling_level))
    {
      working_set_sizes.push_back(problem::GetMaxWorkingSetSizes(dimension_sizes));
      tiling_level++;
    }
  }

  ASSERT(working_set_sizes.size() == storage_tiling_boundaries_.size());
  return working_set_sizes;
}

problem::PerDataSpace<std::vector<tiling::TileInfo>>
NestAnalysis::GetWorkingSets()
{
  if (!working_sets_computed_)
  {
    ComputeWorkingSets();
  }
  ASSERT(working_sets_computed_);
  return working_sets_;
}

tiling::BodyInfo NestAnalysis::GetBodyInfo()
{
  if (!working_sets_computed_)
  {
    ComputeWorkingSets();
  }
  ASSERT(working_sets_computed_);
  return body_info_;
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

    // Recursive call starting from the last element of the list.
    num_epochs_ = 1;
    ComputeWorkingSetsRecursive_(nest_state_.rbegin());

    CollectWorkingSets();
  }

  // Done.
  working_sets_computed_ = true;
}

// Internal helper methods

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
  
  body_info_.Reset();

  for (auto loop = nest_state_.rbegin(); loop != nest_state_.rend(); loop++)
  {
    if (!loop::IsSpatial(loop->descriptor.spacetime_dimension) ||
        master_spatial_level_[loop->level])
    {
      // we don't need live state for non-master spatial levels
      loop->live_state.resize(num_spatial_elems_[loop->level]);
    }

    for (auto& it : loop->live_state)
    {
      it.Reset();
      for (auto& acc : it.accesses)  // for each problem variable
      {
        acc.resize(spatial_fanouts_[loop->level]);
      }
      for (auto& sf : it.scatter_factors)
      {
        sf.resize(spatial_fanouts_[loop->level]);
      }
      for (auto& ch : it.cumulative_hops)
      {
        ch.resize(spatial_fanouts_[loop->level]);
      }
      if (linked_spatial_level_[loop->level])
      {
        it.prev_point_sets.resize(analysis::ElementState::MAX_TIME_LAPSE);
        for (auto& elem : it.prev_point_sets)
        {
          elem.resize(spatial_fanouts_[loop->level]);
        }
      }
    }
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
      analysis::ElementState condensed_state;
      for (int pv = 0; pv < int(problem::DataType::Num); pv++)
      {
        // Sanity check: All elements in a given level should
        // have similar working sets, accesses etc.
        // TODO Can we leverage this assertion to avoid redundant simulation
        // by only simulating one spatial element per level?
        for (std::uint64_t i = 1; i < cur.live_state.size(); i++)
        {
          ASSERT(cur.live_state[i].accesses[pv] ==
                 cur.live_state[i - 1].accesses[pv]);
          ASSERT(cur.live_state[i].max_size[pv] ==
                 cur.live_state[i - 1].max_size[pv]);
          ASSERT(cur.live_state[i].link_transfers[pv] ==
                 cur.live_state[i - 1].link_transfers[pv]);
        }

        // Since, all elements have the same properties, use the properties
        // of the first element to build condensed_state
        const uint64_t REPR_ELEM_ID = 0;  // representative element id.
        condensed_state.accesses[pv] =
            cur.live_state[REPR_ELEM_ID].accesses[pv];
        condensed_state.scatter_factors[pv] =
            cur.live_state[REPR_ELEM_ID].scatter_factors[pv];
        condensed_state.cumulative_hops[pv] =
            cur.live_state[REPR_ELEM_ID].cumulative_hops[pv];
        condensed_state.max_size[pv] =
            cur.live_state[REPR_ELEM_ID].max_size[pv];
        condensed_state.link_transfers[pv] =
            cur.live_state[REPR_ELEM_ID].link_transfers[pv];

        // account for write-backs of non-read-only data types
        // multiply by a factor of '2' to account for writes.
        // *** UPDATE *** this is now handled within the main
        // ComputeWorkingSets() recursive loop.
        //
        // if (problem::IsReadWriteDataType(problem::DataType(pv))) {
        //   for (uint64_t i = 0; i < condensed_state.accesses[pv].size(); i++) {
        //     condensed_state.accesses[pv][i] *= 2;
        //   }
        // }
      }

      // Compute the size of the dataspace partition.
      const uint64_t REPR_ELEM_ID = 0;  // representative element id.
      condensed_state.dataspace_partition_size =
        cur.live_state[REPR_ELEM_ID].dataspace_partition.GetSizes();

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
      for (int pv = 0; pv < int(problem::DataType::Num); pv++)
      {
        tiling::TileInfo tile;
        tile.size                   = condensed_state.max_size[pv];
        tile.partition_size         = condensed_state.dataspace_partition_size[pv];
        tile.accesses               = condensed_state.accesses[pv]; // network accesses
        tile.fills                  = 0; // will be set later
        tile.scatter_factors        = condensed_state.scatter_factors[pv];
        tile.cumulative_hops        = condensed_state.cumulative_hops[pv];
        tile.content_accesses       = tile.GetTotalAccesses();
        tile.link_transfers         = condensed_state.link_transfers[pv];
        tile.subnest                = subnest;
        tile.replication_factor     = num_spatial_elems_[cur.level];
        tile.fanout                 = spatial_fanouts_[cur.level];
        tile.is_on_storage_boundary = storage_boundary_level_[cur.level];
        working_sets_[pv].push_back(tile);
      }

    } // if (valid_level)
  } // for (nest)

  // Extract body data from innermost spatial level.
  for (auto& cur : nest_state_)
  {
    // All spatial levels that are not a master-spatial level are not valid
    bool valid_level = !loop::IsSpatial(cur.descriptor.spacetime_dimension) || master_spatial_level_[cur.level];
    if (valid_level)
    {
      body_info_.replication_factor = num_spatial_elems_[cur.level] * spatial_fanouts_[cur.level];
      break;
    }
  }
}

// Working set computation (recursive call).
// Unless skip_delta is true, returns the delta between the working set of the
// previous iteration and the current iteration of the current level.
problem::OperationSpace NestAnalysis::ComputeWorkingSetsRecursive_(
    std::vector<analysis::LoopState>::reverse_iterator cur, bool skip_delta)
{
  ASSERT(cur != nest_state_.rend());
  ASSERT(spatial_id_ < cur->live_state.size());

  auto& cur_state = cur->live_state[spatial_id_];
  
  // The point set for this invocation. Note that we do *not* initialize this to
  // the last-seen state at the end of the prior invocation. Doing so causes the
  // state at this level to grow indefinitely, which isn't what we're trying to
  // model. The responsibility of this level is to supply all the deltas
  // demanded by the next-inner level for this invocation.
  problem::OperationSpace point_set(workload_config_);

  if (loop::IsSpatial(cur->descriptor.spacetime_dimension))
  {
    ComputeSpatialWorkingSet(cur, point_set);
  }
  else
  {
    ComputeTemporalWorkingSet(cur, point_set, cur_state);
  }

  int level = cur->level;

  // Record the maximum point set size ever seen across all invocations
  // of this level.
  // Need to be done only for levels which will map to physical storage levels
  // after we run collapseTiles.
  if (storage_boundary_level_[level])
  {
    auto sizes = point_set.GetSizes();
    std::transform(sizes.begin(), sizes.end(), cur_state.max_size.begin(),
                   cur_state.max_size.begin(),
                   [](std::size_t x, std::size_t y) { return std::max(x, y); });

    // Track the complete dataspace partition that this element walks through
    // over the course of execution of the full workload. Instead of using the
    // += operator or the Add() method, we use the ExtrudeAdd() method.
    cur_state.dataspace_partition += point_set;
  }

  // Reset indices
  indices_[level] = cur->descriptor.start;

  bool dump = false; // (level >= 4);
  if (dump)
  {
    std::cout << "--------------------\n";
    std::cout << "LEVEL " << level << " (DeltaCalc)" << std::endl;
    std::cout << "--------------------\n";

    std::cout << "    Last:\n";
    cur_state.last_point_set.Print();
    std::cout << "    New:\n";
    point_set.Print();
  }
  
  // Calculate delta to send up to caller.
  problem::OperationSpace delta(workload_config_);
  if (!skip_delta)
  {
    delta = point_set - cur_state.last_point_set;
  }

  if (dump)
  {
    std::cout << "    Delta:\n";
    delta.Print();
    //std::cout << "    New after Minus op:\n";
    //point_set.Print();
  }    

  // Update last-seen point set for this level.
  cur_state.last_point_set = point_set;

  return delta;
}

void NestAnalysis::ComputeTemporalWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur,
                                     problem::OperationSpace& point_set,
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
  
  //
  // Step I: Compute Temporal Working Set.
  //

  problem::OperationPoint low_problem_point;
  problem::OperationPoint high_problem_point;

  // We use the pre-computed molds within this level range.
  // Above this level range, we use the transform problem-point to
  // translate, rotate or otherwise transform the mold.
  for (unsigned dim = 0; dim < unsigned(problem::NumDimensions); dim++)
  {
    low_problem_point[dim] = cur_transform_[dim] + mold_low_[level][dim];
    high_problem_point[dim] = cur_transform_[dim] + mold_high_[level][dim];
  }

  // Compute the polyhedron between the low and high problem
  // points (exclusive). Note that this special constructor
  // is only available for certain point-set implementations.
  point_set += problem::OperationSpace(workload_config_, low_problem_point, high_problem_point);

  if (dump)
  {
    std::cout << "Final point set:\n    ";
    point_set.Print();
  }

  //
  // Step II: Compute Accesses by accumulating deltas returned by inner levels.
  //
  std::uint64_t num_iterations = 1 +
    ((cur->descriptor.end - 1 - cur->descriptor.start) /
     cur->descriptor.stride);

  if (level == 0) // base
  {
    auto body_iterations = num_iterations * num_epochs_;
    // macs_ += body_iterations;
    if (spatial_id_ == 0)
    {
      // To avoid double counting of compute_cycles when there are multiple PEs.
      // compute_cycles_ += body_iterations;
      body_info_.accesses += body_iterations;
    }

    for (int pv = 0; pv < int(problem::DataType::Num); pv++)
    {
      // Write-backs of read-modify-write data types consume 2
      // accesses *except* for the first write.
      if (problem::IsReadWriteDataType(problem::DataType(pv)) &&
          cur_state.accesses[pv][0] != 0)
      {
        cur_state.accesses[pv][0] += body_iterations; // (2 * body_iterations); This fixup now happens in model/buffer.cpp.
      }
      else
      {
        cur_state.accesses[pv][0] += body_iterations;
      }

      // Set scatter factor (otherwise it will stay at 0 for temporal levels).
      cur_state.scatter_factors[pv][0] = 1;

      // Set cumulative hops for temporal levels.
      cur_state.cumulative_hops[pv][0] = 0.0;
    }
  }
  else // recurse
  {
    std::vector<problem::PerDataSpace<std::size_t>> temporal_delta_sizes;
    std::vector<std::uint64_t> temporal_delta_scale;

    bool EXTRAPOLATE_UNIFORM = true;
    bool RUN_LAST_ITERATION = false;
      
    if (EXTRAPOLATE_UNIFORM)
    {
      // What we would like to do is to *NOT* iterate through the entire loop
      // for this level, but instead fire iterations #0, #1 and #last, and
      // extrapolate the remainder based on the result of iteration #1.
      //
      // Sadly, skipping iterations causes entire sub-trees of calls to get
      // short-circuited, which leads to incorrect access counts.
      //
      // Therefore, instead of skipping the recursive calls altogether, we make
      // the calls but with a flag to skip the *local* delta calculation
      // within the call - which happens to be the single costliest step.

      int dim = int(cur->descriptor.dimension);
      int scale = per_level_dim_scales_[level][dim];
      auto saved_transform = cur_transform_[dim];

      // Iteration #0.
      indices_[level] = cur->descriptor.start;
      if (num_iterations >= 1)
      {
        // Invoke next (inner) loop level.
        ++cur;
        auto temporal_delta = ComputeWorkingSetsRecursive_(cur, false);
        --cur;

        temporal_delta_sizes.push_back(temporal_delta.GetSizes());
        temporal_delta_scale.push_back(1);
        cur_transform_[dim] += scale;

        indices_[level] += cur->descriptor.stride;
      }

      // Iterations #1 through #last-1/last.
      if ((RUN_LAST_ITERATION && num_iterations >= 3) ||
          (!RUN_LAST_ITERATION && num_iterations >= 2))
      {
        // Invoke next (inner) loop level, scaling up the number of epochs
        // by the number of virtual iterations we want to simulate.
        std::uint64_t virtual_iterations =
          RUN_LAST_ITERATION ? num_iterations - 2 : num_iterations - 1;

        auto saved_epochs = num_epochs_;
        num_epochs_ *= virtual_iterations;

        ++cur;
        auto temporal_delta = ComputeWorkingSetsRecursive_(cur, false);
        --cur;

        num_epochs_ = saved_epochs;

        temporal_delta_sizes.push_back(temporal_delta.GetSizes());
        temporal_delta_scale.push_back(virtual_iterations);

        cur_transform_[dim] += (scale * virtual_iterations);

        indices_[level] += (cur->descriptor.stride * virtual_iterations);
      }

      // Iteration #last.
      if (RUN_LAST_ITERATION && num_iterations >= 2)
      {
        // Invoke next (inner) loop level.
        ++cur;
        auto temporal_delta = ComputeWorkingSetsRecursive_(cur, false);
        --cur;

        // If we ran the virtual-iteration logic above, we shouldn't actually
        // use this returned delta, because we will receive the delta between
        // iteration #2 and #last. Instead, we just re-use the last delta by
        // increasing the #virtual iterations (scale) by 1.
        if (num_iterations >= 3)
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
      }

      cur_transform_[dim] = saved_transform;
    }
    else // not EXTRAPOLATE_UNIFORM
    {
      int dim = int(cur->descriptor.dimension);
      int scale = per_level_dim_scales_[level][dim];

      auto saved_transform = cur_transform_[dim];

      for (indices_[level] = cur->descriptor.start;
           indices_[level] < cur->descriptor.end;
           indices_[level] += cur->descriptor.stride)
      {
        // Invoke next (inner) loop level.
        ++cur;
        auto temporal_delta = ComputeWorkingSetsRecursive_(cur);
        --cur;

        temporal_delta_sizes.push_back(temporal_delta.GetSizes());
        temporal_delta_scale.push_back(1);

        cur_transform_[dim] += scale;
      }

      cur_transform_[dim] = saved_transform;
    } // EXTRAPOLATE_UNIFORM
    
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
        for (int pv = 0; pv < int(problem::DataType::Num); pv++)
        {
          final_delta_sizes[pv] += (temporal_delta_sizes[i][pv] * temporal_delta_scale[i]);
        }
      }

      for (int pv = 0; pv < int(problem::DataType::Num); pv++)
      {
        // Write-backs of read-modify-write data types consume 2
        // accesses *except* for the first write.
        if (problem::IsReadWriteDataType(problem::DataType(pv)) &&
            cur_state.accesses[pv][0] != 0)
        {
          cur_state.accesses[pv][0] += final_delta_sizes[pv] * num_epochs_; // (2 * final_delta_sizes[pv] * num_epochs_); This fixup now happens in model/buffer.cpp.
        }
        else
        {
          cur_state.accesses[pv][0] += final_delta_sizes[pv] * num_epochs_;
        }

        // Set scatter factor (otherwise it will stay at 0 for temporal levels).
        cur_state.scatter_factors[pv][0] = 1;

        // Set cumulative hops for temporal levels.
        cur_state.cumulative_hops[pv][0] = 0.0;

        // Update delta histogram. Hypothesis is we only need to do this for temporal levels.
        cur_state.delta_histograms[pv][final_delta_sizes[pv]] += num_epochs_;
        
      } // for (datatype)
    } // storage boundary
    
  } // level > 0
}

void NestAnalysis::ComputeSpatialWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur,
                                            problem::OperationSpace& point_set)
{
  int level = cur->level;
  ASSERT(master_spatial_level_[level]);

  std::uint64_t num_spatial_elems = spatial_fanouts_[level];
  spatial_id_ *= num_spatial_elems;

  // Deltas needed by each of the spatial elements.
  // This array will be filled by recursive calls.
  std::vector<problem::OperationSpace> spatial_deltas(num_spatial_elems,
                                                    problem::OperationSpace(workload_config_));

  // Indicates if each of the elements of the array above, was ever updated
  // by a recursive call. Only needed to ensure correctness.
  std::vector<bool> valid_delta(num_spatial_elems, false);

  FillSpatialDeltas(cur, point_set, spatial_deltas, valid_delta, 0 /* base_index */);
  
  // Check if each element of spatial_deltas was updated by recursive calls.
  for (auto it : valid_delta)
  {
    ASSERT(it);
  }

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

  std::vector<problem::PerDataSpace<bool>> unaccounted_delta;
  unaccounted_delta.resize(num_spatial_elems);
  for (uint64_t i = 0; i < num_spatial_elems; i++)
  {
    unaccounted_delta[i].fill(true);
  }

  // auto& cur_state = cur->live_state[spatial_id_];
  //  auto& accesses = nest_state_[cur->level].live_state[spatial_id_].accesses;
  auto& cur_state = nest_state_[cur->level].live_state[spatial_id_];

  problem::PerDataSpace<std::vector<std::uint64_t>>
    accesses_without_link_transfers, accesses_with_link_transfers,
    scatter_factors_without_link_transfers, scatter_factors_with_link_transfers,
    cumulative_hops_without_link_transfers, cumulative_hops_with_link_transfers;

  problem::PerDataSpace<std::vector<std::uint64_t>*>
    accesses, scatter_factors, cumulative_hops;
  
  for (unsigned pvi = 0; pvi < int(problem::DataType::Num); pvi++)
  {
    accesses_without_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    accesses_with_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    
    scatter_factors_without_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    scatter_factors_with_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    
    cumulative_hops_without_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    cumulative_hops_with_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    
    for (unsigned i = 0; i < accesses_without_link_transfers[pvi].size(); i++)
    {
      accesses_without_link_transfers[pvi][i] = 0;
      accesses_with_link_transfers[pvi][i] = 0;

      scatter_factors_without_link_transfers[pvi][i] = 0;
      scatter_factors_with_link_transfers[pvi][i] = 0;
      
      cumulative_hops_without_link_transfers[pvi][i] = 0;
      cumulative_hops_with_link_transfers[pvi][i] = 0;
    }

    // Default: do not use link transfers.
    accesses[pvi] = &accesses_without_link_transfers[pvi];
    scatter_factors[pvi] = &scatter_factors_without_link_transfers[pvi];
    cumulative_hops[pvi] = &cumulative_hops_without_link_transfers[pvi];
  }
  
  ComputeAccurateMulticastedAccesses(cur, spatial_deltas, unaccounted_delta,
                                     accesses_without_link_transfers,
                                     scatter_factors_without_link_transfers,
                                     cumulative_hops_without_link_transfers);

  static bool warning_printed = false;
  if (!warning_printed)
  {
    std::cerr << "WARNING: disabling link transfer computations. Link transfers "
              << "cause the multicast/scatter signature to change. We need to "
              << "record the impact of each potential multicast/scatter signature. "
              << "FIXME." << std::endl;
    warning_printed = true;
  }
  if (false && linked_spatial_level_[level])
  {
    // Reset unaccounted delta, and now count with link transfers.
    for (uint64_t i = 0; i < num_spatial_elems; i++)
    {
      unaccounted_delta[i].fill(true);
    }

    problem::PerDataSpace<std::uint64_t> link_transfers;

    ComputeNetworkLinkTransfers(cur, spatial_deltas, unaccounted_delta, link_transfers);

    ComputeAccurateMulticastedAccesses(cur, spatial_deltas, unaccounted_delta,
                                       accesses_with_link_transfers,
                                       scatter_factors_with_link_transfers,
                                       cumulative_hops_with_link_transfers);

    // Compare.
    for (unsigned pvi = 0; pvi < int(problem::DataType::Num); pvi++)
    {
      // if (problem::DataType(pvi) == problem::DataType::Weight)
      // {
      //   std::cout << "ACCESSES *WITH* LINK TRANSFERS\n";
      //   for (unsigned i = 0; i < accesses_with_link_transfers[pvi].size(); i++)
      //   {
      //     std::cout << "  " << i << ": " << accesses_with_link_transfers[pvi][i]
      //               << ", scatter: " << scatter_factors_with_link_transfers[pvi][i] << std::endl;
      //   }
      //   std::cout << "ACCESSES *WITHOUT* LINK TRANSFERS\n";
      //   for (unsigned i = 0; i < accesses_without_link_transfers[pvi].size(); i++)
      //   {
      //     std::cout << "  " << i << ": " << accesses_without_link_transfers[pvi][i]
      //               << ", scatter: " << scatter_factors_without_link_transfers[pvi][i] << std::endl;
      //   }
      // }
      
      std::uint64_t total_without = std::accumulate(accesses_without_link_transfers[pvi].begin(),
                                                    accesses_without_link_transfers[pvi].end(),
                                                    static_cast<std::uint64_t>(0));
      std::uint64_t total_with = std::accumulate(accesses_with_link_transfers[pvi].begin(),
                                                 accesses_with_link_transfers[pvi].end(),
                                                 static_cast<std::uint64_t>(0));
      if (total_with < total_without)
      {
        cur_state.link_transfers[pvi] += link_transfers[pvi];
        
        accesses[pvi] = &accesses_with_link_transfers[pvi];
        scatter_factors[pvi] = &scatter_factors_with_link_transfers[pvi];
        cumulative_hops[pvi] = &cumulative_hops_with_link_transfers[pvi];
      }
    }
  }

  for (unsigned pvi = 0; pvi < int(problem::DataType::Num); pvi++)
  {
    for (unsigned i = 0; i < cur_state.accesses[pvi].size(); i++)
    {
      cur_state.accesses[pvi][i] += (*accesses[pvi])[i];
        
      // Careful: overwriting scatter factor. The multicast/scatter signature must
      // either be un-initialized, or the accesses must be 0 (special case), or
      // it must match with the updated signature.
      if ((*accesses[pvi])[i] > 0)
      {
        if (cur_state.scatter_factors[pvi][i] == 0)
        {
          cur_state.scatter_factors[pvi][i] = (*scatter_factors[pvi])[i];
          cur_state.cumulative_hops[pvi][i] = (*cumulative_hops[pvi])[i];
        }
        else
        {
          // ****** FIXME ****** track multiple multicast/scatter signatures.
          assert(cur_state.scatter_factors[pvi][i] == (*scatter_factors[pvi])[i]);
        }
      }
    }      
  }

  //  auto& accesses = nest_state_[cur->level].live_state[spatial_id_].accesses;

  // Check that all deltas were accounted for correctly.
  for (uint64_t i = 0; i < num_spatial_elems; i++)
  {
    for (auto& it : unaccounted_delta[i])
    {
      ASSERT(!it);
    }
  }

  // Consistency check.
  for (unsigned pvi = 0; pvi < int(problem::DataType::Num); pvi++)
  {
    std::uint64_t fanout = 0;
    for (unsigned i = 0; i < cur_state.accesses[pvi].size(); i++)
    {
      fanout += (i+1) * cur_state.scatter_factors[pvi][i];
    }
    
    if (fanout != spatial_fanouts_[cur->level])
    {
      std::cerr << "FATAL: fanout mismatch, computed = " << fanout
                << " actual = " << spatial_fanouts_[cur->level] << std::endl;
      exit(1);
    }
  }  
  
  bool dump = false; // (level >= 4);
  if (dump)
  {
    std::cout << "-------\n";
    std::cout << "SPATIAL LEVEL " << level << std::endl;
    std::cout << "-------\n";

    std::cout << "analysis::LoopState:\n";
    for (int l = level; l < int(nest_state_.size()); l++)
    {
      std::cout << "    Level " << l << ": "
                << nest_state_[l].descriptor.dimension
                << " = " << indices_[l] << std::endl;
    }
    std::cout << "Final Spatial Point Set:\n    ";
    point_set.Print();
  }
}

// Computes deltas needed by the spatial elements in the next level.
// Will update a subset of the elements of spatial_deltas
void NestAnalysis::FillSpatialDeltas(std::vector<analysis::LoopState>::reverse_iterator cur,
                             problem::OperationSpace& point_set,
                             std::vector<problem::OperationSpace>& spatial_deltas,
                             std::vector<bool>& valid_delta,
                             std::uint64_t base_index,
                             int depth)
{
  int level = cur->level;

  // base_index determines which element of spatial_deltas
  // is going to be updated at the last recursive call to FillSpatialDeltas.
  // It's value is updated as we recursively call FillSpatialDeltas.
  // Very similar to how spatial_id_ is used to identify the spatial element
  // that we are currently computing the working set for.
  base_index *= cur->descriptor.end;

  // Sum of all point sets that are filled at this level.
  // We need to do this accumulation on a per-level basis
  // to makes sure point_set always has nice cuboidal shapes.
  problem::OperationSpace cur_level_point_set(workload_config_);

  bool dump = false; // (level >= 4);
  if (dump)
  {
    std::cout << "----------------------------\n";
    std::cout << "LEVEL " << level << " depth " << depth << std::endl;
    std::cout << "----------------------------\n";
  }
  
  if (level == 0)
  {
    // std::uint64_t body_iterations = (cur->descriptor.end - cur->descriptor.start) * num_epochs_;
    // macs_ += body_iterations;
    // to avoid double counting of compute_cycles_
    if (base_index == 0 && spatial_id_ == 0)
    {
      // compute_cycles_ += num_epochs_;
      body_info_.accesses += num_epochs_;
    }

    // No more recursive calls, directly update spatial_deltas.
    for (indices_[level] = cur->descriptor.start;
         indices_[level] < cur->descriptor.end;
         indices_[level] += cur->descriptor.stride)
    {
      std::uint64_t spatial_delta_index = base_index + indices_[level];
      ASSERT(spatial_delta_index < spatial_deltas.size());
      ASSERT(!valid_delta[spatial_delta_index]);

      spatial_deltas[spatial_delta_index] += IndexToOperationPoint_(indices_);
      valid_delta[spatial_delta_index] = true;
      cur_level_point_set += spatial_deltas[spatial_delta_index];
    }
  }
  else // level > 0
  {
    auto next = cur + 1;
    int dim = int(cur->descriptor.dimension);
    int scale = per_level_dim_scales_[level][dim];

    if (loop::IsSpatial(next->descriptor.spacetime_dimension))
    {
      // Next-inner loop level is spatial.
      for (indices_[level] = cur->descriptor.start;
           indices_[level] < cur->descriptor.end;
           indices_[level] += cur->descriptor.stride)
      {
        ++cur;

        FillSpatialDeltas(cur, cur_level_point_set, spatial_deltas, valid_delta,
                          base_index + indices_[level], depth+1);

        --cur;
        cur_transform_[dim] += scale;
      }
      cur_transform_[dim] -= scale * (cur->descriptor.end - cur->descriptor.start); // FIXME: stride.      
    }
    else // Next-inner loop level is temporal.
    {
      // make a backup before modifying it inside the loop
      uint64_t orig_spatial_id = spatial_id_;
      for (indices_[level] = cur->descriptor.start;
           indices_[level] < cur->descriptor.end;
           indices_[level] += cur->descriptor.stride)
      {
        ++cur;

        std::uint64_t spatial_delta_index = base_index + indices_[level];
        ASSERT(spatial_delta_index < spatial_deltas.size());
        ASSERT(!valid_delta[spatial_delta_index]);

        // Entering temporal dimension
        spatial_id_ = orig_spatial_id + spatial_delta_index;
        spatial_deltas[spatial_delta_index] = ComputeWorkingSetsRecursive_(cur);

        //std::cout << "    Received Spatial Delta from Recursive call (B):\n        ";
        //spatial_deltas[spatial_delta_index].Print();

        valid_delta[spatial_delta_index] = true;

        --cur;
        cur_transform_[dim] += scale;        
      }
      cur_transform_[dim] -= scale * (cur->descriptor.end - cur->descriptor.start);
      
      // restore to original value
      spatial_id_ = orig_spatial_id;
    }

    //
    // Compute Spatial Working Set using the polyhedral approach.
    //

    // Prepare low and high corners of problem space.
    auto indices_low = indices_;
    auto indices_high = indices_;

    // For every level below and including this one,
    // low and high points cover the full problem subspace
    // that the subnests walk over.
    for (int l = 0; l <= level; l++)
    {
      ASSERT(nest_state_[l].level == l);
      indices_low[l] = nest_state_[l].descriptor.start;
      indices_high[l] = nest_state_[l].descriptor.end - nest_state_[l].descriptor.stride;
    }

    auto low_problem_point = IndexToOperationPoint_(indices_low);
    auto high_problem_point = IndexToOperationPoint_(indices_high);
    
    // Compute the polyhedron between the low and high problem
    // points (exclusive). Note that this special constructor
    // is only available for certain point-set implementations.
    // Note: we aren't using +=. This means we're ignoring subvolumes
    // returned to us by recursive FillSpatialDeltas calls.
    cur_level_point_set = problem::OperationSpace(workload_config_, low_problem_point, high_problem_point);
  } // level > 0

  if (dump)
  {
    std::cout << "-------\n";
    std::cout << "LEVEL " << level << std::endl;
    std::cout << "-------\n";

    std::cout << "analysis::LoopState:\n";
    for (int l = level; l < int(nest_state_.size()); l++)
    {
      std::cout << "    Level " << l << ": "
                << nest_state_[l].descriptor.dimension
                << " = " << indices_[l] << std::endl;
    }
    std::cout << "Spatial Point Set Before Add:\n    ";
    point_set.Print();
    std::cout << "Adding:\n    ";
    cur_level_point_set.Print();
  }
  
  point_set += cur_level_point_set;

  if (dump)
  {
    std::cout << "Spatial Point Set After Add:\n    ";
    point_set.Print();
  }
}

// Exhaustively compare all pairs of deltas and infer multicast opportunities.
void NestAnalysis::ComputeAccurateMulticastedAccesses(
    std::vector<analysis::LoopState>::reverse_iterator cur,
    const std::vector<problem::OperationSpace>& spatial_deltas,
    std::vector<problem::PerDataSpace<bool>>& unaccounted_delta,
    problem::PerDataSpace<std::vector<std::uint64_t>>& accesses,
    problem::PerDataSpace<std::vector<std::uint64_t>>& scatter_factors,
    problem::PerDataSpace<std::vector<std::uint64_t>>& cumulative_hops)
{
  std::uint64_t num_deltas = spatial_deltas.size();

  // For each data type, records the number of unaccounted deltas
  // that the current delta matches with. This will be used
  // to infer the multicast factor for a specific delta.
  // reused across loop iterations to avoid initialization overheads.
  problem::PerDataSpace<uint64_t> num_matches;

  // For each datatype, records a ve
  
  // std::cout << "-----------------------------\n";
  // std::cout << "       COMPUTE MULTICAST     \n";
  // std::cout << "-----------------------------\n";
  // std::cout << "Epochs = " << num_epochs_ << std::endl;
  // std::cout << "Num deltas = " << num_deltas << std::endl;

  // for (std::uint64_t i = 0; i < num_deltas; i++)
  // {
  //   auto pv = problem::DataType::Weight;
  //   if (unaccounted_delta[i][int(pv)])
  //     std::cout << "UNACCOUNTED: ";
  //   else
  //     std::cout << "  ACCOUNTED: ";
  //   spatial_deltas[i].Print(pv);
  // }
  
  auto h_size = horizontal_sizes_[cur->level];
  auto v_size = vertical_sizes_[cur->level];

  for (std::uint64_t i = 0; i < num_deltas; i++)
  {
    num_matches.fill(0);
    
    problem::PerDataSpace<std::vector<std::uint64_t>> match_set;

    for (int pv = 0; pv < int(problem::DataType::Num); pv++)
    {
      if (!unaccounted_delta[i][pv])
      {
        // this delta was already accounted for,
        // skip the comparisons.
        continue;
      }

      unaccounted_delta[i][pv] = false;
      num_matches[pv] = 1;  // we match with ourselves.
      match_set[pv].push_back(i);

      for (std::uint64_t j = i + 1; j < num_deltas; j++)
      {
        if (unaccounted_delta[j][pv])
        {
          if (spatial_deltas[i].CheckEquality(spatial_deltas[j], pv))
          {
            // We have a match, record it
            unaccounted_delta[j][pv] = false;
            num_matches[pv]++;
            match_set[pv].push_back(j);
          }
        }
      }
    }

    // update the number of accesses at different multicast factors.
    for (int pv = 0; pv < int(problem::DataType::Num); pv++)
    {
      if (num_matches[pv] > 0)
      {
        accesses[pv][num_matches[pv] - 1] += (spatial_deltas[i].GetSize(pv) * num_epochs_);
        scatter_factors[pv][num_matches[pv] - 1]++;

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
        cumulative_hops[pv][num_matches[pv] - 1] += hops;
      }
    }
  }
}

// Compares two deltas, and if they are equal,
// records the opportunity for inter-PE link transfers.
void CompareSpatioTemporalDeltas(
    const std::vector<problem::OperationSpace>& cur_spatial_deltas,
    const std::vector<problem::OperationSpace>& prev_spatial_deltas,
    const std::uint64_t cur_spatial_index,
    const std::uint64_t prev_spatial_index,
    std::vector<problem::PerDataSpace<bool>>& inter_elem_reuse)
{
  auto& cur_delta = cur_spatial_deltas[cur_spatial_index];
  auto& prev_delta = prev_spatial_deltas[prev_spatial_index];

  for (int pv = 0; pv < int(problem::DataType::Num); pv++)
  {
    if (!cur_delta.IsEmpty(pv))
    {
      if (cur_delta.CheckEquality(prev_delta, pv))
      {
        // ASSERT(!inter_elem_reuse[cur_spatial_index][pv]);
        inter_elem_reuse[cur_spatial_index][pv] = true;
      }
    }
  }
}

void NestAnalysis::ComputeNetworkLinkTransfers(
    std::vector<analysis::LoopState>::reverse_iterator cur,
    const std::vector<problem::OperationSpace>& cur_spatial_deltas,
    std::vector<problem::PerDataSpace<bool>>&
    unaccounted_delta,
    problem::PerDataSpace<std::uint64_t>& link_transfers)
{
  // std::cout << "-----------------------------\n";
  // std::cout << "         LINK TRANSFERS      \n";
  // std::cout << "-----------------------------\n";
  
  // std::cout << "CUR BEFORE:" << std::endl;
  // for (std::uint64_t i = 0; i < cur_spatial_deltas.size(); i++)
  // {
  //   auto pv = problem::DataType::Weight;
  //   if (unaccounted_delta[i][int(pv)])
  //     std::cout << "  UNACCOUNTED: ";
  //   else
  //     std::cout << "    ACCOUNTED: ";
  //   cur_spatial_deltas[i].Print(pv);
  // }
  
  auto h_size = horizontal_sizes_[cur->level];
  auto v_size = vertical_sizes_[cur->level];

  // Imagine origin (0,0) at the top-left corner of a 2D spatial array.
  // Horizontal ids grow from left to right.
  // Vertical ids grow from top to bottom.
  auto GetLinearIndex = [&h_size, &v_size](std::uint64_t h_id, std::uint64_t v_id)
    {
      ASSERT(h_id < h_size && v_id < v_size);
      std::uint64_t linearIndex = v_id * h_size + h_id;  // row major layout
      return linearIndex;
    };

  auto& cur_state = cur->live_state[spatial_id_];
  auto& prev_spatial_deltas = cur_state.prev_point_sets[0];
  ASSERT(cur_spatial_deltas.size() == prev_spatial_deltas.size());
  int num_spatial_elems = spatial_fanouts_[cur->level];

  // std::cout << "PREV:" << std::endl;
  // for (std::uint64_t i = 0; i < prev_spatial_deltas.size(); i++)
  // {
  //   auto pv = problem::DataType::Weight;
  //   std::cout << "  "; prev_spatial_deltas[i].Print(pv);
  // }
  
  // for each spatial elements, this array records if the data
  // needed by the element can be obtained from any of the neighboring elements.
  std::vector<problem::PerDataSpace<bool>> inter_elem_reuse;
  inter_elem_reuse.resize(num_spatial_elems);
  for (int i = 0; i < num_spatial_elems; i++)
  {
    inter_elem_reuse[i].fill(false);
  }

  // FIXME The loops below can be codified in some way to avoid redundant LOC.

  // Test for a few hard-coded transfer patterns in horizontal and vertical
  // dimensions.

  // downward vertical transfers in each column
  for (std::uint64_t h_id = 0; h_id < h_size; h_id++)
  {
    for (std::uint64_t v_id = 1; v_id < v_size; v_id++)
    {
      auto cur_spatial_index = GetLinearIndex(h_id, v_id);
      auto prev_spatial_index = GetLinearIndex(h_id, v_id - 1);
      CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                  cur_spatial_index, prev_spatial_index,
                                  inter_elem_reuse);
    }
  }

  // upward vertical transfers in each column
  for (std::uint64_t h_id = 0; h_id < h_size; h_id++)
  {
    for (std::uint64_t v_id = 0; v_id < v_size - 1; v_id++)
    {
      auto cur_spatial_index = GetLinearIndex(h_id, v_id);
      auto prev_spatial_index = GetLinearIndex(h_id, v_id + 1);
      CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                  cur_spatial_index, prev_spatial_index,
                                  inter_elem_reuse);
    }
  }

  // horizontal transfers in each row from left to right
  for (std::uint64_t v_id = 0; v_id < v_size; v_id++)
  {
    for (std::uint64_t h_id = 1; h_id < h_size; h_id++)
    {
      auto cur_spatial_index = GetLinearIndex(h_id, v_id);
      auto prev_spatial_index = GetLinearIndex(h_id - 1, v_id);
      CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                  cur_spatial_index, prev_spatial_index,
                                  inter_elem_reuse);
    }
  }

  // horizontal transfers in each row from right to left
  for (std::uint64_t v_id = 0; v_id < v_size; v_id++)
  {
    for (std::uint64_t h_id = 0; h_id < h_size - 1; h_id++)
    {
      auto cur_spatial_index = GetLinearIndex(h_id, v_id);
      auto prev_spatial_index = GetLinearIndex(h_id + 1, v_id);
      CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                  cur_spatial_index, prev_spatial_index,
                                  inter_elem_reuse);
    }
  }

  // Compute the total number of accesses that can be bypassed
  // by using link transfers
  for (int i = 0; i < num_spatial_elems; i++)
  {
    for (int pv = 0; pv < int(problem::DataType::Num); pv++)
    {
      if (inter_elem_reuse[i][pv])
      {
        link_transfers[pv] += (cur_spatial_deltas[i].GetSize(pv) * num_epochs_);
        ASSERT(unaccounted_delta[i][pv]);
        unaccounted_delta[i][pv] = false;
      }
    }
  }

  // Time-shift the data in prev_point_sets array
  for (std::uint64_t i = 1; i < analysis::ElementState::MAX_TIME_LAPSE; i++)
  {
    for (int j = 0; j < num_spatial_elems; j++)
    {
      cur_state.prev_point_sets[i - 1][j] = cur_state.prev_point_sets[i][j];
    }
  }

  for (int j = 0; j < num_spatial_elems; j++)
  {
    cur_state.prev_point_sets[analysis::ElementState::MAX_TIME_LAPSE - 1][j] =
        cur_spatial_deltas[j];
  }

  // std::cout << "AFTER:" << std::endl;
  // for (std::uint64_t i = 0; i < cur_spatial_deltas.size(); i++)
  // {
  //   auto pv = problem::DataType::Weight;
  //   if (unaccounted_delta[i][int(pv)])
  //     std::cout << "  UNACCOUNTED: ";
  //   else
  //     std::cout << "    ACCOUNTED: ";
  //   cur_spatial_deltas[i].Print(pv);
  // }
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

#if 0
  std::cout << "Number of spatial elements at each level" << std::endl;
  for (int i = num_spatial_elems_.size() - 1; i >= 0; i--)
  {
    std::cout << num_spatial_elems_[i];
    if (master_spatial_level_[i]) std::cout << "(master)";
    if (linked_spatial_level_[i]) std::cout << "(linked)";
    std::cout << ", ";
  }
  std::cout << std::endl;
#endif
}

void NestAnalysis::InitStorageBoundaries()
{
  storage_boundary_level_.resize(nest_state_.size(), false);
  for (auto& i : storage_tiling_boundaries_)
  {
    ASSERT(i < storage_boundary_level_.size());
    storage_boundary_level_[i] = true;
  }
}

void NestAnalysis::InitSpatialFanouts()
{
  spatial_fanouts_.resize(nest_state_.size(), 1);
  horizontal_sizes_.resize(nest_state_.size(), 1);
  vertical_sizes_.resize(nest_state_.size(), 1);
  for (int cur_level = nest_state_.size() - 1; cur_level >= 0; cur_level--)
  {
    if (!loop::IsSpatial(nest_state_[cur_level].descriptor.spacetime_dimension))
    {
      spatial_fanouts_[cur_level] = 1;
    }
    else if (!master_spatial_level_[cur_level])
    {
      spatial_fanouts_[cur_level] = 0;
    }
    else
    {
      int next_temporal_level = cur_level;
      int scale_factor = 1;
      while (loop::IsSpatial(nest_state_[next_temporal_level].descriptor.spacetime_dimension))
      {
        if (loop::IsSpatialX(nest_state_[next_temporal_level].descriptor.spacetime_dimension))
        {
          horizontal_sizes_[cur_level] *=
              nest_state_[next_temporal_level].descriptor.end;
        }
        else
        {
          vertical_sizes_[cur_level] *=
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

      spatial_fanouts_[cur_level] =
          num_spatial_elems_[next_temporal_level] / num_spatial_elems_[cur_level];
      spatial_fanouts_[cur_level] *= scale_factor;

      ASSERT(spatial_fanouts_[cur_level] ==
             horizontal_sizes_[cur_level] * vertical_sizes_[cur_level]);
    }
  }

#if 0
  std::cout << "Spatial fanouts at each level" << std::endl;
  for (int i = num_spatial_elems_.size() - 1; i >= 0; i--)
  {
    std::cout << spatial_fanouts_[i];
    std::cout << ", ";
  }
  std::cout << std::endl;
#endif
}

void NestAnalysis::InitPerLevelDimScales()
{
  for (unsigned dim = 0; dim < problem::NumDimensions; dim++)
  {
    cur_transform_[dim] = 0;
  }

  std::uint64_t num_levels = nest_state_.size();

  per_level_dim_scales_.resize(num_levels);
  mold_low_.resize(num_levels);
  mold_high_.resize(num_levels);

  // running scale maintained for each dimension.
  problem::PerProblemDimension<std::uint64_t> cur_scale;
  cur_scale.fill(1);

  for (std::uint64_t level = 0; level < num_levels; level++)
  {
    auto desc = nest_state_[level].descriptor;
    int dim = int(desc.dimension);

    for (std::uint64_t dim = 0; dim < problem::NumDimensions; dim++)
    {
      per_level_dim_scales_[level][dim] = cur_scale[dim];
    }

    cur_scale[dim] *= (desc.end - desc.start);  // FIXME: assuming stride = 1

    for (std::uint64_t dim = 0; dim < problem::NumDimensions; dim++)
    {
      mold_low_[level][dim] = desc.start;
      mold_high_[level][dim] = cur_scale[dim] - 1; // FIXME: this is wrong.
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
  for (unsigned dim = 0; dim < problem::NumDimensions; dim++)
  {
    point[dim] = 0;
  }

  for (unsigned int level = 0; level < indices.size(); level++)
  {
    auto desc = nest_state_[level].descriptor;
    int dim = int(desc.dimension);
    point[dim] += (per_level_dim_scales_[level][dim] * indices[level]);
  }

  return point;
}

// A heuristic way to infer multicast opportunities.
// Will correctly identify multicasts when data type
// indices don't depend on multiple problem indices.
// (Ex. Weights and Outputs)
// When data type indices depend on multiple problem indices
// (Ex. Inputs), we break the assumption that multicast
// inference can be done at a per-level basis.

void NestAnalysis::ComputeApproxMulticastedAccesses(
    std::vector<analysis::LoopState>::reverse_iterator cur,
    const std::vector<problem::OperationSpace>& spatial_deltas)
{
  // Find number of spatial levels that correspond to this master spatial level.
  int master_level = cur->level;
  uint64_t num_spatial_levels;
  {
    int next_temporal_level = master_level;
    while (loop::IsSpatial(nest_state_[next_temporal_level].descriptor.spacetime_dimension) &&
           next_temporal_level > 0)
    {
      next_temporal_level--;
    }
    if (next_temporal_level == 0 && loop::IsSpatial(nest_state_[0].descriptor.spacetime_dimension))
    {
      next_temporal_level--;
    }
    num_spatial_levels = cur->level - next_temporal_level;
  }

  // for each level, stores if the tiling at that level results in multicasting
  // for any of the problem variables.
  problem::PerDataSpace<std::vector<bool>>
      is_multicast_level;  // per-pv, per-level
  for (auto& it : is_multicast_level)
  {
    it.resize(num_spatial_levels, false);
  }

  std::vector<uint64_t> max_vals(num_spatial_levels);
  std::vector<uint64_t> cur_vals(num_spatial_levels, 0);
  for (uint64_t i = 0; i < num_spatial_levels; i++)
  {
    max_vals[i] = nest_state_[master_level - i].descriptor.end;
  }

  auto GetSpatialIndex = [&max_vals, &cur_vals]() {
    uint64_t final_index = 0;
    uint64_t scale = 1;
    for (int i = max_vals.size() - 1; i >= 0; i--)
    {
      final_index += scale * cur_vals[i];
      scale *= max_vals[i];
    }
    return final_index;
  };

  for (uint64_t level = 0; level < num_spatial_levels; level++)
  {
    std::vector<uint64_t> indices_to_compare;
    for (uint64_t j = 0; j < max_vals[level]; j++)
    {
      cur_vals[level] = j;
      indices_to_compare.push_back(GetSpatialIndex());
    }
    cur_vals[level] = 0;  // reset

    problem::PerDataSpace<bool> is_multicast;
    is_multicast.fill(true);
    for (int pv = 0; pv < int(problem::DataType::Num); pv++)
    {
      for (uint64_t i = 1; i < indices_to_compare.size(); i++)
      {
        auto lhs_index = indices_to_compare[i];
        auto rhs_index = indices_to_compare[i - 1];
        if (!spatial_deltas[lhs_index]
                 .CheckEquality(spatial_deltas[rhs_index], pv))
        {
          is_multicast[pv] = false;
          break;
        }
      }
    }

    for (int pv = 0; pv < int(problem::DataType::Num); pv++)
    {
      is_multicast_level[pv][level] = is_multicast[pv];
    }
  }

  problem::PerDataSpace<std::size_t> summed_deltas;
  summed_deltas.fill(0);
  for (uint64_t i = 0; i < spatial_deltas.size(); i++)
  {
    auto delta_sizes = spatial_deltas[i].GetSizes();
    for (int pv = 0; pv < int(problem::DataType::Num); pv++)
    {
      summed_deltas[pv] += delta_sizes[pv];
    }
  }

  problem::PerDataSpace<std::size_t> multicast_factors;
  for (int pv = 0; pv < int(problem::DataType::Num); pv++)
  {
    uint64_t product_of_multicast_levels = 1;
    for (uint64_t level = 0; level < num_spatial_levels; level++)
    {
      if (is_multicast_level[pv][level])
      {
        product_of_multicast_levels *= max_vals[level];
      }
    }
    multicast_factors[pv] = product_of_multicast_levels;
  }

  // compute and update the number of accesses at various multicast factors.
  auto& accesses = nest_state_[master_level].live_state[spatial_id_].accesses;
  for (int pv = 0; pv < int(problem::DataType::Num); pv++)
  {
    ASSERT(accesses[pv].size() == spatial_deltas.size());
    ASSERT(summed_deltas[pv] % multicast_factors[pv] == 0);
    accesses[pv][multicast_factors[pv] - 1] +=
        (summed_deltas[pv] / multicast_factors[pv] * num_epochs_);
  }
}

} // namespace analysis
