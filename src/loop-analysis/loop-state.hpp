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

#include "mapping/loop.hpp"
#include "workload/problem-shape.hpp"
#include "workload/operation-space.hpp"

namespace analysis
{

// ---------------------------------------------------------------
// Live state for a single spatial element in a single loop level.
// ---------------------------------------------------------------
struct ElementState
{
  problem::OperationSpace last_point_set;
  problem::PerDataSpace<std::size_t> max_size;

  // Multicast functionality
  // Stores accesses with various multicast factors for each data type
  problem::PerDataSpace<std::vector<unsigned long>> accesses;
  problem::PerDataSpace<std::vector<unsigned long>> scatter_factors;
  problem::PerDataSpace<std::vector<double>> cumulative_hops;
  problem::PerDataSpace<std::map<unsigned long, unsigned long>> delta_histograms;

  // PE activity
  static constexpr std::uint64_t MAX_TIME_LAPSE = 1;
  // Records the data transferred to the next level in
  // last few loop iterations in time dimension
  // One for each spatial element in next level

  // time * element_id
  std::vector<std::vector<problem::OperationSpace>> prev_point_sets;

  // Number of transfers using links between spatial elements
  problem::PerDataSpace<unsigned long> link_transfers;

  void Reset()
  {
    last_point_set.Reset();
    max_size.fill(0);
    for (auto& it : accesses)
    {
      it.resize(0);
    }
    for (auto& it : scatter_factors)
    {
      it.resize(0);
    }
    for (auto& it : cumulative_hops)
    {
      it.resize(0);
    }
    for (auto& it : delta_histograms)
    {
      it.clear();
    }
    link_transfers.fill(0);

    for (uint64_t i = 0; i < prev_point_sets.size(); i++)
    {
      prev_point_sets[i].resize(0);
    }
    prev_point_sets.resize(0);
  }

  ElementState()
  {
    Reset();
  }
};

// -----------------------------------------------------------------
// Live state for a single loop level (across all spatial elements).
// -----------------------------------------------------------------
class LoopState
{
 public:
  int level;
  loop::Descriptor descriptor;
  std::vector<ElementState> live_state; // one for each spatial element

  LoopState() {}

  // Serialization
  friend class boost::serialization::access;
  
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0) 
  {
    if(version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(level);
      ar& BOOST_SERIALIZATION_NVP(descriptor);
    }
  }
};


} // namespace analysis
