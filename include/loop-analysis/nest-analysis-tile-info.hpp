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

#include <boost/serialization/vector.hpp>

#include "mapping/loop.hpp"
#include "workload/util/per-data-space.hpp"

namespace analysis
{

// data structures for nest-analysis to store datamovement and compute info
struct DataMovementInfo
{
  // Serialization.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0);

  std::size_t size;
  // std::size_t partition_size;
  bool distributed_multicast;
  std::vector<std::uint64_t> accesses;   // accesses at various multicast factors.
  std::vector<std::uint64_t> scatter_factors;
  std::vector<double> cumulative_hops;
  std::uint64_t link_transfers;
  std::vector<loop::Descriptor> subnest;
  std::uint64_t replication_factor;      // number of spatial elements at this level.
  std::uint64_t fanout;                  // per-element fanout to next-level.
  std::uint64_t distributed_fanout;      // max range of fanout if distributed multicast is used.
  bool is_on_storage_boundary;
  bool is_master_spatial;
  
  std::uint64_t GetTotalAccesses() const;
  
  std::uint64_t GetWeightedAccesses() const;

  void Reset();

  void Validate();
};

struct ComputeInfo
{
  std::uint64_t replication_factor;      // number of spatial elements at this level.
  std::uint64_t accesses;
  
  ComputeInfo();

  void Reset();
};

// compound tile info types to capture per-dataspace info
typedef problem::PerDataSpace<std::vector<DataMovementInfo>> CompoundDataMovementNest ; 
typedef std::vector<ComputeInfo> CompoundComputeNest;  // single vector, each element for a nest level, no fine-grained op type should be considered here
struct CompoundTileNest
{
   CompoundDataMovementNest compound_data_movement_info_nest;
   CompoundComputeNest compound_compute_info_nest;
};

} //namespace
