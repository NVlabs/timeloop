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

#include <bitset>

#include "mapping/loop.hpp"
#include "util/numeric.hpp"
#include "workload/problem-shape.hpp"
#include "workload/per-data-space.hpp"
#include "workload/workload.hpp"
#include "nest-analysis-tile-info.hpp"

namespace tiling
{

const int MaxTilingLevels = 16;

struct DataMovementInfo
{
  friend class boost::serialization::access;

  // Serialization.
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0) 
  {
    if(version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(size);
      //ar& BOOST_SERIALIZATION_NVP(accesses);
      ar& BOOST_SERIALIZATION_NVP(subnest);
    }
  }

  std::size_t size;
  std::size_t partition_size;
  std::size_t compressed_size;
  bool distributed_multicast;
  AccessStatMatrix access_stats;
  std::uint64_t content_accesses;
  std::uint64_t fills;
  std::uint64_t reads;
  std::uint64_t updates;
  std::uint64_t metadata_updates;
  std::uint64_t metadata_fills;
  std::uint64_t metadata_reads;
  std::uint64_t temporal_reductions;
  std::uint64_t link_transfers;
  std::uint64_t peer_accesses;           // number of accesses caused by link transfers in the previous level 
  std::uint64_t peer_fills;              // number of fills caused by link transfers in the previous level
  std::vector<loop::Descriptor> subnest;
  std::uint64_t replication_factor;      // number of spatial elements at this level.
  std::uint64_t fanout;                  // per-element fanout to next-level.
  std::uint64_t distributed_fanout;      // max range of fanout if distributed multicast is used.
  bool is_on_storage_boundary;
  bool is_master_spatial;
  //double partition_fraction;
  std::size_t partition_fraction_denominator;
  // tile density
  std::shared_ptr<problem::DensityDistribution> tile_density;  // statistical representation of tile data density
  // fine grained actions, names defined in operation-type.hpp
  std::map<std::string, std::uint64_t> fine_grained_accesses;

  // compression related
  bool compressed;
  std::string metadata_format;
  // std::uint64_t metadata_tile_size; // move population to buffer.cpp due to confidence
  // for CSR only
  std::vector<problem::Shape::DimensionID> rank0_list;
  std::vector<problem::Shape::DimensionID> rank1_list;
  std::uint64_t dense_rank1_fills;
  std::uint64_t dense_rank0_fills;

  // parent/child level for inferring decompression/compression overhead
  unsigned parent_level;
  std::string parent_level_name;
  unsigned child_level;
  bool parent_level_compressed;
  bool child_level_compressed;
  uint64_t child_level_tile_size;

  void Reset()
  {
    size = 0;
    partition_size = 0;
    access_stats.clear();
    content_accesses = 0;
    fills = 0;
    link_transfers = 0;
    peer_accesses = 0;
    peer_fills = 0;
    subnest.resize(0);
    replication_factor = 0;
    fanout = 0;
    distributed_fanout = 0;
    compressed = false;
    compressed_size = 0;
    tile_density = NULL;
    metadata_format.resize(0);
    parent_level=std::numeric_limits<unsigned>::max();
    child_level=std::numeric_limits<unsigned>::max();
    parent_level_compressed = false;
    child_level_compressed = false;
    fine_grained_accesses.clear();
    rank1_list.resize(0);
    rank0_list.resize(0);
    dense_rank1_fills = 0;
    dense_rank0_fills = 0;
    metadata_updates = 0;
    metadata_fills = 0;
    metadata_reads = 0;
  }

  void Validate()
  {
    // std::uint64_t f = 0;
    // for (std::uint64_t i = 0; i < fanout; i++)
    // {
    //   if (accesses[i] != 0)
    //   {
    //     auto multicast_factor = i + 1;
    //     auto scatter_factor = scatter_factors[i];
    //     f += (multicast_factor * scatter_factor);
    //   }
    // }

    // if (f != fanout)
    // {
    //   std::cerr << "ERROR: sigma(multicast * scatter) != fanout." << std::endl;
    //   std::cerr << "  dumping (multicast, scatter) pairs:" << std::endl;
    //   for (std::uint64_t i = 0; i < fanout; i++)
    //   {
    //     if (accesses[i] != 0)
    //     {
    //       auto multicast_factor = i + 1;
    //       auto scatter_factor = scatter_factors[i];
    //       std::cerr << "    " << multicast_factor << ", " << scatter_factor << std::endl;
    //     }
    //   }
    //   std::cerr << "  sigma(multicast, scatter) = " << f << std::endl;
    //   std::cerr << "  fanout = " << fanout << std::endl;
    //   exit(1);
    // }
  }

};

struct ComputeInfo
{
  std::uint64_t replication_factor;      // number of spatial elements at this level.
  std::uint64_t accesses;
  std::uint64_t compute_cycles;

  // fine grained actions, names defined in operation-type.hpp
  std::map<std::string, std::uint64_t> fine_grained_accesses; 
  
  ComputeInfo() { Reset(); }

  void Reset()
  {
    replication_factor = 0;
    accesses = 0;
    compute_cycles = 0;
  }
};

// datatypes needed before transpose
// indexing order: [datatype/optype, nest_level]
typedef problem::PerDataSpace<std::vector<DataMovementInfo>> CompoundDataMovementNest ; 
typedef std::vector<ComputeInfo> ComputeNest;
struct CompoundTileNest{
   CompoundDataMovementNest compound_data_movement_info_nest;
   ComputeNest compute_info_nest;
};


// datatypes needed after transpose
typedef problem::PerDataSpace<DataMovementInfo> CompoundDataMovementInfo;

// indexing order: [nest_level, datatype/optype]
struct CompoundTile{
  CompoundDataMovementInfo data_movement_info;
  ComputeInfo compute_info;
};

typedef std::vector<CompoundTile> NestOfCompoundTiles;
typedef problem::PerDataSpace<bool> CompoundMask;

typedef problem::PerDataSpace<std::bitset<MaxTilingLevels>> CompoundMaskNest;
typedef std::vector<CompoundMask> NestOfCompoundMasks;

} //namespace
