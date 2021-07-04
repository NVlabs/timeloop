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
#include "workload/shape-models/problem-shape.hpp"
#include "workload/util/per-data-space.hpp"
#include "workload/workload.hpp"
#include "workload/format-models/metadata-format.hpp"
#include "nest-analysis-tile-info.hpp"
#include "coordinate-space-tile-info.hpp"
#include "model/sparse-optimization-info.hpp"

namespace tiling
{

const int MaxTilingLevels = 16;

// each item stands for a rank, each rank as associated metadata occupancy
typedef std::vector<problem::PerRankMetaDataTileOccupancy> MetaDataTileOccupancy;


//
// DataMovementInfo
//

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
      ar& BOOST_SERIALIZATION_NVP(accesses);
      ar& BOOST_SERIALIZATION_NVP(subnest);
    }
  }
  CoordinateSpaceTileInfo coord_space_info;  // carries information such as the shape of the tile, and eventually the point set
  // Information particularly useful for tensors with metadata
  // all of the vectors below should have the same length... which is the fiber tree depth
  // note that, if a tensor is uncompressed and have no associated metadata (e.g., for eyeriss-style data gating),
  //      the tensor representation is just a dense tensor, which is already pre-analyzed in dense modeling
  std::vector<std::shared_ptr<problem::MetaDataFormat>> metadata_models; // metadata models (if any) for each rank of the tile
  std::vector<bool> rank_compressed; // if each rank is compressed
  std::vector<std::string> rank_formats; // each rank of the tensor should have metadata format, none for uncompressed
  std::size_t size; // for backward compatibility TODO: eventually we should use shape
  std::size_t shape;
  double expected_data_occupancy;
  MetaDataTileOccupancy expected_metadata_occupancy;
  problem::Shape::DataSpaceID dataspace_id ; // which dataspace does this tile belong to
  std::size_t partition_size;
  bool distributed_multicast;
  std::vector<std::uint64_t> accesses;       // accesses at various multicast factors.
  std::vector<std::uint64_t> scatter_factors;
  std::vector<double> cumulative_hops;
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
  // Tile density
  std::shared_ptr<problem::DensityDistribution> tile_density;  // statistical representation of tile data density
  // Fine grained actions, names defined in operation-type.hpp
  std::map<std::string, std::uint64_t> fine_grained_accesses;
  double expected_density;

  // Compression related
  bool compressed;
  bool has_metadata;

  // Only needed when tile has metadata
  std::vector<std::vector<loop::Descriptor>> metadata_subnest;
  std::vector<std::uint64_t> metadata_subtile_shape;
  std::vector<std::uint64_t> fiber_shape;
  double child_level_metadata_occupancy_ratio;

  // Parent child level records
  unsigned parent_level;
  std::string parent_level_name;
  unsigned child_level;
  DataMovementInfo* child_level_ptr;
  DataMovementInfo* parent_level_ptr;

  std::uint64_t GetTotalAccesses() const
  {
    return std::accumulate(accesses.begin(), accesses.end(), static_cast<std::uint64_t>(0));
  }
  
  std::uint64_t GetWeightedAccesses() const
  {
    std::uint64_t total = 0;
    for (std::uint32_t i = 0; i < accesses.size(); i++)
    {
      total += accesses[i] * (i + 1);
    }
    return total;
  }

  void Reset()
  {
    size = 0;
    shape = 0;
    expected_data_occupancy = std::numeric_limits<unsigned>::max();
    expected_metadata_occupancy = {};
    partition_size = 0;
    accesses.resize(0);
    scatter_factors.resize(0);
    cumulative_hops.resize(0);
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
    has_metadata = false;
    parent_level = std::numeric_limits<unsigned>::max();
    child_level = std::numeric_limits<unsigned>::max();
    parent_level_ptr = NULL;
    child_level_ptr = NULL;
    child_level_metadata_occupancy_ratio = 0;
    fine_grained_accesses.clear();
    metadata_updates = 0;
    metadata_fills = 0;
    metadata_reads = 0;
    metadata_subnest.clear();
    metadata_subtile_shape.clear();
    fiber_shape.clear();
    coord_space_info.Clear();
    tile_density = NULL;
    expected_density = 0;
  }

  void Validate()
  {
    std::uint64_t f = 0;
    for (std::uint64_t i = 0; i < fanout; i++)
    {
      if (accesses[i] != 0)
      {
        auto multicast_factor = i + 1;
        auto scatter_factor = scatter_factors[i];
        f += (multicast_factor * scatter_factor);
      }
    }

    if (f != fanout)
    {
      std::cerr << "ERROR: sigma(multicast * scatter) != fanout." << std::endl;
      std::cerr << "  dumping (multicast, scatter) pairs:" << std::endl;
      for (std::uint64_t i = 0; i < fanout; i++)
      {
        if (accesses[i] != 0)
        {
          auto multicast_factor = i + 1;
          auto scatter_factor = scatter_factors[i];
          std::cerr << "    " << multicast_factor << ", " << scatter_factor << std::endl;
        }
      }
      std::cerr << "  sigma(multicast, scatter) = " << f << std::endl;
      std::cerr << "  fanout = " << fanout << std::endl;
      exit(1);
    }
  }

  void SetDensityModel(std::shared_ptr<problem::DensityDistribution> tile_density_ptr)
  {
    tile_density = tile_density_ptr;
  }

  //void SetSubTileShapes(std::vector<std::size_t> subtile_shapes){subtile_shapes_ = subtile_shapes;}
  void SetTensorRepresentation(const sparse::PerDataSpaceCompressionInfo& compression_opt_spec,
                               const std::vector<loop::Descriptor> subnests);
  void SetTensorRepresentation(const sparse::PerDataSpaceCompressionInfo& compression_opt_spec);
  void SetTensorRepresentation(); // for default dense tensors

  std::string GetDataSpaceName() const { return problem::GetShape()->DataSpaceIDToName.at(dataspace_id);}
  bool GetHasMetaData() const { return has_metadata;}
  std::string GetDensityType() const
  {
    return tile_density->GetDistributionType();
  }
  std::string GetMetaDataFormatName() const;
  std::uint64_t GetNumMetaDataRanks() const
  {
    if (! has_metadata) return 0;
    else return metadata_models.size();
  }
  CoordinateSpaceTileInfo GetCoordinateSpaceInfo() const;
  CoordinateSpaceTileInfo GetChildTileCoordinateSpaceInfo() const;

  // do not use this unless super necessary,
  // as density model interface change will break the logic external to sparse modeling step
  std::shared_ptr<problem::DensityDistribution> GetTileDensityModel() const { return tile_density; }


  // More involved getter functions
  // get data tile occupancy
  std::uint64_t GetMaxDataTileOccupancyByConfidence(const double confidence = 1.0) const;
  double GetDataTileOccupancyProbability(const std::uint64_t occupancy) const;
  double GetChildLevelDataTileOccupancyProbability(const std::uint64_t occupancy) const;
  std::uint64_t GetMinDataTileOccupancy() const;

  // get metadata tile occupancy
  MetaDataTileOccupancy GetMetaDataTileOccupancyGivenDataTile(const CoordinateSpaceTileInfo& cur_coord_tile) const;
  MetaDataTileOccupancy GetMaxMetaDataTileOccupancyByConfidence(const double confidence = 1.0) const;
  double GetExpectedAggregatedMetaDataTileOccupancy() const;

  // density value related
  double GetMaxTileDensityByConfidence(const double confidence = 1.0) const;
  double GetExpectedTileDensity() const;
};

//
// Compute info
//

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