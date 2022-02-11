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

#include <bitset>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <iostream>

#include "compound-config/compound-config.hpp"
#include "workload/workload.hpp"
#include "workload/shape-models/problem-shape.hpp"
#include "workload/density-models/density-distribution.hpp"

namespace problem
{

//---------------------------------------//
//   API with sparse modeling step       //
//---------------------------------------//

// query struct received from sparse modeling
struct MetaDataOccupancyQuery
{
  std::uint64_t max_number_of_fibers;
  std::uint64_t cur_rank_fiber_shape;
  tiling::CoordinateSpaceTileInfo cur_rank_coord_tile;
  tiling::CoordinateSpaceTileInfo next_rank_coord_tile;
  std::shared_ptr<problem::DensityDistribution> tile_density_ptr;
  double confidence = 1.0;

  // constructors
  MetaDataOccupancyQuery();
  MetaDataOccupancyQuery(std::uint64_t max_num_of_fibers,
                         std::uint64_t cur_rank_fiber_shape,
                         tiling::CoordinateSpaceTileInfo cur_rank_coord_tile,
                         tiling::CoordinateSpaceTileInfo next_rank_coord_tile,
                         std::shared_ptr<problem::DensityDistribution> density_ptr);
  MetaDataOccupancyQuery(std::uint64_t max_num_of_fibers,
                         std::uint64_t cur_rank_fiber_shape,
                         tiling::CoordinateSpaceTileInfo cur_rank_coord_tile,
                         tiling::CoordinateSpaceTileInfo next_rank_coord_tile,
                         std::shared_ptr<problem::DensityDistribution> density_ptr,
                         double confidence);

  // Getters
  std::uint64_t MaxNumFibers() const;
  tiling::CoordinateSpaceTileInfo CurRankCoordTile() const;
  tiling::CoordinateSpaceTileInfo NextRankCoordTile() const;
  std::shared_ptr<problem::DensityDistribution> TileDensityPtr() const;
  std::shared_ptr<problem::DensityDistribution> NextRankTileDensityPtr() const;
  std::uint64_t CurRankFiberShape() const;
};

// occupancy object returned to sparse modeling
struct PerRankMetaDataTileOccupancy
{
  double metadata_units;
  double payload_units;
  std::uint32_t metadata_word_bits;  // user-specified metadata word bits ( e.g., runlength width for RLE)
  std::uint32_t payload_word_bits;   // user-specified payload word bits (memory pointers)

  // Setters
  void SetEmpty();
  void SetPayloadUnits(const std::uint64_t units);

  // API
  double MetaDataUnits() const;
  double PayloadUnits() const;
  std::uint32_t MetaDataWordBits() const;
  std::uint32_t PayloadWordBits() const;
  double TotalMetDataAndPayloadUnits() const;

  void Scale(double s);
  void Add(PerRankMetaDataTileOccupancy m);

  bool IsEmpty();

  // Serialization.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(payload_units);
      ar& BOOST_SERIALIZATION_NVP(metadata_units);
      ar& BOOST_SERIALIZATION_NVP(payload_word_bits);
      ar& BOOST_SERIALIZATION_NVP(metadata_word_bits);
    }
  }
};


//-------------------------------------------------//
//             MetaData Format Specs               //
//-------------------------------------------------//

struct MetaDataFormatSpecs
{
  virtual ~MetaDataFormatSpecs();

  virtual std::shared_ptr<MetaDataFormatSpecs> Clone() const = 0;

  virtual const std::string Name() const = 0;
  virtual bool RankCompressed() const = 0;
  virtual std::vector<problem::Shape::FlattenedDimensionID> DimensionIDs() const = 0;

  virtual std::uint32_t MetaDataWordBits() const = 0;
  virtual std::uint32_t PayloadWordBits() const = 0;
  virtual void SetMetaDataWordBits(std::uint32_t word_bits) = 0;
  virtual void SetPayloadWordBits(std::uint32_t word_bits) = 0;
  
  std::string name = "UNSET";
  std::uint32_t payload_word_bits = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t metadata_word_bits = std::numeric_limits<std::uint32_t>::max();

  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(name);
    }
  }
};  // struct MetaDataFormatSpecs

BOOST_SERIALIZATION_ASSUME_ABSTRACT(MetaDataFormatSpecs)


//-------------------------------------------------//
//      MetaDataFormat (base class)                //
//-------------------------------------------------//

class MetaDataFormat{

public:
  // destructor
  virtual ~MetaDataFormat();

  // API
  virtual PerRankMetaDataTileOccupancy GetOccupancy(const MetaDataOccupancyQuery& query) const = 0;
  virtual bool RankCompressed() const = 0;
  virtual bool CoordinatesImplicit() const = 0;
  virtual std::vector<problem::Shape::FlattenedDimensionID> GetDimensionIDs() const = 0;
  virtual std::string GetFormatName() const = 0;
  virtual bool MetaDataImplicitAsLowestRank() const = 0;

  // Serialization.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    (void) ar;
    (void) version;
  }

}; // class MetaDataFormat

BOOST_SERIALIZATION_ASSUME_ABSTRACT(MetaDataFormat)

} // namespace problem
