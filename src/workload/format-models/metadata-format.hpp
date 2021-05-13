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
                            std::shared_ptr<problem::DensityDistribution> density_ptr):
                            max_number_of_fibers(max_num_of_fibers),
                            cur_rank_fiber_shape(cur_rank_fiber_shape),
                            cur_rank_coord_tile(cur_rank_coord_tile),
                            next_rank_coord_tile(next_rank_coord_tile),
                            tile_density_ptr(density_ptr){}
     MetaDataOccupancyQuery(std::uint64_t max_num_of_fibers,
                           std::uint64_t cur_rank_fiber_shape,
                           tiling::CoordinateSpaceTileInfo cur_rank_coord_tile,
                           tiling::CoordinateSpaceTileInfo next_rank_coord_tile,
                           std::shared_ptr<problem::DensityDistribution> density_ptr,
                           double confidence):
                           max_number_of_fibers(max_num_of_fibers),
                           cur_rank_fiber_shape(cur_rank_fiber_shape),
                           cur_rank_coord_tile(cur_rank_coord_tile),
                           next_rank_coord_tile(next_rank_coord_tile),
                           tile_density_ptr(density_ptr),
                           confidence(confidence){}

     // Getters
     std::uint64_t MaxNumFibers() const {return max_number_of_fibers;}
     tiling::CoordinateSpaceTileInfo CurRankCoordTile() const {return cur_rank_coord_tile;}
     tiling::CoordinateSpaceTileInfo NextRankCoordTile() const {return next_rank_coord_tile;}
     std::shared_ptr<problem::DensityDistribution> TileDensityPtr() const {return tile_density_ptr;}
     std::shared_ptr<problem::DensityDistribution> NextRankTileDensityPtr() const {return tile_density_ptr;}
     std::uint64_t CurRankFiberShape() const {return cur_rank_fiber_shape;}

   };

  // occupancy object returned to sparse modeling
  struct PerRankMetaDataTileOccupancy
  {
    double metadata_units;
    double payload_units;
    // if any of the datawidth is -1, it means unspecified but needed, will thus use the hardware attribute later
    int metadata_width;  // user-specified metadata payload width ( e.g., runlength width for RLE)
    int payload_width;   // user-specified payload width (memory pointers)

    // Setters
    void SetEmpty()
    {
      metadata_units = 0;
      payload_units = 0;
      metadata_width = -1;
      payload_width = -1;
    }

    void SetPayloadUnits(const std::uint64_t units)
    {
      payload_units = units;
    }

    // API
    double MetaDataUnits() const {return metadata_units;}
    double PayloadUnits() const {return payload_units;}
    int MetaDataWidth() const {return metadata_width;}
    int PayloadWidth() const {return payload_width;}
    double TotalMetDataAndPayloadUnits() const {return metadata_units + payload_units;}

    void Scale( double s)
    {
      metadata_units *= s;
      payload_units *= s;
    }

    void Add ( PerRankMetaDataTileOccupancy m)
    {
      metadata_units += m.MetaDataUnits();
      payload_units += m.PayloadUnits();
    }

    friend class boost::serialization::access;

    // Serialization.
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version=0)
    {
      if(version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(payload_units);
        ar& BOOST_SERIALIZATION_NVP(metadata_units);
        ar& BOOST_SERIALIZATION_NVP(payload_width);
        ar& BOOST_SERIALIZATION_NVP(metadata_width);
      }
    }

  };


//-------------------------------------------------//
//         MetaData Format Specs                   //
//-------------------------------------------------//

  struct MetaDataFormatSpecs{

    virtual ~MetaDataFormatSpecs() { }

    virtual std::shared_ptr<MetaDataFormatSpecs> Clone() const = 0;

    virtual const std::string Name() const = 0;
    virtual bool RankCompressed() const = 0;
    virtual std::vector<problem::Shape::DimensionID> DimensionIDs() const = 0;

    std::string name = "UNSET";

    // Serialization
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0){
      if (version == 0){
        ar& BOOST_SERIALIZATION_NVP(name);
      }
    }
  };  // struct MetaDataFormatSpecs

  BOOST_SERIALIZATION_ASSUME_ABSTRACT(MetaDataFormatSpecs)

//-------------------------------------------------//
//      MetaDataFormat (base class)          //
//-------------------------------------------------//

class MetaDataFormat{

public:
  // destructor
  virtual ~MetaDataFormat(){}

  // API
  virtual PerRankMetaDataTileOccupancy GetOccupancy(const MetaDataOccupancyQuery& query) const = 0;
  virtual bool GetRankCompressed() const = 0;
  virtual std::vector<problem::Shape::DimensionID> GetDimensionIDs() const = 0;
  virtual std::string GetFormatName() const = 0;
  virtual bool ImplicitMetadata() const = 0;

  // Serialization.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0){
    (void) ar;
    (void) version;
  }

}; // class MetaDataFormat

BOOST_SERIALIZATION_ASSUME_ABSTRACT(MetaDataFormat)

} // namespace problem