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

#include "workload/format-models/metadata-format.hpp"

namespace problem
{

//---------------------------------------//
//   API with sparse modeling step       //
//---------------------------------------//

MetaDataOccupancyQuery::MetaDataOccupancyQuery(std::uint64_t max_num_of_fibers,
                                               std::uint64_t cur_rank_fiber_shape,
                                               tiling::CoordinateSpaceTileInfo cur_rank_coord_tile,
                                               tiling::CoordinateSpaceTileInfo next_rank_coord_tile,
                                               std::shared_ptr<problem::DensityDistribution> density_ptr):
    max_number_of_fibers(max_num_of_fibers),
    cur_rank_fiber_shape(cur_rank_fiber_shape),
    cur_rank_coord_tile(cur_rank_coord_tile),
    next_rank_coord_tile(next_rank_coord_tile),
    tile_density_ptr(density_ptr)
{
}

MetaDataOccupancyQuery::MetaDataOccupancyQuery(std::uint64_t max_num_of_fibers,
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
    confidence(confidence)
{
}

// Getters
std::uint64_t MetaDataOccupancyQuery::MaxNumFibers() const
{
  return max_number_of_fibers;
}

tiling::CoordinateSpaceTileInfo MetaDataOccupancyQuery::CurRankCoordTile() const
{
  return cur_rank_coord_tile;
}

tiling::CoordinateSpaceTileInfo MetaDataOccupancyQuery::NextRankCoordTile() const
{
  return next_rank_coord_tile;
}

std::shared_ptr<problem::DensityDistribution> MetaDataOccupancyQuery::TileDensityPtr() const
{
  return tile_density_ptr;
}

std::shared_ptr<problem::DensityDistribution> MetaDataOccupancyQuery::NextRankTileDensityPtr() const
{
  return tile_density_ptr;
}

std::uint64_t MetaDataOccupancyQuery::CurRankFiberShape() const
{
  return cur_rank_fiber_shape;
}


// Setters
void PerRankMetaDataTileOccupancy::SetEmpty()
{
  metadata_units = 0;
  payload_units = 0;
  metadata_word_bits = 0;
  payload_word_bits = 0;
}

void PerRankMetaDataTileOccupancy::SetPayloadUnits(const std::uint64_t units)
{
  payload_units = units;
}

// API
double PerRankMetaDataTileOccupancy::MetaDataUnits() const
{
  return metadata_units;
}

double PerRankMetaDataTileOccupancy::PayloadUnits() const
{
  return payload_units;
}

std::uint32_t PerRankMetaDataTileOccupancy::MetaDataWordBits() const
{
  return metadata_word_bits;
}

std::uint32_t PerRankMetaDataTileOccupancy::PayloadWordBits() const
{
  return payload_word_bits;
}

double PerRankMetaDataTileOccupancy::TotalMetDataAndPayloadUnits() const
{
  return metadata_units + payload_units;
}

void PerRankMetaDataTileOccupancy::Scale(double s)
{
  metadata_units *= s;
  payload_units *= s;
}

void PerRankMetaDataTileOccupancy::Add(PerRankMetaDataTileOccupancy m)
{

  if (metadata_units != 0 && m.metadata_units != 0)
    assert(metadata_word_bits == m.metadata_word_bits);
  if (payload_units != 0 && m.payload_units != 0)
    assert(payload_word_bits == m.payload_word_bits);

  metadata_units += m.MetaDataUnits();
  payload_units += m.PayloadUnits();
}

bool PerRankMetaDataTileOccupancy::IsEmpty()
{
  if (MetaDataUnits() == 0 && PayloadUnits() == 0)
  {
    return true;
  }
  else
  {
    return false;
  }
}


//-------------------------------------------------//
//             MetaData Format Specs               //
//-------------------------------------------------//

MetaDataFormatSpecs::~MetaDataFormatSpecs()
{
}


//-------------------------------------------------//
//          MetaDataFormat (base class)            //
//-------------------------------------------------//

MetaDataFormat::~MetaDataFormat()
{
}


} // namespace problem
