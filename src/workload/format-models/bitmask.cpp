/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "workload/format-models/bitmask.hpp"

namespace problem
{

Bitmask::Bitmask() {}
Bitmask::Bitmask(const Specs& specs) : specs_(specs){ is_specced_ = true;}
Bitmask::~Bitmask(){}

Bitmask::Specs Bitmask::ParseSpecs(config::CompoundConfigNode format_specs)
{
  Bitmask::Specs specs;
  // by default, no special attributes need to be set manually by the users
  specs.payload_word_bits = std::numeric_limits<std::uint32_t>::max();
  specs.metadata_word_bits = 1; // default to 1 bit

  if (format_specs.exists("payload-word-bits"))
  {
    format_specs.lookupValue("payload-word-bits", specs.payload_word_bits);
  }

  if (format_specs.exists("metadata-word-bits"))
  {
    format_specs.lookupValue("metadata-word-bits", specs.metadata_word_bits);
  }

  return specs;
}

PerRankMetaDataTileOccupancy Bitmask::GetOccupancy(const MetaDataOccupancyQuery& query) const
{

  double prob_empty_fibers = query.TileDensityPtr()->GetTileOccupancyProbability(query.CurRankCoordTile(), 0);
  double number_of_fibers = query.MaxNumFibers() * (1-prob_empty_fibers);
  double prob_empty_coord = query.TileDensityPtr()->GetTileOccupancyProbability(query.NextRankCoordTile(), 0);
  double number_of_nnz_coord_per_fiber = ceil(query.CurRankFiberShape() * (1 - prob_empty_coord));

  PerRankMetaDataTileOccupancy occupancy;
  occupancy.payload_word_bits = specs_.payload_word_bits;
  occupancy.metadata_word_bits = specs_.metadata_word_bits;
  occupancy.metadata_units = number_of_fibers * query.CurRankFiberShape();
  occupancy.payload_units = number_of_fibers * number_of_nnz_coord_per_fiber;

  return occupancy;
}

bool Bitmask::RankCompressed() const
{
  assert(is_specced_);
  return specs_.rank_compressed;
}

bool Bitmask::CoordinatesImplicit() const
{
  assert(is_specced_);
  return specs_.coordinates_implicit;
}


std::vector<problem::Shape::FlattenedDimensionID> Bitmask::GetDimensionIDs() const
{
  assert(is_specced_);
  return specs_.dimension_ids;
}

std::string Bitmask::GetFormatName() const
{
  assert(is_specced_);
  return specs_.name;
}

const MetaDataFormatSpecs& Bitmask::GetSpecs() const
{
  assert(is_specced_);
  return  specs_;
}

} // namespace problem
