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

#include "tiling-tile-info.hpp"

namespace tiling
{

void DataMovementInfo::SetTensorRepresentation()
{

  // set all fields to empty for default dense tensors with no metadata as these info will not be useful
  rank_compressed = {};
  rank_formats = {};
  metadata_models_ = {};
  coord_space_info.Set(shape, dataspace_id);

  // for safety, set the quick accessors (should be initialized just right)
  compressed = false;
  has_metadata = false;

}

void DataMovementInfo::SetTensorRepresentation(const sparse::PerDataSpaceCompressionInfo& compression_opt_spec,
                                               const std::vector <loop::Descriptor> subnests)
{

  // About the inputs:
  // 1) since we assume the mapping must be fully concordant,
  //     the rank order in the compression opt must fully specify all necessary compression format info
  // 2) subnests[0] is the inner most loop, subnests.size()-1 is the outer most loop in this level
  rank_compressed = compression_opt_spec.rank_compressed;
  rank_formats = compression_opt_spec.rank_formats;
  metadata_models_ = compression_opt_spec.metadata_models;
  coord_space_info.Set(shape, subnests, dataspace_id);

  compressed = compression_opt_spec.tensor_compressed;
  has_metadata = true;

}

void DataMovementInfo::SetTensorRepresentation(const sparse::PerDataSpaceCompressionInfo& compression_opt_spec)
{

  // About the inputs:
  // 1) since we assume the mapping must be fully concordant,
  //     the rank order in the compression opt must fully specify all necessary compression format info
  rank_compressed = compression_opt_spec.rank_compressed;
  rank_formats = compression_opt_spec.rank_formats;
  metadata_models_ = compression_opt_spec.metadata_models;
  coord_space_info.Set(shape, dataspace_id);

  compressed = compression_opt_spec.tensor_compressed;
  has_metadata = true;

}

std::string DataMovementInfo::GetMetaDataFormatName() const
{
  if (has_metadata)
  {
    std::string format;
    for (int i = metadata_models_.size() - 1; i >= 0; i--)
    {
      format += metadata_models_[i]->GetFormatName();
      format += " ";
    }
    return format;
  } else
  {
    return "none";
  }
}

CoordinateSpaceTileInfo DataMovementInfo::GetCoordinateSpaceInfo() const
{
  return coord_space_info;
}

CoordinateSpaceTileInfo DataMovementInfo::GetChildTileCoordinateSpaceInfo() const
{
  if (child_level != std::numeric_limits<unsigned>::max())
  {
    // there is a child level for this tile
    return child_level_ptr->GetCoordinateSpaceInfo();
  } else
  {
    // lowest level of its data type, next level is compute
    CoordinateSpaceTileInfo singleton_tile;
    singleton_tile.Set(1, dataspace_id);
    return singleton_tile;
  }
}

std::uint64_t DataMovementInfo::GetMaxDataTileOccupancyByConfidence(const double confidence) const
{
  std::uint64_t data_tile_occupancy = !compressed ? coord_space_info.GetShape() :
                                      tile_density->GetMaxTileOccupancyByConfidence(coord_space_info, confidence);
  return data_tile_occupancy;
}

// Move to sparse-analysis.cpp
// double DataMovementInfo::GetExpectedDataTileOccupancy()
// {
//   if (!compressed) return (double)shape;
//   else if (expected_data_occupancy != std::numeric_limits<unsigned>::max()) return expected_data_occupancy;
//   else
//   {
//     expected_data_occupancy = tile_density->GetExpectedTileOccupancy(coord_space_info);
//     return expected_data_occupancy;
//   }
// }

double DataMovementInfo::GetDataTileOccupancyProbability(const std::uint64_t occupancy) const
{
  return tile_density->GetTileOccupancyProbability(coord_space_info, occupancy);
}

double DataMovementInfo::GetChildLevelDataTileOccupancyProbability(const std::uint64_t occupancy) const
{
  double prob;
  if (child_level != std::numeric_limits<unsigned>::max())
  {
    // regular case: this storage level has a lower level
    prob = child_level_ptr->GetDataTileOccupancyProbability(occupancy);
  } else
  {
    // last level, no child
    auto child_tile_coord_space_info = GetChildTileCoordinateSpaceInfo();
    prob = tile_density->GetTileOccupancyProbability(child_tile_coord_space_info, occupancy);

  }
  return prob;
}


MetaDataTileOccupancy DataMovementInfo::GetMetaDataTileOccupancyGivenDataTile(const CoordinateSpaceTileInfo& coord_tile) const
{

  // go through each rank to query the metadata models
  // for each metadata model at rank r, we provide a query object (defined in metadata_format.h), which carries
  // 1) # max number of fibers in this rank
  // 2) # number of coordinates in each fiber
  // 3) current rank's coordinate space tile representation
  // 4) next rank's coordinate space tile representation
  // 5) density model pointer

  // initialize all the internal states for the top rank in the tile
  std::uint64_t max_number_of_fibers_in_rank = 1; // the highest rank will only have one fiber

  // 3), 4), 5) are useful for compressed ranks to calculate # of fibers per rank, # of coords per fiber info
  MetaDataTileOccupancy metadata_tile_occupancy;
  //inits related to next rank, will be defined and updated in the loop
  CoordinateSpaceTileInfo cur_coord_tile = coord_tile;
  CoordinateSpaceTileInfo next_coord_tile;

  for (int r_id = metadata_models_.size() - 1; r_id >= 0; r_id--)
  {

    std::uint64_t cur_rank_fiber_shape = fiber_shape[r_id];
    next_coord_tile.Set(metadata_subtile_shape[r_id], dataspace_id, cur_coord_tile.GetExtraConstraintInfo());
    problem::PerRankMetaDataTileOccupancy per_rank_metadata_occupancy;
     if (cur_rank_fiber_shape == 1)
     {
        // trivial rank (rank related to trivial loop)
        // a good compression setup eliminate a trivial rank and it is reasonble to assume that
        // the underlying hardware is also programmable to skip the traversal of trivial loops as well
        // FIXME: check  if this is a good assumption of common practice
        per_rank_metadata_occupancy.SetEmpty();
     }
     else
     {
      // significant rank
      // construct query
      problem::MetaDataOccupancyQuery query(max_number_of_fibers_in_rank,
                                            fiber_shape[r_id],
                                            cur_coord_tile,
                                            next_coord_tile,
                                            tile_density,
                                            1.0);

      // std::cout << problem::GetShape()->DataSpaceIDToName.at(dataspace_id) << "  rid: " << r_id << " format: " << metadata_models_[r_id]->GetFormatName()
      //           << " cur_rank_fiber_shape: " << cur_rank_fiber_shape << " max_num_fibers: " << max_number_of_fibers_in_rank
      //           << " next coord tile shape: "  <<metadata_subtile_shape[r_id]
      //           << std::endl;

      per_rank_metadata_occupancy = metadata_models_[r_id]->GetOccupancy(query);
      if (r_id == 0)
      { per_rank_metadata_occupancy.SetPayloadUnits(0); } // last rank's payload is not metadata
    }

    metadata_tile_occupancy.push_back(per_rank_metadata_occupancy);

    // prepare for next round
    max_number_of_fibers_in_rank *= cur_rank_fiber_shape;
    cur_coord_tile = next_coord_tile;

    // std::cout << " payloads: " << per_rank_metadata_occupancy.PayloadUnits()
    // << " metadata: " << per_rank_metadata_occupancy.MetaDataUnits() << std::endl;
  }

  return metadata_tile_occupancy;
}


MetaDataTileOccupancy DataMovementInfo::GetMaxMetaDataTileOccupancyByConfidence(const double confidence) const
{
  MetaDataTileOccupancy metadata_tile_occupancy = {}; //empty means no metadata

  if (has_metadata)
  {

    CoordinateSpaceTileInfo cur_coord_tile;
    cur_coord_tile.Set(metadata_subtile_shape.back(), dataspace_id);

    std::uint64_t max_tile_occupancy = tile_density->GetMaxTileOccupancyByConfidence(cur_coord_tile, confidence);
    ExtraTileConstraintInfo extra_tile_constraint_info;
    extra_tile_constraint_info.Set(metadata_subtile_shape.back(), max_tile_occupancy);
    cur_coord_tile.Set(metadata_subtile_shape.back(), dataspace_id, extra_tile_constraint_info);

    metadata_tile_occupancy = GetMetaDataTileOccupancyGivenDataTile(cur_coord_tile);

  }
  return metadata_tile_occupancy;
}

// Move to sparse-analysis.cpp
// MetaDataTileOccupancy DataMovementInfo::GetExpectedMetaDataTileOccupancy()
// {
//   MetaDataTileOccupancy metadata_tile_occupancy = {}; //empty means no metadata
//   if (!compressed) expected_data_occupancy = (double) shape;
//   else expected_data_occupancy = 0;
//
//   if (has_metadata)
//   {
//
//     CoordinateSpaceTileInfo cur_coord_tile;
//     cur_coord_tile.Set(metadata_subtile_shape.back(), dataspace_id);
//
//     std::uint64_t abs_max_tile_occupancy = tile_density->GetMaxTileOccupancyByConfidence(cur_coord_tile, 1.0);
//     for (std::uint64_t possible_occupancy = 0; possible_occupancy <= abs_max_tile_occupancy; possible_occupancy++)
//     {
//       double p = tile_density->GetTileOccupancyProbability(coord_space_info, possible_occupancy);
//       if ( p != 0)
//       {
//         ExtraTileConstraintInfo extra_tile_constraint_info;
//         extra_tile_constraint_info.Set(metadata_subtile_shape.back(), possible_occupancy);
//         cur_coord_tile.Set(metadata_subtile_shape.back(), dataspace_id, extra_tile_constraint_info);
//         auto occupancy = GetMetaDataTileOccupancyGivenDataTile(cur_coord_tile);
//
//         if (compressed) expected_data_occupancy += p * possible_occupancy; // update the data tile occupancy as well
//
//         for (unsigned r = 0; r < occupancy.size(); r++)
//         {
//           auto per_rank_occupancy = occupancy[r];
//           per_rank_occupancy.Scale(p);
//           if (metadata_tile_occupancy.size() == r)
//           {
//             metadata_tile_occupancy.push_back(per_rank_occupancy);
//           } else
//           {
//             metadata_tile_occupancy[r].Add(per_rank_occupancy);
//           }
//         }
//       }
//     }
//     expected_metadata_occupancy = metadata_tile_occupancy;
//   }
//
//   return metadata_tile_occupancy;
// }


 double DataMovementInfo::GetExpectedAggregatedMetaDataTileOccupancy() const
 {
   double aggregated_occupancy = 0;
   if (has_metadata)
   {
     assert(expected_metadata_occupancy.size() > 0);
     for (auto iter = expected_metadata_occupancy.begin(); iter != expected_metadata_occupancy.end(); iter++)
     {
       aggregated_occupancy += iter->TotalMetDataAndPayloadUnits();
     }
   }
   return aggregated_occupancy;
 }

double DataMovementInfo::GetMaxTileDensityByConfidence(const double confidence) const
{
  return tile_density->GetMaxTileDensityByConfidence(coord_space_info, confidence);
}


double DataMovementInfo::GetExpectedTileDensity() const
{
  return expected_data_occupancy/shape;
}

}