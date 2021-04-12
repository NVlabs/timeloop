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

namespace tiling{


  void DataMovementInfo::SetTensorRepresentation()
  {

    // set all fields to empty for default dense tensors with no metadata as these info will not be useful
    rank_compressed = {};
    rank_formats = {};
    metadata_models_ = {};
    coord_space_info.Set(shape);

    // for safety, set the quick accessors (should be initialized just right)
    compressed = false;
    has_metadata = false;

  }


  void DataMovementInfo::SetTensorRepresentation(const sparse::PerDataSpaceCompressionInfo& compression_opt_spec,
                                                 const std::vector<loop::Descriptor> subnests)
  {

    // About the inputs:
    // 1) since we assume the mapping must be fully concordant,
    //     the rank order in the compression opt must fully specify all necessary compression format info
    // 2) subnests[0] is the inner most loop, subnests.size()-1 is the outer most loop in this level
    rank_compressed = compression_opt_spec.rank_compressed;
    rank_formats = compression_opt_spec.rank_formats;
    metadata_models_ = compression_opt_spec.metadata_models;
    coord_space_info.Set(shape, subnests);

    compressed = compression_opt_spec.tensor_compressed;
    has_metadata= true;

  }

void DataMovementInfo::SetTensorRepresentation(const sparse::PerDataSpaceCompressionInfo& compression_opt_spec)
{

  // About the inputs:
  // 1) since we assume the mapping must be fully concordant,
  //     the rank order in the compression opt must fully specify all necessary compression format info
  rank_compressed = compression_opt_spec.rank_compressed;
  rank_formats = compression_opt_spec.rank_formats;
  metadata_models_ = compression_opt_spec.metadata_models;
  coord_space_info.Set(shape);

  compressed = compression_opt_spec.tensor_compressed;
  has_metadata= true;

}

  std::string DataMovementInfo::GetMetaDataFormatName() const
  {
    if (has_metadata)
    {
      std::string format;
      for (int i = metadata_models_.size()-1; i >= 0; i--)
      {
        format += metadata_models_[i]->GetFormatName();
        format += " ";
      }
      return format;
    } else {
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
    }
    else
    {
     // lowest level of its data type, next level is compute
     CoordinateSpaceTileInfo singleton_tile;
     singleton_tile.Set(1);
     return singleton_tile;
    }
  }



  std::uint64_t DataMovementInfo::GetDataTileOccupancyByConfidence(const double confidence) const
  {
    std::uint64_t data_tile_occupancy = !compressed ? coord_space_info.GetShape():
                                         tile_density->GetMaxTileOccupancyByConfidence(coord_space_info, confidence);
    return data_tile_occupancy;
  }

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
    }
    else
    {
      // last level, no child
      auto child_tile_coord_space_info = GetChildTileCoordinateSpaceInfo();
      prob = tile_density->GetTileOccupancyProbability(child_tile_coord_space_info, occupancy);

    }
    return prob;
  }


  MetaDataTileOccupancy DataMovementInfo::GetMetaDataTileOccupancyByConfidence(const double confidence) const
  {
    MetaDataTileOccupancy metadata_tile_occupancy = {}; //empty means no metadata

    if (has_metadata)
    {

      // go through each rank to query the metadata models
      // for each metadata model at rank r, we provide a query object (defined in metadata_format.h), which carries
      // 1) # max number of fibers in this rank
      // 2) # number of coordinates in each fiber
      // 3) current rank's coordinate space tile representation
      // 4) next rank's coordinate space tile representation
      // 5) density model pointer

      // 3), 4), 5) are useful for compressed ranks to calculate # of fibers per rank, # of coords per fiber info


      // initialize all the internal states for the top rank in the tile
      std::uint64_t max_number_of_fibers_in_rank = 1; // the highest rank will only have one fiber

      CoordinateSpaceTileInfo cur_coord_tile;
      cur_coord_tile.Set(metadata_subtile_shape.back());

      //inits related to next rank, will be defined and updated in the loop
      CoordinateSpaceTileInfo next_coord_tile;

      for (int r_id = metadata_models_.size()-1; r_id >= 0 ; r_id--)
      {

        std::uint64_t cur_rank_fiber_shape = fiber_shape[r_id];
        next_coord_tile.Set(metadata_subtile_shape[r_id]);

        // construct query
        problem::MetaDataOccupancyQuery query(max_number_of_fibers_in_rank,
                                              fiber_shape[r_id],
                                              cur_coord_tile,
                                              next_coord_tile,
                                              tile_density,
                                              confidence);

        //std::cout << "rid: " << r_id << " format: " << metadata_models_[r_id]->GetFormatName()
        //<< " cur_rank_fiber_shape: " << cur_rank_fiber_shape << " max_num_fibers: " << max_number_of_fibers_in_rank
        //<< " next coord tile shape: "  <<metadata_subtile_shape[r_id]
        //<< std::endl;

        problem::PerRankMetaDataTileOccupancy per_rank_metadata_occupancy = metadata_models_[r_id]->GetOccupancy(query);
        if (r_id == 0) {per_rank_metadata_occupancy.SetPayloadUnits(0);} // last rank's payload is not metadata

        metadata_tile_occupancy.push_back(per_rank_metadata_occupancy);

        // prepare for next round
        max_number_of_fibers_in_rank *= cur_rank_fiber_shape;
        cur_coord_tile = next_coord_tile;

       // std::cout << " payloads: " << per_rank_metadata_occupancy.PayloadUnits()
       // << " metadata: " << per_rank_metadata_occupancy.MetaDataUnits() << std::endl;
      }
    }
    return metadata_tile_occupancy;
  }

  double DataMovementInfo::GetTileDensityByConfidence(const double confidence) const
  {
    return tile_density->GetTileDensityByConfidence(coord_space_info, confidence);
  }


  void DataMovementInfo::ComputeExpectedMetaDataAccesses (std::map<std::string, double>& avg_non_empty_intersection_probs)
  {
	// initialize all fine-grained metadata related accesses to 0
	fine_grained_accesses["metadata_read"] = 0;
	fine_grained_accesses["gated_metadata_read"] = 0;
	fine_grained_accesses["metadata_fill"] = 0;
	fine_grained_accesses["gated_metadata_fill"] = 0;
	fine_grained_accesses["metadata_update"] = 0;
	fine_grained_accesses["gated_metadata_update"] = 0;

	if (shape == 0 || !has_metadata) return; // no processing needed if empty tile or no metadata

    MetaDataTileOccupancy expected_metadata_tile_occupancy = GetExpectedMetaDataTileOccupancy();

    std::uint64_t num_child_metadata_ranks;
    if (child_level != std::numeric_limits<unsigned>::max())
    {
      // upon a read, only ranks associated with the child level will be sent out
      num_child_metadata_ranks = child_level_ptr->GetNumMetaDataRanks();
    }
    else
    {
      // upon a read, last level storage sends all metadata
      num_child_metadata_ranks = GetNumMetaDataRanks();
    }

    double total_metadata_payload_units_per_tile = 0;
    double child_metadata_payload_units_per_tile = 0;

    for (unsigned r_id = 0; r_id < expected_metadata_tile_occupancy.size(); r_id++)
    {
      total_metadata_payload_units_per_tile += expected_metadata_tile_occupancy[r_id].MetaDataUnits();
      total_metadata_payload_units_per_tile += expected_metadata_tile_occupancy[r_id].PayloadUnits();

      if (r_id < num_child_metadata_ranks)
      {
        child_metadata_payload_units_per_tile += expected_metadata_tile_occupancy[r_id].MetaDataUnits();
        child_metadata_payload_units_per_tile += expected_metadata_tile_occupancy[r_id].PayloadUnits();
      }
    }

    // calculate how many rounds did the tile get read/fill/update, then scale the metadata accesses per tile accordingly
    double read_ratio = (double) reads/shape;
    double fill_ratio = (double) fills/shape;
    double update_ratio = (double) updates/shape;
    metadata_fills = ceil(total_metadata_payload_units_per_tile * fill_ratio);
    metadata_reads = ceil(total_metadata_payload_units_per_tile * read_ratio);
    metadata_updates = ceil(total_metadata_payload_units_per_tile * update_ratio);


    // std::cout << GetDataSpaceName() << ": " << total_metadata_payload_units_per_tile << "  read ratio: " << read_ratio
    // << "  metadata reads: " << total_metadata_payload_units_per_tile * read_ratio << std::endl;

    //
    // process the impact of sparse optimizations
    //

    fine_grained_accesses["gated_metadata_fill"] = floor(metadata_fills * (1 - avg_non_empty_intersection_probs["gated_fill"]));
    fine_grained_accesses["skipped_metadata_fill"] = floor(metadata_fills * (1 - avg_non_empty_intersection_probs["skipped_fill"]));
    fine_grained_accesses["gated_metadata_update"] = floor(metadata_updates * (1 - avg_non_empty_intersection_probs["gated_update"]));
    fine_grained_accesses["skipped_metadata_update"] = floor(metadata_updates * (1 -  avg_non_empty_intersection_probs["skipped_update"]));

    // current-level only metadata ranks are required to read out for performing intersections
    // child-level metadata can be gated/skipped
    if (avg_non_empty_intersection_probs["gated_read"] != 1.0)
    {
      auto num_intersection_reads = total_metadata_payload_units_per_tile - child_metadata_payload_units_per_tile;
      auto num_child_level_transfers = child_metadata_payload_units_per_tile * avg_non_empty_intersection_probs["gated_read"];
      fine_grained_accesses["gated_metadata_read"] = num_intersection_reads + num_child_level_transfers;
    }

    if (avg_non_empty_intersection_probs["skipped_read"] != 1.0)
     {
       auto num_intersection_reads = total_metadata_payload_units_per_tile - child_metadata_payload_units_per_tile;
       auto num_child_level_transfers = child_metadata_payload_units_per_tile * avg_non_empty_intersection_probs["gated_read"];
       fine_grained_accesses["skipped_metadata_read"] = num_intersection_reads + num_child_level_transfers;
    }

    fine_grained_accesses["metadata_fill"] = metadata_fills - fine_grained_accesses["gated_metadata_fill"] - fine_grained_accesses["skipped_metadata_fill"] ;
    fine_grained_accesses["metadata_read"] = metadata_reads - fine_grained_accesses["gated_metadata_read"] - fine_grained_accesses["skipped_metadata_read"];
    fine_grained_accesses["metadata_update"] = metadata_updates - fine_grained_accesses["gated_metadata_update"] - fine_grained_accesses["skipped_metadata_update"];
  }


  void DataMovementInfo::ComputeExpectedDataAccesses ( std::map<std::string, double>& avg_non_empty_intersection_probs)
  {


    // check if all tiles are filled
    // if some data tiles are manually gated/skipped udring the fill process in the first place
    // (because of its parent level gating/skipping behavior), reads/updates must be affected

    double sread_caused_by_sfill = 0;
    double gread_caused_by_gfill = 0;
    double supdate_caused_by_sfill = 0;
    double gupdate_caused_by_gfill = 0;

    if (avg_non_empty_intersection_probs["gated_fill"] != 1.0)
	{
      gread_caused_by_gfill = reads * ( 1 - avg_non_empty_intersection_probs["gated_fill"]);
      gupdate_caused_by_gfill = updates * ( 1 - avg_non_empty_intersection_probs["gated_fill"]);
	}
	else if (avg_non_empty_intersection_probs["skipped_fill"] != 1.0)
	{
	  sread_caused_by_sfill = reads * ( 1 - avg_non_empty_intersection_probs["skipped_fill"]);
	  supdate_caused_by_sfill = updates * ( 1 - avg_non_empty_intersection_probs["skipped_fill"]);
	}

	double max_total_reads = reads - sread_caused_by_sfill - gread_caused_by_gfill;
	double max_total_updates = updates - supdate_caused_by_sfill - gupdate_caused_by_gfill;


    //
    // auto infer compression optimization's impact on average number of data tile accesses
    //
    if (compressed)
    {
      double expected_density = GetExpectedTileDensity();
      fine_grained_accesses["skipped_read"] = max_total_reads * (1 - expected_density);
      fine_grained_accesses["skipped_fill"] = fills * (1 - expected_density);
      fine_grained_accesses["skipped_update"] = max_total_updates * (1 - expected_density);

    }

    //
    // evaluate the impact of user-defined skipping or gating optimizations
    //

    // optimizations should be applied on-top-of
    //  (1) parent-level optimization related skipped/gated reads/updates
    //  (2) skipped read/fill/update caused by compression

    fine_grained_accesses["gated_fill"] = floor((fills - fine_grained_accesses["skipped_fill"]) * (1 - avg_non_empty_intersection_probs["gated_fill"]));
    fine_grained_accesses["gated_read"] += floor((max_total_reads - fine_grained_accesses["skipped_read"]) * (1 - avg_non_empty_intersection_probs["gated_read"]));
    fine_grained_accesses["gated_update"] += floor((max_total_updates - fine_grained_accesses["skipped_update"]) * (1 - avg_non_empty_intersection_probs["gated_update"]));

    fine_grained_accesses["skipped_fill"] +=  floor((fills - fine_grained_accesses["skipped_fill"]) * (1 - avg_non_empty_intersection_probs["skipped_fill"]));
    fine_grained_accesses["skipped_read"] +=  floor((max_total_reads - fine_grained_accesses["skipped_read"]) * (1 - avg_non_empty_intersection_probs["skipped_read"]));
    fine_grained_accesses["skipped_update"] += floor((max_total_updates - fine_grained_accesses["skipped_update"]) * (1 - avg_non_empty_intersection_probs["skipped_update"]));

    fine_grained_accesses["skipped_read"] += sread_caused_by_sfill;
    fine_grained_accesses["gated_read"] += gread_caused_by_gfill;
    fine_grained_accesses["skipped_update"] += supdate_caused_by_sfill;
    fine_grained_accesses["gated_update"] += gupdate_caused_by_gfill;

    fine_grained_accesses["random_fill"] = fills - fine_grained_accesses["gated_fill"] - fine_grained_accesses["skipped_fill"] ;
    fine_grained_accesses["random_read"] = reads - fine_grained_accesses["gated_read"] - fine_grained_accesses["skipped_read"];
    fine_grained_accesses["random_update"] = updates - fine_grained_accesses["gated_update"] - fine_grained_accesses["skipped_update"];

  }


}