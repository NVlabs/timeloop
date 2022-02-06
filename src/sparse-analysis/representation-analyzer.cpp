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

#include "loop-analysis/coordinate-space-tile-info.hpp"
#include "sparse-analysis/state.hpp"
#include "mapping/loop.hpp"

namespace sparse
{

void CalculateExpectedMetaDataAccesses(tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                       const model::Topology::Specs& topology_specs)
{
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    for (int l = topology_specs.NumStorageLevels() - 1; l >= 0; l--)
    {

     auto& data_movement_info = compound_data_movement_nest[pv][l];
     // std::cout << "storage level: " << topology_specs.GetStorageLevel(l)->level_name
     // << "  dataspace name: " << problem::GetShape()->DataSpaceIDToName.at(pv)
     // << " shape: " << data_movement_info.shape << "  has metadata: " << data_movement_info.has_metadata
     // << std::endl;

      if (data_movement_info.shape == 0 || !data_movement_info.has_metadata) continue;
      
      double read_ratio = (double)data_movement_info.reads / data_movement_info.shape;
      double fill_ratio = (double)data_movement_info.fills / data_movement_info.shape;
      double update_ratio = (double)data_movement_info.updates / data_movement_info.shape;
      
      tiling::MetaDataTileOccupancy expected_metadata_tile_occupancy = data_movement_info.expected_metadata_occupancy;

      //std::uint64_t num_child_metadata_ranks;
      //if (data_movement_info.child_level != std::numeric_limits<unsigned>::max())
      //{
      //  // upon a read, only ranks associated with the child level will be sent out
      //  num_child_metadata_ranks =
      //    compound_data_movement_nest[pv][data_movement_info.child_level].GetNumMetaDataRanks();
      //} else
      //{
      //  // upon a read, last level storage sends all metadata
      //  num_child_metadata_ranks = data_movement_info.GetNumMetaDataRanks();
      //}

      //double total_metadata_payload_units_per_tile = 0.0;
      //double child_metadata_payload_units_per_tile = 0.0;
      std::uint64_t md_accesses, pl_accesses;
      
      for (unsigned r_id = 0; r_id < expected_metadata_tile_occupancy.size(); r_id++)
      {
       
        double md_units = expected_metadata_tile_occupancy[r_id].MetaDataUnits();
        double pl_units = expected_metadata_tile_occupancy[r_id].PayloadUnits();

        // std::cout << " === rank id: " << r_id << " ===" << std::endl;
        // std::cout << " expected  md units: " << md_units << "  pl units: " << pl_units << std::endl;

        md_accesses = ceil(md_units * fill_ratio);
        pl_accesses = ceil(pl_units * fill_ratio);
        data_movement_info.format_fills.push_back({md_accesses, pl_accesses});
         
        // std::cout << "  fill ratio: " << fill_ratio << " md fills: " << md_accesses <<" pl fills: " << pl_accesses << std::endl;
 
        md_accesses = ceil(md_units * read_ratio);
        pl_accesses = ceil(pl_units * read_ratio);
        data_movement_info.format_reads.push_back({md_accesses, pl_accesses});  
 
        // std::cout << "  read ratio: " << read_ratio << " md reads: " << md_accesses <<" pl reads: " << pl_accesses << std::endl;
        
        md_accesses = ceil(md_units * update_ratio);
        pl_accesses = ceil(pl_units * update_ratio);        
        data_movement_info.format_updates.push_back({md_accesses, pl_accesses});       
       
        // total_metadata_payload_units_per_tile += md_units; 
        // total_metadata_payload_units_per_tile += pl_units;
        
        //initialize the fine_grained_accesses entries with the proper number of ranks
        for (auto iter = data_movement_info.fine_grained_format_accesses.begin(); 
             iter != data_movement_info.fine_grained_format_accesses.end(); iter++)
        { iter->second.push_back({0, 0}); }
      }

     // calculate how many rounds did the tile get read/fill/update, then scale the metadata accesses per tile accordingly
     // data_movement_info.metadata_fills = ceil(total_metadata_payload_units_per_tile * fill_ratio);
     // data_movement_info.metadata_reads = ceil(total_metadata_payload_units_per_tile * read_ratio);
     // data_movement_info.metadata_updates = ceil(total_metadata_payload_units_per_tile * update_ratio);

     // if (total_metadata_payload_units_per_tile == 0)
     // {
     //   data_movement_info.child_level_metadata_occupancy_ratio = 0;
     // } else
     // {
     //   data_movement_info.child_level_metadata_occupancy_ratio =
     //     child_metadata_payload_units_per_tile / total_metadata_payload_units_per_tile;
     // }
    }
  }
}
 
void CalculateExpectedOccupancy(tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                const model::Topology::Specs& topology_specs)
{

  // Expected occupancy is a weights sum of possible occupancies
  //    note that for metadata, the occupancy is a function of the data tile occupancy
  //    for for each possible data tile occupancy, we need to recalculate the metadata occupancy

  for (unsigned pv = 0; pv < unsigned(problem::GetShape()->NumDataSpaces); pv++)
  {
    for (unsigned level = 0; level < topology_specs.NumStorageLevels(); level++)
    {
      auto& pv_data_movement_info = compound_data_movement_nest[pv][level];
      double total_non_empty_payloads = 0;

      // Initialize occupancy holders
      tiling::MetaDataTileOccupancy expected_metadata_occupancy = {}; //empty means no metadata

      if (pv_data_movement_info.shape != 0)
      {
        std::uint64_t abs_max_tile_occupancy = pv_data_movement_info.GetMaxDataTileOccupancyByConfidence(1.0);
        std::uint64_t abs_min_tile_occupancy = pv_data_movement_info.GetMinDataTileOccupancy();

        std::vector<double> occupancy_probabilities;
        
        std::vector<bool> exist_masks; // mask possible occupancies to reudce size of occupancy probabilities vector
        exist_masks.reserve(abs_max_tile_occupancy - abs_min_tile_occupancy + 1);
        
        // query density model for all probs to maximize density model reuse ( some densities model might use internal recording mechanisims)
        for (std::uint64_t possible_occupancy = abs_min_tile_occupancy;
             possible_occupancy <= abs_max_tile_occupancy; possible_occupancy++)
        {
          double p = pv_data_movement_info.GetDataTileOccupancyProbability(possible_occupancy);
            exist_masks.emplace_back(p != 0);
          if (p != 0)
          {
            occupancy_probabilities.emplace_back(p);
          }
        }
 
        // std::cout << "Storage: " << topology_specs.GetStorageLevel(level)->level_name
        // << "  dataspace: " << problem::GetShape()->DataSpaceIDToName.at(pv)
        // << "  tile shape: " << pv_data_movement_info.shape << std::endl;

        unsigned equivalent_i = 0;
        for (unsigned i = 0; i < exist_masks.size(); i++)
        {
          if (exist_masks[i])
          {
            double p = occupancy_probabilities[equivalent_i];
            std::uint64_t possible_occupancy = abs_min_tile_occupancy + i;

            // update expected occupancy accordingly
            total_non_empty_payloads += p * (double)possible_occupancy;
            if (pv_data_movement_info.has_metadata)
            {
              // Calculate the resulted metadata occupancy for this specific potential data tile size
              // the exact possible occupancy serves as additional information about the tile
              // it is upto the density models to determine whether this addition information is useful
 
              tiling::ExtraTileConstraintInfo extra_constraint_info;
              extra_constraint_info.Set(pv_data_movement_info.shape, possible_occupancy);
              extra_constraint_info.SetMold(*pv_data_movement_info.coord_space_info.tile_point_set_mold_);

              tiling::CoordinateSpaceTileInfo possible_coord_tile;
              possible_coord_tile.Set(*pv_data_movement_info.coord_space_info.tile_point_set_mold_, pv, extra_constraint_info);
              auto occupancy = pv_data_movement_info.GetMetaDataTileOccupancyGivenDataTile(possible_coord_tile);

              // update the metadata tile occupancy record (each item in the record correspond to a rank)
              for (unsigned r = 0; r < occupancy.size(); r++)
              {
                auto per_rank_occupancy = occupancy[r];
                // std::cout << "rid: " << r << " md units: " << per_rank_occupancy.MetaDataUnits()
                //   << "  pl units: " << per_rank_occupancy.PayloadUnits() << std::endl;
                per_rank_occupancy.Scale(p);
                if (expected_metadata_occupancy.size() == r) expected_metadata_occupancy.push_back(per_rank_occupancy);
                else expected_metadata_occupancy[r].Add(per_rank_occupancy);
              }
            }
            equivalent_i += 1;
          }
        }
      }

      // Finished calculating the weighted sum of all possible occupancies, update record
      // density is a fact, uncompressed could also have density < 1.0
      pv_data_movement_info.expected_density = total_non_empty_payloads / pv_data_movement_info.shape;
      pv_data_movement_info.expected_metadata_occupancy = expected_metadata_occupancy;
      pv_data_movement_info.expected_data_occupancy = total_non_empty_payloads;
 
      // std::cout << " expected data occupancy: " << pv_data_movement_info.expected_data_occupancy << std::endl;
      // std::cout << " expected metdata occupancy: " << std::endl;
      // for (unsigned r = 0; r < pv_data_movement_info.expected_metadata_occupancy.size(); r++)
      // {
      //   auto per_rank_occupancy = pv_data_movement_info.expected_metadata_occupancy[r];
      //   std::cout << "   rid: " << r << "   md units: " << per_rank_occupancy.MetaDataUnits()
      //     << "    pl units: " << per_rank_occupancy.PayloadUnits() << std::endl;
      // }
    }
  }
}

bool ApplyRanksOuterToInner(std::uint64_t inner_rank_id,
                            const std::vector <loop::Descriptor>& singleton_metadata_subnest,
                            const std::vector<PointSet>& singleton_metadata_subtile_point_set,
                            const sparse::PerDataSpaceCompressionInfo& pv_compression_info,
                            tiling::DataMovementInfo& pv_data_movement_info)
{
  std::vector <loop::Descriptor> flattened_rank_nest;
  std::vector <problem::Shape::FlattenedDimensionID> flattening_rule;

  bool pv_has_metadata = pv_compression_info.HasMetaData();
  std::uint64_t cur_level_num_ranks = pv_has_metadata ? pv_compression_info.rank_formats.size() : 1;

  assert(singleton_metadata_subnest.size() == singleton_metadata_subtile_point_set.size());
  
  // start by applying the outermost rank to the outermost loop
  // if there are extra inner ranks supported, all of the these ranks will cost no overhead
  int loop_id = singleton_metadata_subnest.size() - 1;
  int r_id = cur_level_num_ranks;
  
  std::uint32_t point_set_order = problem::GetShape()->DataSpaceOrder.at(pv_data_movement_info.dataspace_id);
  Point unit(point_set_order);
  PointSet scalar_point_set(point_set_order, unit);
  PointSet corresponding_tile_point_set = scalar_point_set;

  // std::cout << "total number of ranks: " << cur_level_num_ranks
  // << "  inner rank id: " << inner_rank_id
  // << " total loops: " << singleton_metadata_subnest.size() 
  // << " has metadata: " << pv_has_metadata  << std::endl;

  while (r_id > (int)inner_rank_id && loop_id >= 0)
  {
    
    bool trivial_loop = true;

    while (trivial_loop && loop_id >= 0)  // get rid of the first set of consecutive trivial loops
    {
      auto loop = singleton_metadata_subnest[loop_id];
      trivial_loop = (loop.start + loop.stride) >= loop.end;
      if (!trivial_loop) break;
      loop_id--;
    }

    if (loop_id >= 0) //there are some non-trivial loop(s) that can potenitally be mapped to next rank
    {
      r_id--; // next inner rank
      flattened_rank_nest.clear();

      auto loop = singleton_metadata_subnest[loop_id];
      // std::cout << "trying to map loop below to rank " << r_id << std::endl;
      // std::cout << loop << std::endl;
      
      // reset flattening rule
      flattening_rule = {};     
      bool in_flattened_list = false;
      std::vector<problem::Shape::FlattenedDimensionID>::iterator flatten_iter;

      if (!trivial_loop)
      {
        // if there is any flattening rule set for the rank, test if we can map the loop the to rank
        if (pv_has_metadata && pv_compression_info.ExistFlatteningRule(r_id))
        {
          if (pv_compression_info.FoundDimensionInFlatteningRule(r_id, loop.dimension, flattening_rule))
          {
            in_flattened_list = true;
            flatten_iter = std::find(flattening_rule.begin(), flattening_rule.end(), loop.dimension);
          }
          else
          {
            // std::cout << "flattening rule specified but dimension not in there, this rank cannot be mapped" << std::endl; 
            std::vector <loop::Descriptor> tmp_loop = {};
            pv_data_movement_info.metadata_subnest.push_back(tmp_loop);
            pv_data_movement_info.metadata_subtile_point_set.emplace_back(singleton_metadata_subtile_point_set.at(0));
            continue;
          }
        }
       
        // if uncompressed, by default, the dimension is in flattening rule
        if (!pv_has_metadata)
        {
          in_flattened_list = true;
        }
        
        flattened_rank_nest.push_back(loop);
        corresponding_tile_point_set = singleton_metadata_subtile_point_set.at(loop_id);
      }
      // next inner loop
      loop_id--;

      if (loop_id >= 0)
      {
        // std::cout << "rest of dims in the flattening rule: " << in_flattened_list << std::endl;
        // if (pv_has_metadata)
        // {
        //   for (auto dim = flattening_rule.begin(); dim != flattening_rule.end(); dim++)
        //   {
        //     std::cout << problem::GetShape()->DimensionIDToName.at(*dim) << "  ";
        //   }
        //   std::cout << std::endl;
        // }

        // if default uncompressed, then all dimensions in a list as well
        if (!pv_has_metadata || in_flattened_list)
        {
          // this loop is already the next inner loop of the loop that is in a flattened list
          auto loop = singleton_metadata_subnest[loop_id];
          trivial_loop = (loop.start + loop.stride) >= loop.end;

          flatten_iter = std::find(flattening_rule.begin(), flattening_rule.end(), loop.dimension);

          while (!pv_has_metadata || (flatten_iter != flattening_rule.end()) || trivial_loop)
          {
            if (!trivial_loop)
            {
              // std::cout << "mapping loop below to rank " << r_id << std::endl;
              // std::cout << loop << std::endl;
              flattened_rank_nest.push_back(loop);
              // corresponding_tile_shape = singleton_metadata_subtile_shape[loop_id];

              // remove loop dimension from flatten list, as one item in the list can only be used once
              // if (pv_has_metadata) flattening_rule.erase(flatten_iter);
            }
            loop_id--;

            // check if there are anymore loops (overall and in this flattening rule)
            if (loop_id >= 0)
            {
              loop = singleton_metadata_subnest[loop_id];
              flatten_iter = std::find(flattening_rule.begin(), flattening_rule.end(),
                                       singleton_metadata_subnest[loop_id].dimension);
              trivial_loop = (loop.start + loop.stride) >= loop.end;
              //std::cout << "next loop: " << singleton_metadata_subnest[loop_id] << std::endl;
            } else
              break;
          }
        }
      }
      pv_data_movement_info.metadata_subnest.emplace(pv_data_movement_info.metadata_subnest.begin(), flattened_rank_nest);
      pv_data_movement_info.metadata_subtile_point_set.emplace(pv_data_movement_info.metadata_subtile_point_set.begin(),
          corresponding_tile_point_set);
    }

    // reset to 1
    // corresponding_tile_shape = 1;
    corresponding_tile_point_set = scalar_point_set;

  }


  // if the last used rank is not the innermost, fill in the extra inner supported rank (if any)
  while (r_id > (int)inner_rank_id)
  {
    r_id--;
    std::vector <loop::Descriptor> tmp_loop = {};
    pv_data_movement_info.metadata_subnest.emplace(pv_data_movement_info.metadata_subnest.begin(), tmp_loop);
    pv_data_movement_info.metadata_subtile_point_set.emplace(pv_data_movement_info.metadata_subtile_point_set.begin(),
                                                             singleton_metadata_subtile_point_set.at(0));

    // std::cout << "Warning: more supported ranks then non-trivial loops, "
    //              "the extra inner rank is turned into a dummy rank: "
    //           << pv_compression_info.rank_formats[r_id] << std::endl; 
  }

  // skip the trailing trivial loops (if any)
  bool more_compression_ranks_needed = false;
  while (loop_id >= 0)
  {
    auto loop = singleton_metadata_subnest[loop_id];
    bool trivial_loop = (loop.start + loop.stride) >= loop.end;
    if (!trivial_loop)
    {
      // if any non-trivial loops occurs, then the hardware support is not compatible
      more_compression_ranks_needed = true;
      break;
    }
    loop_id--;
  }
  
  return more_compression_ranks_needed;
}

bool ApplyRanksInnerToOuter(std::uint64_t inner_rank_id,
                            const std::vector <loop::Descriptor>& singleton_metadata_subnest,
                            const std::vector<PointSet>& singleton_metadata_subtile_point_set,
                            const sparse::PerDataSpaceCompressionInfo& pv_compression_info,
                            tiling::DataMovementInfo& pv_data_movement_info)
{

  std::vector <loop::Descriptor> flattened_rank_nest;
  std::vector <problem::Shape::FlattenedDimensionID> flattening_rule;

  bool pv_has_metadata = pv_compression_info.HasMetaData();
  int  cur_level_num_ranks = pv_has_metadata ? (int)pv_compression_info.rank_formats.size() : 1;

  assert(singleton_metadata_subnest.size() == singleton_metadata_subtile_point_set.size());
  
  // start by applying the innermost rank to the innermost loop
  // if there are extra outer ranks supported, all of the these ranks will cost no overhead
  unsigned loop_id = 0; 
  int r_id = inner_rank_id - 1;
  
  auto point_set_order = problem::GetShape()->DataSpaceOrder.at(pv_data_movement_info.dataspace_id);
  Point unit(point_set_order);
  PointSet scalar_point_set(point_set_order, unit);
  
  PointSet corresponding_tile_point_set = scalar_point_set;
  
  //std::cout << "total number of ranks: " << cur_level_num_ranks
  //<< "  inner rank id: " << inner_rank_id
  //<< " total loops: " << singleton_metadata_subnest.size() << std::endl;

  while (r_id < cur_level_num_ranks - 1 && loop_id < singleton_metadata_subnest.size())
  {
    bool trivial_loop = true;

    while (trivial_loop && loop_id < singleton_metadata_subnest.size())  //get rid of the first set of consecutive trivial loops
    {
      auto loop = singleton_metadata_subnest[loop_id];
      trivial_loop = (loop.start + loop.stride) >= loop.end;
      if (!trivial_loop) break;
      loop_id++;
    }
    
    if (loop_id < singleton_metadata_subnest.size()) //there are some non-trivial loop(s) that can be mapped to next rank
    {
      r_id++; // next outer rank
      flattened_rank_nest.clear();
      auto loop = singleton_metadata_subnest[loop_id];
      // std::cout << "trying mapping loop below to rank " << r_id << std::endl;
      // std::cout << loop << std::endl;

      bool in_flattened_list = false;
      // reset flattening rule
      flattening_rule = {};
      std::vector<problem::Shape::FlattenedDimensionID>::iterator flatten_iter;
      
      if (!trivial_loop)
      {
        // if there is any flattening rule set for the rank, test if we can map the loop the to rank
        // std::cout << "has metadata: " << pv_has_metadata << "  exist flat rule: " <<  pv_compression_info.ExistFlatteningRule(r_id) <<std::endl;
        if (pv_has_metadata && pv_compression_info.ExistFlatteningRule(r_id))
        {
          
          if (pv_compression_info.FoundDimensionInFlatteningRule(r_id, loop.dimension, flattening_rule))
          {
            in_flattened_list = true;
            flatten_iter = std::find(flattening_rule.begin(), flattening_rule.end(), loop.dimension);
          }
          else
          {
            // std::cout << "flattening rule specified but dimension not in there, this rank cannot be mapped" << std::endl; 
            // std::vector <loop::Descriptor> tmp_loop();
            std::vector <loop::Descriptor> tmp_loop = { loop };
            tmp_loop[0].end = 1;
            tmp_loop[0].residual_end = 1;
            tmp_loop[0].dimension = pv_compression_info.GetFlatteningRule(r_id);
            pv_data_movement_info.metadata_subnest.emplace_back(tmp_loop);
            pv_data_movement_info.metadata_subtile_point_set.emplace_back(corresponding_tile_point_set);
            continue;
          }
        }
     
        // if uncompressed, by default, the dimension is in flattening rule
        if (!pv_has_metadata)
        {
          in_flattened_list = true;
        } 
        
        // we are able to map the loop to the specific rank we are looking at
        flattened_rank_nest.push_back(loop);
        corresponding_tile_point_set = singleton_metadata_subtile_point_set.at(loop_id);
      }
      
      // next outer loop
      loop_id++;
      
      if (loop_id < singleton_metadata_subnest.size())
      {
        //std::cout << "rest of dims in the flattening rule: " << in_flattened_list << std::endl;
        //if (pv_has_metadata)
        //{
        //  for (auto dim = flattening_rule.begin(); dim != flattening_rule.end(); dim++)
        //  {
        //    std::cout << problem::GetShape()->FlattenedDimensionIDToName.at(*dim) << "  ";
        //  }
        //  std::cout << std::endl;
        //}

        // if default uncompressed, then all dimensions in a list as well
        if (!pv_has_metadata || in_flattened_list)
        {
          // this loop is already the next inner loop of the loop that is in a flattened list
          auto loop = singleton_metadata_subnest[loop_id];
          trivial_loop = (loop.start + loop.stride) >= loop.end;

          flatten_iter = std::find(flattening_rule.begin(), flattening_rule.end(), loop.dimension);

          while (!pv_has_metadata || (flatten_iter != flattening_rule.end()) || trivial_loop)
          {
            if (!trivial_loop)
            {
              //std::cout << "mapping loop below to rank " << r_id << std::endl;
              //std::cout << loop << std::endl;
              flattened_rank_nest.push_back(loop);
              // corresponding_tile_shape = singleton_metadata_subtile_shape[loop_id];

              // remove loop dimension from flatten list, as one item in the list can only be used once
              //if (pv_has_metadata) flattening_rule.erase(flatten_iter);
            }
            loop_id++;

            // check if there are anymore loops (overall and in this flattening rule)
            if (loop_id < singleton_metadata_subnest.size())
            {
              loop = singleton_metadata_subnest[loop_id];
              flatten_iter = std::find(flattening_rule.begin(), flattening_rule.end(),
                                       singleton_metadata_subnest[loop_id].dimension);
              trivial_loop = (loop.start + loop.stride) >= loop.end;
              //std::cout << "next loop: " << singleton_metadata_subnest[loop_id] << std::endl;
            } else
              break;
          }
        }
      }
      pv_data_movement_info.metadata_subnest.push_back(flattened_rank_nest);
      pv_data_movement_info.metadata_subtile_point_set.emplace_back(corresponding_tile_point_set);
    }
    
    // reset to 1
    // corresponding_tile_shape = 1;
    corresponding_tile_point_set = scalar_point_set;
  }

  // fill in the extra outer supported rank (if any)
  while (r_id < cur_level_num_ranks - 1)
  {
    r_id++;
    std::vector <loop::Descriptor> tmp_loop = {};
    pv_data_movement_info.metadata_subnest.push_back(tmp_loop);
    pv_data_movement_info.metadata_subtile_point_set.emplace_back(singleton_metadata_subtile_point_set[0]);
    //std::cout << "Warning: more supported ranks then non-trivial loops, "
    //             "the extra outer rank is turned into a dummy rank: "
    //          << pv_compression_info.rank_formats[r_id] << std::endl;
  }
  
  // skip the trailing trivial loops (if any)
  bool more_compression_ranks_needed = false;

  while (loop_id < singleton_metadata_subnest.size())
  {
    auto loop = singleton_metadata_subnest[loop_id];
    bool trivial_loop = (loop.start + loop.stride) >= loop.end;
    if (!trivial_loop)
    {
      // if any non-trivial loops occurs, then the hardware support is not compatible
      more_compression_ranks_needed = true;
      break;
    }
    loop_id++;
  }
  
  return more_compression_ranks_needed;
}

bool DefineCompressionFormatModels(SparseAnalysisState& state,
                                   tiling::CompoundDataMovementNest& compound_data_movement_nest,
                                   const model::Topology::Specs& topology_specs,
                                   std::vector <model::EvalStatus>& eval_status,
                                   const bool break_on_failure)
{

  bool success = true;
  auto compression_info = state.sparse_optimization_info_->compression_info;

  // nothing needs to be done if no metadata involved
  if (compression_info.all_ranks_default_dense) return success;

  std::ostringstream fail_reason;

  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {

    auto& pv_data_movement_nest = compound_data_movement_nest[pv];
    auto dim_ids_in_proj = problem::GetShape()->DataSpaceIDToDimensionIDVector[pv];

    unsigned level = 0;
    while (level < topology_specs.NumStorageLevels())
    {

      if (pv_data_movement_nest[level].shape == 0)
      {
        // have not seen the first level that storages the datatype pv yet
        level++;
        continue;
      }

      // get the architecture level of the storage level
      unsigned overall_level_id = topology_specs.StorageMap(level);

      // collapse all the bypassed levels between parent and child into metadata subnests
      std::uint64_t child_level_id = pv_data_movement_nest[level].child_level;

      // handle special case: no child level
      // then we need to accumulate all levels below (if any)
      std::uint64_t inner_most_level = child_level_id == std::numeric_limits<unsigned>::max() ? 0 : child_level_id + 1;

      // check if pre-tiling is required for this level
      // if so, current level hardware must support the necessary number of ranks associated with the metadata subnest of both this level and its child level
      // thus, we need to add all the metadata subnest in the child level to this level
      bool pre_tiling_required = false;
      bool cur_level_has_metadata = compression_info.has_metadata_masks.at(level).at(pv);
      bool cur_level_compressed = compression_info.compressed_masks.at(level).at(pv);
      // if current level is default uncompressed, it has one rank that all rankIDs are flattened into a single rank
      unsigned cur_level_num_ranks = cur_level_has_metadata ?
        compression_info.per_level_info_map.at(level).at(pv).rank_formats.size() : 1;

      // std::cout << "define format: "<< topology_specs.GetStorageLevel(level)->level_name
      // << " dataspace: " << problem::GetShape() -> DataSpaceIDToName.at(pv)
      // << " has metadata: " << cur_level_has_metadata
      // << std::endl;

      if (cur_level_has_metadata)
      {
        // update tile information to reflect sparse optimization's impact
        compound_data_movement_nest[pv][level].SetTensorRepresentation(compression_info.per_level_info_map.at(level).at(pv));
      }

      bool child_level_has_metadata = false;
      unsigned child_level_num_ranks = 0;
      bool child_level_compressed = false;

      if (child_level_id != std::numeric_limits<unsigned>::max())
      {
        child_level_has_metadata = compression_info.has_metadata_masks.at(child_level_id).at(pv);
        child_level_compressed = compression_info.compressed_masks.at(child_level_id).at(pv);
        child_level_num_ranks = child_level_has_metadata ?
          compression_info.per_level_info_map.at(child_level_id).at(pv).rank_formats.size() : 1;
      }

      // Pretiling checks
      // only if current level is compressed && current level is not inner most level && inner level has metadata
      // && no online tile partition supported, do we need pre-tiling
      // && child level is not payload
      // if child level has a tile shape of 1, then it is just looking at a value payload, no need to pre-tile for the payload
      // by pretiling, we specifically mean that this level needs to consider inner level nontrivial loops

      if (cur_level_compressed && child_level_id != std::numeric_limits<unsigned>::max()
          && !compression_info.tile_partition_supported_masks[level] && pv_data_movement_nest[child_level_id].shape > 1)
      {

        pre_tiling_required = true;

        if (child_level_num_ranks > cur_level_num_ranks)
        {
          fail_reason << "pretiling for " << problem::GetShape()->DataSpaceIDToName.at(pv)
                      << "required but level compression format does not align: "
                      << topology_specs.GetStorageLevel(level)->level_name << " "
                      << topology_specs.GetStorageLevel(child_level_id)->level_name
                      << std::endl;

          success = false;
          eval_status[overall_level_id].success = false;
          eval_status[overall_level_id].fail_reason = fail_reason.str();
          if (break_on_failure) return success;

        }
        for (unsigned r_id = 0; r_id < child_level_num_ranks; r_id++)
        {
          if (child_level_compressed &&
              compression_info.per_level_info_map.at(level).at(pv).rank_formats[r_id] !=
              compression_info.per_level_info_map.at(child_level_id).at(pv).rank_formats[r_id])
          {
            fail_reason << "pretiling for " << problem::GetShape()->DataSpaceIDToName.at(pv)
                        << "required but level compression format does not align: "
                        << topology_specs.GetStorageLevel(level)->level_name << " "
                        << topology_specs.GetStorageLevel(child_level_id)->level_name
                        << std::endl;
            success = false;
            eval_status[overall_level_id].success = false;
            eval_status[overall_level_id].fail_reason = fail_reason.str();
            if (break_on_failure) return success;
          }
        }
      }

      // singleton subnests for current level and bypassed level
      std::vector <loop::Descriptor> singleton_metadata_subnest;
      std::vector <PointSet> singleton_metadata_subtile_point_set;
      problem::OperationPoint origin;
      problem::OperationSpace scalar_mold(state.workload_, origin, origin);
      
      // Go through the corresponding storage levels to retrieve info
      for (int l = level; l >= int(inner_most_level); l--)
      {
        for (int loop_id = state.complete_subnests_[l].size() - 1; loop_id >= 0; loop_id--)
        {
          auto loop = state.complete_subnests_[l][loop_id];
          // bool trivial_loop = state.trivial_nest_masks_[l][loop_id];

          // pick out loops that are relevant (trivial and non-trivial, non-trivial loops will eventually be removed)
          if (dim_ids_in_proj.find(loop.dimension) != dim_ids_in_proj.end())
          {
            singleton_metadata_subnest.insert(singleton_metadata_subnest.begin(), loop);
            problem::OperationSpace maxtile_mold(state.workload_, origin, state.maxtile_molds_high_[l][loop_id]);
            singleton_metadata_subtile_point_set.insert(singleton_metadata_subtile_point_set.begin(), maxtile_mold.GetDataSpace(pv));
          }
        }
      }

      if (!pre_tiling_required)
      {

        // without pretiling, the subnests associated with the current tile
        // must be bounded to the include all subtile sizes in the inner levels
        // only looking at the subnest bounds at the current level is not enough

        // 1) collect all inner level subnests to get the global bound for each dimension
        // FIXME: temporal and spatial loop bounds might get multiplied together and
        //  the loop type will be set to whichever that is at the top,
        //  this behavior does not affect the correctness in terms of fiber tree construction,
        //  but making it cleaner would be helpful
        problem::PerFlattenedDimension <std::uint64_t> dimension_sizes;
        problem::PerFlattenedDimension <std::uint64_t> residual_sizes;
        for (unsigned i = 0; i < dimension_sizes.size(); i++)
        {
          dimension_sizes[i] = 1;
          residual_sizes[i] = 1;
        }

        for (unsigned i = 0; i < inner_most_level; i++)
        {
          auto subnest = state.complete_subnests_[i];
          for (auto loop = subnest.begin(); loop != subnest.end(); loop++)
          {
            dimension_sizes[loop->dimension] *= ceil((loop->end - loop->start) / loop->stride);
            residual_sizes[loop->dimension] *= loop->residual_end;
          }
        }

        // 2) scale the current level bound accordingly
        // note that after this step, there can be trivial loops left
        for (unsigned subnest_id = 0; subnest_id < singleton_metadata_subnest.size(); subnest_id++)
        {
          auto& loop = singleton_metadata_subnest[subnest_id];
          loop.end = loop.end * loop.stride * dimension_sizes[loop.dimension];
          loop.residual_end = loop.residual_end * residual_sizes[loop.dimension];

          // if there are two loops of the same dim in cur level subnest, the upper loop will not be scaled
          dimension_sizes[loop.dimension] = 1;
          residual_sizes[loop.dimension] = 1;
        }
      }

      // Map the non-trivial loops to the hardware supported ranks:
      // 1) Get rid of potential trivial loops
      // 2) Flatten necessary loops according to flattening rule
      std::uint64_t inner_rank_id = pre_tiling_required ? child_level_num_ranks : 0;
      sparse::PerDataSpaceCompressionInfo pv_compression_info;
      if (cur_level_has_metadata)
      {
        pv_compression_info = compression_info.per_level_info_map.at(level).at(pv);
      }

      bool more_compression_ranks_needed;
      if (!pv_compression_info.apply_rank_inner_to_outer)
      {
        more_compression_ranks_needed = ApplyRanksOuterToInner(inner_rank_id, singleton_metadata_subnest,
                                                               singleton_metadata_subtile_point_set,
                                                               pv_compression_info,
                                                               pv_data_movement_nest[level]);
      } else
      {
        more_compression_ranks_needed = ApplyRanksInnerToOuter(inner_rank_id, singleton_metadata_subnest,
                                                               singleton_metadata_subtile_point_set,
                                                               pv_compression_info,
                                                               pv_data_movement_nest[level]);
      }

      pv_data_movement_nest[level].apply_rank_inner_to_outer = pv_compression_info.apply_rank_inner_to_outer;

      if (more_compression_ranks_needed)
      {

        fail_reason << "more compression ranks needed than supported in hardware."
                    << " dataspace name: " << problem::GetShape()->DataSpaceIDToName.at(pv);
        success = false;
        eval_status[overall_level_id].success = false;
        eval_status[overall_level_id].fail_reason = fail_reason.str();
        if (break_on_failure) return success;
      }

      if (pre_tiling_required)
      {

        // pre-tiling requires the tile to maintain the order and bounds of the metadata subnests from its child level
        for (int loop_id = pv_data_movement_nest[child_level_id].metadata_subnest.size() - 1; loop_id >= 0; loop_id--)
        {
          auto loop = pv_data_movement_nest[child_level_id].metadata_subnest[loop_id];
          pv_data_movement_nest[level].metadata_subnest.insert(pv_data_movement_nest[level].metadata_subnest.begin(),
                                                               loop);

          pv_data_movement_nest[level].metadata_subtile_point_set.emplace(pv_data_movement_nest[level].metadata_subtile_point_set.begin(),
                                                                          pv_data_movement_nest[child_level_id].metadata_subtile_point_set.at(loop_id + 1));

        }
        // subtile shape must have one more element than subtile nest
        // see assert below for more
        pv_data_movement_nest[level].metadata_subtile_point_set.emplace(pv_data_movement_nest[level].metadata_subtile_point_set.begin(),
                                                                   pv_data_movement_nest[child_level_id].metadata_subtile_point_set[0]);
      } 
      else
      {
        pv_data_movement_nest[level].metadata_subtile_point_set.emplace(pv_data_movement_nest[level].metadata_subtile_point_set.begin(), scalar_mold.GetDataSpace(pv));
      }

      if (pv_data_movement_nest[level].metadata_subnest.size() != cur_level_num_ranks)
      {
        std::cout << topology_specs.GetStorageLevel(level)  << ": metadata models defined incorrectly, dataspace name: " << problem::GetShape()->DataSpaceIDToName.at(pv) << std::endl;
        std::cout << "defined number of ranks: " << pv_data_movement_nest[level].metadata_subnest.size() 
          << " expected number of ranks: " << cur_level_num_ranks << std::endl;
        for (unsigned rank_id = 0; rank_id < pv_data_movement_nest[level].metadata_subnest.size(); rank_id++)
        {
          std::cout << " --- flattened loops --- " << std::endl;
          for (auto loop = pv_data_movement_nest[level].metadata_subnest[rank_id].begin();
        	   loop != pv_data_movement_nest[level].metadata_subnest[rank_id].end(); loop++)
          {
            std::cout << *loop << std::endl;
          }
        }
      }

      // validity check on if the required number of ranks == number of hardware supported ranks
      assert(pv_data_movement_nest[level].metadata_subnest.size() == cur_level_num_ranks);
      
      // calculate the fiber shapes at each rank of metadata
      for (unsigned rank_id = 0; rank_id < cur_level_num_ranks; rank_id++)
      {
        problem::OperationPoint origin;
        problem::OperationPoint dimension_sizes;
        dimension_sizes.IncrementAllDimensions(); // initialize to { 1, 1, 1... }

        // going through the flattened ranks (if size is 1, then there is not flattened ranks
        for (auto loop = pv_data_movement_nest[level].metadata_subnest[rank_id].begin();
             loop != pv_data_movement_nest[level].metadata_subnest[rank_id].end(); loop++)
        {
          dimension_sizes[loop->dimension] *= ceil((loop->end - loop->start) / loop->stride);
        }

        // project to operation space to get fiber shape
        problem::OperationPoint high = dimension_sizes;
        high.IncrementAllDimensions(-1);
        problem::OperationSpace offset_tile(state.workload_, origin, high);
        compound_data_movement_nest[pv][level].fiber_shape.push_back(offset_tile.GetSize(pv));
      }

      // subtile shape must have one more element than subtile nest
      // as it includes the tile size of the child level:
      //     important for compressed metadata models to get the prob of empty coordinates in the last level of metadata
      assert(pv_data_movement_nest[level].metadata_subnest.size() + 1 
             == pv_data_movement_nest[level].metadata_subtile_point_set.size());

      // print info for sanity checks

      //std::cout << "\nDataspace name: " << problem::GetShape()->DataSpaceIDToName.at(pv)
      //          << "  level: " << level << " " << topology_specs.GetStorageLevel(level)->level_name
      //          << "   pretiling required: " << pre_tiling_required
      //          << " compressed: " << compression_info.compressed_masks[level][pv] << std::endl;
      //for (unsigned i = 0; i < pv_data_movement_nest[level].metadata_subnest.size(); i++)
      //{
      //  std::cout << " ----- rank: " << i << " ------" << std::endl;
      //  if (compression_info.compressed_masks[level][pv])
      //    std::cout << "   rank format: " << compression_info.per_level_info_map.at(level).at(pv).rank_formats[i]
      //              << std::endl;
      //  std::cout << "   rank tile shape: " << pv_data_movement_nest[level].metadata_subtile_point_set[i + 1].size() << std::endl;
      //  std::cout << "   rank subtile shape: " << pv_data_movement_nest[level].metadata_subtile_point_set[i].size() << std::endl;
      //  std::cout << "   fiber shape: " << pv_data_movement_nest[level].fiber_shape[i] << std::endl;
      //  std::cout << "   flattened nests: " << pv_data_movement_nest[level].metadata_subnest[i].size() << std::endl;

      //for (auto iter = pv_data_movement_nest[level].metadata_subnest[i].begin();
      //       iter != pv_data_movement_nest[level].metadata_subnest[i].end(); iter++)
      //  {
      //    std::cout << "\t" << *iter << std::endl;
      //  }
      //}

      // look at parent directly in the next round, as we know the levels in the middle are bypassed
      auto parent_level = pv_data_movement_nest[level].parent_level;
      level = parent_level;
    }
  }
  return success;
}

} // namespace
