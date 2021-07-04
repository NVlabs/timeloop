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

#include <cassert>
#include <sstream>

#include "loop-analysis/operation-type.hpp"
#include "loop-analysis/tiling.hpp"

namespace tiling
{

bool operator < (const DataMovementInfo& a, const DataMovementInfo& b)
{
  // Logic doesn't matter as long as we provide a way to detect inequality.
  return (a.size < b.size) ||
         (a.size == b.size && a.GetTotalAccesses() < b.GetTotalAccesses());
}

std::ostream& operator << (std::ostream& out, const DataMovementInfo& info)
{
  out << "size = " << info.size << " accesses = " << info.GetTotalAccesses()
      << " fanout = " << info.fanout << " repfactor = " << info.replication_factor
      << " linkxfers = " << info.link_transfers << std::endl;
  for (std::uint32_t i = 0; i < info.accesses.size(); i++)
  {
    out << "    [" << i << "] = " << info.accesses[i] << " @scatter = "
        << info.scatter_factors[i] << " hops = " << info.cumulative_hops[i] << std::endl;
  }
  return out;
}

} // namespace tiling

namespace tiling
{

void SetParentLevel(std::vector<DataMovementInfo>& tile_nest){
  unsigned num_tiling_levels = tile_nest.size();

  for (unsigned cur = 0; cur < num_tiling_levels; cur++)
  {

    // Skip if this tile level has 0 size or 0 accesses.
    if (tile_nest[cur].size == 0)
    {
      continue;
    }

    // Initialize parent level to max.
    tile_nest[cur].parent_level = std::numeric_limits<unsigned>::max();

    // Find next (outer) non-zero level.
    problem::Shape::DataSpaceID outer;
    for (outer = cur + 1; outer < num_tiling_levels && tile_nest[outer].size == 0; outer++)
    {
      // Body is empty.
    }

    if (outer == num_tiling_levels)
    {
      // No outer tiling level.
      continue;
    }

    tile_nest[cur].parent_level = outer;
  }

}

void SetChildLevel(std::vector<DataMovementInfo>& tile_nest){
  unsigned num_tiling_levels = tile_nest.size();

  for (unsigned cur = 0; cur < num_tiling_levels; cur++)
  {
    // Skip
    //     if this tile level has 0 size or 0 accesses.
    //     if this tile level is the inner most level
    if (cur == 0 || tile_nest[cur].size == 0)
    {
      continue;
    }

    // Initialize child level to max.
    tile_nest[cur].child_level = std::numeric_limits<unsigned>::max();
    // Find next (inner) non-zero level.
    int inner;
    for (inner = cur-1; inner >= 0 && tile_nest[inner].size == 0; inner--)
    {
      // Body is empty.
    }

    if ( inner != -1)
    {
      tile_nest[cur].child_level = inner;
    }

    // better checking for child level
    if (inner != 0 && inner != -1){
    } else if (inner != -1) {
      // child level is compute
      // tile size is basically one fetch from the memory, which is dependent on this level's block size
    }
  }
}

// Helper function: find the multicast factor.
uint64_t FindMulticastFactor(const DataMovementInfo& tile)
{
  uint64_t multicast_factor = 1;
  bool multicast_found = false;
  for (uint64_t i = 0; i < tile.fanout; i++)
  {
    if (tile.accesses[i] != 0)
    {
      assert(!multicast_found);
      multicast_found = true;
      multicast_factor = i + 1;
    }
  }
  assert(multicast_found);
  return multicast_factor;
}

// Mask Tiles.
void MaskTiles(std::vector<DataMovementInfo>& tile_nest, std::bitset<MaxTilingLevels> mask)
{
  // std::cout << "***** BEFORE *****" << std::endl;
  // std::cout << "Tile nest = " << std::endl;
  // for (auto & tile : tile_nest)
  //   std::cout << "    " << tile << std::endl;
  // std::cout << "Tile mask = " << mask << std::endl;
  
  int num_tiling_levels = tile_nest.size();

  for (int cur = 0; cur < num_tiling_levels; cur++)
  {
    // Skip if this tile level already has 0 size, or if it's not masked.
    if (tile_nest[cur].size == 0 || mask[cur])
    {
      continue;
    }

    // Special case: if the last tiling level is forcibly masked, simply
    // discard it. Caller is responsible to make sure this doesn't mess
    // counts up. WARNING!
    if (cur == num_tiling_levels - 1)
    {
      tile_nest[cur].Reset();
      continue;
    }

    // Regular cases: inner tiling levels.
    // Okay, we need to mask this tile level. Find next (outer) non-zero level.
    int outer;
    for (outer = cur + 1; outer < num_tiling_levels && tile_nest[outer].size == 0; outer++)
    {
      // Body is empty.
    }

    if (outer == num_tiling_levels)
    {
      continue;
    }

    //
    // Handle link transfers.
    //
    if (tile_nest[cur].link_transfers > 0)
    {
      // Level cur was using link transfers. This means the outer level had
      // fewer accesses. Since we're deleting cur, we're promoting all accesses
      // from cur's child level up to the outer level. So the fact that cur
      // was taking advantage of link transfers is immaterial (FIXME: prove).
    }

    //
    // Handle multicast.
    //
    
    // First, determine the actual utilized multicast factors. This can be
    // 1, the max fanout, or somewhere in between. It's possible to constrain
    // mappings such that the utilized fanout is either 1 or max fanout, but
    // we won't make that assumption here.
    
    tile_nest[outer].content_accesses = 0;
    
    for (uint64_t i = 0; i < tile_nest[outer].fanout; i++)
    {
      if (tile_nest[outer].accesses[i] != 0)
      {
        // The outer content (buffer) will now be accessed as frequently
        // as the inner content was. However, if the outer level had a fanout, then
        // these accesses may be further amplified. Multicasts along the fanout
        // do not amplify accesses, but scatters do. For non-spatial-sliding-windows
        // we can compute scatter factor as fanout/multicast. For spatial sliding
        // windows, the relationship between content accesses, scatter, multicast
        // and max fanout is complex.
        // - FIXME 1: spatial sliding windows.
        // - FIXME 2: outer-cur > 1.
        auto multicast_factor = i + 1;
        auto scatter_factor = tile_nest[outer].scatter_factors[i];
        
        tile_nest[outer].content_accesses +=
          tile_nest[cur].content_accesses * scatter_factor;
      
        // The outer network will now be energized as frequently as
        // the inner content was accessed.
        tile_nest[outer].accesses[i] = 0;
        tile_nest[outer].scatter_factors[i] = 0;
        tile_nest[outer].accesses[multicast_factor-1] =
        tile_nest[cur].content_accesses * scatter_factor;
        tile_nest[outer].scatter_factors[multicast_factor-1] = scatter_factor;

        // Note: partition size for outer does not change.
      }
    }

    // Obliterate the buffer stats (*not* the network stats) for the cur tiling level.
    tile_nest[cur].size = 0;
    tile_nest[cur].shape = 0;
    tile_nest[cur].partition_size = 0;
    tile_nest[cur].content_accesses = 0;
    tile_nest[cur].fills = 0;

    // It appears fills are not affected by masking (FIXME: verify).
  }

  // std::cout << "***** AFTER *****" << std::endl;
  // std::cout << "Tile nest = " << std::endl;
  // for (auto & tile : tile_nest)
  //   std::cout << "    " << tile << std::endl;  
}

// Convert multicasts into scatter->distributed-multicasts if certain conditions
// are met.
void DistributeTiles(std::vector<DataMovementInfo>& tile_nest,
                     const std::bitset<MaxTilingLevels>& distribution_supported)
{
  int num_tiling_levels = tile_nest.size();
  for (int inner = 0; inner < num_tiling_levels-1; inner++)
  {
    // Skip if this tile level has 0 size (i.e., was masked), or if it
    // doesn't support distributed multicast.
    if (tile_nest[inner].size == 0 || !distribution_supported[inner])
    {
      continue;
    }

    // Find next outer non-zero level.
    int outer;
    for (outer = inner + 1; outer < num_tiling_levels && tile_nest[outer].size == 0; outer++)
    {
      // Body is empty.
    }

    if (outer == num_tiling_levels)
    {
      continue;
    }

    uint64_t outer_multicast_factor = FindMulticastFactor(tile_nest[outer]);
    uint64_t inner_multicast_factor = FindMulticastFactor(tile_nest[inner]);

    // If outer has a >1 multicast factor, then it has a choice to not multicast
    // but to scatter and use a distributed multicast the next level down.
    // This is not a binary choice; I can factorize the multicast factor into
    // distributed/private factors, but for now we only consider a fully spread-out
    // scatter.
    if (outer_multicast_factor > 1)
    {
      // The outer tile's per-instance access count is unchanged, but its
      // multicast factor changes to 1 (effectively turning it into a
      // full scatter fanout).
      tile_nest[outer].accesses[0] = tile_nest[outer].accesses[outer_multicast_factor-1];
      tile_nest[outer].scatter_factors[0] = tile_nest[outer].fanout;
      tile_nest[outer].accesses[outer_multicast_factor-1] = 0;
      tile_nest[outer].scatter_factors[outer_multicast_factor-1] = 0;

      // The inner tile's per-instance size, partition size and content-access count
      // reduces by the outer multicast factor. Note that this may not be a perfect
      // factor, so do a ceil-div.
      tile_nest[inner].size = 1 + (tile_nest[inner].size - 1) / outer_multicast_factor;
      tile_nest[inner].partition_size = 1 +
        (tile_nest[inner].partition_size - 1) / outer_multicast_factor;
      tile_nest[inner].content_accesses = 1 +
        (tile_nest[inner].content_accesses - 1) / outer_multicast_factor;

      // The inner tile's network accesses will now happen at a distributed-multicast
      // factor of outer_multicast_factor. These alterations will magically trigger
      // all the right computations at the model evaluation stage.
      uint64_t distributed_multicast_factor = outer_multicast_factor * inner_multicast_factor;
      if (distributed_multicast_factor > tile_nest[inner].fanout)
      {
        tile_nest[inner].accesses.resize(distributed_multicast_factor);
        tile_nest[inner].scatter_factors.resize(distributed_multicast_factor);
      }
      tile_nest[inner].distributed_multicast = true;
      tile_nest[inner].distributed_fanout = distributed_multicast_factor * tile_nest[inner].fanout;
      tile_nest[inner].accesses[distributed_multicast_factor-1] = 1 + 
        (tile_nest[inner].accesses[inner_multicast_factor-1] - 1) / outer_multicast_factor;
      tile_nest[inner].scatter_factors[distributed_multicast_factor-1] = tile_nest[inner].scatter_factors[inner_multicast_factor-1];
      tile_nest[inner].accesses[inner_multicast_factor-1] = 0;
      tile_nest[inner].scatter_factors[inner_multicast_factor-1] = 0;

      // ***** FIXME ***** make this work with PRECISE_MULTICAST, which means we
      // need to update cumulative_hops.
      
      // We should be doing this process hierarchically along the entire tile stack,
      // but for the moment just support doing this once.
      break;
    }
  }  
}

// Compute Fills.
void ComputeFills(std::vector<DataMovementInfo>& tile_nest)
{
  int num_tiling_levels = tile_nest.size();

  for (int cur = 0; cur < num_tiling_levels; cur++)
  {
    
    // Skip if this tile level has 0 size or 0 accesses.
    if (tile_nest[cur].size == 0)
    {
      continue;
    }

    // Initialize fills to 0.
    tile_nest[cur].fills = 0;
    
    // Find next (outer) non-zero level.
    int outer;
    for (outer = cur + 1; outer < num_tiling_levels && tile_nest[outer].size == 0; outer++)
    {
      // Body is empty.
    }

    if (outer == num_tiling_levels)
    {
      // No outer tiling level.
      continue;
    }

    // std::cerr << "  cur = " << cur << std::endl;
    // std::cerr << "  outer = " << outer << std::endl;

    // Found an outer level.
    for (uint64_t i = 0; i < tile_nest[outer].fanout; i++)
    {
      if (tile_nest[outer].accesses[i] != 0)
      {
        // FIXME: is this correct in the face of spatial sliding windows (e.g. Input halos)?
        // If scatter factors are calculated on fragments, then this will be correct, because
        // the halos will be counted as "multicast" data. However, scatter factor calculation
        // via spatial deltas does not look at fragments of delivered temporal deltas, the
        // code compares complete temporal deltas delivered to peer spatial instances.
        // To fix this, we should be able to use the new overlap-fraction based method used to
        // calculate partition sizes in some way.
        tile_nest[cur].fills += tile_nest[outer].accesses[i] / tile_nest[outer].scatter_factors[i];
        // std::cerr << "    mcast = " << i+1 << std::endl;
        // std::cerr << "      outer accesses = " << tile_nest[outer].accesses[i] << std::endl;
        // std::cerr << "      outer scatter = " << tile_nest[outer].scatter_factors[i] << std::endl;
        // std::cerr << "      cur fills incr = " << tile_nest[outer].accesses[i] / tile_nest[outer].scatter_factors[i] << std::endl;
        // std::cerr << "      cur upd fills = " << tile_nest[cur].fills << std::endl;
        // std::cerr << "      cur accesses = " << tile_nest[cur].accesses[i] << std::endl;
      }
    }

    // assert(tile_nest[cur].fills <= tile_nest[cur].GetTotalAccesses());
  }

}

// Compute partition sizes.
void ComputePartitionSizes(std::vector<DataMovementInfo>& tile_nest)
{
  int num_tiling_levels = tile_nest.size();

  // Outermost level must contain the full dataspace partition in
  // its single instance.
  std::size_t partition_size = tile_nest[num_tiling_levels-1].size;
  tile_nest[num_tiling_levels-1].partition_size = partition_size;

  for (int cur = num_tiling_levels-2; cur >= 0; cur--)
  {
    if (tile_nest[cur].partition_fraction_denominator != 0)
    {
    partition_size = (partition_size * tile_nest[cur].size) /
      tile_nest[cur].partition_fraction_denominator;
    }
    tile_nest[cur].partition_size = partition_size;
  }  
}

// Compute the extra fills and accesses due to link transfers in the previous
// level. Link transfers are handled at the network model, and the extra buffer
// accesses should charge the buffer model.
void ComputePeerAccesses(std::vector<DataMovementInfo>& tile_nest)
{
  // Loop through all levels and update peer_{accesses, fills}.
  //
  int num_tiling_levels = tile_nest.size();

  // pair-wise comparison
  for (int cur = num_tiling_levels-1; cur > 0; cur--)
  {
    if (tile_nest[cur].link_transfers != 0)
    {
      // FIXME: For now our assumption is that all spatial units in a level
      // are responsible for all peer communication, even though there can be
      // some optimizations. For example, one read for PE x is multicast to
      // two other PEs, saving one read to the buffer.  Simially, we also
      // assume read and fill comes in pair. However, there can be some other
      // optimizations that break this assumption.
      auto spatial_size = tile_nest[cur - 1].replication_factor;
      assert(spatial_size > 1);
      auto access_per_element = tile_nest[cur].link_transfers / spatial_size;
      auto fills_per_element = tile_nest[cur].link_transfers / spatial_size;
      tile_nest[cur - 1].peer_accesses += access_per_element;
      tile_nest[cur - 1].peer_fills += fills_per_element;
    }
  }

  return;
}

// FIXME: check the if logic for hardware reduction support is still in the loop

void ComputeReadUpdateReductionAccesses(std::vector<DataMovementInfo>& tile_nest, problem::Shape::DataSpaceID pv){
  // Loop through all levels and update reads, writes, updates.
  //
  int num_tiling_levels = tile_nest.size();

  for (int cur = 0; cur < num_tiling_levels; cur++){
    
    if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      // First epoch is an Update, all subsequent epochs are Read-Modify-Update.

      // The following assertion is *incorrect* for coefficients (e.g. stride, pad) > 1.
      // FIXME: find a safety check that works with coefficients > 1.
      // assert(tile[pvi].size == 0 || tile[pvi].content_accesses % tile[pvi].size == 0);

      tile_nest[cur].reads = tile_nest[cur].content_accesses - tile_nest[cur].partition_size + tile_nest[cur].peer_accesses;
      // std::cout << "TILING LEVEL = " << cur << std::endl;
      // std::cout << "  content = " << tile_nest[cur].content_accesses << std::endl;
      // std::cout << "  partition size = " << tile_nest[cur].partition_size << std::endl;
      // std::cout << "  peer accesses = " << tile_nest[cur].peer_accesses << std::endl;
      // std::cout << "  reads = " << tile_nest[cur].reads << std::endl << std::endl;

      tile_nest[cur].updates = tile_nest[cur].content_accesses;
      tile_nest[cur].fills = tile_nest[cur].fills + tile_nest[cur].peer_fills;
      //tile.address_generations[pv] = stats_.updates[pv] + stats_.fills[pv]; // scalar

      // FIXME: temporal reduction and network costs if hardware reduction isn't
      // supported appears to be wonky - network costs may need to trickle down
      // all the way to the level that has the reduction hardware.
      tile_nest[cur].temporal_reductions = tile_nest[cur].content_accesses - tile_nest[cur].partition_size;

      // std::cout << "tile: reads, updates, fills " 
      // << tile.reads << " " <<  tile.updates<< " " << tile.fills <<std::endl;
    }
    else // Read-only data type.
    {
      tile_nest[cur].reads = tile_nest[cur].content_accesses + tile_nest[cur].peer_accesses;
      tile_nest[cur].updates = 0;
      tile_nest[cur].fills = tile_nest[cur].fills + tile_nest[cur].peer_fills;
      //tile.address_generations = tile.reads + tile.fills; // scalar
      tile_nest[cur].temporal_reductions = 0;
    }
  }

  return;
}


void ComputeWorkloadTensorSizes(std::vector<DataMovementInfo>& tile_nest, problem::Shape::DataSpaceID pv, problem::Workload* workload){
   if (! workload->IsWorkloadTensorSizesSet()){
      unsigned num_tiling_levels = tile_nest.size();

      uint64_t max_tensor_size = 0;

      for (unsigned cur = 0; cur < num_tiling_levels; cur++)
      {

        // Skip if this tile level has 0 size or 0 accesses.
        if (tile_nest[cur].size >= max_tensor_size)
        {
          max_tensor_size = tile_nest[cur].size;
        }
      }

      assert(max_tensor_size != 0); //workload tensor size cannot be zero
      workload->SetWorkloadTensorSize(pv, max_tensor_size);
   }

}

// place holder function that performs post processing of the # of fills for
// the backing storage of each data space.
// theoretically, we should reorder ComputeFills and MaskTiles to achieve this
// and the logic for cascaded multicast calculation with bypassed storage level
// in the middle needs to be updated
void ResetBackingStorageFillsPlaceHolder(std::vector<DataMovementInfo>& tile_nest)
{
  unsigned num_tiling_levels = tile_nest.size();

  for (int cur = num_tiling_levels - 1; cur >=0 ; cur--)
  {
    if (tile_nest[cur].size > 0)
    {
      tile_nest[cur].fills = 0;
      break;
    }
  }
}


tiling::CompoundTileNest CollapseTiles(analysis::CompoundTileNest& tiles,
                                       int num_tiling_levels,
                                       const CompoundMaskNest& tile_mask,
                                       const CompoundMaskNest& distribution_supported,
                                       problem::Workload* workload){

  CompoundDataMovementNest collapsed_compound_data_nest = CollapseDataMovementNest(tiles.compound_data_movement_info_nest,
                                                                                   num_tiling_levels,
                                                                                   tile_mask,
                                                                                   distribution_supported, 
                                                                                   workload);
  ComputeNest collapsed_compound_compute_nest = CollapseComputeNest(tiles.compound_compute_info_nest, num_tiling_levels); 
  tiling::CompoundTileNest solution;
  solution.compound_data_movement_info_nest = collapsed_compound_data_nest;
  solution.compute_info_nest = collapsed_compound_compute_nest;
  solution.compute_info_nest[0].compute_cycles = solution.compute_info_nest[0].accesses;
  return solution;
}


ComputeNest CollapseComputeNest(analysis::CompoundComputeNest& tiles, int num_tiling_levels){
  ComputeNest solution;
  
  for (int level=0; level < num_tiling_levels; level++){
    
    ComputeInfo collapsed_tile;
    if (level == 0 ){
      // compute info is only valid for the inner most level
      collapsed_tile.replication_factor = tiles[0].replication_factor;
      collapsed_tile.accesses = tiles[0].accesses;
    } else {
      collapsed_tile.replication_factor = 0;
      collapsed_tile.accesses = 0;
    }
    solution.push_back(collapsed_tile);
  }

  return solution;
}


// Collapse tiles into a given number of levels.
// Input and output are both arrays of tile nests,
// with one nest per problem::Shape::DataSpaceID.
CompoundDataMovementNest CollapseDataMovementNest(analysis::CompoundDataMovementNest& tiles, 
                                                  int num_tiling_levels,
                                                  const CompoundMaskNest& tile_mask,
                                                  const CompoundMaskNest& distribution_supported,
                                                  problem::Workload* workload)
{
  // Constructing an array of tile nests, one for each problem::Shape::DataSpaceID.
  // From the tile data, select the size and accesses at the boundaries of each
  // storage level. Size comes from the outermost tile within the storage level,
  // and accesses comes from the innermost tile within the storage level.
  CompoundDataMovementNest solution;
  for (int pv = 0; pv < int(problem::GetShape()->NumDataSpaces); pv++)
  {
    int processed_loop_count = 0;  // number of loops that have been collapsed
    int cur_tiling_level = 0;
    int total_loops = tiles[pv].size();

    while (processed_loop_count < total_loops)
    {
      // Form a new physical tiling level.
      DataMovementInfo collapsed_tile;

      // Find the last loop that belongs to the current tile.
      int boundary_loop_id = processed_loop_count;

      while (!tiles[pv][boundary_loop_id].is_on_storage_boundary &&
             boundary_loop_id < total_loops - 1)
      {
        boundary_loop_id++;
      }

      // Build the subnest of the current tile.
      for (int loop_id = processed_loop_count; loop_id <= boundary_loop_id; loop_id++)
      {
        collapsed_tile.subnest.insert(collapsed_tile.subnest.end(),
                                      tiles[pv][loop_id].subnest.begin(),
                                      tiles[pv][loop_id].subnest.end());
      }

      auto outermost_loop = boundary_loop_id;
      auto innermost_loop = processed_loop_count;

      // Compute various properties of the collapsed tile.
      // NOTE! some properties are taken from the innermost loop's tile, while others
      // are taken from the outermost loop's tile.
      collapsed_tile.size = tiles[pv][outermost_loop].size;
      collapsed_tile.shape = tiles[pv][outermost_loop].size; // shape is the coord space representation
      collapsed_tile.dataspace_id = (unsigned)pv;
      collapsed_tile.partition_size = 0;
      collapsed_tile.distributed_multicast = false;
      collapsed_tile.accesses = tiles[pv][innermost_loop].accesses;
      collapsed_tile.scatter_factors = tiles[pv][innermost_loop].scatter_factors;
      collapsed_tile.cumulative_hops = tiles[pv][innermost_loop].cumulative_hops;
      collapsed_tile.content_accesses = tiles[pv][innermost_loop].GetTotalAccesses();
      collapsed_tile.link_transfers = tiles[pv][innermost_loop].link_transfers;
      collapsed_tile.peer_accesses = 0;
      collapsed_tile.peer_fills = 0;
      collapsed_tile.replication_factor = tiles[pv][outermost_loop].replication_factor;
      collapsed_tile.fanout = tiles[pv][innermost_loop].fanout;

      //place holder initializations
      collapsed_tile.metadata_reads = 0;
      collapsed_tile.metadata_fills = 0;
      collapsed_tile.metadata_updates = 0;
      collapsed_tile.SetTensorRepresentation(); // default to uncompressed

      collapsed_tile.parent_level = std::numeric_limits<unsigned>::max();
      collapsed_tile.child_level = std::numeric_limits<unsigned>::max();

      if (!solution[pv].empty())
      {
        auto& inner_tile = solution[pv].back();

        inner_tile.partition_fraction_denominator =
          tiles[pv][innermost_loop].is_master_spatial ?
          tiles[pv][innermost_loop].size :
          inner_tile.size;
      }

      solution[pv].push_back(collapsed_tile);

      processed_loop_count = boundary_loop_id + 1;
      cur_tiling_level++;
    }
    assert(cur_tiling_level == num_tiling_levels);

    // Compute partition sizes.
    ComputePartitionSizes(solution[pv]);

    // Calculate fills.
    ComputeFills(solution[pv]);

    // Mask each solution according to the provided bit mask.
    MaskTiles(solution[pv], tile_mask[pv]);

    // Set backing storage fill to zero
    // place holder
    ResetBackingStorageFillsPlaceHolder(solution[pv]);

    // Perform distributed-multicast if supported.
    DistributeTiles(solution[pv], distribution_supported[pv]);

    // Calculate the extra accesses and fills due to link transfers
    ComputePeerAccesses(solution[pv]);

    // split the accesses to read and update and generate reduction
    ComputeReadUpdateReductionAccesses(solution[pv], pv);

    // calculate workload tensor size
    ComputeWorkloadTensorSizes(solution[pv], pv, workload);

    // find the parent and child levels for later compression/decompression logic
    SetParentLevel(solution[pv]);
    SetChildLevel(solution[pv]);

  }

  // flip the workload tensor set flag if necessary
  if (! workload->IsWorkloadTensorSizesSet()) {workload->AllTensorsSet();}

  return solution;
}

void SetParentChildPointers(tiling::NestOfCompoundTiles& nest_of_compound_tiles)
{
  unsigned num_levels = nest_of_compound_tiles.size();
  for (unsigned level = 0; level < num_levels; level++)
  {
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      tiling::CompoundDataMovementInfo& compound_data_movement = nest_of_compound_tiles.at(level).data_movement_info;

      //
      // populate parent/child level compression specifications
      //
      problem::Shape::DataSpaceID parent_level = compound_data_movement[pv].parent_level;
      problem::Shape::DataSpaceID child_level = compound_data_movement[pv].child_level;

      if (parent_level != std::numeric_limits<unsigned>::max())
      {
        // set parent level pointer
        compound_data_movement[pv].parent_level_ptr = &nest_of_compound_tiles.at(parent_level).data_movement_info[pv];
      }

      if (child_level != std::numeric_limits<unsigned>::max())
      {
        // set child level pointer
        compound_data_movement[pv].child_level_ptr = &nest_of_compound_tiles.at(child_level).data_movement_info[pv];
      }

    } // next dataspace
  } // next level
}


NestOfCompoundTiles TransposeTiles(const CompoundTileNest& tiles)
{
  NestOfCompoundTiles retval;

  const CompoundDataMovementNest& data_movement_nest =  tiles.compound_data_movement_info_nest;
  const ComputeNest& compute_nest = tiles.compute_info_nest;

  unsigned num_levels = data_movement_nest[0].size();
  CompoundTile tile_level;
  ComputeInfo compute_info = compute_nest[0];

  // transpose all the tiles
  for (unsigned level = 0; level < num_levels; level++){
    //  Datamovement
    for (int pv = 0; pv < int(problem::GetShape()->NumDataSpaces); pv++){
      tile_level.data_movement_info[pv] = data_movement_nest[pv][level];
    }
    //  Compute
    tile_level.compute_info = compute_nest[level];
    retval.push_back(tile_level);
  }

  // set pointers inside each tile object, so that it is more convenient to perform overbooking analysis in model
  SetParentChildPointers(retval);

  return retval;
}

NestOfCompoundMasks TransposeMasks(const CompoundMaskNest& masks)
{
  NestOfCompoundMasks retval;
  
  for (std::size_t level = 0; level < MaxTilingLevels; level++)
  {
    CompoundMask mask_level;
    for (int pv = 0; pv < int(problem::GetShape()->NumDataSpaces); pv++)
    {
      mask_level[pv] = masks[pv].test(level);
    }
    retval.push_back(mask_level);
  }

  return retval;
}

}  // namespace tiling
