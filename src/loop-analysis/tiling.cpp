/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "tiling.hpp"

namespace tiling
{

bool operator < (const TileInfo& a, const TileInfo& b)
{
  // Logic doesn't matter as long as we provide a way to detect inequality.
  return (a.size < b.size) ||
         (a.size == b.size && a.GetTotalAccesses() < b.GetTotalAccesses());
}

std::ostream& operator << (std::ostream& out, const TileInfo& info)
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

// Helper function: find the multicast factor.
uint64_t FindMulticastFactor(const TileInfo& tile)
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
void MaskTiles(std::vector<TileInfo>& tile_nest, std::bitset<MaxTilingLevels> mask)
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
void DistributeTiles(std::vector<TileInfo>& tile_nest,
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
void ComputeFills(std::vector<TileInfo>& tile_nest)
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
        // the halos will be counted as "multicast" data. However, we are not sure if scatter
        // factor calculation via spatial deltas looks at fragments of delivered temporal
        // deltas, or entire temporal volumes.
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

// Collapse tiles into a given number of levels.
// Input and output are both arrays of tile nests,
// with one nest per problem::Shape::DataSpaceID.
CompoundTileNest CollapseTiles(CompoundTileNest& tiles, int num_tiling_levels,
                               const CompoundMaskNest& tile_mask,
                               const CompoundMaskNest& distribution_supported)
{
  // Constructing an array of tile nests, one for each problem::Shape::DataSpaceID.
  // From the tile data, select the size and accesses at the boundaries of each
  // storage level. Size comes from the outermost tile within the storage level,
  // and accesses comes from the innermost tile within the storage level.
  CompoundTileNest solution;
  for (int pv = 0; pv < int(problem::GetShape()->NumDataSpaces); pv++)
  {
    int processed_loop_count = 0;  // number of loops that have been collapsed
    int cur_tiling_level = 0;
    int total_loops = tiles[pv].size();

    while (processed_loop_count < total_loops)
    {
      // Form a new physical tiling level.
      TileInfo collapsed_tile;

      // Figure out the last loop that belongs to the current tile.
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
      collapsed_tile.partition_size = tiles[pv][outermost_loop].partition_size;
      collapsed_tile.distributed_multicast = false;
      collapsed_tile.accesses = tiles[pv][innermost_loop].accesses;
      collapsed_tile.scatter_factors = tiles[pv][innermost_loop].scatter_factors;
      collapsed_tile.cumulative_hops = tiles[pv][innermost_loop].cumulative_hops;
      collapsed_tile.content_accesses = tiles[pv][innermost_loop].content_accesses;
      collapsed_tile.link_transfers = tiles[pv][innermost_loop].link_transfers;
      collapsed_tile.replication_factor = tiles[pv][outermost_loop].replication_factor;
      collapsed_tile.fanout = tiles[pv][innermost_loop].fanout;

      solution[pv].push_back(collapsed_tile);

      processed_loop_count = boundary_loop_id + 1;
      cur_tiling_level++;
    }
    assert(cur_tiling_level == num_tiling_levels);

    // Calculate fills.
    ComputeFills(solution[pv]);

    // Mask each solution according to the provided bit mask.
    MaskTiles(solution[pv], tile_mask[pv]);

    // Perform distributed-multicast if supported.
    DistributeTiles(solution[pv], distribution_supported[pv]);
  }
  return solution;
}

// Collapse with default tile mask.
// CompoundTileNest CollapseTiles(CompoundTileNest& tiles, int num_tiling_levels)
// {
//   std::bitset<MaxTilingLevels> all_ones;
//   all_ones.set();
//   CompoundMaskNest all_enabled;
//   all_enabled.fill(all_ones);
//   return CollapseTiles(tiles, num_tiling_levels, all_enabled);
// }

NestOfCompoundTiles TransposeTiles(const CompoundTileNest & tiles)
{
  NestOfCompoundTiles retval;
  
  std::size_t num_levels = tiles[0].size();
  for (std::size_t level = 0; level < num_levels; level++)
  {
    CompoundTile tile_level;
    for (int pv = 0; pv < int(problem::GetShape()->NumDataSpaces); pv++)
    {
      tile_level[pv] = tiles[pv][level];
    }
    retval.push_back(tile_level);
  }

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
