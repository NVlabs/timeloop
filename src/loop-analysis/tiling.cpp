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

#include "loop-analysis/tiling.hpp"
#include "workload/workload.hpp"
#include "loop-analysis/operation-type.hpp"

namespace tiling
{

bool gEnableFirstReadElision =
  (getenv("TIMELOOP_ENABLE_FIRST_READ_ELISION") == NULL) ||
  (strcmp(getenv("TIMELOOP_ENABLE_FIRST_READ_ELISION"), "0") != 0);

bool gUpdatedRMW =
  (getenv("TIMELOOP_ENABLE_UPDATED_RMW") != NULL) &&
  (strcmp(getenv("TIMELOOP_ENABLE_UPDATED_RMW"), "0") != 0);

bool operator < (const DataMovementInfo& a, const DataMovementInfo& b)
{
  // Logic doesn't matter as long as we provide a way to detect inequality.
  return (a.size < b.size) ||
    (a.size == b.size && a.access_stats.TotalAccesses() < b.access_stats.TotalAccesses());
}

std::ostream& operator << (std::ostream& out, const DataMovementInfo& info)
{
  out << "size = " << info.size << " accesses = " << info.access_stats.TotalAccesses()
      << " fanout = " << info.fanout << " repfactor = " << info.replication_factor
      << " linkxfers = " << info.link_transfers << std::endl;
  out << info.access_stats;
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

    // Regular cases: inner tiling levels.
    // Okay, we need to mask this tile level. Find next (outer) non-zero level.
    int outer;
    for (outer = cur + 1; outer < num_tiling_levels && tile_nest[outer].size == 0; outer++)
    {
      // Body is empty.
    }

    if (outer == num_tiling_levels)
    {
      // We did not find any outer unmasked tiling level. This means that
      // the cur tiling level was (so far) the outermost unmasked tiling level.
      // The mapping is asking us to mask this level. We need to do something
      // special for these outermost-masked levels, but we do that in a
      // subsequent pass. For now, simply obliterate the buffer stats (*not*
      // the network stats) for the cur tiling level, just like we do for all
      // other masked levels.

      // WARNING! this is dangerous: masking out an outer level is only
      // correct if the next-inner level contains the entire tensor. We
      // do not check for this here and assume it is the user's responsibility.
      tile_nest[cur].size = 0;
      tile_nest[cur].shape = 0;
      tile_nest[cur].partition_size = 0;
      tile_nest[cur].content_accesses = 0;
      tile_nest[cur].parent_access_share = 0;

      // Break out, we are done since there is no other unmasked level.
      break;
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
    
    // tile_nest[outer].content_accesses = 0;

    double all_children_content_accesses = tile_nest[cur].content_accesses * tile_nest[outer].fanout;

    // Warning! It's not clear if this child temporal factor is precise enough
    // if there is temporal imperfection.
    double child_temporal_factor = all_children_content_accesses / tile_nest[outer].access_stats.WeightedAccesses();

    for (auto& x: tile_nest[outer].access_stats.stats)
    {
      if (x.second.accesses == 0)
      {
        // Multicast detection code used to generate records with 0 accesses,
        // which messes up this logic. We've fixed that, so we technically
        // don't need this check, but we'll keep it for double safety.
        continue;
      }

      // The outer content (buffer) will now be accessed as frequently
      // as the inner content was. However, if the outer level had a fanout, then
      // these accesses may be further amplified. Multicasts along the fanout
      // do not amplify accesses, but scatters do.
      // - FIXME: outer - cur > 1.
      // auto scatter_factor = x.first.second;

      // tile_nest[outer].content_accesses +=
      //   tile_nest[cur].content_accesses * scatter_factor;

      // The outer network will now be energized as frequently as
      // the inner content was accessed.
      x.second.accesses *= child_temporal_factor;
      // x.second.accesses = tile_nest[cur].content_accesses * scatter_factor;

      // If a child level gets masked out, all _its_ children will now make
      // accesses to the parent. This means the link transfers are pointless – all
      // the data will be directly accessed at the parent. For RMW data spaces, if
      // the child level disappears (remember, it must have supported in-place
      // reduction) but the parent doesn’t support reduction, the grandchildren
      // must perform the fill-reduce-update cycle. Therefore, reduction
      // processing must happen after masking. Note that parents store the link
      // transfer counts for children. We don't have to check if outer==cur+1.
      tile_nest[outer].link_transfers = 0;

      // If a parent level gets masked out, we keep the link transfer stats
      // intact (as part of the network stats, which don’t get wiped out on being
      // masked). During peer transfer computation we’ll update child stats based
      // on these network stats. This means that in this code, we don't have to
      // check on our children.

      // Note: partition size for outer does not change.
    }

    // Recompute outer content accesses.
    tile_nest[outer].content_accesses = tile_nest[outer].access_stats.TotalAccesses();

    // Outer or child's parent access share is not affected by masking

    // Obliterate the buffer stats (*not* the network stats) for the cur tiling level.
    tile_nest[cur].size = 0;
    tile_nest[cur].shape = 0;
    tile_nest[cur].SetTensorRepresentation();
    tile_nest[cur].partition_size = 0;
    tile_nest[cur].content_accesses = 0;
    tile_nest[cur].parent_access_share = 0;

  }

  // std::cout << "***** AFTER *****" << std::endl;
  // std::cout << "Tile nest = " << std::endl;
  // for (auto & tile : tile_nest)
  //   std::cout << "    " << tile << std::endl;  
}

// Special post-processing for outermost-masked tile levels.
void ProcessOuterMaskedLevels(std::vector<DataMovementInfo>& tile_nest, std::bitset<MaxTilingLevels> mask)
{
  for (int cur = int(tile_nest.size())-1; cur >= 0; cur--)
  {
    // Work on all outermost masked levels until we find an unmasked level.
    if (!mask[cur])
    {
      // Blow up *all* stats (including network stats).
      tile_nest[cur].Reset();
    }
    else
    {
      // First unmasked level: set parent_access_share to 0.

      // WARNING! the updated RMW path uses parent_access_share in a more
      // sophisticated way. FIXME: revisit and make sure we do not have to do
      // anything different for that path.
      // if (gUpdatedRMW) { }
      tile_nest[cur].parent_access_share = 0;
      break;
    }
  }  
}

// Helper function: find the multicast factor. An issue is that many recent
// features (skew, flattening) introduce irregularity, which introduces
// complex multicast signatures with multiple multicast factors at each
// level. This function should be deprecated, but for the moment we
// support a version that works with a simple multicast signature.
// uint64_t FindMulticastFactor(const DataMovementInfo& tile)
// {
//   uint64_t multicast_factor = 1;
//   bool multicast_found = false;
//   for (uint64_t i = 0; i < tile.fanout; i++)
//   {
//     if (tile.accesses[i] != 0)
//     {
//       assert(!multicast_found);
//       multicast_found = true;
//       multicast_factor = i + 1;
//     }
//   }
//   assert(multicast_found);
//   return multicast_factor;
// }

// Convert multicasts into scatter->distributed-multicasts if certain conditions
// are met.
// FIXME: This entire logic breaks in the face of irregular multicast signatures
// that may be introduced by new features such as skew and flattening. We should
// re-implement distribution during the T-function computation (in
// nest-analysis) as opposed to a post-processing step on the Delta-function.
// The code below works for simple multicast signatures, but should be
// deprecated.
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

    if (tile_nest[outer].access_stats.stats.size() != 1 || tile_nest[inner].access_stats.stats.size() != 1)
    {
      std::cerr << "ERROR: complex multicast signature detected, and we cannot yet compute a distributed multicast pattern for this." << std::endl;
      std::exit(1);
    }
    
    auto outer_access_stat_ref = tile_nest[outer].access_stats.stats.begin();
    auto inner_access_stat_ref = tile_nest[inner].access_stats.stats.begin();

    uint64_t outer_multicast_factor = outer_access_stat_ref->first.first;
    uint64_t inner_multicast_factor = inner_access_stat_ref->first.first;

    uint64_t outer_scatter_factor = outer_access_stat_ref->first.second;
    uint64_t inner_scatter_factor = inner_access_stat_ref->first.second;

    AccessStats outer_stats = outer_access_stat_ref->second;
    AccessStats inner_stats = inner_access_stat_ref->second;

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
      uint64_t multicast_factor = 1;
      uint64_t scatter_factor = outer_multicast_factor * outer_scatter_factor; // fanout?

      tile_nest[outer].access_stats.stats[std::make_pair<>(multicast_factor, scatter_factor)] =
        { .accesses = outer_stats.accesses,
          .hops = outer_stats.hops }; // FIXME: recompute stats.hops for PRECISE_MULTICAST.

      tile_nest[outer].access_stats.stats.erase(outer_access_stat_ref);

      // The inner tile's per-instance size, partition size and content-access count
      // reduces by the outer multicast factor. Note that this may not be a perfect
      // factor, so do a ceil-div.
      tile_nest[inner].size = 1 + (tile_nest[inner].size - 1) / outer_multicast_factor;
      tile_nest[inner].partition_size = 1 +
        (tile_nest[inner].partition_size - 1) / outer_multicast_factor;
      tile_nest[inner].content_accesses = 1 +
        (tile_nest[inner].content_accesses - 1) / outer_multicast_factor;
      tile_nest[inner].parent_access_share = 1 +
        (tile_nest[inner].parent_access_share - 1) / outer_multicast_factor;

      // The inner tile's network accesses will now happen at a distributed-multicast
      // factor of outer_multicast_factor. These alterations will magically trigger
      // all the right computations at the model evaluation stage.
      uint64_t distributed_multicast_factor = outer_multicast_factor * inner_multicast_factor;
      tile_nest[inner].distributed_multicast = true;
      tile_nest[inner].distributed_fanout = distributed_multicast_factor * tile_nest[inner].fanout;

      ASSERT(distributed_multicast_factor > inner_multicast_factor);

      tile_nest[inner].access_stats.stats[std::make_pair<>(distributed_multicast_factor, inner_scatter_factor)] =
        { .accesses = (inner_stats.accesses - 1) / outer_multicast_factor,
          .hops = inner_stats.hops }; // FIXME: recompute stats.hops for PRECISE_MULTICAST.

      tile_nest[inner].access_stats.stats.erase(inner_access_stat_ref);

      // We should be doing this process hierarchically along the entire tile stack,
      // but for the moment just support doing this once.
      break;
    }
  }
}

// Compute parent access share.
void ComputeParentAccessShare(std::vector<DataMovementInfo>& tile_nest)
{
  int num_tiling_levels = tile_nest.size();

  for (int cur = 0; cur < num_tiling_levels; cur++)
  {
    // Skip if this tile level has 0 size or 0 accesses.
    if (tile_nest[cur].size == 0)
    {
      continue;
    }

    // Initialize parent_access_share to 0.
    tile_nest[cur].parent_access_share = 0;
   
    // Find next (outer) non-zero level.
    int outer;
    for (outer = cur + 1; outer < num_tiling_levels && tile_nest[outer].size == 0; outer++)
    {
      // Body is empty.
    }

    if (outer == num_tiling_levels)
    {
      if (gUpdatedRMW)
      {
        // No outer tiling level: parent access share is my partition size.
        tile_nest[cur].parent_access_share = tile_nest[cur].partition_size;
      }
      continue;
    }

    // std::cerr << "  cur = " << cur << std::endl;
    // std::cerr << "  outer = " << outer << std::endl;

    // Found an outer level.
    for (auto& x: tile_nest[outer].access_stats.stats)
    {
      // FIXME: is this correct in the face of spatial sliding windows (e.g. Input halos)?
      // If scatter factors are calculated on fragments, then this will be correct, because
      // the halos will be counted as "multicast" data. However, scatter factor calculation
      // via spatial deltas does not look at fragments of delivered temporal deltas, the
      // code compares complete temporal deltas delivered to peer spatial instances.
      // To fix this, we should be able to use the new overlap-fraction based method used to
      // calculate partition sizes in some way.
      auto multicast_factor = x.first.first;
      auto accesses = x.second.accesses;

      // We were using an older formula of parent_access_share = (accesses / scatter).
      // However, Link transfers and irregular sets result in fanout != (multicast * scatter)
      // so we use a new formula: parent_access_share = (accesses * multicast) / fanout.
      // This calculates an *average* number of parent_access_share per child instance (the
      // reality is that some child instances, such as edge instances, will receive more
      // parent_access_share).
      tile_nest[cur].parent_access_share += (accesses * multicast_factor) / tile_nest[outer].fanout;

      // Note: using a floating-point parent_access_share here fixes a rounding
      // in older code that was accumulating directly into an int field fills.
    }

    // assert(tile_nest[cur].parent_access_share <= tile_nest[cur].GetTotalAccesses());
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
    //std::cout << "level: " << cur << " pfd: " << tile_nest[cur].partition_fraction_denominator << " tile nest: " << tile_nest[cur] << std::endl;
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
      // We don't have to find the next-inner level, it's guaranteed to be
      // cur-1. If cur-1 were bypassed, our link_transfers would have been
      // reset to 0.
      assert(tile_nest[cur-1].size > 0);

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
void ComputeReadUpdateReductionAccesses_Legacy(std::vector<DataMovementInfo>& tile_nest, problem::Shape::DataSpaceID pv)
{
  // Loop through all levels and update reads, writes, updates.
  //
  int num_tiling_levels = tile_nest.size();

  for (int cur = 0; cur < num_tiling_levels; cur++)
  {
    if (tile_nest[cur].size == 0)
    {
      // This level was bypassed.
      tile_nest[cur].reads = 0;
      tile_nest[cur].updates = 0;
      tile_nest[cur].fills = 0;
      //tile.address_generations = tile.reads + tile.fills0; // scalar
      tile_nest[cur].temporal_reductions = 0;
      continue;
    }

    if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      // First epoch is an Update, all subsequent epochs are Read-Modify-Update.

      // The following assertion is *incorrect* for coefficients (e.g. stride, pad) > 1.
      // FIXME: find a safety check that works with coefficients > 1.
      // assert(tile[pvi].size == 0 || tile[pvi].content_accesses % tile[pvi].size == 0);

      assert((tile_nest[cur].content_accesses + tile_nest[cur].peer_accesses) >= tile_nest[cur].partition_size);

      // FIXME: temporal reduction and network costs if hardware reduction isn't
      // supported appears to be wonky - network costs may need to trickle down
      // all the way to the level that has the reduction hardware.
      tile_nest[cur].updates = std::round(tile_nest[cur].content_accesses);
      if (gEnableFirstReadElision)
      {
        tile_nest[cur].reads = std::round(tile_nest[cur].content_accesses + tile_nest[cur].peer_accesses - tile_nest[cur].partition_size);
        tile_nest[cur].temporal_reductions = std::round(tile_nest[cur].content_accesses + tile_nest[cur].peer_accesses - tile_nest[cur].partition_size);
        tile_nest[cur].fills = std::round(tile_nest[cur].parent_access_share + tile_nest[cur].peer_fills - tile_nest[cur].partition_size);
      }
      else
      {
        tile_nest[cur].reads = std::round(tile_nest[cur].content_accesses + tile_nest[cur].peer_accesses);
        tile_nest[cur].temporal_reductions = std::round(tile_nest[cur].content_accesses + tile_nest[cur].peer_accesses);
        tile_nest[cur].fills = std::round(tile_nest[cur].parent_access_share + tile_nest[cur].peer_fills);
      }

      //tile.address_generations[pv] = stats_.updates[pv] + stats_.fills[pv]; // scalar
    }
    else // Read-only data type.
    {
      tile_nest[cur].reads = std::round(tile_nest[cur].content_accesses + tile_nest[cur].peer_accesses);
      tile_nest[cur].updates = 0;
      tile_nest[cur].fills = tile_nest[cur].parent_access_share + tile_nest[cur].peer_fills;
      //tile.address_generations = tile.reads + tile.fills; // scalar
      tile_nest[cur].temporal_reductions = 0;
    }
  }

  return;
}

// Split the accesses to read and update and generate reduction.
void ComputeReadUpdateReductionAccesses_UpdatedRMW(std::vector<DataMovementInfo>& tile_nest, problem::Shape::DataSpaceID pv)
{
  // Loop through all levels and update reads, writes, updates.
  //
  int num_tiling_levels = tile_nest.size();

  // UGGGGGHHHH.... this entire code is hard-coded to assume that the outermost
  // level does not support hardware reduction and all other levels support
  // hardware reduction. We do have the makings of a generalized algorithm to
  // propagate reductions into inner levels but it is not finalized yet.
  bool outermost_found = false;
  bool second_outer_found = false;

  for (int cur = num_tiling_levels-1; cur >= 0; cur--)
  {
    if (tile_nest[cur].size == 0)
    {
      // This level was bypassed.
      tile_nest[cur].reads = 0;
      tile_nest[cur].updates = 0;
      tile_nest[cur].fills = 0;
      //tile.address_generations = tile.reads + tile.fills0; // scalar
      tile_nest[cur].temporal_reductions = 0;
      continue;
    }

// #define BYPASS_LAST_UPDATE

    if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      // Start with the general case: hardware reduction supported.
      tile_nest[cur].fills = 0;
      tile_nest[cur].reads = std::round(tile_nest[cur].content_accesses - tile_nest[cur].parent_access_share);
      tile_nest[cur].temporal_reductions = tile_nest[cur].reads;
      tile_nest[cur].updates = std::round(tile_nest[cur].content_accesses);
#ifdef BYPASS_LAST_UPDATE
      tile_nest[cur].updates -= tile_nest[cur].parent_access_share;
#endif

      // std::cout << "--------------------------------\n";
      // std::cout << "Level " << cur << std::endl;
      // std::cout << "  reads = trs = updates = " << tile_nest[cur].reads << std::endl;

      if (cur == num_tiling_levels-1)
      {
        // No hardware reduction support.
        tile_nest[cur].temporal_reductions = 0;
        // std::cout << "  DRAM level, setting trs = 0\n";
      }

      if (!outermost_found)
      {
#ifdef BYPASS_LAST_UPDATE
        tile_nest[cur].updates += tile_nest[cur].parent_access_share;
#endif
        // std::cout << "  outermost level, updates = " << tile_nest[cur].updates << std::endl;
      }

      if (outermost_found && !second_outer_found)
      {
        double tax = 0;
        // Second-outer: perform reductions on behalf of outermost level.
        if(gEnableFirstReadElision)
          tax = tile_nest[cur].parent_access_share - tile_nest[cur].partition_size;
        else
          tax = tile_nest[cur].parent_access_share;

        tile_nest[cur].fills += tax;
        tile_nest[cur].reads += tax;
        tile_nest[cur].temporal_reductions += tax;
        second_outer_found = true;

        // std::cout << "  GBuf, adding parent tax = " << tax << std::endl;
        // std::cout << "  updated fills = " << tile_nest[cur].fills << std::endl;
        // std::cout << "  updated reads = " << tile_nest[cur].reads << std::endl;
        // std::cout << "  updated trs = " << tile_nest[cur].temporal_reductions << std::endl;
      }

      // Accumulate peer accesses.
      if (tile_nest[cur].peer_accesses > 0)
      {
        ASSERT(tile_nest[cur].peer_accesses >= tile_nest[cur].partition_size);
        tile_nest[cur].reads += tile_nest[cur].peer_accesses - tile_nest[cur].partition_size;
        tile_nest[cur].temporal_reductions += tile_nest[cur].peer_accesses - tile_nest[cur].partition_size;
        tile_nest[cur].updates += tile_nest[cur].peer_accesses;
#ifdef BYPASS_LAST_UPDATE
        tile_nest[cur].updates -= tile_nest[cur].partition_size;
#endif
      }

      // std::cout << "  +peer reads, updated = " << tile_nest[cur].reads << std::endl;
      // std::cout << "  +peer trs, updated = " << tile_nest[cur].temporal_reductions << std::endl;
      // std::cout << "  +peer updates, updated = " << tile_nest[cur].updates << std::endl;
    }
    else // Read-only data type.
    {
      tile_nest[cur].reads = std::round(tile_nest[cur].content_accesses + tile_nest[cur].peer_accesses);
      tile_nest[cur].updates = 0;
      if (!outermost_found)
        tile_nest[cur].fills = tile_nest[cur].peer_fills;
      else
        tile_nest[cur].fills = tile_nest[cur].parent_access_share + tile_nest[cur].peer_fills;
      //tile.address_generations = tile.reads + tile.fills; // scalar
      tile_nest[cur].temporal_reductions = 0;
    }

    if (!outermost_found)
    {
      outermost_found = true;
    }
  }

  return;
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
                                       problem::Workload* workload)
{
  CompoundDataMovementNest collapsed_compound_data_nest = CollapseDataMovementNest(tiles.compound_data_movement_info_nest,
                                                                                   num_tiling_levels,
                                                                                   tile_mask,
                                                                                   distribution_supported, 
                                                                                   workload);
  ComputeNest collapsed_compound_compute_nest = CollapseComputeNest(tiles.compound_compute_info_nest, num_tiling_levels); 
  tiling::CompoundTileNest solution;
  solution.compound_data_movement_info_nest = collapsed_compound_data_nest;
  solution.compute_info_nest = collapsed_compound_compute_nest;

  // -- FIXME -- we don't need both compute_cycles and max_temporal iterations.
  // The latter is a temporary hack for Ruby. The former is needed for sparse
  // analysis. We need to merge these.
  solution.compute_info_nest[0].compute_cycles = solution.compute_info_nest[0].accesses;
  return solution;
}


ComputeNest CollapseComputeNest(analysis::CompoundComputeNest& tiles, int num_tiling_levels)
{
  ComputeNest solution;
  
  for (int level=0; level < num_tiling_levels; level++)
  {  
    ComputeInfo collapsed_tile;
    if (level == 0)
    {
      // compute info is only valid for the inner most level
      collapsed_tile.replication_factor = tiles[0].replication_factor;
      collapsed_tile.accesses = tiles[0].accesses;
      // -- FIXME -- we don't need both compute_cycles and max_temporal iterations.
      // The latter is a temporary hack for Ruby. The former is needed for sparse
      // analysis. We need to merge these.
      collapsed_tile.max_temporal_iterations = tiles[0].max_temporal_iterations;
    }
    else
    {
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
  (void) workload;
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
      collapsed_tile.access_stats = tiles[pv][innermost_loop].access_stats;
      collapsed_tile.content_accesses = tiles[pv][innermost_loop].access_stats.TotalAccesses();
      collapsed_tile.link_transfers = tiles[pv][innermost_loop].link_transfers;
      collapsed_tile.peer_accesses = 0;
      collapsed_tile.peer_fills = 0;
      collapsed_tile.replication_factor = tiles[pv][outermost_loop].replication_factor;
      collapsed_tile.fanout = tiles[pv][innermost_loop].fanout;
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

        // if (pv == 0)
        // {
        //   std::cout << "cur_tiling_level = " << cur_tiling_level << std::endl;
        //   std::cout << "inner_tile level = " << solution[pv].size()-1 << std::endl;
        //   std::cout << "innermost loop is_master_spatial = " << tiles[pv][innermost_loop].is_master_spatial << std::endl;
        //   std::cout << "innermost loop size = " << tiles[pv][innermost_loop].size << std::endl;
        //   std::cout << "inner_tile size = " << inner_tile.size << std::endl;
        //   std::cout << std::endl;
        // }
      }

      solution[pv].push_back(collapsed_tile);

      processed_loop_count = boundary_loop_id + 1;
      cur_tiling_level++;
    }
    assert(cur_tiling_level == num_tiling_levels);

    // Compute partition sizes.
    ComputePartitionSizes(solution[pv]);

    // Calculate share of parent accesses (used to compute fills).
    ComputeParentAccessShare(solution[pv]);

    // Mask each solution according to the provided bit mask.
    MaskTiles(solution[pv], tile_mask[pv]);

    // Additional step for outermost masked levels.
    ProcessOuterMaskedLevels(solution[pv], tile_mask[pv]);

    // Set backing storage fill to zero place holder
    ResetBackingStorageFillsPlaceHolder(solution[pv]);

    // Perform distributed-multicast if supported.
    DistributeTiles(solution[pv], distribution_supported[pv]);

    // Calculate the extra accesses and fills due to link transfers
    ComputePeerAccesses(solution[pv]);

    // Split the accesses to read and update and generate reduction.
    if (gUpdatedRMW)
      ComputeReadUpdateReductionAccesses_UpdatedRMW(solution[pv], pv);
    else
      ComputeReadUpdateReductionAccesses_Legacy(solution[pv], pv);
    
    // Find the parent and child levels for later compression/decompression logic
    SetParentLevel(solution[pv]);
    SetChildLevel(solution[pv]);

  }

  // flip the workload tensor set flag if necessary
  // if (! workload->IsWorkloadTensorSizesSet()) {workload->AllTensorsSet();}

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

// check if any fo the dataspace is not stored anywhere
bool CheckMaskValidity(const CompoundMaskNest& masks)
{
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    if (masks[pv].none()) return false; 
  }
  return true;
}

}  // namespace tiling
