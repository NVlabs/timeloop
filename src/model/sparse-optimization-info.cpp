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

#include "model/sparse-optimization-info.hpp"

namespace sparse
{

bool PerDataSpaceCompressionInfo::HasMetaData() const
{
  if (tensor_compressed) return true; // compressed tensor must have metadata
  if (metadata_models.size() > 1) return true; // intermediate levels must have payloads
  if (metadata_models.size() == 1 && !metadata_models[0]->MetaDataImplicitAsLowestRank())
    return true; // single level with metadata
  return false; // single level default dense
}

bool PerDataSpaceCompressionInfo::ExistFlatteningRule(std::uint64_t rank_id) const
{
  // determine if there is any flattening rule for a specific rank
  return flattened_rankIDs[rank_id].size() > 0 ? true :false;
}

problem::Shape::FlattenedDimensionID PerDataSpaceCompressionInfo::GetFlatteningRule(std::uint64_t rank_id, std::uint64_t rule_idx) const
{

  assert(flattened_rankIDs[rank_id].size() > rule_idx);
  return flattened_rankIDs.at(rank_id).at(rule_idx).at(rule_idx);
}

bool PerDataSpaceCompressionInfo::FoundDimensionInFlatteningRule(std::uint64_t rank_id, problem::Shape::FlattenedDimensionID dim_id,
                                                                 std::vector<problem::Shape::FlattenedDimensionID> &rule_item) const
{
  for (auto iter = flattened_rankIDs[rank_id].begin(); iter != flattened_rankIDs[rank_id].end(); iter++)
  {
    if (std::find(iter->begin(), iter->end(), dim_id) != iter->end())
    {
      rule_item = *iter;
      return true;
    }
  }
  return false;
}

bool CompressionInfo::GetDataSpaceCompressionInfo(unsigned level, unsigned pv, PerDataSpaceCompressionInfo &info)
{
  if (has_metadata_masks[level][pv])
  {
    info = per_level_info_map[level][pv];
    return true;
  }
  return false;
}

bool CompressionInfo::GetStorageLevelCompressionInfo(unsigned level, PerStorageLevelCompressionInfo &info)
{
  if (per_level_info_map.find(level) != per_level_info_map.end())
  {
    info = per_level_info_map[level];
    return true;
  }
  return false;
}

bool CompressionInfo::GetDataSpaceCompressed(unsigned level, unsigned pv)
{
  return compressed_masks[level][pv];
}

} // namespace
