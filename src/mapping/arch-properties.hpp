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

#pragma once

#include "model/engine.hpp"

//
// Derived Architecture properties that are relevant to the mapping.
// 

class ArchProperties
{
 private:
  model::Engine::Specs specs_;
  
  uint64_t num_temporal_tiling_levels_;
  uint64_t num_spatial_tiling_levels_;
  uint64_t num_total_tiling_levels_;  // temporal + spatial
  
  std::vector<bool> spatial_mask_;       // across all levels
  std::vector<bool> twoD_spatial_mask_;  // across all levels

  // Maps to index between the different tiling spaces.
  std::map<unsigned, unsigned> temporal_to_tiling_map_;
  std::map<unsigned, unsigned> spatial_to_tiling_map_;
  std::map<unsigned, unsigned> tiling_to_storage_map_;

 public:
  ArchProperties() { }
  
  void Construct(model::Engine::Specs arch_specs)
  {
    specs_ = arch_specs;
    
    auto num_storage_levels = specs_.topology.NumStorageLevels();
    
    // one temporal partition for each storage level
    num_temporal_tiling_levels_ = num_storage_levels;

    uint64_t cur_tiling_level = 0;
    for (uint64_t i = 0; i < num_storage_levels; i++)
    {
      // Peek at the fanout in the arch specs to figure out if this is a
      // purely temporal level, a 1D spatial level or a 2D spatial level.

      // For partitioned levels, we have to look at all partitions. If
      // any of the partitions have a spatial fanout, then we treat
      // this as a spatial level.
      bool is_spatial = false;
      bool is_spatial_2D = false;

      auto& specs = *specs_.topology.GetStorageLevel(i);
      auto lambda = [&] (problem::Shape::DataSpaceID pv)
        {
          if (specs.Fanout(pv).Get() > 1)
            is_spatial = true;
          if (specs.FanoutX(pv).Get() > 1 && specs.FanoutY(pv).Get() > 1)
            is_spatial_2D = true;
        };
      model::BufferLevel::ForEachDataSpaceID(lambda, specs.sharing_type);

      if (is_spatial)
      {
        // This is a spatial level.
        spatial_mask_.push_back(true);
        twoD_spatial_mask_.push_back(is_spatial_2D);
        spatial_to_tiling_map_[i] = cur_tiling_level;
        tiling_to_storage_map_[cur_tiling_level] = i;
        cur_tiling_level++;
      }
      
      // There is always a temporal level
      spatial_mask_.push_back(false);
      twoD_spatial_mask_.push_back(false);

      temporal_to_tiling_map_[i] = cur_tiling_level;
      tiling_to_storage_map_[cur_tiling_level] = i;
      cur_tiling_level++;      
    }

    num_total_tiling_levels_ = spatial_mask_.size();
    assert(twoD_spatial_mask_.size() == num_total_tiling_levels_);    
  }

  //
  // Accessors.
  //
  
  unsigned& TemporalToTiling(const unsigned l)
  {
    return temporal_to_tiling_map_.at(l);
  }

  unsigned& SpatialToTiling(const unsigned l)
  {
    return spatial_to_tiling_map_.at(l);
  }

  unsigned& TilingToStorage(const unsigned l)
  {
    return tiling_to_storage_map_.at(l);
  }
  
  unsigned TilingLevels() const
  {
    return num_total_tiling_levels_;
  }

  unsigned StorageLevels() const
  {
    return specs_.topology.NumStorageLevels();
  }

  model::Engine::Specs& Specs()
  {
    return specs_;
  }

  bool IsSpatial(int level)
  {
    return spatial_mask_[level];
  }  
};
