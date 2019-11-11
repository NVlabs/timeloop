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

  // Storage level to fanout map.
  std::map<unsigned, std::uint64_t> fanoutX_map_; 
  std::map<unsigned, std::uint64_t> fanoutY_map_; 

 public:
  ArchProperties()
  { }

  ArchProperties(const model::Engine::Specs& arch_specs)
  {
    Construct(arch_specs);
  }

  void DeriveFanouts()
  {
    // Assumption here is that level i always connects to level
    // i-1 via a 1:1 or fanout network. The network module will
    // eventually be factored out, at which point we can make the
    // interconnection more generic and specifiable.

    // NOTE! The following code only works for Shared topologies.
    // We will be getting rid of Partitioned topologies.

    for (unsigned i = 0; i < specs_.topology.NumStorageLevels(); i++)
    {
      std::uint64_t inner_meshX, inner_meshY;
      std::uint64_t outer_meshX, outer_meshY;

      if (i == 0)
      {
        inner_meshX = specs_.topology.GetArithmeticLevel()->MeshX().Get();
        inner_meshY = specs_.topology.GetArithmeticLevel()->MeshY().Get();
      }
      else
      {
        inner_meshX = specs_.topology.GetStorageLevel(i-1)->MeshX().Get();
        inner_meshY = specs_.topology.GetStorageLevel(i-1)->MeshY().Get();        
      }

      outer_meshX = specs_.topology.GetStorageLevel(i)->MeshX().Get();
      outer_meshY = specs_.topology.GetStorageLevel(i)->MeshY().Get();        

      if ((inner_meshX % outer_meshX) != 0)
      {
        if (i == 0)
          std::cerr << "inner MACC meshX = " << inner_meshX << std::endl;
        else
          std::cerr << "inner " << StorageLevelName(i-1) << " meshX = " << inner_meshX << std::endl;
        std::cerr << "outer " << StorageLevelName(i) << " meshX = " << outer_meshX << std::endl;
      }

      if ((inner_meshY % outer_meshY) != 0)
      {
        if (i == 0)
          std::cerr << "inner MACC meshY = " << inner_meshY << std::endl;
        else
          std::cerr << "inner " << StorageLevelName(i-1) << " meshY = " << inner_meshY << std::endl;
        std::cerr << "outer " << StorageLevelName(i) << " meshY = " << outer_meshY << std::endl;
      }

      assert(inner_meshX % outer_meshX == 0);
      assert(inner_meshY % outer_meshY == 0);

      fanoutX_map_[i] = inner_meshX / outer_meshX;
      fanoutY_map_[i] = inner_meshY / outer_meshY;
    }
  }

  void Construct(const model::Engine::Specs& arch_specs)
  {
    specs_ = arch_specs;

    // Derive fanouts.
    DeriveFanouts();
    
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
      bool is_spatial = (Fanout(i) > 1);
      bool is_spatial_2D = (FanoutX(i) > 1 && FanoutY(i) > 1);

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
  
  std::uint64_t FanoutX(unsigned storage_level_id)
  {
    return fanoutX_map_.at(storage_level_id);
  }
  
  std::uint64_t FanoutY(unsigned storage_level_id)
  {
    return fanoutY_map_.at(storage_level_id);
  }
  
  std::uint64_t Fanout(unsigned storage_level_id)
  {
    return fanoutX_map_.at(storage_level_id) * fanoutY_map_.at(storage_level_id);
  }
  
  const unsigned& TemporalToTiling(const unsigned l) const
  {
    return temporal_to_tiling_map_.at(l);
  }

  const unsigned& SpatialToTiling(const unsigned l) const
  {
    return spatial_to_tiling_map_.at(l);
  }

  const unsigned& TilingToStorage(const unsigned l) const
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

  bool IsSpatial(int level) const
  {
    return spatial_mask_.at(level);
  }

  bool IsSpatial2D(int level) const
  {
    return twoD_spatial_mask_.at(level);
  }

  //
  // Helpers.
  //
  std::string StorageLevelName(unsigned l) const
  {
    return specs_.topology.GetStorageLevel(l)->level_name;
  }

  std::string TilingLevelName(unsigned l) const
  {
    std::string retval;
    auto& storage_level = TilingToStorage(l);
    retval = specs_.topology.GetStorageLevel(storage_level)->level_name;
    retval += IsSpatial(l) ? " (spatial)" : " (temporal)";
    return retval;
  }
};
