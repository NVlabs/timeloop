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
  ArchProperties();
  ArchProperties(const model::Engine::Specs& arch_specs);

  void DeriveFanouts();

  void Construct(const model::Engine::Specs& arch_specs);

  //
  // Accessors.
  //
  
  std::uint64_t FanoutX(unsigned storage_level_id);
  std::uint64_t FanoutY(unsigned storage_level_id);
  std::uint64_t Fanout(unsigned storage_level_id);
  const std::map<unsigned, std::uint64_t>& FanoutX() const;
  const std::map<unsigned, std::uint64_t>& FanoutY() const;
  
  const unsigned& TemporalToTiling(const unsigned l) const;
  const unsigned& SpatialToTiling(const unsigned l) const;
  const unsigned& TilingToStorage(const unsigned l) const;
  
  unsigned TilingLevels() const;
  unsigned StorageLevels() const;

  model::Engine::Specs& Specs();

  bool IsSpatial(int level) const;
  bool IsSpatial2D(int level) const;

  //
  // Helpers.
  //
  std::string StorageLevelName(unsigned l) const;
  std::string TilingLevelName(unsigned l) const;
};
