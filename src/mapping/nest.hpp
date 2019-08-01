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

#pragma once

#include <list>
#include <vector>

#include "loop-analysis/loop-state.hpp"
#include "loop-analysis/tiling.hpp"
#include "workload/problem-shape.hpp"

namespace loop {

// ----------
// NestConfig
// ----------

typedef std::vector<std::vector<Descriptor>> NestConfig;

std::ostream& operator << (std::ostream& out, const NestConfig& nest);


// ---------
// Loop nest
// ---------

class Nest
{
 public:
  // Nest structure.
  std::vector<Descriptor> loops;
  std::vector<uint64_t> storage_tiling_boundaries;

 public:
  Nest();

  bool operator == (const Nest& n) const; 

  void AddLoop(Descriptor descriptor);
  void AddLoop(problem::Shape::DimensionID dimension, int start, int end, int stride,
               spacetime::Dimension spacetime_dimension);
  bool AddStorageTilingBoundary();

  friend std::ostream& operator << (std::ostream& out, const Nest& nest);

  void PrettyPrint(std::ostream& out, const std::vector<std::string>& storage_level_names,
                   const tiling::NestOfCompoundMasks& mask_nest,
                   const std::vector<problem::PerDataSpace<std::uint64_t>>& tile_sizes);

  void PrintWhoopNest(std::ostream& out, const std::vector<std::string>& storage_level_names,
                      const tiling::NestOfCompoundMasks& mask_nest,
                      const std::vector<problem::PerDataSpace<std::uint64_t>>& tile_sizes);
};

} // namespace loop
