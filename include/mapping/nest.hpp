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

#include <list>
#include <vector>
#include <unordered_map>

#include "loop-analysis/loop-state.hpp"
#include "loop-analysis/tiling.hpp"
#include "workload/shape-models/problem-shape.hpp"

namespace loop
{

// ----------
// NestConfig
// ----------

typedef std::vector<std::vector<Descriptor>> NestConfig;

//std::ostream& operator << (std::ostream& out, const NestConfig& nest);


// ---------
// Loop nest
// ---------

/**
 * @brief A nest of loops.
 * 
 * Loops are organized in order of inner to outer.
 * 
 * Also holds storage boundaries (i.e., indices of loops just under storage
 * levels), skew descriptors, link transfer, multicast, and temporal reuse
 * flags.
 */
class Nest
{
 public:
  // Skew specs.
  struct SkewDescriptor
  {
    struct Term
    {
      struct DimSpec
      {
        problem::Shape::FlattenedDimensionID dimension = problem::GetShape()->NumFlattenedDimensions;
        bool is_spatial;
      };
      // Each skew term can have a constant, a loop bound, and a loop variable
      int constant = 1;
      DimSpec variable;
      DimSpec bound;
    };
    std::vector<Term> terms;
    int modulo;
  };

  // Nest structure.

  /**
   * @brief Loops in order or inner to outer.
   */
  std::vector<Descriptor> loops;
  /**
   * @brief Indices of loops just below storage levels.
   */
  std::vector<uint64_t> storage_tiling_boundaries;
  std::unordered_map<unsigned, SkewDescriptor> skew_descriptors;
  /**
   * @brief A mapping from loop index to link transfer flags per data space.
   */
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> no_link_transfer;
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> no_multicast;
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> no_temporal_reuse;
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> rmw_first_update;
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> no_coalesce;

  problem::Shape problem_shape;

 public:
  Nest() = default;
  Nest(const problem::Shape& shape);

  bool operator == (const Nest& n) const; 

  void AddLoop(Descriptor descriptor);
  void AddLoop(problem::Shape::FlattenedDimensionID dimension, int start, int end, int stride,
               spacetime::Dimension spacetime_dimension, int residual_end = 0);
  bool AddStorageTilingBoundary();

  friend std::ostream& operator << (std::ostream& out, const Nest& nest);

  void PrettyPrint(std::ostream& out, const std::vector<std::string>& storage_level_names,
                   const tiling::NestOfCompoundMasks& mask_nest,
                   const std::vector<problem::PerDataSpace<std::uint64_t>>& utilized_capacities,
                   const std::vector<problem::PerDataSpace<std::uint64_t>>& tile_sizes,
                   const std::string _indent = "");

  void PrintWhoopNest(std::ostream& out, const std::vector<std::string>& storage_level_names,
                      const tiling::NestOfCompoundMasks& mask_nest,
                      const std::vector<problem::PerDataSpace<std::uint64_t>>& tile_sizes,
                      const std::vector<problem::PerDataSpace<std::uint64_t>>& utilized_instances);

  std::string PrintCompact(const tiling::NestOfCompoundMasks& mask_nest);

  void PrintTenssella(std::ostream& out, const tiling::NestOfCompoundMasks& mask_nest);

};

} // namespace loop
