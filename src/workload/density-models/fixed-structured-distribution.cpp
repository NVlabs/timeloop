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

#include "workload/density-models/fixed-structured-distribution.hpp"

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

BOOST_CLASS_EXPORT(problem::FixedStructuredDistribution)

namespace problem{

FixedStructuredDistribution::FixedStructuredDistribution(){ }

FixedStructuredDistribution::FixedStructuredDistribution(const Specs& specs) : specs_(specs){
  is_specced_ = true;
}

FixedStructuredDistribution::~FixedStructuredDistribution() { }

FixedStructuredDistribution::Specs FixedStructuredDistribution::ParseSpecs(config::CompoundConfigNode density_config){

  Specs specs;
  std::string type;
  double fixed_density;

  density_config.lookupValue("distribution", type);
  if( !density_config.lookupValue("density", fixed_density))
  { 
    std::cerr << "ERROR: missing density value specification for " << type << " distribution" << std::endl;
    exit(1);
  }
  
  specs.type = "fixed-structured";
  specs.fixed_density = fixed_density;

  return specs;
}

void FixedStructuredDistribution::SetWorkloadTensorSize(const std::uint64_t size ){
  // setter that allows workload tensor size at a latter stage (topology.cpp, PreEvaluationCheck)
  specs_.workload_tensor_size = size;
}

std::uint64_t FixedStructuredDistribution::GetTileOccupancyByConfidence(const std::uint64_t tile_shape,
                                                              const double confidence) const
{

  double exact_nnzs = tile_shape * specs_.fixed_density;
  // now we need to decide how to assign the prob of ceil and floor so that the expected nnzs == exact_nnzs

  // assume prob of ceil is pc, then the prob of floor is pf = 1 - pc
  //  => pc*ceil + (1-pc)*floor = exact_nnzs
  //  => pc = exact_nnzs-floor

  double prob_ceil = exact_nnzs - floor(exact_nnzs);
  double prob_floor = 1 - prob_ceil;

  return confidence <= prob_floor ? floor(exact_nnzs) : ceil(exact_nnzs);
}

std::uint64_t FixedStructuredDistribution::GetMaxTileOccupancyByConfidence(const tiling::CoordinateSpaceTileInfo& tile,
                                                                 const double confidence) const
{
  std::uint64_t tile_shape = tile.GetShape();
  return FixedStructuredDistribution::GetTileOccupancyByConfidence(tile_shape, confidence);
}

// place holder lightweight version
std::uint64_t FixedStructuredDistribution::GetMaxTileOccupancyByConfidence_LTW (const std::uint64_t tile_shape,
                                                   const double confidence) const
{
  return FixedStructuredDistribution::GetTileOccupancyByConfidence(tile_shape, confidence);
}

std::uint64_t FixedStructuredDistribution::GetWorkloadTensorSize() const{
  return specs_.workload_tensor_size;
};

std::string FixedStructuredDistribution::GetDistributionType() const{
  return specs_.type;
}


double FixedStructuredDistribution::GetMaxTileDensityByConfidence(const tiling::CoordinateSpaceTileInfo tile,
                                                                  const double confidence) const
{
  (void) confidence;

  if (tile.GetShape() == 0) return 0;
  return specs_.fixed_density;
}


double FixedStructuredDistribution::GetMinTileDensity(const tiling::CoordinateSpaceTileInfo tile) const
{
  if (tile.GetShape() == 0)
  {
    return 0;
  }
  std::uint64_t floor_nnzs =  floor(tile.GetShape() * specs_.fixed_density);
  return floor_nnzs/tile.GetShape();
}

double FixedStructuredDistribution::GetTileOccupancyProbability(const tiling::CoordinateSpaceTileInfo& tile,
                                                                const std::uint64_t occupancy) const
{
  

  assert(is_specced_);
  double prob, exact_nnzs;
  std::uint64_t tile_shape = tile.GetShape();

  if (tile.HasExtraConstraintInfo())
  {
    auto extra_constraint_info = tile.GetExtraConstraintInfo();
    double constrained_density = (double)extra_constraint_info.GetOccupancy()/extra_constraint_info.GetShape();
    exact_nnzs = tile_shape * constrained_density;
  }
  else
  {
    exact_nnzs = tile_shape * specs_.fixed_density;
  }

   // now we need to decide how to assign the prob of ceil and floor so that the expected nnzs == exact_nnzs

  // assume prob of ceil is pc, then the prob of floor is pf = 1 - pc
  //  => pc*ceil + (1-pc)*floor = exact_nnzs
  //  => pc = exact_nnzs-floor

  double prob_ceil = exact_nnzs - floor(exact_nnzs);
  double prob_floor = 1 - prob_ceil;

  if (floor(exact_nnzs) == occupancy)
  {
    return prob_floor;

  } else if (ceil(exact_nnzs) == occupancy)
  {
    return prob_ceil;

  } else
  {
    return 0.0;
  }
 
  
  return prob;
}


double FixedStructuredDistribution::GetExpectedTileOccupancy (const tiling::CoordinateSpaceTileInfo tile) const
{
  return tile.GetShape() * specs_.fixed_density;
}

}
