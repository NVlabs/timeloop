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

#include <bitset>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/math/distributions/hypergeometric.hpp>
#include <iostream>
#include <exception>
#include <boost/math/special_functions/binomial.hpp>

#include "workload/density-models/hypergeometric-distribution.hpp"

BOOST_CLASS_EXPORT(problem::HypergeometricDistribution)

namespace problem
{

HypergeometricDistribution::HypergeometricDistribution()
{}

HypergeometricDistribution::HypergeometricDistribution(const Specs& specs)
    : specs_(specs)
{
  is_specced_ = true;
  if (specs.workload_tensor_size != 0)
  {
    workload_tensor_size_set_ = true;
  }
}

HypergeometricDistribution::~HypergeometricDistribution()
{}

HypergeometricDistribution::Specs HypergeometricDistribution::ParseSpecs(config::CompoundConfigNode density_config)
{

  Specs specs;
  assert(density_config.lookupValue("distribution", specs.type));
  assert(density_config.lookupValue("density", specs.average_density));

  long long workload_tensor_size;
  if (density_config.lookupValue("workload_tensor_size", workload_tensor_size))
  {
    specs.total_nnzs = ceil(specs.average_density * workload_tensor_size);
    specs.workload_tensor_size = workload_tensor_size;

  } else
  {
    specs.total_nnzs = 0;
    specs.workload_tensor_size = 0;
  }

  return specs;
}

void HypergeometricDistribution::SetWorkloadTensorSize(const problem::DataSpace& point_set)
{
  assert(is_specced_);
  specs_.workload_tensor_size = point_set.size();
  specs_.total_nnzs = ceil(specs_.average_density * specs_.workload_tensor_size);
  workload_tensor_size_set_ = true;
}

std::uint64_t HypergeometricDistribution::GetTileOccupancyByConfidence(const std::uint64_t tile_shape,
                                                                       const double confidence) const
{

  std::uint64_t tile_occupancy;

  if (confidence == 1.0)
  {
    if (tile_shape >= specs_.total_nnzs)
    {
      tile_occupancy = specs_.total_nnzs;
    } else
    {
      tile_occupancy = tile_shape;
    }
  } else
  {
    std::uint64_t r = specs_.total_nnzs;
    std::uint64_t n = tile_shape;
    std::uint64_t N = specs_.workload_tensor_size;
    boost::math::hypergeometric_distribution<double> distribution(r, n, N);
    tile_occupancy = quantile(distribution, confidence);
  }

  return tile_occupancy;
}

std::uint64_t HypergeometricDistribution::GetMaxTileOccupancyByConfidence(const tiling::CoordinateSpaceTileInfo& tile,
                                                                          const double confidence) const
{
  std::uint64_t tile_shape = tile.GetShape();
  return HypergeometricDistribution::GetTileOccupancyByConfidence(tile_shape, confidence);
}

std::uint64_t HypergeometricDistribution::GetMaxTileOccupancyByConfidence_LTW(const std::uint64_t tile_shape,
                                                                              const double confidence) const
{
  return HypergeometricDistribution::GetTileOccupancyByConfidence(tile_shape, confidence);
}

std::uint64_t HypergeometricDistribution::GetWorkloadTensorSize() const
{
  assert(workload_tensor_size_set_);
  return specs_.workload_tensor_size;
};

std::string HypergeometricDistribution::GetDistributionType() const
{
  return specs_.type;
}

double HypergeometricDistribution::GetMaxTileDensityByConfidence(const tiling::CoordinateSpaceTileInfo tile,
                                                                 const double confidence) const
{

  if (confidence == 0.5)
    { return specs_.average_density; } //shortcut for faster calculations

  std::uint64_t tile_shape = tile.GetShape();
  std::uint64_t percentile_occupancy = HypergeometricDistribution::GetTileOccupancyByConfidence(tile_shape, confidence);
  return (double)percentile_occupancy / tile_shape;

}

double HypergeometricDistribution::GetMinTileDensity(const tiling::CoordinateSpaceTileInfo tile) const
{
  if (tile.GetShape() <= specs_.workload_tensor_size - specs_.total_nnzs)
  {
    return 0;
  }
  else
  {
    return (tile.GetShape() - specs_.workload_tensor_size + specs_.total_nnzs)/tile.GetShape();
  }
}

double HypergeometricDistribution::GetTileExpectedDensity(const uint64_t tile_shape) const
{

  (void)tile_shape;
  assert(is_specced_);
  return specs_.average_density;
}

// ------------------------------------------------------------------------
// Calculating the probability directly, 
// based on this mathematical definition: 
// https://www.statisticshowto.com/hypergeometric-distribution-examples/ 
// ------------------------------------------------------------------------
double HypergeometricDistribution::CalculateProbability(const std::uint64_t nnz_vals,
                                                        const std::uint64_t r,
                                                        const std::uint64_t n,
                                                        const std::uint64_t N
  ) const 
{

  double prob;
  try 
  {
    long double r_choose_x = boost::math::binomial_coefficient<long double>(r, nnz_vals);
    long double Nr_choose_nx = boost::math::binomial_coefficient<long double>(N-r, n-nnz_vals);
    long double N_choose_n = boost::math::binomial_coefficient<long double>(N, n);
    prob =  (double) (r_choose_x*Nr_choose_nx)/N_choose_n;
  } catch (std::exception& e) 
  {
    std::cout << "ERROR: Integer overflow occured. Check your workload sizes and density." 
              << std::endl;
    exit(-1);
  }

  return prob;
}

double HypergeometricDistribution::GetProbability(const std::uint64_t tile_shape,
                                                  const std::uint64_t nnz_vals) const
{

  assert(is_specced_);
  assert(workload_tensor_size_set_);

  std::uint64_t r = specs_.total_nnzs;
  std::uint64_t n = tile_shape;
  std::uint64_t N = specs_.workload_tensor_size;

  if (((n + r > N) && (nnz_vals < n + r - N)) | (nnz_vals > r))
    { return 0; }

  boost::math::hypergeometric_distribution<double> distribution(r, n, N);
  
  //Workaround/fix for domain error for large workload size. Compute the PDF directly
  //This might return an overflow error, but it occurs less times than using the boost pdf call  
  double prob;
  try 
  {
    prob = pdf(distribution, nnz_vals);
  } catch (std::exception& e) 
  {
    prob = CalculateProbability(nnz_vals, r, n, N);
  }
  
  return prob;

}

double HypergeometricDistribution::GetProbability(const std::uint64_t tile_shape, const std::uint64_t nnz_vals,
                                                  const std::uint64_t constraint_tensor_shape,
                                                  const std::uint64_t constraint_tensor_occupancy) const
{

  std::uint64_t r = constraint_tensor_occupancy;
  std::uint64_t n = tile_shape;
  std::uint64_t N = constraint_tensor_shape;
  
  if (((n + r > N) && (nnz_vals < n + r - N)) | (nnz_vals > r))
    { return 0; }

  boost::math::hypergeometric_distribution<double> distribution(r, n, N);

  // Workaround/fix for domain error for large workload size. Compute the PDF directly
  // This might return an overflow error, but it occurs less times than using the boost pdf call
  double prob;
  try 
  {
    prob = pdf(distribution, nnz_vals);
  } catch (std::exception& e) {
    prob = CalculateProbability(nnz_vals, r, n, N);
  } 
  
  return prob;

}

double HypergeometricDistribution::GetTileOccupancyProbability(const tiling::CoordinateSpaceTileInfo& tile,
                                                               const std::uint64_t occupancy) const
{
  std::uint64_t tile_shape = tile.GetShape();
  double prob;

  if (tile.HasExtraConstraintInfo())
  {
    auto extra_constraint_info = tile.GetExtraConstraintInfo();
    auto occupancy_constraint = extra_constraint_info.GetOccupancy();
    auto shape_constraint = extra_constraint_info.GetShape();
    prob = GetProbability(tile_shape, occupancy, shape_constraint, occupancy_constraint);
  } else
  {
    prob = GetProbability(tile_shape, occupancy);
  }

  return prob;
}

double HypergeometricDistribution::GetExpectedTileOccupancy(const tiling::CoordinateSpaceTileInfo tile) const
{

  std::uint64_t tile_shape = tile.GetShape();
  double expected_occupancy = 0.0;

  if (tile.HasExtraConstraintInfo())
  {
    auto extra_constraint_info = tile.GetExtraConstraintInfo();
    auto occupancy_constraint = extra_constraint_info.GetOccupancy();
    auto shape_constraint = extra_constraint_info.GetShape();
    std::uint64_t max_occupancy = (tile_shape <= occupancy_constraint) ? tile_shape : occupancy_constraint;
    for (std::uint64_t occupancy = 0; occupancy <= max_occupancy; occupancy++)
    {
      double prob = GetProbability(tile_shape, occupancy, shape_constraint, occupancy_constraint);
      expected_occupancy += prob * occupancy;
    }
  } else
  {
    std::uint64_t max_occupancy = (tile_shape <= specs_.total_nnzs) ? tile_shape : specs_.total_nnzs;
    for (std::uint64_t occupancy = 0; occupancy <= max_occupancy; occupancy++)
    {
      double prob = GetProbability(tile_shape, occupancy);
      expected_occupancy += prob * occupancy;
    }
  }
  return expected_occupancy;
}

}
