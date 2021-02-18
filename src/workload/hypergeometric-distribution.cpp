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

#include "hypergeometric-distribution.hpp"

BOOST_CLASS_EXPORT(problem::HypergeometricDistribution)

namespace problem {

HypergeometricDistribution::HypergeometricDistribution() {}

HypergeometricDistribution::HypergeometricDistribution(const Specs &specs) : specs_(specs) {
  is_specced_ = true;
  if (specs.workload_tensor_size != 0) {
    workload_tensor_size_set_ = true;
  }
}

HypergeometricDistribution::~HypergeometricDistribution() {}

HypergeometricDistribution::Specs HypergeometricDistribution::ParseSpecs(config::CompoundConfigNode density_config) {

  Specs specs;
  assert(density_config.lookupValue("distribution", specs.type));
  assert(density_config.lookupValue("density", specs.average_density));

  long long workload_tensor_size;
  if (density_config.lookupValue("workload_tensor_size", workload_tensor_size)) {
    specs.total_nnzs = ceil(specs.average_density * workload_tensor_size);
    specs.workload_tensor_size = workload_tensor_size;

  } else {
    specs.total_nnzs = 0;
    specs.workload_tensor_size = 0;
  }

  return specs;
}

void HypergeometricDistribution::SetDensity(double density) {
  specs_.average_density = density;
}

void HypergeometricDistribution::SetWorkloadTensorSize(std::uint64_t size) {
  // setter that allows workload tensor size at a latter stage (topology.cpp, PreEvaluationCheck)
  assert(is_specced_);
  specs_.workload_tensor_size = size;
  specs_.total_nnzs = ceil(specs_.average_density * size);
  workload_tensor_size_set_ = true;
}

double HypergeometricDistribution::GetTileConfidenceByAllocatedCapacity(std::uint64_t tile_shape,
                                                                        std::uint64_t allocated_buffer_size) const {
  // given a specific buffer size, return the confidence of tile shape with workload density fitting in
  // the buffer

  float confidence;

  if (allocated_buffer_size >= tile_shape) {

    confidence = 1.0; // for sure the tile will fit, no matter what the density is

  } else if (allocated_buffer_size >= specs_.total_nnzs) {

    // can store more than total number of nonzeros, tile must fit, check on upper bound min(n,r)
    confidence = 1.0;

  } else if (allocated_buffer_size < tile_shape + specs_.total_nnzs - specs_.workload_tensor_size) {

    // check on lower bound max(0, n+r-N)
    confidence = 0.0;

  } else if (specs_.average_density == 1.0) {
    // if not larger than tile shape or total number of nonzeros, and has a density of 1.0, for sure will not fit
    confidence = 0.0;

  } else {  // buffer size < tile shape, buffer size < total number of nonzeros and sparse data

    // construct a hypergeometric distribution according to
    //    1) N: population of objects -> workload size
    //    2) n: sample size -> tile size
    //    3) r: defective objects -> total number of zero data

    std::uint64_t r = specs_.total_nnzs;
    std::uint64_t n = tile_shape;
    std::uint64_t N = specs_.workload_tensor_size;
    boost::math::hypergeometric_distribution<double> distribution(r, n, N);
    double cdf_val = cdf(distribution, 1.0 * allocated_buffer_size);
    // confidence = cdf_val >= 0.9999 ? 1.0 : cdf_val;
    confidence = cdf_val;
  }

  return confidence;

}

std::uint64_t HypergeometricDistribution::GetTileOccupancyByConfidence(std::uint64_t tile_shape, double confidence) {

  std::uint64_t tile_occupancy;

  if (confidence == 1.0) {
    if (tile_shape >= specs_.total_nnzs) {
      tile_occupancy = specs_.total_nnzs;
    } else {
      tile_occupancy = tile_shape;
    }
  } else {
    std::uint64_t r = specs_.total_nnzs;
    std::uint64_t n = tile_shape;
    std::uint64_t N = specs_.workload_tensor_size;
    boost::math::hypergeometric_distribution<double> distribution(r, n, N);
    tile_occupancy = quantile(distribution, confidence);
  }

  return tile_occupancy;
}

std::uint64_t HypergeometricDistribution::GetWorkloadTensorSize() const {
  assert(workload_tensor_size_set_);
  return specs_.workload_tensor_size;
};

std::string HypergeometricDistribution::GetDistributionType() const {
  return specs_.type;
}

double HypergeometricDistribution::GetTileDensityByConfidence(std::uint64_t tile_shape,
                                                              double confidence, uint64_t allocated_capacity) const {
  double tile_density;

  if (specs_.average_density == 1.0) {
    // dense data does not need distribution
    tile_density = 1.0;

  } else if (tile_shape == 0) {
    // simply assign the average density for nonexistent tiles
    tile_density = specs_.average_density;

  } else if (confidence == 1.0) {

    if (allocated_capacity <= tile_shape && allocated_capacity > specs_.total_nnzs){
      tile_density = specs_.total_nnzs/tile_shape;
    } else if (allocated_capacity <= tile_shape) {
      tile_density = double(allocated_capacity)/tile_shape;
    } else if (allocated_capacity > tile_shape && tile_shape > specs_.total_nnzs){
      tile_density = specs_.total_nnzs/tile_shape;
    } else if (allocated_capacity > tile_shape){
      tile_density = 1.0;
    } else {
      assert(false);
    }

  } else {
    // we don't need to re-derive the percentile
    // for this case, we know that we are not confident that the entire tile will fit
    // even if we give the entire allocated_capacity to the tile
    // so the max density supported is just allocated_capacity/tile shape
    tile_density = 1.0 * allocated_capacity / tile_shape;
  }

  return tile_density;

}

double HypergeometricDistribution::GetTileExpectedDensity(uint64_t tile_shape) const {

  (void) tile_shape;
  assert(is_specced_);
  return specs_.average_density;
}


double HypergeometricDistribution::GetProbability(std::uint64_t tile_shape, std::uint64_t nnz_vals) const {

  assert(is_specced_);
  assert(workload_tensor_size_set_);

  std::uint64_t r = specs_.total_nnzs;
  std::uint64_t n = tile_shape;
  std::uint64_t N = specs_.workload_tensor_size;

  if (((n + r > N) && (nnz_vals < n + r - N)) | (nnz_vals > r)) { return 0; }

  boost::math::hypergeometric_distribution<double> distribution(r, n, N);
  double prob = pdf(distribution, nnz_vals);

  return prob;

}

}