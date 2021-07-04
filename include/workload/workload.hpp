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

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "loop-analysis/point-set.hpp"
#include "compound-config/compound-config.hpp"

#include "shape-models/problem-shape.hpp"
#include "density-models/density-distribution.hpp"
#include "density-models/density-distribution-factory.hpp"

namespace problem
{

// ======================================== //
//              Shape instance              //
// ======================================== //
// Sadly, this has to be a static instance outside Workload for now
// because a large section of the codebase was written assuming there would
// only be one active shape instance. To fix this, we need to make the shape
// instance a member of Workload, and pass pointers to the Workload
// object *everywhere*. The most problematic classes are PerDataSpace and
// PerProblemDimension. If we can figure out a clean implementation of these
// classes that does not require querying Shape::NumDimensions or
// Shape::NumDataSpaces then most of the problem is possibly solved.

const Shape* GetShape();

// ======================================== //
//                 Workload                 //
// ======================================== //

class Workload
{
 public:
  typedef std::map<Shape::DimensionID, Coordinate> Bounds;
  typedef std::map<Shape::CoefficientID, int> Coefficients;
  typedef std::map<Shape::DataSpaceID, std::shared_ptr<DensityDistribution>> Densities;
  
 protected:
  Bounds bounds_;
  Coefficients coefficients_;
  Densities densities_;
  bool workload_tensor_size_set_ = false;
  bool default_dense_ = true;

 public:
  Workload() {}

  const Shape* GetShape() const
  {
    // Just a trampolene to the global function at the moment.
    return problem::GetShape();
  }

  int GetBound(Shape::DimensionID dim) const
  {
    return bounds_.at(dim);
  }

  int GetCoefficient(Shape::CoefficientID p) const
  {
    return coefficients_.at(p);
  }

  std::shared_ptr<DensityDistribution> GetDensity(Shape::DataSpaceID pv) const
  {
    return densities_.at(pv);
  }

  bool GetDenseDefaultTensor() const
  {
    return default_dense_;
  }

  void SetBounds(const Bounds& bounds)
  {
    bounds_ = bounds;
  }
  
  void SetCoefficients(const Coefficients& coefficients)
  {
    coefficients_ = coefficients;
  }
  
  void SetDensities(const Densities& densities)
  {
    densities_ = densities;
  }

  void SetWorkloadTensorSize(problem::Shape::DataSpaceID id, std::uint64_t tensor_size)
  {
    // hypergeometric distribution specification requires workload tensor sizes
    densities_.at(id)->SetWorkloadTensorSize(tensor_size);
  }

  bool IsWorkloadTensorSizesSet()
  {
    return workload_tensor_size_set_;
  }

  void AllTensorsSet()
  {
    workload_tensor_size_set_ = true;
  }

  void SetDefaultDenseTensorFlag(const bool flag)
  {
    default_dense_ = flag;
  }


 private:
  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(bounds_);
      ar& BOOST_SERIALIZATION_NVP(coefficients_);
      ar& BOOST_SERIALIZATION_NVP(densities_);
    }
  }
};

void ParseWorkload(config::CompoundConfigNode config, Workload& workload);
void ParseWorkloadInstance(config::CompoundConfigNode config, Workload& workload);

} // namespace problem
