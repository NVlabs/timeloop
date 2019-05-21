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

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <libconfig.h++>

#include "global-names.hpp"

namespace problem
{

// ======================================== //
//              WorkloadConfig              //
// ======================================== //

class WorkloadConfig
{
  Bounds bounds_;
  Coefficients coefficients_;
  Densities densities_;

 public:
  WorkloadConfig() {}

  int getBound(problem::Dimension dim) const
  {
    return bounds_.at(dim);
  }

  int getCoefficient(problem::CoefficientID p) const
  {
    return coefficients_.at(p);
  }
  
  double getDensity(problem::DataSpaceID pv) const
  {
    return densities_.at(pv);
  }

  void setBounds(const Bounds& bounds)
  {
    bounds_ = bounds;
  }
  
  void setCoefficients(const Coefficients& coefficients)
  {
    coefficients_ = coefficients;
  }
  
  void setDensities(const Densities& densities)
  {
    densities_ = densities;
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

void ParseWorkload(libconfig::Setting& config, WorkloadConfig& workload);

} // namespace problem
