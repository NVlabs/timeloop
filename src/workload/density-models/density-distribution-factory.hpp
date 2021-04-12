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

#pragma once

#include "density-distribution.hpp"
#include "fixed-distribution.hpp"
#include "hypergeometric-distribution.hpp"
#include "compound-config/compound-config.hpp"


namespace problem
{

//-------------------------------------------------//
//               Density Distribution Factory      //
//-------------------------------------------------//

class DensityDistributionFactory{

public:


  static std::shared_ptr<DensityDistributionSpecs> ParseSpecs(config::CompoundConfigNode density_config){

    std::shared_ptr<DensityDistributionSpecs> specs;
    std::string distribution_type = "None";

    if (density_config.lookupValue("distribution", distribution_type)) {
      if (distribution_type == "fixed") {
        auto fixed_specs = FixedDistribution::ParseSpecs(density_config);
        specs = std::make_shared<FixedDistribution::Specs>(fixed_specs);

      } else if (distribution_type == "hypergeometric"){
        auto hypergeo_specs = HypergeometricDistribution::ParseSpecs(density_config);
        specs = std::make_shared<HypergeometricDistribution::Specs>(hypergeo_specs);

      } else {
        std::cerr << "ERROR: unrecognized density distribution type: " << specs->Type() << std::endl;
        exit(1);
      }

    }

    return specs;
  }


  static std::shared_ptr<DensityDistribution> Construct(std::shared_ptr<DensityDistributionSpecs> specs){

    std::shared_ptr<DensityDistribution> density_distribution;

    if (specs->Type() == "fixed"){
      auto fixed_specs = *std::static_pointer_cast<FixedDistribution::Specs>(specs);
      auto fixed_density_distribution = std::make_shared<FixedDistribution>(fixed_specs);
      density_distribution = std::static_pointer_cast<DensityDistribution>(fixed_density_distribution);

    } else if (specs->Type() == "hypergeometric"){
      auto hypergeo_specs = *std::static_pointer_cast<HypergeometricDistribution::Specs>(specs);
      auto hypergeo_density_distribution = std::make_shared<HypergeometricDistribution>(hypergeo_specs);
      density_distribution = std::static_pointer_cast<DensityDistribution>(hypergeo_density_distribution);

    } else {
      std::cerr << "ERROR: unrecognized density distribution type: " << specs->Type() << std::endl;
      exit(1);
    }

    return density_distribution;
  }


};




} // namespace
