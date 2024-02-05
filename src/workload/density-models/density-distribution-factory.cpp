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

#include "workload/density-models/density-distribution-factory.hpp"

namespace problem
{

//-------------------------------------------------//
//            Density Distribution Factory         //
//-------------------------------------------------//

std::shared_ptr<DensityDistributionSpecs>
DensityDistributionFactory::ParseSpecs(config::CompoundConfigNode density_config)
{

  std::shared_ptr<DensityDistributionSpecs> specs;
  std::string distribution_type = "None";

  if (density_config.lookupValue("distribution", distribution_type)) 
  {
    if (distribution_type == "fixed" || distribution_type == "fixed_structured")
    {
      auto parsed_specs = FixedStructuredDistribution::ParseSpecs(density_config);
      specs = std::make_shared<FixedStructuredDistribution::Specs>(parsed_specs);
    }
    else if (distribution_type == "hypergeometric")
    {
      auto parsed_specs = HypergeometricDistribution::ParseSpecs(density_config);
      specs = std::make_shared<HypergeometricDistribution::Specs>(parsed_specs);
    }
    else if (distribution_type == "banded" || distribution_type == "diagonal")
    {
      auto parsed_specs = BandedDistribution::ParseSpecs(density_config);
      specs = std::make_shared<BandedDistribution::Specs>(parsed_specs);
    }
    else
    {
      std::cerr << "ERROR: unrecognized density distribution type: " << distribution_type << std::endl;
      exit(1);
    }
  }

  return specs;
}


std::shared_ptr<DensityDistribution>
DensityDistributionFactory::Construct(std::shared_ptr<DensityDistributionSpecs> specs)
{
  std::shared_ptr<DensityDistribution> density_distribution;

  if (specs->Type() == "fixed" || specs->Type() == "fixed_structured")
  {
    auto specs_ptr = *std::static_pointer_cast<FixedStructuredDistribution::Specs>(specs);
    auto constructed_distribution = std::make_shared<FixedStructuredDistribution>(specs_ptr);
    density_distribution = std::static_pointer_cast<DensityDistribution>(constructed_distribution);
  }
  else if (specs->Type() == "hypergeometric")
  {
    auto specs_ptr = *std::static_pointer_cast<HypergeometricDistribution::Specs>(specs);
    auto constructed_distribution = std::make_shared<HypergeometricDistribution>(specs_ptr);
    density_distribution = std::static_pointer_cast<DensityDistribution>(constructed_distribution);
  }
  else if (specs->Type().find("banded") != std::string::npos)
  {
    auto specs_ptr = *std::static_pointer_cast<BandedDistribution::Specs>(specs);
    auto constructed_distribution = std::make_shared<BandedDistribution>(specs_ptr);
    density_distribution = std::static_pointer_cast<DensityDistribution>(constructed_distribution);
  }
  else
  {
    std::cerr << "ERROR: unrecognized density distribution type: " << specs->Type() << std::endl;
    assert(false);
  }

  return density_distribution;
}

} // namespace
