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

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "model/arithmetic.hpp"
//BOOST_CLASS_EXPORT(model::ArithmeticUnits::Specs)
BOOST_CLASS_EXPORT(model::ArithmeticUnits)

#include "pat/pat.hpp"

namespace model
{

ArithmeticUnits::ArithmeticUnits(const Specs& specs) :
    specs_(specs)
{
  is_specced_ = true;
  is_evaluated_ = false;
  area_ = pat::MultiplierArea(specs_.WordBits().Get(), specs_.WordBits().Get());
}

ArithmeticUnits::Specs ArithmeticUnits::ParseSpecs(libconfig::Setting& setting)
{
  Specs specs;

  // Instances.
  std::uint32_t instances;
  if (!setting.lookupValue("instances", instances))
  {
    std::cerr << "instances is a required arithmetic parameter" << std::endl;
    assert(false);
  }    
  specs.Instances() = instances;
    
  // Word size (in bits).
  std::uint32_t word_bits;
  if (setting.lookupValue("word-bits", word_bits))
  {
    specs.WordBits() = word_bits;
  }
  else
  {
    specs.WordBits() = Specs::kDefaultWordBits;
  }

  // MeshX.
  std::uint32_t mesh_x;
  if (setting.lookupValue("meshX", mesh_x))
  {
    specs.MeshX() = mesh_x;
  }

  // MeshY.
  std::uint32_t mesh_y;
  if (setting.lookupValue("meshY", mesh_y))
  {
    specs.MeshY() = mesh_y;
  }

  // quick validation
  if (specs.MeshX().IsSpecified())
  {
    if (specs.MeshY().IsSpecified())    // X and Y are both specificed
    {
      assert(specs.MeshX().Get() * specs.MeshY().Get() == specs.Instances().Get());
    }
    else                                // only X specified
    {
      assert(specs.Instances().Get() % specs.MeshX().Get() == 0);
      specs.MeshY() = specs.Instances().Get() / specs.MeshX().Get();
    }
  }
  else
  {
    if (specs.MeshY().IsSpecified())    // only Y specified
    {
      assert(specs.Instances().Get() % specs.MeshY().Get() == 0);
      specs.MeshX() = specs.Instances().Get() / specs.MeshY().Get();
    }
  }

  // Energy (override).
  double energy;
  if (setting.lookupValue("energy", energy))
  {
    specs.EnergyPerOp() = energy;
  }
  else
  {
    specs.EnergyPerOp() =
      pat::MultiplierEnergy(specs.WordBits().Get(), specs.WordBits().Get());
  }
    
  return specs;
}
  
// Accessors.

double ArithmeticUnits::Energy(problem::DataSpaceID pv) const
{
  assert(is_evaluated_);
  assert(pv == problem::NumDataSpaces);
  return energy_;
}

double ArithmeticUnits::Area() const
{
  assert(is_specced_);
  return AreaPerInstance() * specs_.Instances().Get();
}

double ArithmeticUnits::AreaPerInstance() const
{
  assert(is_specced_);
  return area_;
}

std::uint64_t ArithmeticUnits::Cycles() const
{
  assert(is_evaluated_);
  return cycles_;
}

} // namespace model
