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

#include <libconfig.h++>
#include <boost/serialization/export.hpp>

#include "loop-analysis/nest-analysis.hpp"
#include "model/level.hpp"
#include "mapping/mapping.hpp"

namespace model
{

class ArithmeticUnits : public Level
{
 public:
  struct Specs : public LevelSpecs
  {
    static const std::uint64_t kDefaultWordBits = 16;
    const std::string Type() const override { return "ArithmeticUnits"; }
    
    Attribute<std::size_t> instances;
    Attribute<std::size_t> mesh_x;
    Attribute<std::size_t> mesh_y;
    Attribute<std::uint64_t> word_bits;
    Attribute<double> energy_per_op;

    Attribute<std::size_t>& Instances() { return instances; }
    const Attribute<std::size_t>& Instances() const { return instances; }
    Attribute<std::size_t>& MeshX() { return mesh_x; }
    const Attribute<std::size_t>& MeshX() const { return mesh_x; }
    Attribute<std::size_t>& MeshY() { return mesh_y; }
    const Attribute<std::size_t>& MeshY() const { return mesh_y; }
    Attribute<std::uint64_t>& WordBits() { return word_bits; }
    const Attribute<std::uint64_t>& WordBits() const { return word_bits; }
    Attribute<double>& EnergyPerOp() { return energy_per_op; }
    const Attribute<double>& EnergyPerOp() const { return energy_per_op; }

    // Serialization
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(LevelSpecs);
      if (version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(instances);
        ar& BOOST_SERIALIZATION_NVP(mesh_x);
        ar& BOOST_SERIALIZATION_NVP(mesh_y);
        ar& BOOST_SERIALIZATION_NVP(word_bits);
      }
    }
  };
  
  // FIXME: need Spec, Stats, etc.
 private:
  Specs specs_;

  double energy_ = 0;
  double area_ = 0;
  std::uint64_t cycles_ = 0;
  std::size_t utilized_instances_ = 0;
  std::uint64_t maccs_ = 0;

  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Level);    
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(specs_);
      ar& BOOST_SERIALIZATION_NVP(energy_);
      ar& BOOST_SERIALIZATION_NVP(area_);
      ar& BOOST_SERIALIZATION_NVP(cycles_);
      ar& BOOST_SERIALIZATION_NVP(utilized_instances_);
      ar& BOOST_SERIALIZATION_NVP(maccs_);
    }
  }
  
 public:
  ArithmeticUnits() { }
  ArithmeticUnits(const Specs & specs);
  ~ArithmeticUnits() { }
  
  // The hierarchical ParseSpecs functions are static and do not
  // affect the internal specs_ data structure, which is set by
  // the dynamic Spec() call later.
  static Specs ParseSpecs(libconfig::Setting& setting);
  
  double Energy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  double Area() const override;
  double AreaPerInstance() const override;
  std::uint64_t Cycles() const override;

  // --- Unsupported overrides ---
  bool DistributedMulticastSupported() override { return false; }
  bool PreEvaluationCheck(const problem::PerDataSpace<std::size_t> working_set_sizes,
                          const tiling::CompoundMask mask) override
  {
    (void) working_set_sizes;
    (void) mask;
    return true;
  }
  bool Evaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                const double inner_tile_area, const std::uint64_t compute_cycles) override
  {
    (void) tile;
    (void) mask;
    (void) inner_tile_area;
    (void) compute_cycles;
    return false;
  }
  
  std::string Name() const override { return "__NONAME__"; }
  std::uint64_t Accesses(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override
  {
    (void) pv;
    return 0;
  }
  double CapacityUtilization() override { return 0; }
  std::uint64_t MaxFanout() const override { return 0; }
  void Print(std::ostream& out) const override
  {
    (void) out;
  }
    
  // --- Temporary hack interfaces, these will be removed ---
  
  bool HackEvaluate(analysis::NestAnalysis* analysis,
                    const problem::Workload& workload_config)
  {
    assert(is_specced_);

    bool success = true;

    auto body_info = analysis->GetBodyInfo();
    
    utilized_instances_ = body_info.replication_factor;
    auto compute_cycles = body_info.accesses;

    // maccs_ = analysis->GetMACs();
    // utilized_instances_ = maccs_ / analysis->GetComputeCycles(); // Yuck!!! FIXME.
    // assert(body_info.accesses == analysis->GetComputeCycles());
    // assert(body_info.replication_factor == utilized_instances_);

    if (utilized_instances_ <= specs_.Instances().Get())
    {
      cycles_ = compute_cycles;
      maccs_ = utilized_instances_ * compute_cycles;
      energy_ = maccs_ * specs_.EnergyPerOp().Get();

      // Scale energy for sparsity.
      for (unsigned d = 0; d < problem::GetShape()->NumDataSpaces; d++)
      {
        if (!problem::GetShape()->IsReadWriteDataSpace.at(d))
          energy_ *= workload_config.GetDensity(d);
      }
      
      is_evaluated_ = true;    
    }
    else
    {
      success = false;
    }
    
    return success;
  }
  
  std::uint64_t MACCs() const
  {
    assert(is_evaluated_);
    return maccs_;
  }

  double IdealCycles() const
  {
    // FIXME: why would this be different from Cycles()?
    assert(is_evaluated_);
    return double(maccs_) / specs_.Instances().Get();
  }
};

} // namespace model
