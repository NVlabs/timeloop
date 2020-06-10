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

#include <boost/serialization/export.hpp>

#include "loop-analysis/nest-analysis.hpp"
#include "model/level.hpp"
#include "model/network.hpp"
#include "mapping/mapping.hpp"
#include "compound-config/compound-config.hpp"

namespace model
{

class ArithmeticUnits : public Level
{
 public:
  struct Specs : public LevelSpecs
  {
    static const std::uint64_t kDefaultWordBits = 16;
    const std::string Type() const override { return "ArithmeticUnits"; }

    Attribute<std::string> name;
    Attribute<std::size_t> instances;
    Attribute<std::size_t> meshX;
    Attribute<std::size_t> meshY;
    Attribute<std::uint64_t> word_bits;
    Attribute<double> energy_per_op;
    Attribute<double> area;

    Attribute<std::string> operand_network_name;
    Attribute<std::string> result_network_name;

    // Serialization
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(LevelSpecs);
      if (version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(instances);
        ar& BOOST_SERIALIZATION_NVP(meshX);
        ar& BOOST_SERIALIZATION_NVP(meshY);
        ar& BOOST_SERIALIZATION_NVP(word_bits);
      }
    }

   public:
    std::shared_ptr<LevelSpecs> Clone() const override
    {
      return std::static_pointer_cast<LevelSpecs>(std::make_shared<Specs>(*this));
    }
  };
  
 private:
  Specs specs_;

  // Network endpoints.
  std::shared_ptr<Network> network_operand_;
  std::shared_ptr<Network> network_result_;

  // Stats.
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
  
  std::shared_ptr<Level> Clone() const override
  {
    return std::static_pointer_cast<Level>(std::make_shared<ArithmeticUnits>(*this));
  }

  // The hierarchical ParseSpecs functions are static and do not
  // affect the internal specs_ data structure, which is set by
  // the dynamic Spec() call later.
  static Specs ParseSpecs(config::CompoundConfigNode setting, uint32_t nElements);
  static void ValidateTopology(ArithmeticUnits::Specs& specs);
  
  Specs& GetSpecs() { return specs_; }

  // Connect to networks.
  void ConnectOperand(std::shared_ptr<Network> network);
  void ConnectResult(std::shared_ptr<Network> network);

  std::string Name() const override;
  double Energy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  double Area() const override;
  double AreaPerInstance() const override;
  std::uint64_t Cycles() const override;
  std::uint64_t UtilizedInstances(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;

  void Print(std::ostream& out) const override;
    
  // --- Unsupported overrides ---
  bool HardwareReductionSupported() override { return false; }

  EvalStatus PreEvaluationCheck(const problem::PerDataSpace<std::size_t> working_set_sizes,
                                const tiling::CompoundMask mask,
                                const bool break_on_failure) override
  {
    (void) working_set_sizes;
    (void) mask;
    (void) break_on_failure;
    return { true, "" };
  }

  EvalStatus Evaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                      const std::uint64_t compute_cycles,
                      const bool break_on_failure) override
  {
    (void) tile;
    (void) mask;
    (void) compute_cycles;
    (void) break_on_failure;
    return { false, "ArithmeticLevel must use the HackEvaluate() function" };
  }
  
  std::uint64_t Accesses(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override
  {
    (void) pv;
    return 0;
  }

  double CapacityUtilization() const override { return 0; }

  std::uint64_t UtilizedCapacity(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override
  {
    (void) pv;
    return 0;
  }

  // --- Temporary hack interfaces, these will be removed ---
  
  EvalStatus HackEvaluate(analysis::NestAnalysis* analysis)
  {
    assert(is_specced_);

    EvalStatus eval_status;
    eval_status.success = true;

    auto body_info = analysis->GetBodyInfo();
    
    utilized_instances_ = body_info.replication_factor;
    auto compute_cycles = body_info.accesses;

    // maccs_ = analysis->GetMACs();

    if (utilized_instances_ <= specs_.instances.Get())
    {
      cycles_ = compute_cycles;
      maccs_ = utilized_instances_ * compute_cycles;
      energy_ = maccs_ * specs_.energy_per_op.Get();

      // Scale energy for sparsity.
      for (unsigned d = 0; d < problem::GetShape()->NumDataSpaces; d++)
      {
        if (!problem::GetShape()->IsReadWriteDataSpace.at(d))
          //energy_ *= workload.GetDensity(d);
          energy_ *= body_info.data_densities[d];
      }

      is_evaluated_ = true;    
    }
    else
    {
      eval_status.success = false;
      std::ostringstream str;
      str << "mapped Arithmetic instances " << utilized_instances_
          << " exceeds hardware instances " << specs_.instances.Get();
      eval_status.fail_reason = str.str();
    }
    
    return eval_status;
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
    return double(maccs_) / specs_.instances.Get();
  }
};

} // namespace model
