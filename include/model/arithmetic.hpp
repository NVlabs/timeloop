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

//#include "loop-analysis/nest-analysis.hpp"
#include "model/level.hpp"
#include "model/network.hpp"
#include "mapping/mapping.hpp"
#include "compound-config/compound-config.hpp"
#include "loop-analysis/operation-type.hpp"

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

    Attribute<bool> is_sparse_module;

    // for ERT parsing
    std::map<std::string, double> ERT_entries;
    std::map<std::string, double> op_energy_map;

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

    void UpdateOpEnergyViaERT(const std::map<std::string, double>& ERT_entries, const double max_energy) override;
    void UpdateAreaViaART(const double component_area) override;
    
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
  std::uint64_t utilized_instances_ = 0;
  std::uint64_t avg_utilized_instances_ = 0;
  std::uint64_t utilized_x_expansion_ = 0;
  std::uint64_t utilized_y_expansion_ = 0;
  std::uint64_t algorithmic_computes_ = 0; // number of computes defined by the algorithm (loop nests)
  std::uint64_t actual_computes_ = 0; // computes that actually happened, right now, consists of random computes only
  // Fine-grained actions
  // A fine grained action is either effectual and ineffectual
  //     effectual: actual computes: random compute
  //     ineffectual: computes that did not happen: nonexistent compute, skipped compute, gated compute,
  std::uint64_t random_computes_ = 0;
  std::uint64_t skipped_computes_ = 0;
  std::uint64_t gated_computes_ = 0;

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
      ar& BOOST_SERIALIZATION_NVP(avg_utilized_instances_);
      ar& BOOST_SERIALIZATION_NVP(algorithmic_computes_);
      ar& BOOST_SERIALIZATION_NVP(random_computes_);
      ar& BOOST_SERIALIZATION_NVP(gated_computes_);
      ar& BOOST_SERIALIZATION_NVP(skipped_computes_);
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
  static Specs ParseSpecs(config::CompoundConfigNode setting, uint32_t nElements, bool is_sparse_module);
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
                                const problem::Workload* workload,
                                const sparse::PerStorageLevelCompressionInfo per_level_compression_info,
                                const double confidence_threshold,
                                const bool break_on_failure) override
  {
    (void) working_set_sizes;
    (void) mask;
    (void) workload;
    (void) break_on_failure;
    (void) per_level_compression_info;
    (void) confidence_threshold;
    return { true, "" };
  }

  // EvalStatus Evaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
  //                     const std::uint64_t compute_cycles,
  //                     const bool break_on_failure) override
  // {
  //   (void) tile;
  //   (void) mask;
  //   (void) compute_cycles;
  //   (void) break_on_failure;
  //   return { false, "ArithmeticLevel must use the HackEvaluate() function" };
  // }
  
  std::uint64_t Accesses(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override
  {
    (void) pv;
    return 0;
  }

  double CapacityUtilization() const override { return 0; }

  std::uint64_t TileSize(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override
  {
    (void) pv;
    return 0;
  }

  std::uint64_t UtilizedCapacity(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override
  {
    (void) pv;
    return 0;
  }
 
  EvalStatus Evaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                      const double confidence_threshold, const std::uint64_t compute_cycles,
                      const bool break_on_failure)
  {
    assert(is_specced_);

    (void) mask;
    (void) confidence_threshold;
    (void) break_on_failure;
    (void) compute_cycles;

    EvalStatus eval_status;
    eval_status.success = true;

    utilized_instances_ = tile.compute_info.max_x_expansion * tile.compute_info.max_y_expansion;
    avg_utilized_instances_ = tile.compute_info.avg_replication_factor;
    utilized_x_expansion_ = tile.compute_info.max_x_expansion;
    utilized_y_expansion_ = tile.compute_info.max_y_expansion;
    
    // std::cout << specs_.level_name <<": max x expansion: " << utilized_x_expansion_
    //  << "    max y expansion: " << utilized_y_expansion_ << std::endl;

    if (utilized_instances_ > specs_.instances.Get())
    {
      eval_status.success = false;
      std::ostringstream str;
      str << "mapped max Arithmetic instances " << utilized_instances_
          << " exceeds hardware instances " << specs_.instances.Get();
      eval_status.fail_reason = str.str();   
    }
    else if (utilized_x_expansion_ > specs_.meshX.Get())
    {
      eval_status.success = false;
      std::ostringstream str;
      str << "mapped max Arithmetic X expansion " << utilized_x_expansion_ 
          << " exceeds hardware instances " << specs_.meshX.Get();
      eval_status.fail_reason = str.str();   
    }
    else if (utilized_y_expansion_ > specs_.meshY.Get())
    {
      eval_status.success = false;
      std::ostringstream str;
      str << "mapped max Arithmetic Y expansion " << utilized_y_expansion_ 
          << " exceeds hardware instances " << specs_.meshY.Get();
      eval_status.fail_reason = str.str();   
    }
    else // legal case
    {
      energy_ = 0;
      std::uint64_t op_accesses;
      std::string op_name;

      // go through the fine grained actions and reflect the special impacts
      for (unsigned op_id = 0; op_id < tiling::arithmeticOperationTypes.size(); op_id++){
        op_name = tiling::arithmeticOperationTypes[op_id];
        op_accesses = tile.compute_info.fine_grained_accesses.at(op_name);
        energy_ += op_accesses * specs_.op_energy_map.at(op_name);

        // collect stats...
        if (op_name == "random_compute")
        {
          random_computes_ = op_accesses;
        } else if (op_name == "gated_compute")
        {
          gated_computes_ = op_accesses;
        } else if (op_name == "skipped_compute")
        {
          skipped_computes_ = op_accesses;
        }

        actual_computes_ = random_computes_;
      }

      cycles_ = ceil(double(random_computes_ + gated_computes_)/avg_utilized_instances_);
      algorithmic_computes_ = tile.compute_info.replication_factor * tile.compute_info.accesses;
      is_evaluated_ = true;
    }
    
    return eval_status;
  }
  
  std::uint64_t AlgorithmicComputes() const
  {
    assert(is_evaluated_);
    return algorithmic_computes_;
  }

  std::uint64_t ActualComputes() const
  {
    assert(is_evaluated_);
    return random_computes_;
  }

  double IdealCycles() const
  {
    // FIXME: why would this be different from Cycles()?
    assert(is_evaluated_);
    return double(actual_computes_ + gated_computes_)/specs_.instances.Get();
  }
};

} // namespace model
