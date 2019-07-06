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

#include <iostream>
#include <memory>
#include <libconfig.h++>

#include "loop-analysis/tiling.hpp"
#include "loop-analysis/nest-analysis.hpp"
#include "mapping/nest.hpp"
#include "mapping/mapping.hpp"
#include "model/level.hpp"
#include "model/arithmetic.hpp"
#include "model/buffer.hpp"

namespace model
{

class Topology : public Module
{
 public:
  class Specs
  {
   private:
    std::vector<std::shared_ptr<LevelSpecs>> levels;
    std::map<unsigned, unsigned> storage_map;
    unsigned arithmetic_map;

   public:
    unsigned NumLevels() const;
    unsigned NumStorageLevels() const;

    void AddLevel(unsigned typed_id, std::shared_ptr<LevelSpecs> level_specs);

    unsigned StorageMap(unsigned i) const { return storage_map.at(i); }
    unsigned ArithmeticMap() const { return arithmetic_map; }

    std::shared_ptr<LevelSpecs> GetLevel(unsigned level_id) const;
    std::shared_ptr<BufferLevel::Specs> GetStorageLevel(unsigned storage_level_id) const;
    std::shared_ptr<ArithmeticUnits::Specs> GetArithmeticLevel() const;
  };
  
 private:
  std::vector<std::shared_ptr<Level>> levels_;
  
  Specs specs_;
  
  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(levels_);
    }
  }

 public:
  // The hierarchical ParseSpecs functions are static and do not
  // affect the internal specs_ data structure, which is set by
  // the dynamic Spec() call later.
  static Specs ParseSpecs(libconfig::Setting& setting, libconfig::Setting& arithmetic_specs);
  static void Validate(Specs& specs);
  
  void Spec(const Specs& specs);
  unsigned NumLevels() const;
  unsigned NumStorageLevels() const;

  std::shared_ptr<Level> GetLevel(unsigned level_id) const;
  std::shared_ptr<BufferLevel> GetStorageLevel(unsigned storage_level_id) const;
  std::shared_ptr<ArithmeticUnits> GetArithmeticLevel() const;
  
  std::vector<bool> PreEvaluationCheck(const Mapping& mapping, analysis::NestAnalysis* analysis);
  std::vector<bool> Evaluate(Mapping& mapping, analysis::NestAnalysis* analysis, const problem::Workload& workload);

  double Energy() const;
  double Area() const;
  // double Size() const;
  std::uint64_t Cycles() const;
  double Utilization() const;
  std::uint64_t MACCs() const;
  
  friend std::ostream& operator<<(std::ostream& out, const Topology& sh);
};

}  // namespace model
