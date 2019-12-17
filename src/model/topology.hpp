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

#include <iostream>
#include <memory>

#include "loop-analysis/tiling.hpp"
#include "loop-analysis/nest-analysis.hpp"
#include "mapping/nest.hpp"
#include "mapping/mapping.hpp"
#include "model/level.hpp"
#include "model/arithmetic.hpp"
#include "model/buffer.hpp"
#include "compound-config/compound-config.hpp"
#include "network.hpp"
#include "network-legacy.hpp"

namespace model
{

static std::string bufferClasses[4] = { "DRAM",
                                        "SRAM",
                                        "regfile",
                                        "smartbuffer"};

static std::string computeClasses[3] = { "mac",
                                         "intmac",
                                         "fpmac" };

// FIXME: derive these from a statically-instantiated list of class names that
// are auto-populated by each Network class at program init time.
static std::string networkClasses[] = { "XY_NoC",
                                        "Legacy" };

bool isBufferClass(std::string className);
bool isComputeClass(std::string className);
bool isNetworkClass(std::string className);

class Topology : public Module
{
 public:
  class Specs
  {
   private:
    std::vector<std::shared_ptr<LevelSpecs>> levels;
    std::vector<std::shared_ptr<LegacyNetwork::Specs>> inferred_networks;
    std::vector<std::shared_ptr<NetworkSpecs>> networks;
    std::map<unsigned, unsigned> storage_map;
    unsigned arithmetic_map;

   public:
    unsigned NumLevels() const;
    unsigned NumStorageLevels() const;
    unsigned NumNetworks() const;

    std::vector<std::string> LevelNames() const;
    std::vector<std::string> StorageLevelNames() const;

    void ParseAccelergyERT(config::CompoundConfigNode ert);

    void AddLevel(unsigned typed_id, std::shared_ptr<LevelSpecs> level_specs);
    void AddInferredNetwork(std::shared_ptr<LegacyNetwork::Specs> specs);
    void AddNetwork(std::shared_ptr<NetworkSpecs> specs);

    unsigned StorageMap(unsigned i) const { return storage_map.at(i); }
    unsigned ArithmeticMap() const { return arithmetic_map; }

    std::shared_ptr<LevelSpecs> GetLevel(unsigned level_id) const;
    std::shared_ptr<BufferLevel::Specs> GetStorageLevel(unsigned storage_level_id) const;
    std::shared_ptr<ArithmeticUnits::Specs> GetArithmeticLevel() const;
    std::shared_ptr<LegacyNetwork::Specs> GetInferredNetwork(unsigned network_id) const;
    std::shared_ptr<NetworkSpecs> GetNetwork(unsigned network_id) const;
  };
  
 private:
  std::vector<std::shared_ptr<Level>> levels_;
  std::map<std::string, std::shared_ptr<Network>> networks_;
  std::map<unsigned, std::uint64_t> fanout_map_;

  Specs specs_;
  
  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(levels_);
      ar& BOOST_SERIALIZATION_NVP(networks_);
    }
  }

 private:
  std::shared_ptr<Level> GetLevel(unsigned level_id) const;
  std::shared_ptr<BufferLevel> GetStorageLevel(unsigned storage_level_id) const;
  std::shared_ptr<ArithmeticUnits> GetArithmeticLevel() const;
  void DeriveFanouts();

 public:
  // The hierarchical ParseSpecs functions are static and do not
  // affect the internal specs_ data structure, which is set by
  // the dynamic Spec() call later.
  static Specs ParseSpecs(config::CompoundConfigNode setting, config::CompoundConfigNode arithmetic_specs);
  static Specs ParseTreeSpecs(config::CompoundConfigNode designRoot);
  
  void Spec(const Specs& specs);
  unsigned NumLevels() const;
  unsigned NumStorageLevels() const;
  unsigned NumNetworks() const;

  std::vector<EvalStatus> PreEvaluationCheck(const Mapping& mapping, analysis::NestAnalysis* analysis, bool break_on_failure);
  std::vector<EvalStatus> Evaluate(Mapping& mapping, analysis::NestAnalysis* analysis, const problem::Workload& workload, bool break_on_failure);

  double Energy() const;
  double Area() const;
  std::uint64_t Cycles() const;
  double Utilization() const;
  std::vector<problem::PerDataSpace<std::uint64_t>> TileSizes() const;
  std::vector<problem::PerDataSpace<std::uint64_t>> UtilizedInstances() const;
  std::uint64_t MACCs() const;
  std::uint64_t LastLevelAccesses() const;

  friend std::ostream& operator<<(std::ostream& out, const Topology& sh);
};

}  // namespace model
