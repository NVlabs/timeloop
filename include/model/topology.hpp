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
algorithmic contributors may be used tactual or promote products derived
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
#include <algorithm>
#include <fstream>

#include "loop-analysis/tiling.hpp"
#include "loop-analysis/tiling-tile-info.hpp"
#include "loop-analysis/nest-analysis.hpp"
#include "mapping/nest.hpp"
#include "mapping/mapping.hpp"
#include "model/level.hpp"
#include "model/arithmetic.hpp"
#include "model/buffer.hpp"
#include "compound-config/compound-config.hpp"
#include "network.hpp"
#include "network-legacy.hpp"
#include "sparse-optimization-info.hpp"

namespace model
{

// mapping between architectural action names and accelergy's ERT name
// this mapping can be moved out as a separate yaml file that can be read in by timeloop to allow more flexibility
// NOTE: the keys in each map MUST MATCH the operation type names in loop-analysis/operation-type.hpp
// FIXME: regarding the note above: cleanup the setup more so that a unified set of names are used

// format {timeloop_action_name: [priority list of ERT action names]}
static std::map <std::string, std::vector<std::string>> arithmeticOperationMappings
  = {{"random_compute", {"mac_random", "mult_random", "mac", "mult", "compute"}},
     {"skipped_compute", {"mac_skipped", "mult_skipped","compute_skipped", "mac_gated", "mult_gated", "compute_gated", "mac", "mult", "compute"}},
     {"gated_compute", {"mac_gated", "mult_gated", "compute_gated", "mac", "mult", "compute"}}
  };

static std::map <std::string, std::vector<std::string>> storageOperationMappings
  = {
     {"random_read", {"random_read", "read"}},
     {"gated_read", {"gated_read"}},
     {"skipped_read", {"skipped_read", "gated_read"}},
     {"random_metadata_read", {"random_metadata_read", "metadata_read"}},
     {"gated_metadata_read", {"gated_metadata_read"}},
     {"skipped_metadata_read", {"skipped_metadata_read", "gated_metadata_read"}},

     {"random_fill", {"random_fill", "random_write", "fill", "write"}},
     {"gated_fill", {"gated_fill", "gated_write"}},
     {"skipped_fill", {"skipped_fill", "skipped_write", "gated_fill", "gated_write"}},
     {"random_metadata_fill", {"random_metadata_fill", "random_metadata_write", "metadata_fill", "metadata_write"}},
     {"gated_metadata_fill", {"gated_metadata_fill", "gated_metadata_write"}},
     {"skipped_metadata_fill", {"skipped_metadata_fill", "skipped_metadata_write", "gated_metadata_fill", "gated_metadata_write"}},

     {"random_update", {"random_update", "random_write", "update", "write"}},
     {"gated_update", {"gated_update", "gated_write"}},
     {"skipped_update", {"skipped_update", "skipped_write", "gated_update", "gated_write"}},
     {"random_metadata_update", {"random_metadata_update", "random_metadata_write", "metadata_update", "metadata_write"}},
     {"gated_metadata_update", {"gated_metadata_update", "gated_metadata_write"}},
     {"skipped_metadata_update", {"skipped_metadata_update", "skipped_metadata_write", "gated_metadata_update", "gated_metadata_write"}},

     {"decompression_count", {"decompression_count"}},
     {"compression_count", {"compression_count"}},
     {"leak", {"leak"}},
  };

static std::string bufferClasses[5] = { "DRAM",
                                        "SRAM",
                                        "regfile",
                                        "smartbuffer",
                                        "storage"};

static std::string computeClasses[4] = { "mac",
                                         "intmac",
                                         "fpmac",
                                         "compute" };

// FIXME: derive these from a statically-instantiated list of class names that
// are auto-populated by each Network class at program init time.
static std::string networkClasses[] = { "XY_NoC",
                                        "Legacy",
                                        "ReductionTree",
                                        "SimpleMulticast"};

bool isBufferClass(std::string className);
bool isComputeClass(std::string className);
bool isNetworkClass(std::string className);

class Topology : public Module
{
 public:

  //
  // Specs.
  //
  class Specs
  {
   private:
    std::vector<std::shared_ptr<LevelSpecs>> levels;
    std::vector<std::shared_ptr<LegacyNetwork::Specs>> inferred_networks;
    std::vector<std::shared_ptr<NetworkSpecs>> networks;
    std::map<unsigned, unsigned> storage_map;
    unsigned arithmetic_map;

   public:
    // Constructors and assignment operators.
    Specs() = default;
    ~Specs() = default;

    // We need an explicit deep-copy constructor because of shared_ptrs.
    Specs(const Specs& other)
    {
      for (auto& level_p: other.levels)
        levels.push_back(level_p->Clone());

      for (auto& inferred_network_p: other.inferred_networks)
        inferred_networks.push_back(std::make_shared<LegacyNetwork::Specs>(*inferred_network_p));

      for (auto& network_p: other.networks)
        networks.push_back(network_p->Clone());

      storage_map = other.storage_map;
      arithmetic_map = other.arithmetic_map;
    }

    // Copy-and-swap idiom.
    friend void swap(Specs& first, Specs& second)
    {
      using std::swap;
      swap(first.levels, second.levels);
      swap(first.inferred_networks, second.inferred_networks);
      swap(first.networks, second.networks);
      swap(first.storage_map, second.storage_map);
      swap(first.arithmetic_map, second.arithmetic_map);
    }

    Specs& operator = (Specs other)
    {
      swap(*this, other);
      return *this;
    }

    unsigned NumLevels() const;
    unsigned NumStorageLevels() const;
    unsigned NumNetworks() const;

    std::vector<std::string> LevelNames() const;
    std::vector<std::string> StorageLevelNames() const;

    void ParseAccelergyERT(config::CompoundConfigNode ert);
    void ParseAccelergyART(config::CompoundConfigNode art);

    void AddLevel(unsigned typed_id, std::shared_ptr<LevelSpecs> level_specs);
    void AddInferredNetwork(std::shared_ptr<LegacyNetwork::Specs> specs);
    void AddNetwork(std::shared_ptr<NetworkSpecs> specs);

    unsigned StorageMap(unsigned i) const { return storage_map.at(i); }
    unsigned ArithmeticMap() const { return arithmetic_map; }

    std::shared_ptr<LevelSpecs> GetLevel(unsigned level_id) const;
    std::shared_ptr<BufferLevel::Specs> GetStorageLevel(unsigned storage_level_id) const;
    std::shared_ptr<BufferLevel::Specs> GetStorageLevel(std::string level_name) const;
    std::shared_ptr<ArithmeticUnits::Specs> GetArithmeticLevel() const;
    std::shared_ptr<LegacyNetwork::Specs> GetInferredNetwork(unsigned network_id) const;
    std::shared_ptr<NetworkSpecs> GetNetwork(unsigned network_id) const;
  };

  //
  // Stats.
  //
  struct Stats
  {
    double energy;
    double area;
    std::uint64_t cycles;
    double utilization;
    std::vector<problem::PerDataSpace<std::uint64_t>> tile_sizes;
    std::vector<problem::PerDataSpace<std::uint64_t>> utilized_capacities;
    std::vector<problem::PerDataSpace<std::uint64_t>> utilized_instances;
    std::uint64_t algorithmic_computes;
    std::uint64_t actual_computes;
    std::uint64_t last_level_accesses;
    std::vector<std::uint64_t> accesses;
    // FIXME: the following stat should be typed
    // std::vector<problem::PerDataSpace<std::uint64_t>>. However, we do
    // not yet have a PyBind11 wrapper around PerDataSpace<> in PyTimeloop,
    // which is why we are temporarily using a vector-of-vectors.
    std::vector<std::vector<std::uint64_t>> per_tensor_accesses;

    void Reset()
    {
      energy = 0;
      area = 0;
      cycles = 0;
      utilization = 0;
      tile_sizes.clear();
      utilized_capacities.clear();
      utilized_instances.clear();
      algorithmic_computes = 0;
      actual_computes = 0;
      last_level_accesses = 0;
      accesses.clear();
      per_tensor_accesses.clear();
    }
  };

 private:
  std::vector<std::shared_ptr<Level>> levels_;
  std::map<std::string, std::shared_ptr<Network>> networks_;

  // Maps to store the binding relationship between architectural tiling level
  // to actual micro-architecture. The key here is a temporal/spatial tiling
  // level id, and the value is the pointer to the actual micro-architecture
  // (buffer/network) for the tiling level. We might deprecate the above two
  // members in the future.

  // Level map
  std::map<unsigned, std::shared_ptr<Level>> level_map_;
  // Network map. The pair of network connecting two storage levels should
  // share the same spatial tiling id. Note that these read_fill network and
  // drain_update_network can be the same network.
  struct Connection
  {
    std::shared_ptr<Network> read_fill_network;
    std::shared_ptr<Network> drain_update_network;
  };
  std::map<unsigned, Connection> connection_map_;
  uint64_t total_network_latency_;
  std::map<unsigned, double> tile_area_;

  Specs specs_;
  Stats stats_;

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
  
  // Accessing levels.
  friend Level GetLevel(const Topology& topology, unsigned level_id);
  BufferLevel ExposeStorageLevel(unsigned storage_level_id) const;
  friend ArithmeticUnits GetArithmeticLevel(const Topology& topology);

 private:
  std::shared_ptr<Level> GetLevel(unsigned level_id) const;
  std::shared_ptr<BufferLevel> GetStorageLevel(unsigned storage_level_id) const;
  std::shared_ptr<BufferLevel> GetStorageLevel(std::string level_name) const;
  std::shared_ptr<ArithmeticUnits> GetArithmeticLevel() const;

  void FloorPlan();
  void ComputeStats(bool eval_success);

 public:

  // Constructors and assignment operators.
  Topology() = default;
  ~Topology() = default;

  // We need an explicit deep-copy constructor because of shared_ptrs.
  Topology(const Topology& other)
  {
    is_specced_ = other.is_specced_;
    is_evaluated_ = other.is_evaluated_;

    for (auto& level_p: other.levels_)
      levels_.push_back(level_p->Clone());

    for (auto& network_kv: other.networks_)
      networks_[network_kv.first] = network_kv.second->Clone();

    tile_area_ = other.tile_area_;
    specs_ = other.specs_;
    stats_ = other.stats_;
  }

  // Copy-and-swap idiom.
  friend void swap(Topology& first, Topology& second)
  {
    using std::swap;
    swap(first.is_specced_, second.is_specced_);
    swap(first.is_evaluated_, second.is_evaluated_);
    swap(first.levels_, second.levels_);
    swap(first.networks_, second.networks_);
    swap(first.tile_area_, second.tile_area_);
    swap(first.specs_, second.specs_);
    swap(first.stats_, second.stats_);
  }

  Topology& operator = (Topology other)
  {
    swap(*this, other);
    return *this;
  }

  // The hierarchical ParseSpecs functions are static and do not
  // affect the internal specs_ data structure, which is set by
  // the dynamic Spec() call later.
  static Specs ParseSpecs(config::CompoundConfigNode setting, config::CompoundConfigNode arithmetic_specs, bool is_sparse_topology);
  static Specs ParseTreeSpecs(config::CompoundConfigNode designRoot, bool is_sparse_topology);

  void Spec(const Specs& specs);
  void Reset();
  unsigned NumLevels() const;
  unsigned NumStorageLevels() const;
  unsigned NumNetworks() const;

  std::vector<EvalStatus> PreEvaluationCheck(const Mapping& mapping, analysis::NestAnalysis* analysis, sparse::SparseOptimizationInfo* sparse_optimizations, bool break_on_failure);
  std::vector<EvalStatus> Evaluate(Mapping& mapping, analysis::NestAnalysis* analysis, sparse::SparseOptimizationInfo* sparse_optimizations, bool break_on_failure);

  const Stats& GetStats() const { return stats_; }
  const Specs& GetSpecs() const {return specs_;}

  // FIXME: these stat-specific accessors are deprecated and only exist for
  // backwards-compatibility with some applications.
  double Energy() const { return stats_.energy; }
  double Area() const { return stats_.area; }
  std::uint64_t Cycles() const { return stats_.cycles; }
  double Utilization() const { return stats_.utilization; }
  std::vector<problem::PerDataSpace<std::uint64_t>> TileSizes() const { return stats_.tile_sizes; }
  std::vector<problem::PerDataSpace<std::uint64_t>> UtilizedCapacities() const { return stats_.utilized_capacities; }
  std::vector<problem::PerDataSpace<std::uint64_t>> UtilizedInstances() const { return stats_.utilized_instances; }
  std::uint64_t AlgorithmicComputes() const { return stats_.algorithmic_computes; }
  std::uint64_t ActualComputes() const { return stats_.actual_computes; }
  std::uint64_t LastLevelAccesses() const { return stats_.last_level_accesses; }
  void PrintOAVES(std::ostream& out, Mapping& mapping, bool log_oaves_mappings, std::string oaves_prefix, unsigned thread_id) const;
  void OutputOAVESMappingYAML(Mapping& mapping, std::string map_yaml_file_name) const;

  friend std::ostream& operator<<(std::ostream& out, const Topology& sh);
};

}  // namespace model
