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
#include <boost/serialization/export.hpp>

#include "model/util.hpp"
#include "model/level.hpp"
#include "pat/pat.hpp"

#include "model/network.hpp"

namespace model
{

class ReductionTreeNetwork : public Network
{
 public:

  //
  // Specs.
  //
  struct Specs : public NetworkSpecs
  {
    static const std::uint64_t kDefaultWordBits = 16;

    std::string type;
    Attribute<std::uint64_t> word_bits;
    Attribute<double> adder_energy; // let user overwrite the one in pat
    Attribute<double> wire_energy;

    // Network fill and drain latency
    Attribute<std::uint64_t> fill_latency;
    Attribute<std::uint64_t> drain_latency;

    // Post-floorplanning physical attributes.
    Attribute<double> tile_width; // um

    Attribute<bool> is_sparse_module;
    
    const std::string Type() const override { return type; }
    bool SupportAccelergyTables() const override { return false; }
    void ProcessERT(const config::CompoundConfigNode& ERT) override;
    
    // Serialization
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(NetworkSpecs);
      if (version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(word_bits);
        ar& BOOST_SERIALIZATION_NVP(adder_energy);
        ar& BOOST_SERIALIZATION_NVP(wire_energy);
        ar& BOOST_SERIALIZATION_NVP(tile_width);
        ar& BOOST_SERIALIZATION_NVP(fill_latency);
        ar& BOOST_SERIALIZATION_NVP(drain_latency);
      }
    }

   public:
    std::shared_ptr<NetworkSpecs> Clone() const override
    {
      return std::static_pointer_cast<NetworkSpecs>(std::make_shared<Specs>(*this));
    }

  }; // struct Specs

  struct Stats
  {
    problem::PerDataSpace<AccessStatMatrix> ingresses;
    problem::PerDataSpace<unsigned long> spatial_reductions;
    problem::PerDataSpace<double> num_hops;
    problem::PerDataSpace<double> energy_per_hop;
    problem::PerDataSpace<double> energy;
    problem::PerDataSpace<double> spatial_reduction_energy;

    // Network fill and drain latency, can be set by the spec or inferred from outer buffer
    // network_fill_latency and network_drain_latency
    std::uint64_t fill_latency;
    std::uint64_t drain_latency;

    // Redundant stats with outer buffer.
    problem::PerDataSpace<std::uint64_t> utilized_instances;    

    // Serialization
    friend class boost::serialization::access;
      
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      if (version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(ingresses);
        ar& BOOST_SERIALIZATION_NVP(spatial_reductions);
        ar& BOOST_SERIALIZATION_NVP(num_hops);
        ar& BOOST_SERIALIZATION_NVP(energy_per_hop);
        ar& BOOST_SERIALIZATION_NVP(energy);
        ar& BOOST_SERIALIZATION_NVP(spatial_reduction_energy);
      }
    }      
  }; // struct Stats

  //
  // Data
  //

 private:
  
  Specs specs_;
  std::weak_ptr<Level> source_;
  std::weak_ptr<Level> sink_;

 public:
  Stats stats_; // temporarily public.

  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(Network);
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(specs_);
      ar& BOOST_SERIALIZATION_NVP(stats_);
    }
  }

  //
  // API
  //

 public:

  ReductionTreeNetwork(); // Need this to make Boost happy.
  ReductionTreeNetwork(const Specs& specs);
  ~ReductionTreeNetwork();

  Specs& GetSpecs() { return specs_; }

  std::shared_ptr<Network> Clone() const override
  {
    return std::static_pointer_cast<Network>(std::make_shared<ReductionTreeNetwork>(*this));
  }
  
  static Specs ParseSpecs(config::CompoundConfigNode network, std::size_t n_elements, bool is_sparse_module);

  void ConnectSource(std::weak_ptr<Level> source) override;
  void ConnectSink(std::weak_ptr<Level> sink) override;
  void SetName(std::string name) override;
  std::string Name() const override;
  void AddConnectionType(ConnectionType ct) override;
  void ResetConnectionType() override;

  bool DistributedMulticastSupported() const override;

  // Floorplanner interface.
  void SetTileWidth(double width_um) override;
 
  EvalStatus Evaluate(const tiling::CompoundTile& tile,
                      const bool break_on_failure) override;
  // PAT interface.
  static double WireEnergyPerHop(std::uint64_t word_bits, const double hop_distance, double wire_energy_override);
  static double AdderEnergy(std::uint64_t word_bits, double adder_energy_override);

  void Print(std::ostream& out) const override;

  // Ugly abstraction-breaking probes that should be removed.
  std::uint64_t WordBits() const override;
  std::uint64_t FillLatency() const override;
  std::uint64_t DrainLatency() const override;
  void SetFillLatency(std::uint64_t fill_latency) override;
  void SetDrainLatency(std::uint64_t drain_latency) override;

  STAT_ACCESSOR_HEADER(double, Energy) override;

}; // class ReductionTreeNetwork

} // namespace model

