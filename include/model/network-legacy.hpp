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

class LegacyNetwork : public Network
{

 public:

  //
  // Specs.
  //
  struct Specs : public NetworkSpecs
  {
    static const std::uint64_t kDefaultWordBits = 16;

    std::string type;
    std::string legacy_subtype;
    Attribute<std::uint64_t> word_bits;

    // Physical attributes.
    Attribute<double> router_energy;
    Attribute<double> wire_energy;

    // Post-floorplanning physical attributes.
    Attribute<double> tile_width; // um
    Attribute<double> energy_per_hop; //pJ

    const std::string Type() const override { return type; }

    // Serialization
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(NetworkSpecs);
      if (version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(word_bits);
        ar& BOOST_SERIALIZATION_NVP(router_energy);
        ar& BOOST_SERIALIZATION_NVP(wire_energy);
        ar& BOOST_SERIALIZATION_NVP(tile_width);
        ar& BOOST_SERIALIZATION_NVP(energy_per_hop);
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
    problem::PerDataSpace<std::uint64_t> fanout;
    problem::PerDataSpace<std::uint64_t> distributed_fanout;
    problem::PerDataSpace<std::uint64_t> multicast_factor;
    problem::PerDataSpace<AccessStatMatrix> ingresses;
    problem::PerDataSpace<bool> distributed_multicast;
    problem::PerDataSpace<unsigned long> link_transfers;
    problem::PerDataSpace<unsigned long> spatial_reductions;
    problem::PerDataSpace<double> link_transfer_energy;
    problem::PerDataSpace<double> num_hops;
    //problem::PerDataSpace<std::vector<double>> avg_hops;
    problem::PerDataSpace<double> energy_per_hop;
    problem::PerDataSpace<double> energy;
    problem::PerDataSpace<double> spatial_reduction_energy;

    // Redundant stats with outer buffer.
    problem::PerDataSpace<std::uint64_t> utilized_instances;    

    // Serialization
    friend class boost::serialization::access;
      
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      if (version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(fanout);
        ar& BOOST_SERIALIZATION_NVP(distributed_fanout);
        ar& BOOST_SERIALIZATION_NVP(multicast_factor);
        //ar& BOOST_SERIALIZATION_NVP(ingresses);
        ar& BOOST_SERIALIZATION_NVP(distributed_multicast);
        ar& BOOST_SERIALIZATION_NVP(link_transfers);
        ar& BOOST_SERIALIZATION_NVP(spatial_reductions);
        ar& BOOST_SERIALIZATION_NVP(link_transfer_energy);
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

  LegacyNetwork(); // Need this to make Boost happy.
  LegacyNetwork(const Specs& specs);
  ~LegacyNetwork();

  std::shared_ptr<Network> Clone() const override
  {
    return std::static_pointer_cast<Network>(std::make_shared<LegacyNetwork>(*this));
  }

  static Specs ParseSpecs(config::CompoundConfigNode network, std::size_t n_elements);

  void ConnectSource(std::weak_ptr<Level> source);
  void ConnectSink(std::weak_ptr<Level> sink);
  void SetName(std::string name);
  std::string Name() const;
  void AddConnectionType(ConnectionType ct);
  void ResetConnectionType();

  bool DistributedMulticastSupported() const;

  // Floorplanner interface.
  void SetTileWidth(double width_um);

  EvalStatus Evaluate(const tiling::CompoundTile& tile,
                      const bool break_on_failure);

  EvalStatus ComputeAccesses(const tiling::CompoundDataMovementInfo& tile, const bool break_on_failure);
  void ComputeNetworkEnergy();
  void ComputeSpatialReductionEnergy();
  void ComputePerformance();

  std::uint64_t WordBits() const;

  void Print(std::ostream& out) const;

  // PAT interface.
  static double WireEnergyPerHop(std::uint64_t word_bits, const double hop_distance, double wire_energy_override);
  static double NumHops(std::uint32_t multicast_factor, std::uint32_t fanout);

  STAT_ACCESSOR_HEADER(double, NetworkEnergy);
  STAT_ACCESSOR_HEADER(double, SpatialReductionEnergy);
  STAT_ACCESSOR_HEADER(double, Energy);

}; // class Network

} // namespace model

