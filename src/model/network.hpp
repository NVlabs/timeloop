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

namespace model
{

class Network
{

 public:

  //
  // Specs.
  //
  struct Specs
  {
    static const std::uint64_t kDefaultWordBits = 16;

    enum class NetworkType
    {
      OneToOne,
      OneToMany,
      ManyToMany
    };

    DataSpaceIDSharing sharing_type;

    PerDataSpaceOrShared<Attribute<NetworkType>> type;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> word_bits;
    PerDataSpaceOrShared<Attribute<double>> routerEnergy;
    PerDataSpaceOrShared<Attribute<double>> wireEnergy;

    // Serialization
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      if (version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(type);
        ar& BOOST_SERIALIZATION_NVP(word_bits);
        ar& BOOST_SERIALIZATION_NVP(routerEnergy);
        ar& BOOST_SERIALIZATION_NVP(wireEnergy);
      }
    }

    Specs() :
        sharing_type(DataSpaceIDSharing::Shared)
    {
      Init();
    }

    Specs(DataSpaceIDSharing sharing) :
        sharing_type(sharing)
    {
      Init();
    }

    void Init()
    {
      // Initialize all parameters to "unspecified" default.
      if (sharing_type == DataSpaceIDSharing::Partitioned)
      {
        type.SetPerDataSpace();
        word_bits.SetPerDataSpace();
        routerEnergy.SetPerDataSpace();
        wireEnergy.SetPerDataSpace();
      }
      else // sharing_type == DataSpaceIDSharing::Shared
      {
        type.SetShared();
        word_bits.SetShared();
        routerEnergy.SetShared();
        wireEnergy.SetShared();        
      }
    }

    DataSpaceIDSharing SharingType() const { return sharing_type; }

    unsigned DataSpaceIDIteratorStart() const
    {
      return sharing_type == DataSpaceIDSharing::Shared
                             ? unsigned(problem::GetShape()->NumDataSpaces)
                             : 0;
    }

    unsigned DataSpaceIDIteratorEnd() const
    {
      return sharing_type == DataSpaceIDSharing::Shared
                             ? unsigned(problem::GetShape()->NumDataSpaces) + 1
                             : unsigned(problem::GetShape()->NumDataSpaces);
    }

    size_t NumPartitions() const
    {
      return sharing_type == DataSpaceIDSharing::Shared
                             ? 1
                             : size_t(problem::GetShape()->NumDataSpaces);
    }

    ADD_ACCESSORS(Type, type, NetworkType)
    ADD_ACCESSORS(WordBits, word_bits, std::uint64_t)
    ADD_ACCESSORS(RouterEnergy, routerEnergy, double)
    ADD_ACCESSORS(WireEnergy, wireEnergy, double)

    friend std::ostream& operator << (std::ostream& out, const Specs& specs)
    {
      std::string indent = "    ";

      unsigned start_pv = specs.DataSpaceIDIteratorStart();
      unsigned end_pv = specs.DataSpaceIDIteratorEnd();

      for (unsigned pvi = start_pv; pvi < end_pv; pvi++)
      {
        auto pv = problem::Shape::DataSpaceID(pvi);
        
        if (pv == problem::GetShape()->NumDataSpaces)
          out << indent << "Shared:" << std::endl;
        else
          out << indent << problem::GetShape()->DataSpaceIDToName.at(pv) << ":" << std::endl;      
      }

      return out;
    }

  }; // struct Specs

  struct Stats
  {
    problem::PerDataSpace<std::uint64_t> fanout;
    problem::PerDataSpace<std::uint64_t> distributed_fanout;
    problem::PerDataSpace<std::uint64_t> multicast_factor;
    problem::PerDataSpace<std::vector<unsigned long>> ingresses;
    problem::PerDataSpace<bool> distributed_multicast;
    problem::PerDataSpace<unsigned long> link_transfers;
    problem::PerDataSpace<unsigned long> spatial_reductions;
    problem::PerDataSpace<double> link_transfer_energy;
    problem::PerDataSpace<double> num_hops;
    problem::PerDataSpace<std::vector<double>> avg_hops;
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
        ar& BOOST_SERIALIZATION_NVP(ingresses);
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
  std::shared_ptr<Level> outer_ = nullptr;

 public:
  Stats stats_; // temporarily public.

  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    // ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Level);
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

  Network()
  { }

  Network(const Specs& specs) :
      specs_(specs)
  { }

  ~Network()
  { }

  static Specs ParseSpecs(config::CompoundConfigNode network)
  {
    Specs specs(DataSpaceIDSharing::Shared);

    // Network Type.
    std::string network_type;
    if (network.lookupValue("network-type", network_type))
    {
      if (network_type.compare("1:1") == 0)
        specs.Type() = Network::Specs::NetworkType::OneToOne;
      else if (network_type.compare("1:N") == 0)
        specs.Type() = Network::Specs::NetworkType::OneToMany;
      else if (network_type.compare("M:N") == 0)
        specs.Type() = Network::Specs::NetworkType::ManyToMany;
      else
      {
        std::cerr << "ERROR: Unrecognized network type: " << network_type << std::endl;
        exit(1);
      }
    }
    
    // Word Bits.
    std::uint32_t word_bits;
    if (network.lookupValue("network-word-bits", word_bits))
    {
      specs.WordBits() = word_bits;
    }
    else if (network.lookupValue("word-bits", word_bits) ||
             network.lookupValue("word_width", word_bits) ||
             network.lookupValue("datawidth", word_bits) )
    {
      // FIXME. Derive this from the buffer I'm connected to in the topology
      // instead of cheating and reading it directly from its specs config.
      specs.WordBits() = word_bits;
    }
    else
    {
      specs.WordBits() = Specs::kDefaultWordBits;
    }

    // Router energy.
    double router_energy = 0;
    network.lookupValue("router-energy", router_energy);
    specs.RouterEnergy() = router_energy;

    // Wire energy.
    double wire_energy = 0.0;
    network.lookupValue("wire-energy", wire_energy);
    specs.WireEnergy() = wire_energy;

    return specs;
  }

  const Specs& GetSpecs() const
  {
    return specs_;
  }

  void Connect(std::shared_ptr<Level> outer)
  {
    outer_ = outer;
  }

  bool DistributedMulticastSupported()
  {
    bool retval = true;

    for (unsigned pvi = 0; pvi < specs_.NumPartitions(); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      retval &= (specs_.Type(pv).IsSpecified() &&
                 specs_.Type(pv).Get() == Network::Specs::NetworkType::ManyToMany);
    }

    return retval;
  }

  EvalStatus Evaluate(const tiling::CompoundTile& tile,
                      const double inner_tile_area,
                      const bool break_on_failure)
  {
    auto eval_status = ComputeAccesses(tile, break_on_failure);
    if (!break_on_failure || eval_status.success)
    {
      ComputeNetworkEnergy(inner_tile_area);
      ComputeSpatialReductionEnergy();
      ComputePerformance();
    }
    return eval_status;
  }

  EvalStatus ComputeAccesses(const tiling::CompoundTile& tile, const bool break_on_failure)
  {
    bool success = true;
    std::ostringstream fail_reason;

    //
    // 1. Collect stats (stats are always collected per-DataSpaceID).
    //
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      
      stats_.utilized_instances[pv] = tile[pvi].replication_factor;      

      if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
      {
        // Network access-count calculation for Read-Modify-Write datatypes depends on
        // whether the unit receiving a Read-Write datatype has the ability to do
        // a Read-Modify-Write (e.g. accumulate) locally. If the unit isn't capable
        // of doing this, we need to account for additional network traffic.

        // FIXME: need to account for the case when this level is bypassed. In this
        //        case we'll have to query a different level. Also size will be 0,
        //        we may have to maintain a network_size.
        if (outer_->HardwareReductionSupported(pv))
        {
          stats_.ingresses[pv] = tile[pvi].accesses;
        }
        else
        {
          stats_.ingresses[pv].resize(tile[pvi].accesses.size());
          for (unsigned i = 0; i < tile[pvi].accesses.size(); i++)
          {
            if (tile[pvi].accesses[i] > 0)
            {
              assert(tile[pvi].size == 0 || tile[pvi].accesses[i] % tile[pvi].size == 0);
              stats_.ingresses[pv][i] = 2*tile[pvi].accesses[i] - tile[pvi].partition_size;
            }
            else
            {
              stats_.ingresses[pv][i] = 0;
            }
          }
        } // hardware reduction not supported
      }
      else // Read-only data space.
      {
        stats_.ingresses[pv] = tile[pvi].accesses;
      }

      stats_.spatial_reductions[pv] = 0;
      stats_.distributed_multicast[pv] = tile[pvi].distributed_multicast;
      stats_.avg_hops[pv].resize(tile[pvi].accesses.size());
      for (unsigned i = 0; i < tile[pvi].accesses.size(); i++)
      {
        if (tile[pvi].accesses[i] > 0)
        {
          stats_.avg_hops[pv][i] = tile[pvi].cumulative_hops[i] / double(tile[pvi].scatter_factors[i]);
        }
      }
    
      // FIXME: issues with link-transfer modeling:
      // 1. link transfers should result in buffer accesses to a peer.
      // 2. should reductions via link transfers be counted as spatial or temporal?
      stats_.link_transfers[pv] = tile[pvi].link_transfers;
      if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
      {
        stats_.spatial_reductions[pv] += tile[pvi].link_transfers;
      }

      stats_.fanout[pv] = tile[pvi].fanout;
      if (stats_.distributed_multicast.at(pv))
        stats_.distributed_fanout[pv] = tile[pvi].distributed_fanout;
      else
        stats_.distributed_fanout[pv] = 0;

      // FIXME: multicast factor can be heterogeneous. This is correctly
      // handled by energy calculations, but not correctly reported out
      // in the stats.
      stats_.multicast_factor[pv] = 0;

      for (unsigned i = 0; i < stats_.ingresses[pv].size(); i++)
      {
        if (stats_.ingresses[pv][i] > 0)
        {
          auto factor = i + 1;
          if (factor > stats_.multicast_factor[pv])
          {
            stats_.multicast_factor[pv] = factor;
          }
          if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
          {
            stats_.spatial_reductions[pv] += (i * stats_.ingresses[pv][i]);
          }
        }
      }

    } // loop over pv.
    
    //
    // 2. Derive/validate architecture specs based on stats.
    //
    if (specs_.sharing_type == DataSpaceIDSharing::Partitioned)
    {
      for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
      {
        // Bandwidth constraints cannot be checked/inherited at this point
        // because the calculation is a little more involved. We will do
        // this later in the ComputePerformance() function.
      }
    }
    else  // sharing_type == DataSpaceIDSharing::Shared
    {
      // Bandwidth constraints cannot be checked/inherited at this point
      // because the calculation is a little more involved. We will do
      // this later in the ComputePerformance() function.    
    }

    (void) break_on_failure;

    EvalStatus eval_status;
    eval_status.success = success;
    eval_status.fail_reason = fail_reason.str();
    
    return eval_status;

  } // ComputeAccesses()

  //
  // Compute network energy.
  //
  void ComputeNetworkEnergy(const double inner_tile_area)
  {
#define PROBABILISTIC_MULTICAST 0
#define PRECISE_MULTICAST 1
#define EYERISS_HACK_MULTICAST 2  

#define MULTICAST_MODEL PROBABILISTIC_MULTICAST
  
    // NOTE! Stats are always maintained per-DataSpaceID
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);

      double energy_per_hop =
        WireEnergyPerHop(specs_.WordBits(pv).Get(), inner_tile_area);
      double energy_wire = specs_.WireEnergy(pv).Get();
      if (energy_wire != 0.0) { // user provided energy per wire length per bit
        energy_per_hop = specs_.WordBits(pv).Get() * inner_tile_area * energy_wire;
      }
      double energy_per_router = specs_.RouterEnergy(pv).Get();
    
      auto fanout = stats_.distributed_multicast.at(pv) ?
        stats_.distributed_fanout.at(pv) :
        stats_.fanout.at(pv);

      double total_wire_hops = 0;
      std::uint64_t total_routers_touched = 0;
      double total_ingresses = 0;
    
      for (unsigned i = 0; i < stats_.ingresses[pv].size(); i++)
      {
        auto ingresses = stats_.ingresses.at(pv).at(i);
        total_ingresses += ingresses;
        if (ingresses > 0)
        {
          auto multicast_factor = i + 1;

#if MULTICAST_MODEL == PROBABILISTIC_MULTICAST

          auto num_hops = NumHops(multicast_factor, fanout);
          total_routers_touched += (1 + num_hops) * ingresses;

#elif MULTICAST_MODEL == PRECISE_MULTICAST

          (void)fanout;
          (void)multicast_factor;
          if (stats_.distributed_multicast.at(pv))
          {
            std::cerr << "ERROR: precise multicast calculation does not work with distributed multicast." << std::endl;
            exit(1);
          }
          auto num_hops = stats_.avg_hops.at(pv).at(i);
          total_routers_touched += (1 + std::uint64_t(std::floor(num_hops))) * ingresses;

#elif MULTICAST_MODEL == EYERISS_HACK_MULTICAST

          (void)fanout;
          unsigned num_hops = 0;
        
          // Weights are multicast, and energy is already captured in array access.
          if (pv != problem::Shape::DataSpaceID::Weight)
          {
            // Input and Output activations are forwarded between neighboring PEs,
            // so the number of link transfers is equal to the multicast factor-1.
            num_hops = multicast_factor - 1;
          }
        
          // We pick up the router energy from the .cfg file as the "array" energy
          // as defined in the Eyeriss paper, so we don't add a 1 to the multicast
          // factor.
          total_routers_touched += num_hops * ingresses;

#else
#error undefined MULTICAST_MODEL
#endif        

          total_wire_hops += num_hops * ingresses;
        }
      }

      stats_.energy_per_hop[pv] = energy_per_hop;
      stats_.num_hops[pv] = total_ingresses > 0 ? total_wire_hops / total_ingresses : 0;
      stats_.energy[pv] =
        total_wire_hops * energy_per_hop + // wire energy
        total_routers_touched * energy_per_router; // router energy

      stats_.link_transfer_energy[pv] =
        stats_.link_transfers.at(pv) * (energy_per_hop + 2*energy_per_router);
    }
  }

  //
  // Compute spatial reduction energy.
  //
  void ComputeSpatialReductionEnergy()
  {
    // Spatial reduction: add two values in the 
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
      {
        stats_.spatial_reduction_energy[pv] = stats_.spatial_reductions[pv] * 
          pat::AdderEnergy(specs_.WordBits(pv).Get(), specs_.WordBits(pv).Get());
      }
      else
      {
        stats_.spatial_reduction_energy[pv] = 0;
      }
    }    
  }

  void ComputePerformance()
  {
    // FIXME.
    // problem::PerDataSpace<double> unconstrained_read_bandwidth;
    // problem::PerDataSpace<double> unconstrained_write_bandwidth;
    // for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    // {
    //   auto pv = problem::Shape::DataSpaceID(pvi);
    //   auto total_ingresses =
    //     std::accumulate(network_.stats_.ingresses.at(pv).begin(),
    //                     network_.stats_.ingresses.at(pv).end(), static_cast<std::uint64_t>(0));
    // }
  }

  std::uint64_t MaxFanout() const
  {
    // FIXME: remove this function, it's used only once.
    return stats_.fanout.Max();
  }

  //
  // Printers.
  //
  void PrintSpecs(std::ostream& out) const
  {
    std::string indent = "    ";

    out << indent << "NETWORK SPECS" << std::endl;
    out << indent << "-------------" << std::endl;
    out << specs_;

    out << std::endl;
  }
    
  void PrintStats(std::ostream& out) const
  {
    std::string indent = "    ";

    out << indent << "NETWORK STATS" << std::endl;
    out << indent << "-------------" << std::endl;
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      out << indent << problem::GetShape()->DataSpaceIDToName.at(pv) << ":" << std::endl;

      out << indent + indent << "Fanout                                  : "
          << stats_.fanout.at(pv) << std::endl;
      out << indent + indent << "Fanout (distributed)                    : "
          << stats_.distributed_fanout.at(pv) << std::endl;
      if (stats_.distributed_multicast.at(pv))
        out << indent + indent << "Multicast factor (distributed)          : ";
      else
        out << indent + indent << "Multicast factor                        : ";
      out << stats_.multicast_factor.at(pv) << std::endl;
      
      auto total_accesses =
        std::accumulate(stats_.ingresses.at(pv).begin(),
                        stats_.ingresses.at(pv).end(),
                        static_cast<std::uint64_t>(0));
      out << indent + indent << "Ingresses                               : " << total_accesses << std::endl;
    
      std::string mcast_type = "@multicast ";
      if (stats_.distributed_multicast.at(pv))
        mcast_type += "(distributed) ";
      for (std::uint64_t i = 0; i < stats_.ingresses.at(pv).size(); i++)
        if (stats_.ingresses.at(pv)[i] != 0)
          out << indent + indent + indent << mcast_type << i + 1 << ": "
              << stats_.ingresses.at(pv)[i] << std::endl;

      out << indent + indent << "Link transfers                          : "
          << stats_.link_transfers.at(pv) << std::endl;
      out << indent + indent << "Spatial reductions                      : "
          << stats_.spatial_reductions.at(pv) << std::endl;
    
      out << indent + indent << "Average number of hops                  : "
          << stats_.num_hops.at(pv) << std::endl;
    
      out << indent + indent << "Energy (per-hop)                        : "
          << stats_.energy_per_hop.at(pv)*1000 << " fJ" << std::endl;

      out << indent + indent << "Energy (per-instance)                   : "
          << stats_.energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Energy (total)                          : "
          << stats_.energy.at(pv) * stats_.utilized_instances.at(pv)
          << " pJ" << std::endl;
      out << indent + indent << "Link transfer energy (per-instance)     : "
          << stats_.link_transfer_energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Link transfer energy (total)            : "
          << stats_.link_transfer_energy.at(pv) * stats_.utilized_instances.at(pv)
          << " pJ" << std::endl;    
      out << indent + indent << "Spatial Reduction Energy (per-instance) : "
          << stats_.spatial_reduction_energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Spatial Reduction Energy (total)        : "
          << stats_.spatial_reduction_energy.at(pv) * stats_.utilized_instances.at(pv)
          << " pJ" << std::endl;
    }
  
    out << std::endl;
  }

  void Print(std::ostream& out) const
  {
    PrintSpecs(out);
    PrintStats(out);
  }

  friend std::ostream& operator << (std::ostream& out, const Network& network)
  {
    network.Print(out);
    return out;
  }

  //
  // PAT interface.
  //
  static double WireEnergyPerHop(std::uint64_t word_bits, const double inner_tile_area)
  {
    // Assuming square modules
    double inner_tile_width = std::sqrt(inner_tile_area);  // um
    inner_tile_width /= 1000;                              // mm
    return pat::WireEnergy(word_bits, inner_tile_width);
  }

  static double NumHops(std::uint32_t multicast_factor, std::uint32_t fanout)
  {
    // Assuming central/side entry point.
    double root_f = std::sqrt(multicast_factor);
    double root_n = std::sqrt(fanout);
    return (root_n*root_f) + 0.5*(root_n-root_f) - (root_n/root_f) + 0.5;
    // return (root_n*root_f);
  }

  //
  // Accessors.
  //

  STAT_ACCESSOR_INLINE(double, NetworkEnergy,
                       (stats_.link_transfer_energy.at(pv) + stats_.energy.at(pv)) * stats_.utilized_instances.at(pv))
  STAT_ACCESSOR_INLINE(double, SpatialReductionEnergy,
                       stats_.spatial_reduction_energy.at(pv) * stats_.utilized_instances.at(pv))
  
  STAT_ACCESSOR_INLINE(double, Energy,
                       NetworkEnergy(pv) +
                       SpatialReductionEnergy(pv))

}; // class Network

} // namespace model
