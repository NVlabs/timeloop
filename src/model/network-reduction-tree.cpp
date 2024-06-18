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

#include <iostream>

#include "model/util.hpp"
#include "model/level.hpp"
#include "pat/pat.hpp"

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "model/network-reduction-tree.hpp"
BOOST_CLASS_EXPORT(model::ReductionTreeNetwork)

bool gHideInconsequentialStatsNetworkReductionTree =
  (getenv("TIMELOOP_HIDE_INCONSEQUENTIAL_STATS") == NULL) ||
  (strcmp(getenv("TIMELOOP_HIDE_INCONSEQUENTIAL_STATS"), "0") != 0);

namespace model
{

ReductionTreeNetwork::ReductionTreeNetwork() // Need this to make Boost happy.
{ }

ReductionTreeNetwork::ReductionTreeNetwork(const Specs& specs) :
    specs_(specs)
{
  is_specced_ = true;
  is_evaluated_ = false;
}

ReductionTreeNetwork::~ReductionTreeNetwork()
{ }

ReductionTreeNetwork::Specs ReductionTreeNetwork::ParseSpecs(config::CompoundConfigNode network, std::size_t n_elements, bool is_sparse_module)
{
  (void) n_elements; // FIXME.

  Specs specs;

  // Network Type.
  specs.type = "ReductionTree";
  std::string name;
  network.lookupValue("name", name);
  specs.name = name;

  // Sparse Architecture's Module 
  specs.is_sparse_module = is_sparse_module;

  if (network.exists("attributes"))
  {
    network = network.lookup("attributes");
  }

  // Word Bits.
  std::uint32_t word_bits;
  if (network.lookupValue("network_word_bits", word_bits))
  {
    specs.word_bits = word_bits;
  }
  else if (network.lookupValue("word_bits", word_bits) ||
           network.lookupValue("word_width", word_bits) ||
           network.lookupValue("datawidth", word_bits) )
  {
    // FIXME. Derive this from the buffer I'm connected to in the topology
    // instead of cheating and reading it directly from its specs config.
    specs.word_bits = word_bits;
  }
  else
  {
    specs.word_bits = Specs::kDefaultWordBits;
  }

  // adder energy.
  double adder_energy = 0.0;
  network.lookupValue("adder_energy", adder_energy);
  specs.adder_energy = adder_energy;

  // Wire energy.
  double wire_energy = 0.0;
  network.lookupValue("wire_energy", wire_energy);
  specs.wire_energy = wire_energy;

  // Network fill and drain latency
  unsigned long long fill_latency;
  if (network.lookupValue("fill_latency", fill_latency))
  {
    specs.fill_latency = fill_latency;
  }
  else
  {
    specs.fill_latency = 0;
  }

  unsigned long long drain_latency;
  if (network.lookupValue("drain_latency", drain_latency))
  {
    specs.drain_latency = drain_latency;
  }
  else
  {
    specs.drain_latency = 0;
  }

  return specs;
}

void ReductionTreeNetwork::Specs::ProcessERT(const config::CompoundConfigNode& ERT)
{
  (void) ERT;
  assert(false);
}


void ReductionTreeNetwork::ConnectSource(std::weak_ptr<Level> source)
{
  source_ = source;
}

void ReductionTreeNetwork::ConnectSink(std::weak_ptr<Level> sink)
{
  sink_ = sink;
}

void ReductionTreeNetwork::SetName(std::string name)
{
  specs_.name = name;
}

std::string ReductionTreeNetwork::Name() const
{
  return specs_.name;
}

void ReductionTreeNetwork::AddConnectionType(ConnectionType ct)
{
  specs_.cType = static_cast<ConnectionType>(static_cast<int>(specs_.cType) | static_cast<int>(ct));
}

void ReductionTreeNetwork::ResetConnectionType()
{
  specs_.cType = Unused;
}


bool ReductionTreeNetwork::DistributedMulticastSupported() const
{
  return false;
}

// Floorplanner interface.
void ReductionTreeNetwork::SetTileWidth(double width_um)
{
  // Only set this if user didn't specify a pre-floorplanned tile width.
  if (!specs_.tile_width.IsSpecified() || specs_.tile_width.Get() == 0.0)
  {
    specs_.tile_width = width_um;
  }
}

EvalStatus ReductionTreeNetwork::Evaluate(const tiling::CompoundTile& tile,
                                          problem::Workload* workload,
                                          const bool break_on_failure)
{
  workload_ = workload;
  
  tiling::CompoundDataMovementInfo data_movement = tile.data_movement_info;

  (void) break_on_failure;
  assert(specs_.cType == UpdateDrain); // ReductionTreeNetwork can only be used in update-drain connection

  // Get stats from the CompoundTile
  for (unsigned pvi = 0; pvi < unsigned(workload_->GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
      
    stats_.utilized_instances[pv] = data_movement[pvi].replication_factor;

    if (workload_->GetShape()->IsReadWriteDataSpace.at(pv))
    {
      stats_.ingresses[pv] = data_movement[pvi].access_stats;
      for (auto& x: stats_.ingresses[pv].stats)
      {
        auto multicast = x.first.first;
        stats_.spatial_reductions[pv] += (multicast-1) * x.second.accesses;
      }
    }
    else // Read-only data, all zeros
    {
      stats_.ingresses[pv].clear();
    }
  } 

  // Calculate energy
  for (unsigned pvi = 0; pvi < unsigned(workload_->GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);

    double energy_per_hop =
      WireEnergyPerHop(specs_.word_bits.Get(), specs_.tile_width.Get(), specs_.wire_energy.Get());
    double total_wire_hops = 0;
    double total_ingresses = 0;

    for (auto& x: stats_.ingresses.at(pv).stats)
    {
      auto reduction_factor = x.first.first;
      // auto scatter_factor = x.first.second;
      auto ingresses = x.second.accesses;
      // auto hops = x.second.hops;

      total_ingresses += ingresses;
      double num_hops = 0;
      if (ingresses > 0)
      {
        if (workload_->GetShape()->IsReadWriteDataSpace.at(pv))
        {
          // Modeling the reduction tree here!
          num_hops = std::floor(std::log2(reduction_factor)) * 0.5;
        }
      }
      total_wire_hops += num_hops * ingresses;
    }

    if (workload_->GetShape()->IsReadWriteDataSpace.at(pv))
    {
      stats_.energy_per_hop[pv] = energy_per_hop;
      stats_.num_hops[pv] = total_ingresses > 0 ? total_wire_hops / total_ingresses : 0;
      stats_.energy[pv] = total_wire_hops * energy_per_hop;
      stats_.spatial_reduction_energy[pv] = stats_.spatial_reductions[pv] *
          AdderEnergy(specs_.word_bits.Get(), specs_.adder_energy.Get());
    }
  }

  auto eval_status = EvalStatus{true, std::string("")};
  is_evaluated_ = true;
  // std::cout << "ReductionNetwork::Evaluate()" << std::endl;

  return eval_status;
}

// FIXME: Should merge this back to the common abstract Network class
// PAT interface.
//
double ReductionTreeNetwork::WireEnergyPerHop(std::uint64_t word_bits,
                                              const double hop_distance,
                                              double wire_energy_override)
{
  double hop_distance_mm = hop_distance / 1000;
  if (wire_energy_override != 0.0)
  {
    // Internal wire model using user-provided average wire_energy/b/mm.
    return word_bits * hop_distance_mm * wire_energy_override;
  }
  else
  {
    return pat::WireEnergy(word_bits, hop_distance_mm);
  }
}

double ReductionTreeNetwork::AdderEnergy(std::uint64_t word_bits, double adder_energy_override)
{
  if (adder_energy_override != 0.0)
  {
    // Use user-provided adder energy.
    return adder_energy_override;
  }
  else
  {
    return pat::AdderEnergy(word_bits, word_bits);
  }
}

void ReductionTreeNetwork::Print(std::ostream& out) const
{
  // Print network name.
  out << specs_.name << std::endl;  
  out << std::endl;

  std::string indent = "    ";

  out << indent << "SPECS" << std::endl;
  out << indent << "-----" << std::endl;

  out << indent << indent << "Type            : " << specs_.type << std::endl;
  out << indent << indent << "ConnectionType  : " << specs_.cType << std::endl;
  out << indent << indent << "Word bits       : " << specs_.word_bits << std::endl;
  if (specs_.adder_energy.Get() != 0.0)
  {
    out << indent << indent << "Adder energy    : " << specs_.adder_energy << " pJ" << std::endl;
  }
  if (specs_.wire_energy.Get() != 0.0)
  {
    out << indent << indent << "Wire energy     : " << specs_.wire_energy << " pJ/b/mm" << std::endl;
  }
  out << indent << indent << "Fill latency     : " << stats_.fill_latency << std::endl;
  out << indent << indent << "Drain latency     : " << stats_.drain_latency << std::endl;


  out << std::endl;

  out << indent << "STATS" << std::endl;
  out << indent << "-----" << std::endl;
  for (unsigned pvi = 0; pvi < unsigned(workload_->GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    if(gHideInconsequentialStatsNetworkReductionTree && stats_.spatial_reductions.at(pv) == 0) continue;
    out << indent << workload_->GetShape()->DataSpaceIDToName.at(pv) << ":" << std::endl;

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
    out << indent + indent << "Spatial Reduction Energy (per-instance) : "
        << stats_.spatial_reduction_energy.at(pv) << " pJ" << std::endl;
    out << indent + indent << "Spatial Reduction Energy (total)        : "
        << stats_.spatial_reduction_energy.at(pv) * stats_.utilized_instances.at(pv)
        << " pJ" << std::endl;
  }

  out << std::endl;

}

std::uint64_t ReductionTreeNetwork::WordBits() const
{
  return 0;
}

std::uint64_t ReductionTreeNetwork::FillLatency() const
{
  assert(is_specced_);
  return specs_.fill_latency.Get();
}

std::uint64_t ReductionTreeNetwork::DrainLatency() const
{
  assert(is_specced_);
  return specs_.fill_latency.Get();
}

void ReductionTreeNetwork::SetFillLatency(std::uint64_t fill_latency)
{
  stats_.fill_latency = fill_latency;
}

void ReductionTreeNetwork::SetDrainLatency(std::uint64_t drain_latency)
{
  stats_.drain_latency = drain_latency;
}

/*
STAT_ACCESSOR(double, ReductionTreeNetwork, NetworkEnergy,
              (stats_.link_transfer_energy.at(pv) + stats_.energy.at(pv)) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, ReductionTreeNetwork, SpatialReductionEnergy,
              stats_.spatial_reduction_energy.at(pv) * stats_.utilized_instances.at(pv))
*/
STAT_ACCESSOR(double, ReductionTreeNetwork, Energy,
              stats_.energy.at(pv) + stats_.spatial_reduction_energy.at(pv) )

} // namespace model
