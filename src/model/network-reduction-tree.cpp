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

ReductionTreeNetwork::Specs ReductionTreeNetwork::ParseSpecs(config::CompoundConfigNode network, std::size_t n_elements)
{
  (void) n_elements; // FIXME.

  Specs specs;

  // Network Type.
  specs.type = "ReductionTree";
  std::string name;
  network.lookupValue("name", name);
  specs.name = name;

  if (network.exists("attributes"))
  {
    network = network.lookup("attributes");
  }

  // Word Bits.
  std::uint32_t word_bits;
  if (network.lookupValue("network-word-bits", word_bits))
  {
    specs.word_bits = word_bits;
  }
  else if (network.lookupValue("word-bits", word_bits) ||
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
  double adder_energy = 0;
  network.lookupValue("adder-energy", adder_energy);
  specs.adder_energy = adder_energy;

  // Wire energy.
  double wire_energy = 0.0;
  network.lookupValue("wire-energy", wire_energy);
  specs.wire_energy = wire_energy;

  return specs;
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

bool ReductionTreeNetwork::DistributedMulticastSupported() const
{
  return false;
}

EvalStatus ReductionTreeNetwork::Evaluate(const tiling::CompoundTile& tile,
                              const double inner_tile_area,
                              const bool break_on_failure,
                              const bool reduction)
{
  (void) tile;
  (void) inner_tile_area;
  (void) break_on_failure;
  assert(reduction);
  auto eval_status = EvalStatus{true, std::string("")};
  std::cout << "ReductionNetwork::Evaluate()" << std::endl;
  return eval_status;
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
  out << indent << indent << "Word bits       : " << specs_.word_bits << std::endl;
  out << indent << indent << "Adder energy    : " << specs_.adder_energy << " pJ" << std::endl;
  out << indent << indent << "Wire energy     : " << specs_.wire_energy << " pJ/b/mm" << std::endl;

  out << std::endl;

  out << indent << "STATS" << std::endl;
  out << indent << "-----" << std::endl;
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    out << indent << problem::GetShape()->DataSpaceIDToName.at(pv) << ":" << std::endl;

    out << indent + indent << "Fanout                                  : "
        << stats_.fanout.at(pv) << std::endl;
      
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

std::uint64_t ReductionTreeNetwork::WordBits() const
{
  return 0;
}

/*
STAT_ACCESSOR(double, ReductionTreeNetwork, NetworkEnergy,
              (stats_.link_transfer_energy.at(pv) + stats_.energy.at(pv)) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, ReductionTreeNetwork, SpatialReductionEnergy,
              stats_.spatial_reduction_energy.at(pv) * stats_.utilized_instances.at(pv))
*/
STAT_ACCESSOR(double, ReductionTreeNetwork, Energy,
              //NetworkEnergy(pv) +
              //SpatialReductionEnergy(pv))
              0.0)

} // namespace model
