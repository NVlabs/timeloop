/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "model/network-simple-multicast.hpp"
BOOST_CLASS_EXPORT(model::SimpleMulticastNetwork)

namespace model
{

SimpleMulticastNetwork::SimpleMulticastNetwork() // Need this to make Boost happy.
{ }

SimpleMulticastNetwork::SimpleMulticastNetwork(const Specs& specs) :
    specs_(specs)
{
  is_specced_ = true;
  is_evaluated_ = false;
}

SimpleMulticastNetwork::~SimpleMulticastNetwork()
{ }

SimpleMulticastNetwork::Specs SimpleMulticastNetwork::ParseSpecs(config::CompoundConfigNode network, std::size_t n_elements, bool is_sparse_module)
{
  (void) n_elements; // FIXME.

  Specs specs;

  // Network Type.
  specs.type = "SimpleMulticast";
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

  // user-defined action name
  std::string action_name;
  if (network.lookupValue("action_name", action_name)){
     specs.action_name = action_name;
  } else {
     std::cerr << "must specify the multicast action_name to look for in ERT" << std::endl;
     assert(false); // FIXME: perform the default multicast energy calculations
  }

  // user-defined argument name corresponding to the multicast factor
  std::string multicast_factor_argument;
  if (network.lookupValue("multicast_factor_argument", multicast_factor_argument)){
     specs.multicast_factor_argument = multicast_factor_argument;
  } else {
     specs.multicast_factor_argument = "none";
  }

  // whether ERT specification is in terms of data types
  bool per_datatype_ERT;
  if (network.lookupValue("per_datatype_ERT", per_datatype_ERT)){
    specs.per_datatype_ERT = true;
  } else {
    specs.per_datatype_ERT = false;
  }

  return specs;
}

void SimpleMulticastNetwork::ConnectSource(std::weak_ptr<Level> source)
{
  source_ = source;
}

void SimpleMulticastNetwork::ConnectSink(std::weak_ptr<Level> sink)
{
  sink_ = sink;
}

void SimpleMulticastNetwork::SetName(std::string name)
{
  specs_.name = name;
}

std::string SimpleMulticastNetwork::Name() const
{
  return specs_.name;
}

void SimpleMulticastNetwork::AddConnectionType(ConnectionType ct)
{
  specs_.cType = static_cast<ConnectionType>(static_cast<int>(specs_.cType) | static_cast<int>(ct));
}

void SimpleMulticastNetwork::ResetConnectionType()
{
  specs_.cType = Unused;
}


bool SimpleMulticastNetwork::DistributedMulticastSupported() const
{
  return false;
}

// Floorplanner interface.
void SimpleMulticastNetwork::SetTileWidth(double width_um)
{
  // Only set this if user didn't specify a pre-floorplanned tile width.
  if (!specs_.tile_width.IsSpecified() || specs_.tile_width.Get() == 0.0)
  {
    specs_.tile_width = width_um;
  }
}

double SimpleMulticastNetwork::GetOpEnergyFromERT(std::uint64_t multicast_factor, std::string operation_name){
    double opEnergy = 0.0;
    std::vector<std::string> actions;
    specs_.accelergyERT.getMapKeys(actions);
    // use transfer as the keyword for multicast NoC action specification
    if (specs_.accelergyERT.exists(operation_name)){
        auto actionERT = specs_.accelergyERT.lookup(operation_name);
        if (actionERT.isList()){
            assert(specs_.multicast_factor_argument != "none"); // must have multicast factor argument name specified
            for(int i = 0; i < actionERT.getLength(); i ++){
                config::CompoundConfigNode arguments = actionERT[i].lookup("arguments");
                unsigned int num_destinations;
                // use num_destinations as a keyword to perform ERT search (might be updated)
                arguments.lookupValue(specs_.multicast_factor_argument, num_destinations);
                if (num_destinations == multicast_factor){
                  // std::cout << "found correct num destinations" << std::endl;
                  actionERT[i].lookupValue("energy", opEnergy);
                }
             }
        } else {
            // if there is no argument, use the available energy
            actionERT.lookupValue("energy", opEnergy);
        }
    } 
    return opEnergy;
}

double SimpleMulticastNetwork::GetMulticastEnergy(std::uint64_t multicast_factor){
    std::string operation_name = specs_.action_name;
    double opEnergy = GetOpEnergyFromERT(multicast_factor, operation_name);
    return opEnergy;
}

// Parse ERT to get multi-casting energy
double SimpleMulticastNetwork::GetMulticastEnergyByDataType(std::uint64_t multicast_factor, std::string data_space_name){
    std::string operation_name = specs_.action_name + "_" + data_space_name;
    double opEnergy = GetOpEnergyFromERT(multicast_factor, operation_name);
    return opEnergy;
}

EvalStatus SimpleMulticastNetwork::Evaluate(const tiling::CompoundTile& tile,
                              const bool break_on_failure)
{
  (void) tile;
  (void) break_on_failure;

  tiling::CompoundDataMovementInfo data_movement = tile.data_movement_info;
  
  // Get stats from the CompoundTile
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    stats_.utilized_instances[pv] = data_movement[pvi].replication_factor;
    stats_.fanout[pv] = data_movement[pvi].fanout;
    stats_.multicast_factor[pv] = 0;

    std::string data_space_name = problem::GetShape()->DataSpaceIDToName.at(pvi);

    // don't care what type of connection this is
    // only need to count the number of transfers
    stats_.ingresses[pv] = data_movement[pv].access_stats;
    stats_.multicast_factor[pv] = 0;

    for (auto& x: stats_.ingresses.at(pv).stats)
    {
      auto multicast_factor = x.first.first;
      // auto scatter_factor = x.first.second;
      auto ingresses = x.second.accesses;
      // auto hops = x.second.hops;

      if (ingresses > 0)
      {
        if (specs_.per_datatype_ERT)
        {
          stats_.energy[pv] = GetMulticastEnergyByDataType(multicast_factor, data_space_name) * ingresses;
        }
        else
        {
          stats_.energy[pv] = GetMulticastEnergy(multicast_factor) * ingresses;
        }

        if (multicast_factor > stats_.multicast_factor[pv])
        {
          stats_.multicast_factor[pv] = multicast_factor;
        }
      }
    }
  }

  auto eval_status = EvalStatus{true, std::string("")};
  is_evaluated_ = true;

  return eval_status;
}

void SimpleMulticastNetwork::Print(std::ostream& out) const
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
  out << indent << indent << "Action Name       : " << specs_.action_name << std::endl;

  out << std::endl;

  out << indent << "STATS" << std::endl;
  out << indent << "-----" << std::endl;
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    out << indent << problem::GetShape()->DataSpaceIDToName.at(pv) << ":" << std::endl;
    out << indent + indent << "Fanout                                  : "
    << stats_.fanout.at(pv) << std::endl;
    out << indent + indent << "Multicast factor                        : "
        << stats_.multicast_factor.at(pv) << std::endl;
    auto total_accesses = stats_.ingresses.at(pv).TotalAccesses();
    out << indent + indent << "Ingresses                               : "
        << total_accesses << std::endl;
    out << indent + indent << "Energy (per-instance)                   : "
        << stats_.energy.at(pv) << " pJ" << std::endl;
    out << indent + indent << "Energy (total)                          : "
        << stats_.energy.at(pv) * stats_.utilized_instances.at(pv)
        << " pJ" << std::endl;
  }
  out << std::endl;
}

std::uint64_t SimpleMulticastNetwork::WordBits() const
{
  return 0;
}

STAT_ACCESSOR(double, SimpleMulticastNetwork, Energy, stats_.energy.at(pv) * stats_.utilized_instances.at(pv))

} // namespace model
