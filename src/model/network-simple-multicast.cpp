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

SimpleMulticastNetwork::Specs SimpleMulticastNetwork::ParseSpecs(config::CompoundConfigNode network, std::size_t n_elements)
{
  (void) n_elements; // FIXME.

  Specs specs;

  // Network Type.
  specs.type = "SimpleMulticast";
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

EvalStatus SimpleMulticastNetwork::Evaluate(const tiling::CompoundTile& tile,
                              const bool break_on_failure)
{
  (void) tile;
  (void) break_on_failure;

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

  out << std::endl;

  out << indent << "STATS" << std::endl;
  out << indent << "-----" << std::endl;
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    out << indent << problem::GetShape()->DataSpaceIDToName.at(pv) << ":" << std::endl;

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

STAT_ACCESSOR(double, SimpleMulticastNetwork, Energy, stats_.energy.at(pv))

} // namespace model
