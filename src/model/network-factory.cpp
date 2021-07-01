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

#include "model/network-factory.hpp"

namespace model
{

//--------------------------------------------//
//                Network Factory             //
//--------------------------------------------//

// Parse network type and instantiate a Spec object of that network type.
std::shared_ptr<NetworkSpecs> NetworkFactory::ParseSpecs(config::CompoundConfigNode network, uint32_t n_elements)
{
  std::shared_ptr<NetworkSpecs> specs;

  std::string network_class;
  if (network.lookupValue("class", network_class))
  {
    if (network_class.compare("XY_NoC") == 0 || network_class.compare("Legacy") == 0)
    {
      auto legacy_specs = LegacyNetwork::ParseSpecs(network, n_elements);
      specs = std::make_shared<LegacyNetwork::Specs>(legacy_specs);
    }
    else if (network_class.compare("ReductionTree") == 0)
    {
      auto reduction_tree_specs = ReductionTreeNetwork::ParseSpecs(network, n_elements);
      specs = std::make_shared<ReductionTreeNetwork::Specs>(reduction_tree_specs);
    }
    else if (network_class.compare("SimpleMulticast") == 0)
    {
      auto simple_multicast_specs = SimpleMulticastNetwork::ParseSpecs(network, n_elements);
      specs = std::make_shared<SimpleMulticastNetwork::Specs>(simple_multicast_specs);
    }

    else
    {
      std::cerr << "ERROR: unrecognized network class: " << network_class << std::endl;
      exit(1);
    }
  }
  else
  {
    std::cerr << "ERROR: class name not found in network spec." << std::endl;
    exit(1);
  }

  return specs;
}

// Instantiate a network object based on a given spec.
std::shared_ptr<Network> NetworkFactory::Construct(std::shared_ptr<NetworkSpecs> specs)
{
  std::shared_ptr<Network> network;

  if (specs->Type() == "Legacy")
  {
    auto legacy_specs = *std::static_pointer_cast<LegacyNetwork::Specs>(specs);
    auto legacy_network = std::make_shared<LegacyNetwork>(legacy_specs);
    network = std::static_pointer_cast<Network>(legacy_network);
  }
  else if (specs->Type() == "ReductionTree")
  {
    auto reduction_tree_specs = *std::static_pointer_cast<ReductionTreeNetwork::Specs>(specs);
    auto reduction_tree_network = std::make_shared<ReductionTreeNetwork>(reduction_tree_specs);
    network = std::static_pointer_cast<Network>(reduction_tree_network);
  }
  else if (specs->Type() == "SimpleMulticast")
  {
    auto simple_multicast_specs = *std::static_pointer_cast<SimpleMulticastNetwork::Specs>(specs);
    auto simple_multicast_network = std::make_shared<SimpleMulticastNetwork>(simple_multicast_specs);
    network = std::static_pointer_cast<Network>(simple_multicast_network);
  }

  else
  {
    std::cerr << "ERROR: unrecognized network type: " << specs->Type() << std::endl;
    exit(1);
  }

  return network;
}

} // namespace model
