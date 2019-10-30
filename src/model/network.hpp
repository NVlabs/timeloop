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

namespace model
{

class Network
{

 public:

  struct Specs
  {
    enum class NetworkType
    {
      OneToOne,
      OneToMany,
      ManyToMany
    };

    DataSpaceIDSharing sharing_type;

    PerDataSpaceOrShared<Attribute<NetworkType>> type;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> word_bits;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> fanout;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> fanoutX;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> fanoutY;
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
        ar& BOOST_SERIALIZATION_NVP(fanout);
        ar& BOOST_SERIALIZATION_NVP(fanoutX);
        ar& BOOST_SERIALIZATION_NVP(fanoutY);
        ar& BOOST_SERIALIZATION_NVP(routerEnergy);
        ar& BOOST_SERIALIZATION_NVP(wireEnergy);
      }
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
        fanout.SetPerDataSpace();
        fanoutX.SetPerDataSpace();
        fanoutY.SetPerDataSpace();
        routerEnergy.SetPerDataSpace();
        wireEnergy.SetPerDataSpace();
      }
      else // sharing_type == DataSpaceIDSharing::Shared
      {
        type.SetShared();
        word_bits.SetShared();
        fanout.SetShared();
        fanoutX.SetShared();
        fanoutY.SetShared();
        routerEnergy.SetShared();
        wireEnergy.SetShared();        
      }
    }

    ADD_ACCESSORS(Type, type, NetworkType)
    ADD_ACCESSORS(WordBits, word_bits, std::uint64_t)
    ADD_ACCESSORS(Fanout, fanout, std::uint64_t)
    ADD_ACCESSORS(FanoutX, fanoutX, std::uint64_t)
    ADD_ACCESSORS(FanoutY, fanoutY, std::uint64_t)
    ADD_ACCESSORS(RouterEnergy, routerEnergy, double)
    ADD_ACCESSORS(WireEnergy, wireEnergy, double)

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
  };


};

}
