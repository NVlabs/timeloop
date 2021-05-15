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
#pragma once

#include <map>

struct AccessStatMatrix
{
  struct AccessStats
  {
    std::uint64_t accesses = 0;
    double hops = 0.0;
  };

  std::map<std::pair<std::uint64_t,std::uint64_t>, AccessStats> stats;

  void clear()
  {
    stats.clear();
  }

  std::uint64_t TotalAccesses() const
  {
    std::uint64_t total = 0;
    for (auto& x: stats)
    {
      total += x.second.accesses;
    }
    return total;
  }
  
  std::uint64_t WeightedAccesses() const
  {
    std::uint64_t total = 0;
    for (auto& x: stats)
    {
      total += x.second.accesses * x.first.first;
    }
    return total;
  }

  void Accumulate(const AccessStatMatrix& other)
  {
    for (auto& x: other.stats)
    {
      auto multicast = x.first.first;
      auto scatter = x.first.second;

      auto& mine = stats[std::make_pair(multicast, scatter)];
      mine.accesses += x.second.accesses; 
      mine.hops += x.second.hops; 
    }
  }

  void Divide(const std::uint64_t divisor)
  {
    ASSERT(divisor > 0);
    for (auto& x: stats)
    {
      std::cout << "Divide " << x.second.accesses << "/" << divisor << std::endl;
      x.second.accesses /= divisor;
      x.second.hops /= divisor;
    }
  }

  AccessStats& at(std::uint64_t multicast, std::uint64_t scatter)
  {
    return stats.at(std::make_pair(multicast, scatter));
  }

  AccessStats& operator () (std::uint64_t multicast, std::uint64_t scatter)
  {
    return stats[std::make_pair(multicast, scatter)];
  }

  bool operator == (const AccessStatMatrix& other)
  {
    for (auto& x: other.stats)
    {
      auto it = stats.find(std::make_pair(x.first.first, x.first.second));
      if (it == stats.end()) return false;
      if (it->second.accesses != x.second.accesses) return false;
      if (it->second.hops != x.second.hops) return false;
    }
    for (auto& x: stats)
    {
      auto it = other.stats.find(std::make_pair(x.first.first, x.first.second));
      if (it == other.stats.end()) return false;
      if (it->second.accesses != x.second.accesses) return false;
      if (it->second.hops != x.second.hops) return false;
    }
    return true;
  }

  friend std::ostream& operator << (std::ostream& out, const AccessStatMatrix& m)
  {
    for (auto& x: m.stats)
    {
      auto multicast = x.first.first;
      auto scatter = x.first.second;
      out << "    [" << multicast << ", " << scatter << "]: accesses = "
          << x.second.accesses << " hops = " << x.second.hops << std::endl;
    }
    return out;
  }
};


namespace analysis
{
// data structures for nest-analysis to store datamovement and compute info
struct DataMovementInfo
{
  friend class boost::serialization::access;

  // Serialization.
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0) 
  {
    if(version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(size);
      //ar& BOOST_SERIALIZATION_NVP(accesses);
      ar& BOOST_SERIALIZATION_NVP(subnest);
    }
  }

  std::size_t size;
  bool distributed_multicast;
  AccessStatMatrix access_stats;
  std::uint64_t link_transfers;
  std::vector<loop::Descriptor> subnest;
  std::uint64_t replication_factor;      // number of spatial elements at this level.
  std::uint64_t fanout;                  // per-element fanout to next-level.
  std::uint64_t distributed_fanout;      // max range of fanout if distributed multicast is used.
  bool is_on_storage_boundary;
  bool is_master_spatial;
  
  void Reset()
  {
    size = 0;
    access_stats.clear();
    link_transfers = 0;
    subnest.resize(0);
    replication_factor = 0;
    fanout = 0;
    distributed_fanout = 0;
  }

  void Validate()
  {
    // std::uint64_t f = 0;
    // for (std::uint64_t i = 0; i < fanout; i++)
    // {
    //   if (accesses[i] != 0)
    //   {
    //     auto multicast_factor = i + 1;
    //     auto scatter_factor = scatter_factors[i];
    //     f += (multicast_factor * scatter_factor);
    //   }
    // }

    // if (f != fanout)
    // {
    //   std::cerr << "ERROR: sigma(multicast * scatter) != fanout." << std::endl;
    //   std::cerr << "  dumping (multicast, scatter) pairs:" << std::endl;
    //   for (std::uint64_t i = 0; i < fanout; i++)
    //   {
    //     if (accesses[i] != 0)
    //     {
    //       auto multicast_factor = i + 1;
    //       auto scatter_factor = scatter_factors[i];
    //       std::cerr << "    " << multicast_factor << ", " << scatter_factor << std::endl;
    //     }
    //   }
    //   std::cerr << "  sigma(multicast, scatter) = " << f << std::endl;
    //   std::cerr << "  fanout = " << fanout << std::endl;
    //   exit(1);
    // }
  }
};


struct ComputeInfo
{
  std::uint64_t replication_factor;      // number of spatial elements at this level.
  std::uint64_t accesses;
  
  ComputeInfo() { Reset(); }

  void Reset()
  {
    replication_factor = 0;
    accesses = 0;
  }
};

// compound tile info types to capture per-dataspace info
typedef problem::PerDataSpace<std::vector<DataMovementInfo>> CompoundDataMovementNest ; 
typedef std::vector<ComputeInfo> CompoundComputeNest;  // single vector, each element for a nest level, no fine-grained op type should be considered here
struct CompoundTileNest{
   CompoundDataMovementNest compound_data_movement_info_nest;
   CompoundComputeNest compound_compute_info_nest;
};

} //namespace
