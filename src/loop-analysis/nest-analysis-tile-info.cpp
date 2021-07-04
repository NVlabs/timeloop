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

#include "loop-analysis/nest-analysis-tile-info.hpp"

namespace analysis
{

//
// DataMovementInfo
//

template <class Archive>
void DataMovementInfo::serialize(Archive& ar, const unsigned int version)
{
  if (version == 0)
  {
    ar& BOOST_SERIALIZATION_NVP(size);
    ar& BOOST_SERIALIZATION_NVP(accesses);
    ar& BOOST_SERIALIZATION_NVP(subnest);
  }
}

std::uint64_t DataMovementInfo::GetTotalAccesses() const
{
  return std::accumulate(accesses.begin(), accesses.end(), static_cast<std::uint64_t>(0));
}
  
std::uint64_t DataMovementInfo::GetWeightedAccesses() const
{
  std::uint64_t total = 0;
  for (std::uint32_t i = 0; i < accesses.size(); i++)
  {
    total += accesses[i] * (i + 1);
  }
  return total;
}

void DataMovementInfo::Reset()
{
  size = 0;
  accesses.resize(0);
  scatter_factors.resize(0);
  cumulative_hops.resize(0);
  link_transfers = 0;
  subnest.resize(0);
  replication_factor = 0;
  fanout = 0;
  distributed_fanout = 0;
}

void DataMovementInfo::Validate()
{
  std::uint64_t f = 0;
  for (std::uint64_t i = 0; i < fanout; i++)
  {
    if (accesses[i] != 0)
    {
      auto multicast_factor = i + 1;
      auto scatter_factor = scatter_factors[i];
      f += (multicast_factor * scatter_factor);
    }
  }

  if (f != fanout)
  {
    std::cerr << "ERROR: sigma(multicast * scatter) != fanout." << std::endl;
    std::cerr << "  dumping (multicast, scatter) pairs:" << std::endl;
    for (std::uint64_t i = 0; i < fanout; i++)
    {
      if (accesses[i] != 0)
      {
        auto multicast_factor = i + 1;
        auto scatter_factor = scatter_factors[i];
        std::cerr << "    " << multicast_factor << ", " << scatter_factor << std::endl;
      }
    }
    std::cerr << "  sigma(multicast, scatter) = " << f << std::endl;
    std::cerr << "  fanout = " << fanout << std::endl;
    exit(1);
  }
}

//
// ComputeInfo
//

ComputeInfo::ComputeInfo()
{
  Reset();
}

void ComputeInfo::Reset()
{
  replication_factor = 0;
  accesses = 0;
}


} // namespace analysis
