/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <boost/multiprecision/cpp_int.hpp>

#include "mapping/mapping.hpp"
#include "model/engine.hpp"
#include "workload/problem-shape.hpp"

using namespace boost::multiprecision;

//--------------------------------------------//
//             Mapspace Dimensions            //
//--------------------------------------------//

namespace mapspace
{

enum class Dimension
{
  IndexFactorization,  // Factorization of loop bounds across storage levels
  LoopPermutation,     // Permutation of loop nests in each storage level
  Spatial,             // Position of the transition point between horizontal and vertical
                       //   spatial tilings 
  DatatypeBypass,      // Optionally bypass a storage level for a datatype
  Num
};

std::ostream& operator << (std::ostream& out, Dimension d);

typedef CartesianCounter<int(Dimension::Num)> ID;

//--------------------------------------------//
//                  MapSpace                  //
//--------------------------------------------//

class MapSpace
{
 protected:
  model::Engine::Specs arch_specs_;
  const problem::Workload& workload_config_;
  std::array<uint128_t, int(Dimension::Num)> size_;

 public:
  MapSpace(model::Engine::Specs arch_specs,
           const problem::Workload& workload_config) :
      arch_specs_(arch_specs),
      workload_config_(workload_config),
      size_({})
  {}

  virtual ~MapSpace() {}

  virtual std::vector<MapSpace*> Split(std::uint64_t num_splits) = 0;

  virtual void InitPruned(uint128_t local_index_factorization_id) = 0;

  virtual bool ConstructMapping(ID mapping_id, Mapping* mapping) = 0;

  bool ConstructMapping(const uint128_t mapping_id,
                        Mapping* mapping)
  {
    ID cmapping_id(size_);
    cmapping_id.Set(mapping_id);
    return ConstructMapping(cmapping_id, mapping); 
  }

  uint128_t Size(Dimension dim)
  {
    return size_[int(dim)];
  }
  
  uint128_t Size()
  {
    uint128_t size = 1;
    for (int i = 0; i < int(Dimension::Num); i++)
    {
      size *= size_[i];
    }
    return size;
  }

  std::array<uint128_t, int(Dimension::Num)> AllSizes()
  {
    return size_;
  }
};

} // namespace mapspace
