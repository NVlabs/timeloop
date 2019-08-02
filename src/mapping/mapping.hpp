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

#include <iostream>
#include <boost/multiprecision/cpp_int.hpp>
#include <libconfig.h++>

#include "nest.hpp"

using namespace boost::multiprecision;

//--------------------------------------------//
//                  Mapping                   //
//--------------------------------------------//

struct Mapping
{
  uint128_t id;
  loop::Nest loop_nest;
  tiling::CompoundMaskNest datatype_bypass_nest;
  
  // Serialization
  friend class boost::serialization::access;
  
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0)
  {
    if(version == 0)
    {
      ar << boost::serialization::make_nvp(
        "datatype_bypass_nest",
        boost::serialization::make_array(
          datatype_bypass_nest.data(),
          datatype_bypass_nest.size()));
    }
  }

  void FormatAsConstraints(libconfig::Setting& mapspace);
  
  void PrintAsConstraints(std::string filename);

  void PrettyPrint(std::ostream& out, const std::vector<std::string>& storage_level_names,
                   const std::vector<problem::PerDataSpace<std::uint64_t>>& tile_sizes);

  void PrintWhoopNest(std::ostream& out, const std::vector<std::string>& storage_level_names,
                      const std::vector<problem::PerDataSpace<std::uint64_t>>& tile_sizes,
                      const std::vector<problem::PerDataSpace<std::uint64_t>>& utilized_instances);
};

std::ostream& operator << (std::ostream& out, const Mapping& mapping);

