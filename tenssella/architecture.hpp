/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "isl_utils.hpp"

class Architecture
{
 protected:
  std::size_t num_levels_;
  isl_ctx* context_;
  std::map<int, map<string, map<string, isl_map*>>> mem_instance_map_;
  std::map<int, map<string, isl_map*>> comp_instance_map_;

 public:
  Architecture() = delete;

  Architecture(std::size_t num_levels, isl_ctx* context) :
      num_levels_(num_levels),
      context_(context)
  { }

  std::size_t NumLevels()
  {
    return num_levels_;
  }

  isl_map* MemInstanceMap(int hlevel, std::string partition_name,
          std::string compute_space_name)
  {
    return mem_instance_map_.at(hlevel).at(partition_name).at(compute_space_name);
  }

  isl_map* InitInstanceMap(std::string partition_name)
  {
    return pick(mem_instance_map_.at(num_levels_).at(partition_name)).second;
  }

  isl_map* ComputeInstanceMap(int hlevel, std::string compute_space_name)
  {
    return comp_instance_map_.at(hlevel).at(compute_space_name);
  }
};
