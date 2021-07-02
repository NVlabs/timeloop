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

#include <iterator>
#include <unordered_set>
#include <boost/functional/hash.hpp>

#include "mapping/mapping.hpp"
#include "mapspaces/mapspace-base.hpp"
#include "util/misc.hpp"
#include "search/search.hpp"

namespace search
{

class RandomSearch : public SearchAlgorithm
{
 private:
  enum class State
  {
    Ready,
    WaitingForStatus,
    Terminated
  };
  
  // Config.
  mapspace::MapSpace* mapspace_;
  // std::unordered_set<std::uint64_t> bad_;
  std::unordered_set<uint128_t> visited_;
  bool filter_revisits_;

  // Submodules.
  std::array<PatternGenerator128*, int(mapspace::Dimension::Num)> pgens_;
  
  // Live state.
  State state_;
  mapspace::ID mapping_id_;
  uint128_t masking_space_covered_;
  uint128_t valid_mappings_;

  // Roll the dice along a single mapspace dimension.
  void Roll(mapspace::Dimension dim);

 public:
  RandomSearch(config::CompoundConfigNode config, mapspace::MapSpace* mapspace);

  // This class does not support being copied
  RandomSearch(const RandomSearch&) = delete;
  RandomSearch& operator=(const RandomSearch&) = delete;

  ~RandomSearch();
  
  bool Next(mapspace::ID& mapping_id);

  void Report(Status status, double cost = 0);
};

} // namespace search
