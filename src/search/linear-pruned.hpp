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
#include <fstream>
#include <iostream>

#include "mapping/mapping.hpp"
#include "mapspaces/mapspace-base.hpp"
#include "util/misc.hpp"
#include "search/search.hpp"

namespace search
{

class LinearPrunedSearch : public SearchAlgorithm
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
  unsigned id_;

  // Live state.
  State state_;
  std::array<uint128_t, unsigned(mapspace::Dimension::Num)> iterator_;
  uint128_t valid_mappings_;
  std::uint64_t eval_fail_count_;

  double best_cost_;
  std::ofstream best_cost_file_;

  // Order:
  //   DatatypeBypass <- Spatial <- LoopPermutation <- IndexFactorization.
  std::vector<mapspace::Dimension> dim_order_ =
  {
    mapspace::Dimension::DatatypeBypass,
    mapspace::Dimension::Spatial,
    mapspace::Dimension::LoopPermutation,
    mapspace::Dimension::IndexFactorization
  };
  
 public:
  LinearPrunedSearch(config::CompoundConfigNode config, mapspace::MapSpace* mapspace, unsigned id);

  ~LinearPrunedSearch();

  bool IncrementRecursive_(int position = 0);

  bool Next(mapspace::ID& mapping_id);

  void Report(Status status, double cost = 0);
};

} // namespace search
