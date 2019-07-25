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
  
 private:
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
  void Roll(mapspace::Dimension dim)
  {
    // if (dim == mapspace::Dimension::IndexFactorization)
    // {
    //   uint128_t roll;
    //   bool good;
    //   do
    //   {
    //     roll = pgens_[int(dim)]->Next();
    //     good = (bad_.find(static_cast<std::uint64_t>(roll)) == bad_.end());
    //   }
    //   while (!good);
    //   mapping_id_.Set(int(dim), roll);
    // }
    // else
    // {
    mapping_id_.Set(int(dim), pgens_[int(dim)]->Next());
    // }
  }

 public:
  RandomSearch(config::CompoundConfigNode config, mapspace::MapSpace* mapspace) :
      SearchAlgorithm(),
      mapspace_(mapspace),
      state_(State::Ready),
      mapping_id_(mapspace->AllSizes()),
      masking_space_covered_(mapspace_->Size(mapspace::Dimension::DatatypeBypass)),
      valid_mappings_(0)
  {
    filter_revisits_ = false;
    config.lookupValue("filter-revisits", filter_revisits_);    

    pgens_[int(mapspace::Dimension::IndexFactorization)] =
      new RandomGenerator128(mapspace_->Size(mapspace::Dimension::IndexFactorization));
    pgens_[int(mapspace::Dimension::LoopPermutation)] =
      new RandomGenerator128(mapspace_->Size(mapspace::Dimension::LoopPermutation));
    pgens_[int(mapspace::Dimension::Spatial)] =
      new RandomGenerator128(mapspace_->Size(mapspace::Dimension::Spatial));
    pgens_[int(mapspace::Dimension::DatatypeBypass)] =
      new SequenceGenerator128(mapspace_->Size(mapspace::Dimension::DatatypeBypass));
    // std::cout << "Mapping ID base dimensions = " << mapping_id_.Base().size() << std::endl;
    // std::cout << "Mapping ID base = ";
    // Print<>(mapping_id_.Base());
    // std::cout << std::endl;

    // Special case: if the index factorization space has size 0
    // (can happen with residual mapspaces) then we init in terminated
    // state.
    if (mapspace_->Size(mapspace::Dimension::IndexFactorization) == 0)
      state_ = State::Terminated;
  }

  // This class does not support being copied
  RandomSearch(const RandomSearch&) = delete;
  RandomSearch& operator=(const RandomSearch&) = delete;

  ~RandomSearch()
  {
    delete static_cast<RandomGenerator128*>(
            pgens_[int(mapspace::Dimension::IndexFactorization)]);
    delete static_cast<RandomGenerator128*>(
            pgens_[int(mapspace::Dimension::LoopPermutation)]);
    delete static_cast<RandomGenerator128*>(
            pgens_[int(mapspace::Dimension::Spatial)]);
    delete static_cast<SequenceGenerator128*>(
            pgens_[int(mapspace::Dimension::DatatypeBypass)]);
  }
  
  bool Next(mapspace::ID& mapping_id)
  {
    if (state_ == State::Terminated)
    {
      return false;
    }

    assert(state_ == State::Ready);
    
    if (masking_space_covered_ == mapspace_->Size(mapspace::Dimension::DatatypeBypass))
    {
      while (true)
      {
        Roll(mapspace::Dimension::IndexFactorization);
        Roll(mapspace::Dimension::LoopPermutation);
        Roll(mapspace::Dimension::Spatial);
        Roll(mapspace::Dimension::DatatypeBypass);
        if (filter_revisits_)
        {
          if (visited_.find(mapping_id_.Integer()) == visited_.end())
          {
            visited_.insert(mapping_id_.Integer());
            break;
          }
        }
        else
        {
          break;
        }
      }
      masking_space_covered_ = 1;
    }
    else
    {
      Roll(mapspace::Dimension::DatatypeBypass);
      masking_space_covered_++;
    }

    state_ = State::WaitingForStatus;
    
    mapping_id = mapping_id_;
    return true;
  }

  void Report(Status status, double cost = 0)
  {
    (void) cost;
    
    assert(state_ == State::WaitingForStatus);

    if (status == Status::Success)
    {
      valid_mappings_++;
    }
    // else
    // {
    //   bad_.insert(static_cast<std::uint64_t>(mapping_id_[int(mapspace::Dimension::IndexFactorization)]));
    // }
    
    // total_mappings >= mapspace_->Size()
    // if (bad_.size() == mapspace_->Size(mapspace::Dimension::IndexFactorization) ||
    // if (valid_mappings_ >= std::min(search_size_, mapspace_->Size()))
    if (valid_mappings_ >= mapspace_->Size())
    {
      state_ = State::Terminated;
    }
    else
    {
      state_ = State::Ready;
    }
  }
};

} // namespace search
