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

#include "search/random-pruned.hpp"

namespace search
{

RandomPrunedSearch::RandomPrunedSearch(config::CompoundConfigNode config, mapspace::MapSpace* mapspace, unsigned id) :
    SearchAlgorithm(),
    mapspace_(mapspace),
    id_(id),
    if_pgen_(mapspace_->Size(mapspace::Dimension::IndexFactorization)),
    lp_pgen_(mapspace_->Size(mapspace::Dimension::LoopPermutation)),
    state_(State::Ready),
    valid_mappings_(0),
    eval_fail_count_(0),
    best_cost_(0)
{
  (void) id_;
    
  unsigned x = 16;
  config.lookupValue("max-permutations-per-if-visit", x);
  max_permutations_per_if_visit_ = x;
    
  for (unsigned i = 0; i < unsigned(mapspace::Dimension::Num); i++)
  {
    iterator_[i] = 0;
  }

  // Special case: if the index factorization space has size 0
  // (can happen with residual mapspaces) then we init in terminated
  // state.
  if (mapspace_->Size(mapspace::Dimension::IndexFactorization) == 0)
  {
    state_ = State::Terminated;
  }
  else
  {
    // Prepare the first random subspace IDs.
    iterator_[unsigned(mapspace::Dimension::IndexFactorization)] = if_pgen_.Next();      

    // Prune the mapspace for the first time.
    mapspace_->InitPruned(iterator_[unsigned(mapspace::Dimension::IndexFactorization)]);

    // Determine how many loop permutations to evaluate within this index
    // factorization.
    permutations_to_visit_ = std::min(max_permutations_per_if_visit_,
                                      mapspace_->Size(mapspace::Dimension::LoopPermutation));
    permutations_visited_ = 0;

    // Also throw a random number for the first permutation.
    iterator_[unsigned(mapspace::Dimension::LoopPermutation)] = lp_pgen_.Next() %
      mapspace_->Size(mapspace::Dimension::LoopPermutation);
  }

#ifdef DUMP_COSTS
  // Dump best cost for each index factorization.
  best_cost_file_.open("/tmp/timeloop-if-cost.txt");
#endif
}

RandomPrunedSearch::~RandomPrunedSearch()
{
#ifdef DUMP_COSTS
  best_cost_file_.close();
#endif
}

bool RandomPrunedSearch::IncrementRecursive_(int position)
{
  auto dim = dim_order_[position];

  if (dim == mapspace::Dimension::IndexFactorization)
  {
    // Throw a random number to get the next index factorization.
    iterator_[unsigned(dim)] = if_pgen_.Next();
      
    // We just changed the index factorization. Prune the sub-mapspace
    // for this specific factorization index.
    mapspace_->InitPruned(iterator_[unsigned(dim)]);

    // Determine how many loop permutations to evaluate within this index
    // factorization.
    permutations_to_visit_ = std::min(max_permutations_per_if_visit_,
                                      mapspace_->Size(mapspace::Dimension::LoopPermutation));
    permutations_visited_ = 0;

    // Also throw a random number for the first permutation.
    iterator_[unsigned(mapspace::Dimension::LoopPermutation)] = lp_pgen_.Next() %
      mapspace_->Size(mapspace::Dimension::LoopPermutation);
      
#ifdef DUMP_COSTS
    // Dump the best cost observed for this index factorization.
    // Note: best_cost_ == 0 implies this was a bad index factorization
    // that failed mapping. We can choose to not report these, or
    // grep them out in post-processing.
    best_cost_file_ << best_cost_ << std::endl;
#endif
        
    // Reset the best cost.
    best_cost_ = 0;

    return true;
  }
  else if (dim == mapspace::Dimension::LoopPermutation)
  {
    if (permutations_visited_ + 1 < permutations_to_visit_)
    {
      // Throw a random number to get the next loop permutation. However, the
      // pruned permutation-space may be smaller than the range of the RNG, so
      // apply a modulus.
      iterator_[unsigned(dim)] = lp_pgen_.Next() %
        mapspace_->Size(mapspace::Dimension::LoopPermutation);
      permutations_visited_++;
      return true;
    }
    // Carry over to next higher-order mapspace dimension.
    else
    {
      iterator_[unsigned(dim)] = 0;
      return IncrementRecursive_(position + 1);
    }
  }
  else // All other dimensions *except* IndexFactorization and LoopPermutation.
  {
    if (iterator_[unsigned(dim)] + 1 < mapspace_->Size(dim))
    {
      // Move to next integer in this mapspace dimension.
      iterator_[unsigned(dim)]++;
      return true;
    }
    // Carry over to next higher-order mapspace dimension.
    else
    {
      // This cannot be the last position because that is reserved for
      // IndexFactorization.
      assert(position + 1 < int(mapspace::Dimension::Num));
      iterator_[unsigned(dim)] = 0;
      return IncrementRecursive_(position + 1);
    }
  }
}

bool RandomPrunedSearch::Next(mapspace::ID& mapping_id)
{
  if (state_ == State::Terminated)
  {
    return false;
  }

  assert(state_ == State::Ready);

  mapping_id = mapspace::ID(mapspace_->AllSizes());
  for (unsigned i = 0; i < unsigned(mapspace::Dimension::Num); i++)
  {
    mapping_id.Set(i, iterator_[i]);
  }
    
  state_ = State::WaitingForStatus;

  // std::cerr << "MAPPING ID: IF(" << iterator_[unsigned(mapspace::Dimension::IndexFactorization)]
  //           << ") P(" << iterator_[unsigned(mapspace::Dimension::LoopPermutation)]
  //           << ") B(" << iterator_[unsigned(mapspace::Dimension::DatatypeBypass)]
  //           << ") S(" << iterator_[unsigned(mapspace::Dimension::Spatial)]
  //           << ")" << std::endl;
    
  return true;
}

void RandomPrunedSearch::Report(Status status, double cost)
{
  assert(state_ == State::WaitingForStatus);

  bool skip_datatype_bypass = false;
  if (status == Status::Success)
  {
    valid_mappings_++;

    if (best_cost_ == 0)
      best_cost_ = cost;
    else
      best_cost_ = std::min(best_cost_, cost);
  }
  else if (status == Status::MappingConstructionFailure)
  {
    // Accelerate search by invalidating bad spaces.
    // ConstructMapping failure =>
    //   Combination of (IF, LP, S) is bad.
    //   Skip all DBs.
    skip_datatype_bypass = true;
  }
  else if (status == Status::EvalFailure)
  {
    // PreEval/Eval failure (capacity) =>
    //   Combination of (IF, DB) is bad.
    //   If all DBs cause Eval failure for an IF, then that IF is bad,
    //   no need to look at other LP, S combinations.
    eval_fail_count_++;
  }

  if (iterator_[unsigned(mapspace::Dimension::DatatypeBypass)] + 1 ==
      mapspace_->Size(mapspace::Dimension::DatatypeBypass))
  {
    if (eval_fail_count_ == mapspace_->Size(mapspace::Dimension::DatatypeBypass))
    {
      // All DBs failed eval for this combination of IF*LP*S. This means
      // this IF is bad. Skip to the next IF by fast-forwarding to the end of
      // this IF.
      iterator_[unsigned(mapspace::Dimension::Spatial)] =
        mapspace_->Size(mapspace::Dimension::Spatial) - 1;
      permutations_visited_ = permutations_to_visit_ - 1;
    }
    eval_fail_count_ = 0;
  }

  if (skip_datatype_bypass)
  {
    iterator_[unsigned(mapspace::Dimension::DatatypeBypass)] =
      mapspace_->Size(mapspace::Dimension::DatatypeBypass) - 1;
  }

  bool mapspace_remaining = IncrementRecursive_();

  if (mapspace_remaining) //  && valid_mappings_ < search_size_)
  {
    state_ = State::Ready;
  }
  else
  {
    state_ = State::Terminated;
  }
}

} // namespace search
