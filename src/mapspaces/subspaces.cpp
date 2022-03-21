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

#include <functional>
#include <set>

#include "mapspaces/subspaces.hpp"

namespace mapspace
{

//--------------------------------------------//
//           IndexFactorizationSpace          //
//--------------------------------------------//

IndexFactorizationSpace::IndexFactorizationSpace() :
    tiling_counter_(problem::GetShape()->NumFlattenedDimensions)
{ }

void IndexFactorizationSpace::Init(const problem::Workload &workload,
                                   std::map<problem::Shape::FlattenedDimensionID, std::uint64_t> cofactors_order,
                                   std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>> prefactors,
                                   std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>> maxfactors)
{
  problem::PerFlattenedDimension<uint128_t> counter_base;
  for (int idim = 0; idim < int(problem::GetShape()->NumFlattenedDimensions); idim++)
  {
    auto dim = problem::Shape::FlattenedDimensionID(idim);

    if (prefactors.find(dim) == prefactors.end())
      dimension_factors_[idim] = Factors(workload.GetFlattenedBound(dim), cofactors_order[dim]);
    else
      dimension_factors_[idim] = Factors(workload.GetFlattenedBound(dim), cofactors_order[dim], prefactors[dim]);

    if (maxfactors.find(dim) != maxfactors.end())
      dimension_factors_[idim].PruneMax(maxfactors[dim]);

    counter_base[idim] = dimension_factors_[idim].size();
  }

  tiling_counter_.Init(counter_base);

  std::cout << "Initializing Index Factorization subspace." << std::endl;
  for (int dim = 0; dim < int(problem::GetShape()->NumFlattenedDimensions); dim++)
  {
    std::cout << "  Factorization options along problem dimension "
              << problem::GetShape()->FlattenedDimensionIDToName.at(dim) << " = "
              << counter_base[dim] << std::endl;
  }
}

unsigned long IndexFactorizationSpace::GetFactor(uint128_t nest_id, problem::Shape::FlattenedDimensionID dim, unsigned level)
{
  auto idim = unsigned(dim);
  tiling_counter_.Set(nest_id);
  auto cartesian_idx = tiling_counter_.Read();    
  return dimension_factors_[idim][std::uint64_t(cartesian_idx[idim])][level];
}

uint128_t IndexFactorizationSpace::Size() const
{
  return tiling_counter_.EndInteger();
}

//--------------------------------------------//
//      ResidualIndexFactorizationSpace       //
//--------------------------------------------//

ResidualIndexFactorizationSpace::ResidualIndexFactorizationSpace() :
    tiling_counter_(problem::GetShape()->NumFlattenedDimensions)
{ }

void ResidualIndexFactorizationSpace::Init(const problem::Workload &workload,
          std::map<problem::Shape::FlattenedDimensionID, std::uint64_t> cofactors_order,
          std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>> prefactors,
          std::map<problem::Shape::FlattenedDimensionID, std::map<unsigned, unsigned long>> maxfactors,
          std::vector<unsigned long int> remainders,
          std::vector<unsigned long int> remainders_ix)
{
  problem::PerFlattenedDimension<uint128_t> counter_base;
  for (int idim = 0; idim < int(problem::GetShape()->NumFlattenedDimensions); idim++)
  {
    auto dim = problem::Shape::FlattenedDimensionID(idim);


    std::vector<unsigned long int> remainder_bounds;
    std::vector<unsigned long int> remainder_ix;


    //Preparse spatial factors to see if they're already given by user
    for(auto x: remainders){
      remainder_bounds.push_back(x);
    }
    
    for(auto x: remainders_ix){
      remainder_ix.push_back(x);
    }
    if (prefactors.find(dim) != prefactors.end()){
      for(unsigned i = remainder_ix.size(); i > 0; i--){
        for(auto& pref : prefactors[dim]){
          if(pref.first == remainder_ix[i-1]) {
            remainder_bounds.erase(remainder_bounds.begin()+i-1);
            remainder_ix.erase(remainder_ix.begin()+i-1);
          }
        }
      }
      dimension_factors_[idim] = ResidualFactors(workload.GetFlattenedBound(dim), cofactors_order[dim], remainder_bounds, remainder_ix, prefactors[dim]);
    }
    else{
      dimension_factors_[idim] = ResidualFactors(workload.GetFlattenedBound(dim), cofactors_order[dim], remainder_bounds, remainder_ix, {});
    }

    if (maxfactors.find(dim) != maxfactors.end())
      dimension_factors_[idim].PruneMax();

    counter_base[idim] = dimension_factors_[idim].size();

  }

  tiling_counter_.Init(counter_base);

  std::cout << "Initializing Index Factorization subspace." << std::endl;
  for (int dim = 0; dim < int(problem::GetShape()->NumFlattenedDimensions); dim++)
  {
    std::cout << "  Factorization options along problem dimension "
              << problem::GetShape()->FlattenedDimensionIDToName.at(dim) << " = "
              << counter_base[dim] << std::endl;
  }
}

std::vector<unsigned long> ResidualIndexFactorizationSpace::GetFactor(uint128_t nest_id, problem::Shape::FlattenedDimensionID dim, unsigned level)
{
  std::vector<unsigned long> ret;
  auto idim = unsigned(dim);
  tiling_counter_.Set(nest_id);
  auto cartesian_idx = tiling_counter_.Read();
  auto a = dimension_factors_[idim][std::uint64_t(cartesian_idx[idim])];
  ret.push_back(a[0][level]);
  ret.push_back(a[1][level]);
  
  return ret;
}

uint128_t ResidualIndexFactorizationSpace::Size() const
{
  return tiling_counter_.EndInteger();
}

//--------------------------------------------//
//              PermutationSpace              //
//--------------------------------------------//

PermutationSpace::PermutationSpace()
{
  for (unsigned i = 0; i < unsigned(problem::GetShape()->NumFlattenedDimensions); i++)
  {
    canonical_pattern_.push_back(problem::Shape::FlattenedDimensionID(i));
  }
}

void PermutationSpace::Init(uint64_t num_levels)
{
  num_levels_ = num_levels;
  patterns_.clear();
  size_.clear();
}

void PermutationSpace::InitLevelCanonical(uint64_t level)
{
  InitLevel(level, canonical_pattern_);
}

void PermutationSpace::InitLevel(uint64_t level, std::vector<problem::Shape::FlattenedDimensionID> user_prefix,
                                 std::vector<problem::Shape::FlattenedDimensionID> pruned_dimensions)
{
  assert(level < num_levels_);

  // Merge pruned dimensions with user prefix, with all unit factors at the
  // beginning followed by user-specified non-unit factors, i.e.,
  // <unit-factors><user-specified-non-unit-factors><free-non-unit-factors>
  // <unit-factors><user-specified-non-unit-factors> = baked_prefix
  // <free-non-unit-factors> = permutable_suffix
  std::vector<problem::Shape::FlattenedDimensionID> baked_prefix = pruned_dimensions;

  for (auto dim : user_prefix)
    if (std::find(pruned_dimensions.begin(), pruned_dimensions.end(), dim) == pruned_dimensions.end())
      baked_prefix.push_back(dim);

  std::set<problem::Shape::FlattenedDimensionID> unspecified_dimensions;
  for (unsigned i = 0; i < unsigned(problem::GetShape()->NumFlattenedDimensions); i++)
    unspecified_dimensions.insert(problem::Shape::FlattenedDimensionID(i));
    
  for (auto& dim : baked_prefix)
    unspecified_dimensions.erase(dim);
    
  std::vector<problem::Shape::FlattenedDimensionID> permutable_suffix;
  for (auto& dim : unspecified_dimensions)
    permutable_suffix.push_back(dim);

  assert(baked_prefix.size() + permutable_suffix.size() == unsigned(problem::GetShape()->NumFlattenedDimensions));

  patterns_[level] = { baked_prefix, permutable_suffix };
  size_[level] = factoradic_.Factorial(permutable_suffix.size());
}

std::vector<std::vector<problem::Shape::FlattenedDimensionID>>
PermutationSpace::GetPatterns(uint128_t id)
{
  std::vector<std::vector<problem::Shape::FlattenedDimensionID>> retval;

  for (unsigned level = 0; level < num_levels_; level++)
  {
    auto& pattern = patterns_.at(level);
    if (pattern.baked_prefix.size() == unsigned(problem::GetShape()->NumFlattenedDimensions))
    {
      retval.push_back(pattern.baked_prefix);
    }
    else
    {
      std::vector<problem::Shape::FlattenedDimensionID> permuted_suffix = pattern.permutable_suffix;
      factoradic_.Permute(permuted_suffix.data(), permuted_suffix.size(),
                          std::uint64_t(id % size_.at(level)));
      id = id / size_.at(level);
      std::vector<problem::Shape::FlattenedDimensionID> final_pattern = pattern.baked_prefix;
      final_pattern.insert(final_pattern.end(), permuted_suffix.begin(), permuted_suffix.end());

      retval.push_back(final_pattern);
    }
  }

  return retval;
}

uint128_t PermutationSpace::Size() const
{
  uint128_t product = 1;
  for (unsigned level = 0; level < num_levels_; level++)
  {
    product *= size_.at(level);
  }
  return product;
}

//--------------------------------------------//
//              SpatialSplitSpace             //
//--------------------------------------------//

SpatialSplitSpace::SpatialSplitSpace() {}

void SpatialSplitSpace::Init(uint64_t num_levels)
{
  num_levels_ = num_levels;
  is_user_specified_.clear();
  user_splits_.clear();
  size_.clear();
  unit_factors_.clear();
}

void SpatialSplitSpace::InitLevel(uint64_t level, unsigned unit_factors)
{
  assert(level < num_levels_);
  is_user_specified_[level] = false;
  unit_factors_[level] = unit_factors;
  size_[level] = int(problem::GetShape()->NumFlattenedDimensions) + 1 - unit_factors;
}

void SpatialSplitSpace::InitLevelUserSpecified(uint64_t level, std::uint32_t user_split)
{
  assert(level < num_levels_);
  is_user_specified_[level] = true;
  user_splits_[level] = user_split;
  size_[level] = 1;
}

std::map<unsigned, std::uint32_t>
SpatialSplitSpace::GetSplits(uint128_t id)
{
  std::map<unsigned, std::uint32_t> retval;
    
  for (unsigned level = 0; level < num_levels_; level++)
  {
    // Is this a spatial level (i.e., was this level initialized at all)?
    auto it_is_user_specified = is_user_specified_.find(level);
    if (it_is_user_specified != is_user_specified_.end())
    {
      // Was this level user-specified?
      if (it_is_user_specified->second)
      {
        // User-specified
        retval[level] = user_splits_.at(level);
      }
      else
      {
        // Variable
        retval[level] = unit_factors_.at(level) + std::uint32_t(id % size_.at(level));
        id = id / size_.at(level);
      }
    }      
  }

  return retval;
}

uint128_t SpatialSplitSpace::Size() const
{
  uint128_t retval = 1;
  for (auto& it : size_)
    retval *= uint128_t(it.second);
  return retval;
}  

} // namespace mapspace
