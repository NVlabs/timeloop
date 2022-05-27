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

#include <regex>

#include "mapping/constraints.hpp"

namespace mapping
{

//--------------------------------------------//
//                 Constraints                //
//--------------------------------------------//

Constraints::Constraints(const ArchProperties& arch_props,
                         const problem::Workload& workload) :
    arch_props_(arch_props),
    workload_(workload)
{
  factors_.clear();
  max_factors_.clear();
  permutations_.clear();
  spatial_splits_.clear();
  confidence_thresholds_.clear();
  bypass_strings_.clear();
  min_parallelism_ = 0.0;
  min_parallelism_isset_ = false;
  skews_.clear();
  no_multicast_.clear();
  no_link_transfer_.clear();
  no_temporal_reuse_.clear();

  // Initialize user bypass strings to "XXXXX...1" (note the 1 at the end).
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    std::string xxx(arch_props_.StorageLevels(), 'X');
    xxx.back() = '1';
    bypass_strings_[problem::Shape::DataSpaceID(pvi)] = xxx;
  }

  // Initialize confidence thresholds to be 1.0
  // We want to constraint the mapspace as much as possible by default
  for (unsigned storage_level_id = 0; storage_level_id < arch_props_.StorageLevels(); storage_level_id++)
  {
    confidence_thresholds_[storage_level_id] = 1.0;
  }
}

const std::map<unsigned, std::map<problem::Shape::FlattenedDimensionID, int>>&
  Constraints::Factors() const
{
  return factors_;
}

const std::map<unsigned, std::map<problem::Shape::FlattenedDimensionID, int>>&
  Constraints::MaxFactors() const
{
  return max_factors_;
}

const std::map<unsigned, std::vector<problem::Shape::FlattenedDimensionID>>&
  Constraints::Permutations() const
{
  return permutations_;
}

const std::map<unsigned, std::uint32_t>&
Constraints::SpatialSplits() const
{
  return spatial_splits_;
}
  
const problem::PerDataSpace<std::string>&
Constraints::BypassStrings() const
{
  return bypass_strings_;
}

double Constraints::MinParallelism() const
{
  return min_parallelism_;
}

const std::map<unsigned, double>&
Constraints::ConfidenceThresholds() const
{
  return confidence_thresholds_;
}

const std::unordered_map<unsigned, loop::Nest::SkewDescriptor>
Constraints::Skews() const
{
  return skews_;
}

const std::unordered_map<unsigned, problem::PerDataSpace<bool>>
Constraints::NoMulticast() const
{
  return no_multicast_;
}

const std::unordered_map<unsigned, problem::PerDataSpace<bool>>
Constraints::NoLinkTransfers() const
{
  return no_link_transfer_;
}

const std::unordered_map<unsigned, problem::PerDataSpace<bool>>
Constraints::NoTemporalReuse() const
{
  return no_temporal_reuse_;
}

//
// Create a constraints object from a given mapping object. The resultant
// constraints will *only* be satisfied by that mapping.
//
void Constraints::Generate(Mapping* mapping)
{
  auto num_storage_levels = mapping->loop_nest.storage_tiling_boundaries.size();
    
  // Data-space Bypass.
  auto mask_nest = tiling::TransposeMasks(mapping->datatype_bypass_nest);
    
  for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
  {
    auto& compound_mask = mask_nest.at(storage_level);
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      problem::Shape::DataSpaceID pv = problem::Shape::DataSpaceID(pvi);
      if (compound_mask.at(pv))
        bypass_strings_.at(pv).at(storage_level) = '1';
      else
        bypass_strings_.at(pv).at(storage_level) = '0';
    }
  }
    
  // Factors, Permutations and Spatial Split.
  unsigned loop_level = 0;
  for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
  {
    std::map<spacetime::Dimension, std::vector<problem::Shape::FlattenedDimensionID>> permutations;
    std::map<spacetime::Dimension, std::map<problem::Shape::FlattenedDimensionID, int>> factors;
    unsigned spatial_split;
      
    // Collect loop bounds and ordering.
    for (unsigned sdi = 0; sdi < unsigned(spacetime::Dimension::Num); sdi++)
    {
      auto sd = spacetime::Dimension(sdi);
      permutations[sd] = { };
      for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumFlattenedDimensions); idim++)
        factors[sd][problem::Shape::FlattenedDimensionID(idim)] = 1;
    }

    for (; loop_level <= mapping->loop_nest.storage_tiling_boundaries.at(storage_level); loop_level++)
    {
      auto& loop = mapping->loop_nest.loops.at(loop_level);
      if (loop.end > 1)
      {
        factors.at(loop.spacetime_dimension).at(loop.dimension) = loop.end;
        permutations.at(loop.spacetime_dimension).push_back(loop.dimension);
      }
    }

    // Determine X-Y split.
    spatial_split = permutations.at(spacetime::Dimension::SpaceX).size();
    
    // Merge spatial X and Y factors and permutations.
    std::vector<problem::Shape::FlattenedDimensionID> spatial_permutation;
    spatial_permutation = permutations.at(spacetime::Dimension::SpaceX);
    spatial_permutation.insert(spatial_permutation.end(),
                               permutations.at(spacetime::Dimension::SpaceY).begin(),
                               permutations.at(spacetime::Dimension::SpaceY).end());
      
    // Only generate spatial constraints if there is a spatial permutation.
    if (spatial_permutation.size() > 0)
    {
      std::map<problem::Shape::FlattenedDimensionID, int> spatial_factors;
      for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumFlattenedDimensions); idim++)
      {
        auto dim = problem::Shape::FlattenedDimensionID(idim);
        spatial_factors[dim] =
          factors.at(spacetime::Dimension::SpaceX).at(dim) *
          factors.at(spacetime::Dimension::SpaceY).at(dim);

        // If the factor is 1, concatenate it to the permutation.
        if (spatial_factors.at(dim) == 1)
          spatial_permutation.push_back(dim);
      }

      auto tiling_level = arch_props_.SpatialToTiling(storage_level);
      factors_[tiling_level] = spatial_factors;
      permutations_[tiling_level] = spatial_permutation;
      spatial_splits_[tiling_level] = spatial_split;
    }

    auto& temporal_permutation = permutations.at(spacetime::Dimension::Time);
    auto& temporal_factors = factors.at(spacetime::Dimension::Time);
    
    // Temporal factors: if the factor is 1, concatenate it into the permutation.
    for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumFlattenedDimensions); idim++)
    {
      auto dim = problem::Shape::FlattenedDimensionID(idim);
      if (temporal_factors.at(dim) == 1)
        temporal_permutation.push_back(dim);
    }
    
    auto tiling_level = arch_props_.TemporalToTiling(storage_level);
    factors_[tiling_level] = temporal_factors;
    permutations_[tiling_level] = temporal_permutation;
  }
}
  
//
// Check if a given Constraints object is a subset (i.e., more constrained).
//
bool Constraints::operator >= (const Constraints& other) const
{
  // This other constraints must be *equal or more* constrained than us.
  // Walk through each constraint type and check for this property.

  // --- Factors vs. other's Factors ---
  for (auto& level_entry: factors_)
  {
    unsigned level = level_entry.first;
    auto& level_factors = level_entry.second;

    auto other_level_it = other.factors_.find(level);
    if (other_level_it == other.factors_.end())
    {
      // Other doesn't have an entry for this level. This is almost
      // certainly a FAIL, unless our factors object is an empty map.
      if (!level_factors.empty())
      {
        return false;
      }
    }

    auto& other_level_factors = other_level_it->second;
    for (auto& dim_entry: level_factors)
    {
      auto dim = dim_entry.first;
      auto val = dim_entry.second;
        
      auto other_dim_entry = other_level_factors.find(dim);
      if (other_dim_entry == other_level_factors.end())
      {
        // Other doesn't have an entry for this dimension. Ergo, it
        // is less constrained than us.
        return false;
      }
      else if (other_dim_entry->second != val)
      {
        // Constraints are different.
        return false;
      }
    }
  }

  // =================================
  // FIXME: this method is incomplete.
  // =================================
  std::cerr << "ERROR: constraints subset-check implementation is incomplete." << std::endl;
  exit(1);

  // for (auto& level_entry: max_factors_)
  // {
  //   unsigned level = level_entry.first;
  //   auto& level_max_factors = level_entry.second;

  //   auto other_level_it = other.factors_.find(level);
  //   // RESUME HERE.      

  //   auto other_max_level_it = other.max_factors_.find(level);
  //   if (other_max_level_it == other.max_factors_.end())
  //   {
  //     // Other doesn't have an entry for this level. This is almost
  //     // certainly a FAIL, unless our max_factors object is an empty map.
  //     if (!level_max_factors.empty())
  //     {
  //       return false;
  //     }
  //   }

  //   other_level_max_factors = other_level_it->second;
  //   for (auto& dim_entry: level_max_factors)
  //   {
  //     auto dim = dim_entry.first;
  //     auto val = dim_entry.second;
        
  //     auto other_dim_entry = other_level_max_factors.find(dim);
  //     if (other_dim_entry == other_level_max_factors.end())
  //     {
  //       // Other doesn't have an entry for this dimension. Ergo, it
  //       // is less constrained than us.
  //       return false;
  //     }
  //     else if (other_dim_entry->second > val)
  //     {
  //       // Other has an entry, but its max value is greater than our
  //       // max value. Ergo, it is less constrained than us.
  //       return false;
  //     }
  //   }      
  // }

  return true;
}

//
// Check if a given mapping satisfies these constraints.
//
bool Constraints::SatisfiedBy(Mapping* mapping) const
{
  // First generate a constraints object from the mapping.
  Constraints other(arch_props_, workload_);
  other.Generate(mapping);

  // We would like to use the more general subset-check function, but
  // its implementation is incomplete. Instead, we have some narrower checks
  // below which suffices for checking a specific mapping.

  // (NOT IMPLEMENTED) return (*this >= candidate);

  // --- Factors vs. other's Factors ---
  for (auto& level_entry: factors_)
  {
    unsigned level = level_entry.first;
    auto& level_factors = level_entry.second;

    auto other_level_it = other.factors_.find(level);
    assert(other_level_it != other.factors_.end());

    auto other_level_factors = other_level_it->second;
    for (auto& dim_entry: level_factors)
    {
      auto dim = dim_entry.first;
      auto val = dim_entry.second;
        
      auto other_dim_entry = other_level_factors.find(dim);
      assert(other_dim_entry != other_level_factors.end());

      if (other_dim_entry->second != val)
      {
        return false;
      }
    }
  }

  // --- Max Factors vs. other's Factors ---
  // Note: other doesn't have a Max Factors because it was derived from
  // a single mapping.
  for (auto& level_entry: max_factors_)
  {
    unsigned level = level_entry.first;
    auto& level_max_factors = level_entry.second;

    auto other_level_it = other.factors_.find(level);
    assert(other_level_it != other.factors_.end());

    auto& other_level_factors = other_level_it->second;
    for (auto& dim_entry: level_max_factors)
    {
      auto dim = dim_entry.first;
      auto max_val = dim_entry.second;
        
      auto other_dim_entry = other_level_factors.find(dim);
      assert(other_dim_entry != other_level_factors.end());

      if (other_dim_entry->second > max_val)
      {
        return false;
      }
    }
  }
    
  // --- Permutations ---
  for (auto& level_entry: permutations_)
  {
    // This is tricky. We need to ignore unit-factors in other.
    unsigned level = level_entry.first;
    auto& permutation = level_entry.second;

    auto other_factors_level_it = other.factors_.find(level);
    assert(other_factors_level_it != other.factors_.end());
    auto& other_level_factors = other_factors_level_it->second;

    auto other_permutation_level_it = other.permutations_.find(level);
    assert(other_permutation_level_it != other.permutations_.end());
    auto& other_permutation = other_permutation_level_it->second;
      
    unsigned idx = 0, other_idx = 0;
    while (idx < permutation.size() && other_idx < other_permutation.size())
    {
      if (permutation.at(idx) == other_permutation.at(other_idx))
      {
        // So far so good...
        idx++;
        other_idx++;
        continue;
      }

      // We have a mismatch. However, if this is a unit-factor
      // we can skip ahead.
      if (other_level_factors.at(other_permutation.at(idx)) == 1)
      {
        other_idx++;
        continue;
      }

      // Fail.
      return false;
    }
  }
    
  // --- Spatial splits ---
  for (auto& level_entry: spatial_splits_)
  {
    unsigned level = level_entry.first;
    auto split = level_entry.second;

    auto other_level_it = other.spatial_splits_.find(level);
    assert(other_level_it != other.spatial_splits_.end());

    if (split != other_level_it->second)
    {
      return false;
    }
  }

  // --- Bypass strings ---
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto& my_str = bypass_strings_.at(problem::Shape::DataSpaceID(pvi));
    auto& other_str = other.bypass_strings_.at(problem::Shape::DataSpaceID(pvi));

    for (unsigned level = 0; level < arch_props_.StorageLevels(); level++)
    {
      auto& my_char = my_str.at(level);
      auto& other_char = other_str.at(level);

      if (my_char != 'X' && my_char != other_char)
      {
        // FIXME: Default string with last level set to 1 is not right, failed on SCNN IORAM
        //return false;
      }
    }
  }

  return true;
}

//
// Parse user-provided constraints.
//
void Constraints::Parse(config::CompoundConfigNode config)
{
  // This is primarily a wrapper function written to handle various ways to
  // get to the list of constraints. The only reason there are multiple ways
  // to get to this list is because of backwards compatibility.
  if (config.isList())
  {
    // We're already at the constraints list.
    ParseList(config);
  }
  else
  {
    // Constraints can be specified either anonymously as a list, or indirected
    // via a string name.
    std::string name = "";
    if (config.exists("constraints"))
    {
      if (config.lookup("constraints").isList())
        name = "constraints";
      else if (config.lookupValue("constraints", name))
        name = std::string("constraints_") + name;
    }
    else if (config.exists("targets"))
    {
      if (config.lookup("targets").isList())
        name = "targets";
    }

    if (name != "")
    {
      auto constraints = config.lookup(name);
      ParseList(constraints);
    }
    // else: no constraints specified, nothing to do.
  }
}  

//
// Parse a list of constraints.
//
void Constraints::ParseList(config::CompoundConfigNode constraints)
{
  assert(constraints.isList());

  // Iterate over all the constraints/targets.
  int len = constraints.getLength();
  for (int i = 0; i < len; i++)
  {
    // Prepare a set of CompoundConfigNodes that we will be parsing. These
    // nodes can be in different places in the hierarchy for backwards-
    // compatibility reasons.
    config::CompoundConfigNode target, constraint, attributes;

    target = constraints[i];
    if (target.exists("constraints"))
    {
      auto constraints_list = target.lookup("constraints");
      assert(constraints_list.isList());
        
      for (int j = 0; j < constraints_list.getLength(); j++)
      {
        auto constraint = constraints_list[j];

        config::CompoundConfigNode attributes;
        if (constraint.exists("attributes"))
          attributes = constraint.lookup("attributes");
        else
          attributes = constraint; // Backwards compatibility.

        ParseSingleConstraint(target, constraint, attributes);
      }
    }
    else // Backwards compatibility.
    {
      auto constraint = target;

      config::CompoundConfigNode attributes;
      if (constraint.exists("attributes"))
        attributes = constraint.lookup("attributes");
      else
        attributes = constraint; // Backwards compatibility.

      ParseSingleConstraint(target, constraint, attributes);
    }
  }    
}

//
// Parse a single user constraint.
//
void Constraints::ParseSingleConstraint(
  config::CompoundConfigNode target,
  config::CompoundConfigNode constraint,
  config::CompoundConfigNode attributes)
{
  // Find out if this is a temporal constraint or a spatial constraint.
  std::string type;
  assert(constraint.lookupValue("type", type));

  auto level_id = FindTargetTilingLevel(target, type);

  if (type == "temporal" || type == "spatial")
  {
    auto level_factors = ParseFactors(attributes);
    for (auto& factor: level_factors)
    {
      if (factors_[level_id].find(factor.first) != factors_[level_id].end())
      {
        std::cerr << "ERROR: re-specification of factor for dimension "
                  << problem::GetShape()->FlattenedDimensionIDToName.at(factor.first)
                  << " at level " << arch_props_.TilingLevelName(level_id)
                  << ". This may imply a conflict between architecture and "
                  << "mapspace constraints." << std::endl;
        exit(1);
      }
      factors_[level_id][factor.first] = factor.second;
    }

    auto level_max_factors = ParseMaxFactors(attributes);
    for (auto& max_factor: level_max_factors)
    {
      if (max_factors_[level_id].find(max_factor.first) != max_factors_[level_id].end())
      {
        std::cerr << "ERROR: re-specification of max factor for dimension "
                  << problem::GetShape()->FlattenedDimensionIDToName.at(max_factor.first)
                  << " at level " << arch_props_.TilingLevelName(level_id)
                  << ". This may imply a conflict between architecture and "
                  << "mapspace constraints." << std::endl;
        exit(1);
      }
      max_factors_[level_id][max_factor.first] = max_factor.second;
    }

    auto level_permutations = ParsePermutations(attributes);
    if (level_permutations.size() > 0)
    {
      if (permutations_[level_id].size() > 0)
      {
        std::cerr << "ERROR: re-specification of permutation at level "
                  << arch_props_.TilingLevelName(level_id)
                  << ". This may imply a conflict between architecture and "
                  << "mapspace constraints." << std::endl;
        exit(1);
      }
      permutations_[level_id] = level_permutations;
    }

    if (type == "spatial")
    {
      std::uint32_t split;
      if (constraint.lookupValue("split", split))
      {
        if (spatial_splits_.find(level_id) != spatial_splits_.end())
        {
          std::cerr << "ERROR: re-specification of spatial split at level "
                    << arch_props_.TilingLevelName(level_id)
                    << ". This may imply a conflict between architecture and "
                    << "mapspace constraints." << std::endl;
          exit(1);
        }
        spatial_splits_[level_id] = split;
      }
      // No link transfer
      
      if (constraint.exists("no_link_transfer"))
      {
        auto storage_level = arch_props_.TilingToStorage(level_id);
        std::vector<std::string> datatype_strings;
        constraint.lookupArrayValue("no_link_transfer", datatype_strings);
        if (no_link_transfer_.find(storage_level) != no_link_transfer_.end())
        {
          std::cerr << "ERROR: re-specification of no_link_transfer at level "
                    << arch_props_.TilingLevelName(level_id)
                    << ". This may imply a conflict between architecture and "
                    << "mapspace constraints." << std::endl;
          exit(1);
        }
        no_link_transfer_[storage_level] = problem::PerDataSpace<bool>();
        for(unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
          no_link_transfer_[storage_level][pv] = 0;        
        for (const std::string& datatype_string: datatype_strings)
        {
          try
          {
            no_link_transfer_[storage_level].at(
              problem::GetShape()->DataSpaceNameToID.at(datatype_string)) = 1;
          }
          catch (std::out_of_range& oor)
          {
            std::cerr << "ERROR: parsing no_link_transfer setting: data-space " << datatype_string
                      << " not found in problem shape." << std::endl;
            exit(1);
          }
        }
      }
      // No multicast no reduction
      if (constraint.exists("no_multicast_no_reduction"))
      {
        auto storage_level = arch_props_.TilingToStorage(level_id);
        std::vector<std::string> datatype_strings;
        constraint.lookupArrayValue("no_multicast_no_reduction", datatype_strings);
        if (no_multicast_.find(storage_level) != no_multicast_.end())
        {
          std::cerr << "ERROR: re-specification of no_multicast_no_reduction at level "
                    << arch_props_.TilingLevelName(level_id)
                    << ". This may imply a conflict between architecture and "
                    << "mapspace constraints." << std::endl;
          exit(1);
        }
        no_multicast_ [storage_level] = problem::PerDataSpace<bool>();
        for(unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
          no_multicast_[storage_level][pv] = 0;
        for (const std::string& datatype_string: datatype_strings)
        {
          try
          {
            no_multicast_[storage_level].at(
              problem::GetShape()->DataSpaceNameToID.at(datatype_string)) = 1;
          }
          catch (std::out_of_range& oor)
          {
            std::cerr << "ERROR: parsing no_multicast_no_reduction setting: data-space " << datatype_string
                      << " not found in problem shape." << std::endl;
            exit(1);
          }
        }
      }
    }
    if (type == "temporal")
    {
      // No temporal reuse
      if (constraint.exists("no_temporal_reuse"))
      {
        auto storage_level = arch_props_.TilingToStorage(level_id);
        std::vector<std::string> datatype_strings;
        constraint.lookupArrayValue("no_temporal_reuse", datatype_strings);
        if (no_temporal_reuse_.find(storage_level) != no_temporal_reuse_.end())
        {
          std::cerr << "ERROR: re-specification of no_temporal_reuse at level "
                    << arch_props_.TilingLevelName(level_id)
                    << ". This may imply a conflict between architecture and "
                    << "mapspace constraints." << std::endl;
          exit(1);
        }
        no_temporal_reuse_ [storage_level] = problem::PerDataSpace<bool>();
        for(unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
          no_temporal_reuse_[storage_level][pv] = 0;
        for (const std::string& datatype_string: datatype_strings)
        {
          try
          {
            no_temporal_reuse_[storage_level].at(
              problem::GetShape()->DataSpaceNameToID.at(datatype_string)) = 1;
          }
          catch (std::out_of_range& oor)
          {
            std::cerr << "ERROR: parsing no_temporal_reuse setting: data-space " << datatype_string
                      << " not found in problem shape." << std::endl;
            exit(1);
          }
        }
      }

    }
  }
  else if (type == "max_overbooked_proportion")
  {
    double max_overbooked_proportion;
    assert(constraint.lookupValue("proportion", max_overbooked_proportion));
    confidence_thresholds_[level_id] =  1 - max_overbooked_proportion;
  }
  else if (type == "datatype" || type == "bypass" || type == "bypassing")
  {
    // Error handling for re-spec conflicts are inside the parse function.
    ParseDatatypeBypassSettings(attributes, arch_props_.TilingToStorage(level_id));
  }
  else if (type == "utilization" || type == "parallelism")
  {
    if (min_parallelism_isset_)
    {
      std::cerr << "ERROR: re-specification of min parallelism/utilization at level "
                << arch_props_.TilingLevelName(level_id)
                << ". This may imply a conflict between architecture and "
                << "mapspace constraints." << std::endl;
      exit(1);
    }
    assert(attributes.lookupValue("min", min_parallelism_));
    min_parallelism_isset_ = true;
  }
  else
  {
    assert(false);
  }
}

//
// FindTargetTilingLevel()
//
unsigned Constraints::FindTargetTilingLevel(config::CompoundConfigNode constraint, std::string type)
{
  auto num_storage_levels = arch_props_.StorageLevels();
    
  //
  // Find the target storage level. This can be specified as either a name or an ID.
  //
  std::string storage_level_name;
  unsigned storage_level_id;
    
  if (constraint.lookupValue("target", storage_level_name) ||
      constraint.lookupValue("name", storage_level_name))
  {
    // Find this name within the storage hierarchy in the arch specs.
    for (storage_level_id = 0; storage_level_id < num_storage_levels; storage_level_id++)
    {
      if (arch_props_.StorageLevelName(storage_level_id) == storage_level_name)
        break;
    }
    if (storage_level_id == num_storage_levels)
    {
      std::cerr << "ERROR: target storage level not found: " << storage_level_name << std::endl;
      exit(1);
    }
  }
  else
  {
    int id;
    assert(constraint.lookupValue("target", id));
    assert(id >= 0 && id < int(num_storage_levels));
    storage_level_id = static_cast<unsigned>(id);
  }
    
  assert(storage_level_id < num_storage_levels);    

  //
  // Translate this storage ID to a tiling ID.
  //
  unsigned tiling_level_id;
  if (type == "temporal" || type == "datatype" || type == "bypass" || type == "bypassing" || type == "max_overbooked_proportion")
  {
    // This should always succeed.
    tiling_level_id = arch_props_.TemporalToTiling(storage_level_id);
  }
  else if (type == "spatial")
  {
    // This will fail if this level isn't a spatial tiling level.
    try
    {
      tiling_level_id = arch_props_.SpatialToTiling(storage_level_id);
    }
    catch (const std::out_of_range& oor)
    {
      std::cerr << "ERROR: cannot find spatial tiling level associated with "
                << "storage level " << arch_props_.StorageLevelName(storage_level_id)
                << ". This is because the number of instances of the next-inner "
                << "level ";
      if (storage_level_id != 0)
      {
        std::cerr << "(" << arch_props_.StorageLevelName(storage_level_id-1) << ") ";
      }
      std::cerr << "is the same as this level, which means there cannot "
                << "be a spatial fanout." << std::endl;
      exit(1);
    }
  }
  else if (type == "utilization" || type == "parallelism")
  {
    // For now, we only allow parallelism to be specified for level 0.
    // Note that this is the level 0 storage, not arithmetic. Fanout from
    // level 0 to arithmetic is undefined. The parallelism specified here
    // is the cumulative fanout utilization all the way from the top of
    // the storage tree down to level 0.
    if (storage_level_id != 0)
    {
      std::cerr << "ERROR: parallelism cannot be constrained at level "
                << storage_level_name << ". It must be constrained at the "
                << "innermost storage level." << std::endl;
      exit(1);
    }
    tiling_level_id = arch_props_.TemporalToTiling(storage_level_id);
  }
  else
  {
    std::cerr << "ERROR: unrecognized constraint type: " << type << std::endl;
    exit(1);
  }

  return tiling_level_id;
}

//
// Parse user factors.
//
std::map<problem::Shape::FlattenedDimensionID, int>
Constraints::ParseFactors(config::CompoundConfigNode constraint)
{
  std::map<problem::Shape::FlattenedDimensionID, int> retval;

  std::string buffer;
  if (constraint.lookupValue("factors", buffer))
  {
    std::regex re("([A-Za-z]+)[[:space:]]*[=]*[[:space:]]*([0-9]+)", std::regex::extended);
    std::smatch sm;
    std::string str = std::string(buffer);

    while (std::regex_search(str, sm, re))
    {
      std::string dimension_name = sm[1];
      problem::Shape::FlattenedDimensionID dimension;
      try
      {
        dimension = problem::GetShape()->FlattenedDimensionNameToID.at(dimension_name);
      }
      catch (const std::out_of_range& oor)
      {
        std::cerr << "ERROR: parsing factors: " << buffer << ": dimension " << dimension_name
                  << " not found in problem shape." << std::endl;
        exit(1);
      }

      int end = std::stoi(sm[2]);
      if (end == 0)
      {
        std::cerr << "WARNING: Interpreting 0 to mean full problem dimension instead of residue." << std::endl;
        end = workload_.GetFlattenedBound(dimension);
      }
      else if (end > workload_.GetFlattenedBound(dimension))
      {
        std::cerr << "WARNING: Constraint " << dimension_name << "=" << end
                  << " exceeds problem dimension " << dimension_name << "="
                  << workload_.GetFlattenedBound(dimension) << ". Setting constraint "
                  << dimension << "=" << workload_.GetFlattenedBound(dimension) << std::endl;
        end = workload_.GetFlattenedBound(dimension);
      }
      else
      {
        assert(end > 0);
      }

      // Found all the information we need to setup a factor!
      retval[dimension] = end;

      str = sm.suffix().str();
    }
  }

  return retval;
}

//
// Parse user max factors.
//
std::map<problem::Shape::FlattenedDimensionID, int>
Constraints::ParseMaxFactors(config::CompoundConfigNode constraint)
{
  std::map<problem::Shape::FlattenedDimensionID, int> retval;

  std::string buffer;
  if (constraint.lookupValue("factors", buffer))
  {
    std::regex re("([A-Za-z]+)[[:space:]]*<=[[:space:]]*([0-9]+)", std::regex::extended);
    std::smatch sm;
    std::string str = std::string(buffer);

    while (std::regex_search(str, sm, re))
    {
      std::string dimension_name = sm[1];
      problem::Shape::FlattenedDimensionID dimension;
      try
      {
        dimension = problem::GetShape()->FlattenedDimensionNameToID.at(dimension_name);
      }
      catch (const std::out_of_range& oor)
      {
        std::cerr << "ERROR: parsing factors: " << buffer << ": dimension " << dimension_name
                  << " not found in problem shape." << std::endl;
        exit(1);
      }

      int max = std::stoi(sm[2]);
      if (max <= 0)
      {
        std::cerr << "ERROR: max factor must be positive in constraint: " << buffer << std::endl;
        exit(1);
      }

      // Found all the information we need to setup a factor!
      retval[dimension] = max;

      str = sm.suffix().str();
    }
  }

  return retval;
}

//
// Parse user permutations.
//
std::vector<problem::Shape::FlattenedDimensionID>
Constraints::ParsePermutations(config::CompoundConfigNode constraint)
{
  std::vector<problem::Shape::FlattenedDimensionID> retval;
    
  std::string buffer;
  if (constraint.lookupValue("permutation", buffer))
  {
    std::istringstream iss(buffer);
    char token;
    while (iss >> token)
    {
      problem::Shape::FlattenedDimensionID dimension;
      try
      {
        dimension = problem::GetShape()->FlattenedDimensionNameToID.at(std::string(1, token));
      }
      catch (const std::out_of_range& oor)
      {
        std::cerr << "ERROR: parsing permutation: " << buffer << ": dimension " << token
                  << " not found in problem shape." << std::endl;
        exit(1);
      }
      retval.push_back(dimension);
    }
  }

  return retval;
}

//
// Parse user datatype bypass settings.
//
void Constraints::ParseDatatypeBypassSettings(config::CompoundConfigNode constraint, unsigned level)
{
  // Datatypes to "keep" at this level.
  if (constraint.exists("keep"))
  {
    std::vector<std::string> datatype_strings;
    constraint.lookupArrayValue("keep", datatype_strings);
    for (const std::string& datatype_string: datatype_strings)
    {
      problem::Shape::DataSpaceID datatype;
      try
      {
        datatype = problem::GetShape()->DataSpaceNameToID.at(datatype_string);
      }
      catch (std::out_of_range& oor)
      {
        std::cerr << "ERROR: parsing keep setting: data-space " << datatype_string
                  << " not found in problem shape." << std::endl;
        exit(1);
      }
      // FIXME: no error handling for overwriting last-level bypass setting.
      if (level != arch_props_.StorageLevels()-1 &&
          bypass_strings_.at(datatype).at(level) != 'X')
      {
        std::cerr << "ERROR: re-specification of dataspace keep flag at level "
                  << arch_props_.StorageLevelName(level) << ". This may imply a "
                  << "conflict between architecture and mapspace constraints."
                  << std::endl;
        exit(1);          
      }
      bypass_strings_.at(datatype).at(level) = '1';
    }
  }
      
  // Datatypes to "bypass" at this level.
  if (constraint.exists("bypass"))
  {
    std::vector<std::string> datatype_strings;
    constraint.lookupArrayValue("bypass", datatype_strings);
    for (const std::string& datatype_string: datatype_strings)
    {
      problem::Shape::DataSpaceID datatype;
      try
      {
        datatype = problem::GetShape()->DataSpaceNameToID.at(datatype_string);
      }
      catch (std::out_of_range& oor)
      {
        std::cerr << "ERROR: parsing bypass setting: data-space " << datatype_string
                  << " not found in problem shape." << std::endl;
        exit(1);
      }
      if (level != arch_props_.StorageLevels()-1 &&
          bypass_strings_.at(datatype).at(level) != 'X')
      {
        std::cerr << "ERROR: re-specification of dataspace bypass flag at level "
                  << arch_props_.StorageLevelName(level) << ". This may imply a "
                  << "conflict between architecture and mapspace constraints."
                  << std::endl;
        exit(1);          
      }
      bypass_strings_.at(datatype).at(level) = '0';
    }
  }
}

} // namespace mapping
