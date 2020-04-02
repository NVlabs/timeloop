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

#include <map>
#include <regex>

#pragma once

#include "mapping/arch-properties.hpp"

namespace mapping
{

//--------------------------------------------//
//                 Constraints                //
//--------------------------------------------//

class Constraints
{
 protected:

  // Abstract representation of the architecture.
  const ArchProperties& arch_props_;

  // Workload.
  const problem::Workload& workload_;  
 
  // The constraints.
  std::map<unsigned, std::map<problem::Shape::DimensionID, int>> factors_;
  std::map<unsigned, std::map<problem::Shape::DimensionID, int>> max_factors_;
  std::map<unsigned, std::vector<problem::Shape::DimensionID>> permutations_;
  std::map<unsigned, std::uint32_t> spatial_splits_;
  problem::PerDataSpace<std::string> bypass_strings_;
  double min_parallelism_;
  bool min_parallelism_isset_;  

 public:
  Constraints() = delete;

  Constraints(const ArchProperties& arch_props,
              const problem::Workload& workload) :
      arch_props_(arch_props),
      workload_(workload)
  {
    factors_.clear();
    max_factors_.clear();
    permutations_.clear();
    spatial_splits_.clear();
    bypass_strings_.clear();
    min_parallelism_ = 0.0;
    min_parallelism_isset_ = false;

    // Initialize user bypass strings to "XXXXX...1" (note the 1 at the end).
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      std::string xxx(arch_props_.StorageLevels(), 'X');
      xxx.back() = '1';
      bypass_strings_[problem::Shape::DataSpaceID(pvi)] = xxx;
    }
  }

  const std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& Factors() const
  {
    return factors_;
  }

  const std::map<unsigned, std::map<problem::Shape::DimensionID, int>>& MaxFactors() const
  {
    return max_factors_;
  }

  const std::map<unsigned, std::vector<problem::Shape::DimensionID>>& Permutations() const
  {
    return permutations_;
  }

  const std::map<unsigned, std::uint32_t>& SpatialSplits() const
  {
    return spatial_splits_;
  }
  
  const problem::PerDataSpace<std::string>& BypassStrings() const
  {
    return bypass_strings_;
  }

  double MinParallelism()
  {
    return min_parallelism_;
  }

  //
  // Check if a given mapping satisfies these constraints.
  //
  bool SatisfiedBy(Mapping* mapping)
  {
    (void) mapping;
    // bool success = true;

    // auto num_storage_levels = loop_nest.storage_tiling_boundaries.size();
  
    // // Datatype Bypass.
    // auto mask_nest = tiling::TransposeMasks(datatype_bypass_nest);

    // for (unsigned level = 0; level < num_storage_levels; level++)
    // {
    //   auto& compound_mask = mask_nest.at(level);    
    //   for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    //   {
    //     problem::Shape::DataSpaceID pv = problem::Shape::DataSpaceID(pvi);
    //     if (compound_mask.at(pv))
    //     {
    //       success &= (constraints->BypassStrings().at(pv).at(level) == 'X' ||
    //                   constraints->BypassStrings().at(pv).at(level) == '1');
    //     }
    //     else
    //     {
    //       success &= (constraints->BypassStrings().at(pv).at(level) == 'X' ||
    //                   constraints->BypassStrings().at(pv).at(level) == '0');
    //     }
    //   }
    // }

    // if (!success)
    // {
    //   std::cerr << "WARNING: mapping violates dataspace bypass constraints." << std::endl;
    //   return false;
    // }

    // // Factors and Permutations.
    // unsigned loop_level = 0;
    // for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
    // {
    //   std::map<spacetime::Dimension, std::string> permutations;
    //   std::map<spacetime::Dimension, std::map<problem::Shape::DimensionID, unsigned>> factors;
    //   unsigned spatial_split;

    //   for (unsigned sdi = 0; sdi < unsigned(spacetime::Dimension::Num); sdi++)
    //   {
    //     auto sd = spacetime::Dimension(sdi);
    //     permutations[sd] = "";
    //     for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumDimensions); idim++)
    //       factors[sd][problem::Shape::DimensionID(idim)] = 1;
    //   }

    //   for (; loop_level <= loop_nest.storage_tiling_boundaries.at(storage_level); loop_level++)
    //   {
    //     auto& loop = loop_nest.loops.at(loop_level);
    //     if (loop.end > 1)
    //     {
    //       factors.at(loop.spacetime_dimension).at(loop.dimension) = loop.end;
    //       permutations.at(loop.spacetime_dimension) += problem::GetShape()->DimensionIDToName.at(loop.dimension);
    //     }
    //   }

    //   if (constraints_spatial_splits != constraints->SpatialSplits().end())
    //   {
    //     spatial_split = permutations.at(spacetime::Dimension::SpaceX).size();
    //     success &= (spatial_split == *constraints_spatial_splits);
    //   }

    //   if (!success)
    //   {
    //     std::cerr << "WARNING: splatial split constraint violation at level "
    //               << arch_props_.StorageLevelName(storage_level) << std::endl;
    //     return false;
    //   }

    // // Merge spatial X and Y factors and permutations.
    // std::string spatial_permutation =
    //   permutations.at(spacetime::Dimension::SpaceX) +
    //   permutations.at(spacetime::Dimension::SpaceY);
    
    // // Only print spatial constraints if there is a spatial permutation.
    // if (spatial_permutation.size() > 0)
    // {
    //   auto level = arch_props_.SpatialToTiling(storage_level);


    //   std::string spatial_factor_string = "";

    //   std::map<problem::Shape::DimensionID, unsigned> spatial_factors;
    //   for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumDimensions); idim++)
    //   {
    //     auto dim = problem::Shape::DimensionID(idim);
    //     spatial_factors[dim] =
    //       factors.at(spacetime::Dimension::SpaceX).at(dim) *
    //       factors.at(spacetime::Dimension::SpaceY).at(dim);

    //     spatial_factor_string += problem::GetShape()->DimensionIDToName.at(dim);
    //     char factor[8];
    //     sprintf(factor, "%d", spatial_factors.at(dim));
    //     spatial_factor_string += factor;
    //     if (idim != unsigned(problem::GetShape()->NumDimensions)-1)
    //       spatial_factor_string += " ";
        
    //     // If the factor is 1, concatenate it to the permutation.
    //     if (spatial_factors.at(dim) == 1)
    //       spatial_permutation += problem::GetShape()->DimensionIDToName.at(dim);
    //   }
      

    //   auto constraints_factors = constraints->Factors().find(level);
    //   auto constraints_max_factors = constraints->MaxFactors().find(level);
    //   auto constraints_permutations = constraints->Permutations().find(level);
    //   auto constraints_spatial_splits = constraints->SpatialSplits().find(level);
      

      
    //   libconfig::Setting& constraint = constraints.add(libconfig::Setting::TypeGroup);
    //   constraint.add("target", libconfig::Setting::TypeInt) = static_cast<int>(storage_level);
    //   constraint.add("type", libconfig::Setting::TypeString) = "spatial";
    //   constraint.add("factors", libconfig::Setting::TypeString) = spatial_factor_string;
    //   constraint.add("permutation", libconfig::Setting::TypeString) = spatial_permutation;
    //   constraint.add("split", libconfig::Setting::TypeInt) = static_cast<int>(spatial_split);
    // }

    // auto& temporal_permutation = permutations.at(spacetime::Dimension::Time);
    // auto& temporal_factors = factors.at(spacetime::Dimension::Time);
    // std::string temporal_factor_string = "";
    
    // // Temporal factors: if the factor is 1, concatenate it into the permutation.
    // for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumDimensions); idim++)
    // {
    //   auto dim = problem::Shape::DimensionID(idim);

    //   temporal_factor_string += problem::GetShape()->DimensionIDToName.at(dim);
    //   char factor[8];
    //   sprintf(factor, "%d", temporal_factors.at(dim));
    //   temporal_factor_string += factor;
    //   if (idim != unsigned(problem::GetShape()->NumDimensions)-1)
    //     temporal_factor_string += " ";
      
    //   if (temporal_factors.at(dim) == 1)
    //     temporal_permutation += problem::GetShape()->DimensionIDToName.at(dim);
    // }

    // libconfig::Setting& constraint = constraints.add(libconfig::Setting::TypeGroup);
    // constraint.add("target", libconfig::Setting::TypeInt) = static_cast<int>(storage_level);
    // constraint.add("type", libconfig::Setting::TypeString) = "temporal";
    // constraint.add("factors", libconfig::Setting::TypeString) = temporal_factor_string;
    // constraint.add("permutation", libconfig::Setting::TypeString) = temporal_permutation;
    
    // }
    
    return true;
  }

  //
  // Parse user-provided constraints.
  //
  void Parse(config::CompoundConfigNode config)
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
  void ParseList(config::CompoundConfigNode constraints)
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
  void ParseSingleConstraint(
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
                    << problem::GetShape()->DimensionIDToName.at(factor.first)
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
                    << problem::GetShape()->DimensionIDToName.at(max_factor.first)
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
      }
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
  unsigned FindTargetTilingLevel(config::CompoundConfigNode constraint, std::string type)
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
    if (type == "temporal" || type == "datatype" || type == "bypass" || type == "bypassing")
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
  std::map<problem::Shape::DimensionID, int> ParseFactors(config::CompoundConfigNode constraint)
  {
    std::map<problem::Shape::DimensionID, int> retval;

    std::string buffer;
    if (constraint.lookupValue("factors", buffer))
    {
      std::regex re("([A-Za-z]+)[[:space:]]*[=]*[[:space:]]*([0-9]+)", std::regex::extended);
      std::smatch sm;
      std::string str = std::string(buffer);

      while (std::regex_search(str, sm, re))
      {
        std::string dimension_name = sm[1];
        problem::Shape::DimensionID dimension;
        try
        {
          dimension = problem::GetShape()->DimensionNameToID.at(dimension_name);
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
          end = workload_.GetBound(dimension);
        }
        else if (end > workload_.GetBound(dimension))
        {
          std::cerr << "WARNING: Constraint " << dimension_name << "=" << end
                    << " exceeds problem dimension " << dimension_name << "="
                    << workload_.GetBound(dimension) << ". Setting constraint "
                    << dimension << "=" << workload_.GetBound(dimension) << std::endl;
          end = workload_.GetBound(dimension);
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
  std::map<problem::Shape::DimensionID, int> ParseMaxFactors(config::CompoundConfigNode constraint)
  {
    std::map<problem::Shape::DimensionID, int> retval;

    std::string buffer;
    if (constraint.lookupValue("factors", buffer))
    {
      std::regex re("([A-Za-z]+)[[:space:]]*<=[[:space:]]*([0-9]+)", std::regex::extended);
      std::smatch sm;
      std::string str = std::string(buffer);

      while (std::regex_search(str, sm, re))
      {
        std::string dimension_name = sm[1];
        problem::Shape::DimensionID dimension;
        try
        {
          dimension = problem::GetShape()->DimensionNameToID.at(dimension_name);
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
  std::vector<problem::Shape::DimensionID> ParsePermutations(config::CompoundConfigNode constraint)
  {
    std::vector<problem::Shape::DimensionID> retval;
    
    std::string buffer;
    if (constraint.lookupValue("permutation", buffer))
    {
      std::istringstream iss(buffer);
      char token;
      while (iss >> token)
      {
        problem::Shape::DimensionID dimension;
        try
        {
          dimension = problem::GetShape()->DimensionNameToID.at(std::string(1, token));
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
  void ParseDatatypeBypassSettings(config::CompoundConfigNode constraint, unsigned level)
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
};

} // namespace mapping
