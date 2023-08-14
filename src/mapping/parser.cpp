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

#include <regex>

#include "mapping/parser.hpp"
#include "mapping/arch-properties.hpp"

namespace mapping
{

//
// Shared state.
//

ArchProperties arch_props_;

//
// Forward declarations.
//
unsigned FindTargetStorageLevel(config::CompoundConfigNode directive);
unsigned FindTargetTilingLevel(config::CompoundConfigNode constraint, std::string type);
std::map<problem::Shape::FlattenedDimensionID, std::pair<int,int>> ParseUserFactors(
  config::CompoundConfigNode constraint, const problem::Workload& workload);
std::vector<problem::Shape::FlattenedDimensionID> ParseUserPermutations(config::CompoundConfigNode constraint);
void ParseUserDatatypeBypassSettings(config::CompoundConfigNode constraint,
                                     unsigned level,
                                     problem::PerDataSpace<std::string>& user_bypass_strings);
loop::Nest::SkewDescriptor ParseUserSkew(config::CompoundConfigNode directive);

//
// Parse mapping in libconfig format and generate data structure.
//
Mapping ParseAndConstruct(config::CompoundConfigNode config,
                          model::Engine::Specs& arch_specs,
                          const problem::Workload& workload)
{
  arch_props_ = ArchProperties();
  arch_props_.Construct(arch_specs);

  std::map<unsigned, std::map<problem::Shape::FlattenedDimensionID, std::pair<int,int>>> user_factors;
  std::map<unsigned, std::vector<problem::Shape::FlattenedDimensionID>> user_permutations;
  std::map<unsigned, std::uint32_t> user_spatial_splits;
  problem::PerDataSpace<std::string> user_bypass_strings;
  std::map<unsigned, double> confidence_thresholds;
  std::unordered_map<unsigned, loop::Nest::SkewDescriptor> user_skews;
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> no_link_transfer;
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> no_multicast;
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> no_temporal_reuse;
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> rmw_on_first_writeback;
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> passthrough;

  // Initialize user bypass strings to "XXXXX...1" (note the 1 at the end).
  // FIXME: there's probably a cleaner way/place to initialize this.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    std::string xxx(arch_props_.StorageLevels(), 'X');
    xxx.back() = '1';
    user_bypass_strings[problem::Shape::DataSpaceID(pvi)] = xxx;
  }

  // Initialize confidence thresholds to be 0.0
  // We want to allow the model to be able model any mapping as long as memory levels allow overflow
  for (unsigned storage_level_id = 0; storage_level_id < arch_props_.StorageLevels(); storage_level_id++)
  {
    confidence_thresholds[storage_level_id] = 0.0;
  }

  // Parse user-provided mapping.
  assert(config.isList());
  
  // Iterate over all the directives.
  int len = config.getLength();
  for (int i = 0; i < len; i ++)
  {
    auto directive = config[i];
    // Find out if this is a temporal directive or a spatial directive.
    std::string type;
    assert(directive.lookupValue("type", type));

    if (type == "temporal" || type == "spatial")
    {
      auto level_id = FindTargetTilingLevel(directive, type);

      user_factors[level_id] = ParseUserFactors(directive, workload);

      // The following logic was moved to the next per-level block of code.

      // auto level_factors = ParseUserFactors(directive, workload);
      // if (level_factors.size() > 0)
      // {
      //   // Fill in missing factors with default = 1.
      //   for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumFlattenedDimensions); idim++)
      //   {
      //     auto dim = problem::Shape::FlattenedDimensionID(idim);
      //     if (level_factors.find(dim) == level_factors.end())
      //     {
      //       level_factors[dim] = std::make_pair<>(1, 1);
      //     }
      //   }
      //   user_factors[level_id] = level_factors;
      // }
        
      user_permutations[level_id] = ParseUserPermutations(directive);

      // The following logic was moved to the next per-level block of code.

      // auto level_permutations = ParseUserPermutations(directive);
      // if (level_permutations.size() > 0)
      // {
      //   // Fill in missing dimensions with an undetermined order.
      //   for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumFlattenedDimensions); idim++)
      //   {
      //     auto dim = problem::Shape::FlattenedDimensionID(idim);
      //     if (std::find(level_permutations.begin(), level_permutations.end(), dim) == level_permutations.end())
      //       level_permutations.push_back(dim);
      //   }
      //   user_permutations[level_id] = level_permutations;
      // }

      if (type == "spatial")
      {
        // Initialize user spatial splits to map all dimensions to the hardware X-axis.
        std::uint32_t user_split = unsigned(problem::GetShape()->NumFlattenedDimensions);
        directive.lookupValue("split", user_split);
        user_spatial_splits[level_id] = user_split;

        // No link transfer
        if (directive.exists("no_link_transfer"))
        {
          auto storage_level = arch_props_.TilingToStorage(level_id);
          std::vector<std::string> datatype_strings;
          if (directive.lookupArrayValue("no_link_transfer", datatype_strings))
          {
            no_link_transfer[storage_level] = problem::PerDataSpace<bool>();
            
            for(unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
              no_link_transfer[storage_level][pv] = 0;
            for (const std::string& datatype_string: datatype_strings)
            {
              try
              {
                no_link_transfer[storage_level].at(
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
        }

        // No multicast no reduction
        if (directive.exists("no_multicast_no_reduction"))
        {
          auto storage_level = arch_props_.TilingToStorage(level_id);
          std::vector<std::string> datatype_strings;
          if (directive.lookupArrayValue("no_multicast_no_reduction", datatype_strings))
          {
            no_multicast[storage_level] = problem::PerDataSpace<bool>();
            for(unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
              no_multicast[storage_level][pv] = 0;
            for (const std::string& datatype_string: datatype_strings)
            {
              try
              {
                no_multicast[storage_level].at(
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
      }
      if (type == "temporal")
      {
        // No temporal reuse
        if (directive.exists("no_temporal_reuse"))
        {
          auto storage_level = arch_props_.TilingToStorage(level_id);
          std::vector<std::string> datatype_strings;
          if (directive.lookupArrayValue("no_temporal_reuse", datatype_strings))
          {
            no_temporal_reuse[storage_level] = problem::PerDataSpace<bool>();
            for(unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
              no_temporal_reuse[storage_level][pv] = 0;
            for (const std::string& datatype_string: datatype_strings)
            {
              try
              {
                no_temporal_reuse[storage_level].at(
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
        if (directive.exists("rmw_on_first_writeback"))
        {
          auto storage_level = arch_props_.TilingToStorage(level_id);
          std::vector<std::string> datatype_strings;
          if (directive.lookupArrayValue("rmw_on_first_writeback", datatype_strings))
          {
            rmw_on_first_writeback[storage_level] = problem::PerDataSpace<bool>();
            for(unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
              rmw_on_first_writeback[storage_level][pv] = 0;
            for (const std::string& datatype_string: datatype_strings)
            {
              try
              {
                rmw_on_first_writeback[storage_level].at(
                  problem::GetShape()->DataSpaceNameToID.at(datatype_string)) = 1;
              }
              catch (std::out_of_range& oor)
              {
                std::cerr << "ERROR: parsing rmw_on_first_writeback setting: data-space " << datatype_string
                          << " not found in problem shape." << std::endl;
                exit(1);
              }
            }
          }
        }
      }
    }
    else if (type == "datatype" || type == "bypass")
    {
      auto level_id = FindTargetTilingLevel(directive, type);
      ParseUserDatatypeBypassSettings(directive,
                                      arch_props_.TilingToStorage(level_id),
                                      user_bypass_strings);
      if (directive.exists("passthrough"))
      {
        auto storage_level = arch_props_.TilingToStorage(level_id);
        std::vector<std::string> datatype_strings;
        if (directive.lookupArrayValue("passthrough", datatype_strings))
        {
          passthrough[storage_level] = problem::PerDataSpace<bool>();
          for(unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
            passthrough[storage_level][pv] = 0;
          for (const std::string& datatype_string: datatype_strings)
          {
            try
            {
              passthrough[storage_level].at(
                problem::GetShape()->DataSpaceNameToID.at(datatype_string)) = 1;
            }
            catch (std::out_of_range& oor)
            {
              std::cerr << "ERROR: parsing passthrough setting: data-space " << datatype_string
                        << " not found in problem shape." << std::endl;
              exit(1);
            }
          }
        }
      }
    }
    else if (type == "skew")
    {
      // Note: skews are stored by storage level id, not tiling level id.
      auto storage_level_id = FindTargetStorageLevel(directive);
      user_skews[storage_level_id] = ParseUserSkew(directive);
    }
    else
    {
      std::cerr << "ERROR: illegal mapping directive type: " << type << std::endl;
      std::exit(1);
    }
  } // Done iterating through user-provided directives.

  // Construct the mapping from the parsed sub-structures.
  // A set of subnests, one for each tiling level.
  loop::NestConfig subnests(arch_props_.TilingLevels());

  // Construct num_storage_levels loop-nest partitions and assign factors, dimensions
  // and spatial split points.
  for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
  {
    auto permutation = user_permutations.find(level);
    if (permutation == user_permutations.end())
    {
      std::cerr << "WARNING: parsing mapping: permutation not found for level: "
                << arch_props_.TilingLevelName(level) << std::endl;
      user_permutations[level];
      permutation = user_permutations.find(level);
    }

    if (permutation->second.size() != std::size_t(problem::GetShape()->NumFlattenedDimensions))
    {
      // Fill in missing dimensions with an undetermined order.
      std::cerr << "WARNING: parsing mapping: permutation contains insufficient dimensions at level: "
                << arch_props_.TilingLevelName(level) << ", padding with arbitrary order." << std::endl;

      for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumFlattenedDimensions); idim++)
      {
        auto dim = problem::Shape::FlattenedDimensionID(idim);
        if (std::find(permutation->second.begin(), permutation->second.end(), dim) == permutation->second.end())
          permutation->second.push_back(dim);
      }
    }
      
    auto factors = user_factors.find(level);
    if (factors == user_factors.end())
    {
      std::cerr << "WARNING: parsing mapping: factors not found for level: "
                << arch_props_.TilingLevelName(level) << std::endl;
      user_factors[level];
      factors = user_factors.find(level);
    }

    if (factors->second.size() != std::size_t(problem::GetShape()->NumFlattenedDimensions))
    {
      // Fill in missing factors with default = 1.
      std::cerr << "WARNING: parsing mapping: factors not provided for all dimensions at level: "
                << arch_props_.TilingLevelName(level) << ", setting to 1." << std::endl;

      for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumFlattenedDimensions); idim++)
      {
        auto dim = problem::Shape::FlattenedDimensionID(idim);
        if (factors->second.find(dim) == factors->second.end())
        {
          factors->second[dim] = std::make_pair<>(1, 1);
        }
      }
    }

    auto split = user_spatial_splits.find(level);
    if (arch_props_.IsSpatial(level))
    {
      if (split == user_spatial_splits.end())
      {
        std::cerr << "WARNING: parsing mapping: split not found for spatial level: "
                  << arch_props_.TilingLevelName(level) << ", setting to all-X." << std::endl;
        user_spatial_splits[level] = unsigned(problem::GetShape()->NumFlattenedDimensions);
        split = user_spatial_splits.find(level);
      }
    }

    // Each partition has problem::GetShape()->NumFlattenedDimensions loops.
    for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumFlattenedDimensions); idim++)
    {
      loop::Descriptor loop;
      loop.dimension = permutation->second.at(idim);
      loop.start = 0;
      loop.end = factors->second.at(loop.dimension).first;
      loop.residual_end = factors->second.at(loop.dimension).second;
      loop.stride = 1; // FIXME.
      loop.spacetime_dimension = arch_props_.IsSpatial(level)
        ? (idim < split->second ? spacetime::Dimension::SpaceX : spacetime::Dimension::SpaceY)
        : spacetime::Dimension::Time;
      subnests.at(level).push_back(loop);
    }

  }

  // Validity checks.
  std::map<problem::Shape::FlattenedDimensionID, int> prod;
  for (unsigned dim = 0; dim < problem::GetShape()->NumFlattenedDimensions; dim++)
    prod[dim] = 0;

  // (((resP3-1)*P2 + (resP2-1))*P1 + (resP1-1))*P0 + resP0  
  for (uint64_t level = arch_props_.TilingLevels(); level-- > 0; )
  {
    for (auto& loop: subnests.at(level))
      prod[loop.dimension] = prod[loop.dimension]*loop.end + loop.residual_end-1;
  }

  for (unsigned dim = 0; dim < problem::GetShape()->NumFlattenedDimensions; dim++)
    prod[dim]++;

  // All user-provided factors must multiply-up to the dimension size.
  bool fault = false;
  for (unsigned dim = 0; dim < problem::GetShape()->NumFlattenedDimensions; dim++)
  {
    if (prod[dim] != workload.GetFlattenedBound(dim))
    {
      std::cerr << "ERROR: parsing mapping: product of all factors of dimension "
                << problem::GetShape()->FlattenedDimensionIDToName.at(dim) << " is "
                << prod[dim] << ", which is not equal to "
                << "the dimension size of the workload " << workload.GetFlattenedBound(dim)
                << "." << std::endl;
      fault = true;
    }
  }
  if (fault)
  {
    exit(1);
  }

  // Concatenate the subnests to form the final mapping nest.
  Mapping mapping;
  
  for (uint64_t i = 0; i < arch_props_.TilingLevels(); i++)
  {
    uint64_t num_subnests_added = 0;
    for (int dim = 0; dim < int(problem::GetShape()->NumFlattenedDimensions); dim++)
    {
      // Ignore trivial factors
      // This reduces computation time by 1.5x on average.
      if (subnests[i][dim].start + subnests[i][dim].stride < subnests[i][dim].end)
      {
        mapping.loop_nest.AddLoop(subnests[i][dim]);
        num_subnests_added++;
      }
      mapping.complete_loop_nest.AddLoop(subnests[i][dim]);
    }
    if (!arch_props_.IsSpatial(i))
    {
      if (num_subnests_added == 0)
      {
        // Add a trivial temporal nest to make sure
        // we have at least one subnest in each level.
        mapping.loop_nest.AddLoop(problem::Shape::FlattenedDimensionID(int(problem::GetShape()->NumFlattenedDimensions) - 1),
                                   0, 1, 1, spacetime::Dimension::Time);
      }
      mapping.loop_nest.AddStorageTilingBoundary();
      mapping.complete_loop_nest.AddStorageTilingBoundary();
    }
  }

  // The user_mask input is a set of per-datatype strings. Each string has a length
  // equal to num_storage_levels, and contains the characters 0 (bypass), 1 (keep),
  // or X (evaluate both).    
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);

    // Start parsing the user mask string.
    assert(user_bypass_strings.at(pv).length() <= arch_props_.StorageLevels());

    // The first loop runs the length of the user-specified string.
    unsigned level = 0;
    for (; level < user_bypass_strings.at(pv).length(); level++)
    {
      char spec = user_bypass_strings.at(pv).at(level);
      switch (spec)
      {
        case '0':
          mapping.datatype_bypass_nest.at(pvi).reset(level);
          break;
            
        case '1':
          mapping.datatype_bypass_nest.at(pvi).set(level);
          break;

        case 'X':
          // We allow this to be left un-specified by the user. Default is "keep".
          mapping.datatype_bypass_nest.at(pvi).set(level);
          break;          
            
        default:
          assert(false);
          break;
      }
    }
  } // for (pvi)

  mapping.confidence_thresholds = confidence_thresholds;
  mapping.loop_nest.skew_descriptors = user_skews;
  mapping.loop_nest.no_link_transfer = no_link_transfer;
  mapping.loop_nest.no_multicast = no_multicast;
  mapping.loop_nest.no_temporal_reuse = no_temporal_reuse;
  mapping.loop_nest.rmw_on_first_writeback = rmw_on_first_writeback;
  mapping.loop_nest.passthrough = passthrough;
  mapping.id = 0;
  mapping.fanoutX_map = arch_props_.FanoutX();
  mapping.fanoutY_map = arch_props_.FanoutY();

  return mapping;
}

//
// FindTargetStorageLevel()
//
unsigned FindTargetStorageLevel(config::CompoundConfigNode directive)
{
  auto num_storage_levels = arch_props_.StorageLevels();
    
  //
  // Find the target storage level. This can be specified as either a name or an ID.
  //
  std::string storage_level_name;
  unsigned storage_level_id;
    
  if (directive.lookupValue("target", storage_level_name))
  {
    // Find this name within the storage hierarchy in the arch specs.
    for (storage_level_id = 0; storage_level_id < num_storage_levels; storage_level_id++)
    {
      if (arch_props_.Specs().topology.GetStorageLevel(storage_level_id)->level_name == storage_level_name)
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
    assert(directive.lookupValue("target", id));
    assert(id >= 0  && id < int(num_storage_levels));
    storage_level_id = static_cast<unsigned>(id);
  }

  assert(storage_level_id < num_storage_levels);

  return storage_level_id;
}

//
// FindTargetTilingLevel()
//
unsigned FindTargetTilingLevel(config::CompoundConfigNode directive, std::string type)
{
  unsigned storage_level_id = FindTargetStorageLevel(directive);

  //
  // Translate this storage ID to a tiling ID.
  //
  unsigned tiling_level_id;
  if (type == "temporal" || type == "datatype" || type == "bypass")
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
  else
  {
    std::cerr << "ERROR: unrecognized mapping directive type: " << type << std::endl;
    exit(1);
  }

  return tiling_level_id;
}

//
// Parse user factors.
//
std::map<problem::Shape::FlattenedDimensionID, std::pair<int,int>> ParseUserFactors(
  config::CompoundConfigNode directive,
  const problem::Workload& workload)
{
  std::map<problem::Shape::FlattenedDimensionID, std::pair<int,int>> retval;
    
  std::string buffer;
  if (directive.lookupValue("factors", buffer))
  {
    buffer = buffer.substr(0, buffer.find("#"));

    std::regex re("([A-Za-z]+)[[:space:]]*[=]*[[:space:]]*([0-9]+)(,([0-9]+))?", std::regex::extended);
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
        end = workload.GetFlattenedBound(dimension);
      }
      // else if (end > workload.GetBound(dimension))
      // {
      //   std::cerr << "WARNING: Directive " << dimension << "=" << end
      //             << " exceeds problem dimension " << dimension << "="
      //             << workload.GetBound(dimension) << ". Setting directive "
      //             << dimension << "=" << workload.GetBound(dimension) << std::endl;
      //   end = workload_.GetBound(dimension);
      // }
      else
      {
        assert(end > 0);
      }

      int residual_end = end;
      if (sm[4] != "")
      {
        residual_end = std::stoi(sm[4]);
      }

      // Found all the information we need to setup a factor!
      retval[dimension] = std::make_pair<>(end, residual_end);

      str = sm.suffix().str();
    }
  }

  return retval;
}

//
// Parse user permutations.
//
std::vector<problem::Shape::FlattenedDimensionID> ParseUserPermutations(config::CompoundConfigNode directive)
{
  std::vector<problem::Shape::FlattenedDimensionID> retval;
    
  std::string buffer;
  if (directive.lookupValue("permutation", buffer))
  {
    std::istringstream iss(buffer);
    char token;
    while (iss >> token)
    {
      auto dimension = problem::GetShape()->FlattenedDimensionNameToID.at(std::string(1, token)); // note: can fault.
      retval.push_back(dimension);
    }
  }

  return retval;
}

//
// Parse user datatype bypass settings.
//
void ParseUserDatatypeBypassSettings(config::CompoundConfigNode directive,
                                     unsigned level,
                                     problem::PerDataSpace<std::string>& user_bypass_strings)
{
  // Datatypes to "keep" at this level.
  if (directive.exists("keep"))
  {
    std::vector<std::string> datatype_strings;
    directive.lookupArrayValue("keep", datatype_strings);
    for (const std::string& datatype_string: datatype_strings)
    {
      auto datatype = problem::GetShape()->DataSpaceNameToID.at(datatype_string);
      user_bypass_strings.at(datatype).at(level) = '1';
    }
  }
      
  // Datatypes to "bypass" at this level.
  if (directive.exists("bypass"))
  {
    std::vector<std::string> datatype_strings;
    directive.lookupArrayValue("bypass", datatype_strings);
    for (const std::string& datatype_string: datatype_strings)
    {
      auto datatype = problem::GetShape()->DataSpaceNameToID.at(datatype_string);
      user_bypass_strings.at(datatype).at(level) = '0';
    }
  }
}

//
// Parse user skew.
//
loop::Nest::SkewDescriptor ParseUserSkew(config::CompoundConfigNode directive)
{
  loop::Nest::SkewDescriptor skew_descriptor;

  if (!directive.lookupValue("modulo", skew_descriptor.modulo))
  {
    std::cerr << "ERROR: parsing skew directive: no modulo specified." << std::endl;
    std::exit(1);
  }

  if (!directive.exists("terms"))
  {
    std::cerr << "ERROR: parsing skew directive: no terms specified." << std::endl;
    std::exit(1);
  }

  auto expr_cfg = directive.lookup("terms");
  assert(expr_cfg.isList());
  
  int len = expr_cfg.getLength();
  for (int i = 0; i < len; i ++)
  {
    loop::Nest::SkewDescriptor::Term term;
    auto term_cfg = expr_cfg[i];

    if (term_cfg.exists("constant"))
    {
      term_cfg.lookupValue("constant", term.constant);
    }

    if (term_cfg.exists("variable"))
    {
      auto variable = term_cfg.lookup("variable");
      std::string buffer;
      variable.lookupValue("dimension", buffer);
      term.variable.dimension = problem::GetShape()->FlattenedDimensionNameToID.at(buffer);

      variable.lookupValue("type", buffer);
      if (buffer == "spatial")
        term.variable.is_spatial = true;
      else if (buffer == "temporal")
        term.variable.is_spatial = false;
      else
      {
        std::cerr << "ERROR: skew variable type must be spatial or temporal." << std::endl;
        std::exit(1);
      }
    }

    if (term_cfg.exists("bound"))
    {
      auto bound = term_cfg.lookup("bound");
      std::string buffer;
      bound.lookupValue("dimension", buffer);
      term.bound.dimension = problem::GetShape()->FlattenedDimensionNameToID.at(buffer);

      bound.lookupValue("type", buffer);
      if (buffer == "spatial")
        term.bound.is_spatial = true;
      else if (buffer == "temporal")
        term.bound.is_spatial = false;
      else
      {
        std::cerr << "ERROR: skew bound type must be spatial or temporal." << std::endl;
        std::exit(1);
      }
    }

    skew_descriptor.terms.push_back(term);
  }

  return skew_descriptor;
}

} // namespace mapping
