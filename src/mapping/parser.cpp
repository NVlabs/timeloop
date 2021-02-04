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

#include "parser.hpp"
#include "arch-properties.hpp"

namespace mapping
{

//
// Shared state.
//

ArchProperties arch_props_;
problem::Workload workload_;

//
// Forward declarations.
//
unsigned FindTargetTilingLevel(config::CompoundConfigNode constraint, std::string type);
std::map<problem::Shape::DimensionID, int> ParseUserFactors(config::CompoundConfigNode constraint);
std::vector<problem::Shape::DimensionID> ParseUserPermutations(config::CompoundConfigNode constraint);
void ParseUserDatatypeBypassSettings(config::CompoundConfigNode constraint,
                                     unsigned level,
                                     problem::PerDataSpace<std::string>& user_bypass_strings);
//
// Parse mapping in libconfig format and generate data structure.
//
Mapping ParseAndConstruct(config::CompoundConfigNode config,
                          model::Engine::Specs& arch_specs,
                          problem::Workload workload)
{
  arch_props_ = ArchProperties();
  arch_props_.Construct(arch_specs);

  workload_ = workload;
  
  std::map<unsigned, std::map<problem::Shape::DimensionID, int>> user_factors;
  std::map<unsigned, std::vector<problem::Shape::DimensionID>> user_permutations;
  std::map<unsigned, std::uint32_t> user_spatial_splits;
  problem::PerDataSpace<std::string> user_bypass_strings;

  // Initialize user bypass strings to "XXXXX...1" (note the 1 at the end).
  // FIXME: there's probably a cleaner way/place to initialize this.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    std::string xxx(arch_props_.StorageLevels(), 'X');
    xxx.back() = '1';
    user_bypass_strings[problem::Shape::DataSpaceID(pvi)] = xxx;
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

    auto level_id = FindTargetTilingLevel(directive, type);

    if (type == "temporal" || type == "spatial")
    {
      auto level_factors = ParseUserFactors(directive);
      if (level_factors.size() > 0)
      {
        user_factors[level_id] = level_factors;
      }
        
      auto level_permutations = ParseUserPermutations(directive);
      if (level_permutations.size() > 0)
      {
        user_permutations[level_id] = level_permutations;
      }

      if (type == "spatial")
      {
        // Initialize user spatial splits to map all dimensions to the hardware X-axis.
        std::uint32_t user_split = unsigned(problem::GetShape()->NumDimensions);
        directive.lookupValue("split", user_split);
        user_spatial_splits[level_id] = user_split;
      }
    }
    else if (type == "datatype" || type == "bypass")
    {
      ParseUserDatatypeBypassSettings(directive,
                                      arch_props_.TilingToStorage(level_id),
                                      user_bypass_strings);
    }
    else
    {
      assert(false);
    }
  }

  // Validity checks.
  std::map<problem::Shape::DimensionID, int> dimension_factor_products;
  for (unsigned dim = 0; dim < problem::GetShape()->NumDimensions; dim++)
    dimension_factor_products[dim] = 1;

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
      std::cerr << "ERROR: parsing mapping: permutation not found for level: "
                << arch_props_.TilingLevelName(level) << std::endl;
      exit(1);
    }
    if (permutation->second.size() != std::size_t(problem::GetShape()->NumDimensions))
    {
      std::cerr << "ERROR: parsing mapping: permutation contains insufficient dimensions at level: "
                << arch_props_.TilingLevelName(level) << std::endl;
      exit(1);
    }
      
    auto factors = user_factors.find(level);
    if (factors == user_factors.end())
    {
      std::cerr << "ERROR: parsing mapping: factors not found for level: "
                << arch_props_.TilingLevelName(level) << std::endl;
      exit(1);
    }
    if (factors->second.size() != std::size_t(problem::GetShape()->NumDimensions))
    {
      std::cerr << "ERROR: parsing mapping: factors not provided for all dimensions at level: "
                << arch_props_.TilingLevelName(level) << std::endl;
      exit(1);
    }

    // Each partition has problem::GetShape()->NumDimensions loops.
    for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumDimensions); idim++)
    {
      loop::Descriptor loop;
      loop.dimension = permutation->second.at(idim);
      loop.start = 0;
      loop.end = factors->second.at(loop.dimension);
      loop.stride = 1; // FIXME.
      loop.spacetime_dimension = arch_props_.IsSpatial(level)
        ? (idim < user_spatial_splits.at(level) ? spacetime::Dimension::SpaceX : spacetime::Dimension::SpaceY)
        : spacetime::Dimension::Time;
      subnests.at(level).push_back(loop);

      dimension_factor_products[loop.dimension] *= loop.end;
    }
  }

  // All user-provided factors must multiply-up to the dimension size.
  bool fault = false;
  for (unsigned dim = 0; dim < problem::GetShape()->NumDimensions; dim++)
  {
    if (dimension_factor_products[dim] != workload_.GetBound(dim))
    {
      std::cerr << "ERROR: parsing mapping: product of all factors of dimension "
                << problem::GetShape()->DimensionIDToName.at(dim) << " is "
                << dimension_factor_products[dim] << ", which is not equal to "
                << "the dimension size of the workload " << workload_.GetBound(dim)
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
  
  std::uint64_t storage_level = 0;
  for (uint64_t i = 0; i < arch_props_.TilingLevels(); i++)
  {
    uint64_t num_subnests_added = 0;
    for (int dim = 0; dim < int(problem::GetShape()->NumDimensions); dim++)
    {
      // Ignore trivial factors
      // This reduces computation time by 1.5x on average.
      if (subnests[i][dim].start + subnests[i][dim].stride < subnests[i][dim].end)
      {
        mapping.loop_nest.AddLoop(subnests[i][dim]);
        num_subnests_added++;
      }
    }
    if (!arch_props_.IsSpatial(i))
    {
      if (num_subnests_added == 0)
      {
        // Add a trivial temporal nest to make sure
        // we have at least one subnest in each level.
        mapping.loop_nest.AddLoop(problem::Shape::DimensionID(int(problem::GetShape()->NumDimensions) - 1),
                                   0, 1, 1, spacetime::Dimension::Time);
      }
      mapping.loop_nest.AddStorageTilingBoundary();
      storage_level++;
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

    // Finalize mapping.
  mapping.id = 0;

  return mapping;
}

//
// FindTargetTilingLevel()
//
unsigned FindTargetTilingLevel(config::CompoundConfigNode directive, std::string type)
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
std::map<problem::Shape::DimensionID, int> ParseUserFactors(config::CompoundConfigNode directive)
{
  std::map<problem::Shape::DimensionID, int> retval;
    
  std::string buffer;
  if (directive.lookupValue("factors", buffer))
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
      // else if (end > workload_.GetBound(dimension))
      // {
      //   std::cerr << "WARNING: Directive " << dimension << "=" << end
      //             << " exceeds problem dimension " << dimension << "="
      //             << workload_.GetBound(dimension) << ". Setting directive "
      //             << dimension << "=" << workload_.GetBound(dimension) << std::endl;
      //   end = workload_.GetBound(dimension);
      // }
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
// Parse user permutations.
//
std::vector<problem::Shape::DimensionID> ParseUserPermutations(config::CompoundConfigNode directive)
{
  std::vector<problem::Shape::DimensionID> retval;
    
  std::string buffer;
  if (directive.lookupValue("permutation", buffer))
  {
    std::istringstream iss(buffer);
    char token;
    while (iss >> token)
    {
      auto dimension = problem::GetShape()->DimensionNameToID.at(std::string(1, token)); // note: can fault.
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

} // namespace mapping
