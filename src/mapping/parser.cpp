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

#include "parser.hpp"
#include "arch-properties.hpp"

namespace mapping
{

//
// Shared state.
//

ArchProperties arch_props_;
problem::WorkloadConfig problem_config_;

//
// Forward declarations.
//
unsigned FindTargetTilingLevel(libconfig::Setting& constraint, std::string type);
std::map<problem::Dimension, int> ParseUserFactors(libconfig::Setting& constraint);
std::vector<problem::Dimension> ParseUserPermutations(libconfig::Setting& constraint);
void ParseUserDatatypeBypassSettings(libconfig::Setting& constraint,
                                     unsigned level,
                                     problem::PerDataSpace<std::string>& user_bypass_strings);
//
// Parse mapping in libconfig format and generate data structure.
//
Mapping ParseAndConstruct(libconfig::Setting& config,
                          model::Engine::Specs& arch_specs,
                          problem::WorkloadConfig problem_config)
{
  arch_props_.Construct(arch_specs);
  problem_config_ = problem_config;
  
  std::map<unsigned, std::map<problem::Dimension, int>> user_factors;
  std::map<unsigned, std::vector<problem::Dimension>> user_permutations;
  std::map<unsigned, std::uint32_t> user_spatial_splits;
  problem::PerDataSpace<std::string> user_bypass_strings;

  // Initialize user bypass strings to "XXXXX...1" (note the 1 at the end).
  // FIXME: there's probably a cleaner way/place to initialize this.
  for (unsigned pvi = 0; pvi < unsigned(problem::DataType::Num); pvi++)
  {
    std::string xxx(arch_props_.StorageLevels(), 'X');
    xxx.back() = '1';
    user_bypass_strings[problem::DataType(pvi)] = xxx;
  }

  // Parse user-provided mapping.
  assert(config.isList());
  
  // Iterate over all the directives.
  for (auto& directive: config)
  {
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
        std::uint32_t user_split;
        if (directive.lookupValue("split", user_split))
        {
          user_spatial_splits[level_id] = user_split;
        }
      }
    }
    else if (type == "datatype")
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

  // Construct the mapping from the parsed sub-structures.
  // A set of subnests, one for each tiling level.
  loop::NestConfig subnests(arch_props_.TilingLevels());

  // Construct num_storage_levels loop-nest partitions and assign factors, dimensions
  // and spatial split points.
  for (uint64_t level = 0; level < arch_props_.TilingLevels(); level++)
  {
    auto& permutation = user_permutations.at(level);
    assert(permutation.size() == std::size_t(problem::Dimension::Num));
      
    auto& factors = user_factors.at(level);
    assert(factors.size() == std::size_t(problem::Dimension::Num));

    // Each partition has problem::Dimension::Num loops.
    for (unsigned idim = 0; idim < unsigned(problem::Dimension::Num); idim++)
    {
      loop::Descriptor loop;
      loop.dimension = permutation.at(idim);
      loop.start = 0;
      loop.end = factors.at(loop.dimension);
      loop.stride = 1; // FIXME.
      loop.spacetime_dimension = arch_props_.IsSpatial(level)
        ? (idim < user_spatial_splits.at(level) ? spacetime::Dimension::SpaceX : spacetime::Dimension::SpaceY)
        : spacetime::Dimension::Time;
      subnests.at(level).push_back(loop);
    }
  }

  // Concatenate the subnests to form the final mapping nest.
  Mapping mapping;
  
  std::uint64_t storage_level = 0;
  for (uint64_t i = 0; i < arch_props_.TilingLevels(); i++)
  {
    uint64_t num_subnests_added = 0;
    for (int dim = 0; dim < int(problem::Dimension::Num); dim++)
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
        mapping.loop_nest.AddLoop(problem::Dimension(int(problem::Dimension::Num) - 1),
                                   0, 1, 1, spacetime::Dimension::Time);
      }
      mapping.loop_nest.AddStorageTilingBoundary();
      storage_level++;
    }
  }

  // The user_mask input is a set of per-datatype strings. Each string has a length
  // equal to num_storage_levels, and contains the characters 0 (bypass), 1 (keep),
  // or X (evaluate both).    
  for (unsigned pvi = 0; pvi < unsigned(problem::DataType::Num); pvi++)
  {
    auto pv = problem::DataType(pvi);

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
unsigned FindTargetTilingLevel(libconfig::Setting& directive, std::string type)
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
      {
        break;
      }
    }
  }
  else
  {
    int id;
    assert(directive.lookupValue("target", id));
    assert(id >= 0);
    storage_level_id = static_cast<unsigned>(id);
  }
  assert(storage_level_id < num_storage_levels);

  //
  // Translate this storage ID to a tiling ID.
  //
  unsigned tiling_level_id;
  if (type == "temporal" || type == "datatype")
  {
    // This should always succeed.
    tiling_level_id = arch_props_.TemporalToTiling(storage_level_id);
  }
  else if (type == "spatial")
  {
    // This will fail if this level isn't a spatial tiling level.
    tiling_level_id = arch_props_.SpatialToTiling(storage_level_id);
  }
  else
  {
    assert(false);
  }

  return tiling_level_id;
}

//
// Parse user factors.
//
std::map<problem::Dimension, int> ParseUserFactors(libconfig::Setting& directive)
{
  std::map<problem::Dimension, int> retval;
    
  std::string buffer;
  if (directive.lookupValue("factors", buffer))
  {
    std::istringstream iss(buffer);
    char token;
    while (iss >> token)
    {
      auto dimension = problem::DimensionID.at(token); // note: can fault.
        
      int end;
      iss >> end;
      if (end == 0)
      {
        std::cerr << "WARNING: Interpreting 0 to mean full problem dimension instead of residue." << std::endl;
        end = problem_config_.getBound(dimension);
      }
      else if (end > problem_config_.getBound(dimension))
      {
        std::cerr << "WARNING: Directive " << dimension << "=" << end
                  << " exceeds problem dimension " << dimension << "="
                  << problem_config_.getBound(dimension) << ". Setting directive "
                  << dimension << "=" << problem_config_.getBound(dimension) << std::endl;
        end = problem_config_.getBound(dimension);
      }
      else
      {
        assert(end > 0);
      }

      // Found all the information we need to setup a factor!
      retval[dimension] = end;
    }
  }

  return retval;
}

//
// Parse user permutations.
//
std::vector<problem::Dimension> ParseUserPermutations(libconfig::Setting& directive)
{
  std::vector<problem::Dimension> retval;
    
  std::string buffer;
  if (directive.lookupValue("permutation", buffer))
  {
    std::istringstream iss(buffer);
    char token;
    while (iss >> token)
    {
      auto dimension = problem::DimensionID.at(token); // note: can fault.
      retval.push_back(dimension);
    }
  }

  return retval;
}

//
// Parse user datatype bypass settings.
//
void ParseUserDatatypeBypassSettings(libconfig::Setting& directive,
                                     unsigned level,
                                     problem::PerDataSpace<std::string>& user_bypass_strings)
{
  // Datatypes to "keep" at this level.
  if (directive.exists("keep"))
  {
    auto& keep = directive.lookup("keep");
    assert(keep.isArray());
      
    for (const std::string& datatype_string: keep)
    {
      auto datatype = problem::DataTypeID.at(datatype_string);
      user_bypass_strings.at(datatype).at(level) = '1';
    }
  }
      
  // Datatypes to "bypass" at this level.
  if (directive.exists("bypass"))
  {
    auto& bypass = directive.lookup("bypass");
    assert(bypass.isArray());
      
    for (const std::string& datatype_string: bypass)
    {
      auto datatype = problem::DataTypeID.at(datatype_string);
      user_bypass_strings.at(datatype).at(level) = '0';
    }
  }
}

} // namespace mapping
