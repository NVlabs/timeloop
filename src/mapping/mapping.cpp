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

#include "mapping.hpp"

//--------------------------------------------//
//                  Mapping                   //
//--------------------------------------------//

std::ostream& operator << (std::ostream& out, const Mapping& mapping)
{
  out << "Mapping ID = " << mapping.id << std::endl;
  out << "Loop Nest:" << std::endl;
  out << "----------" << std::endl;
  out << mapping.loop_nest << std::endl;
  out << "Datatype Bypass Nest:" << std::endl;
  out << "---------------------" << std::endl;
  out << mapping.datatype_bypass_nest << std::endl;
  return out;
}

//
// FIXME: move to Constraints class.
//
void Mapping::PrintAsConstraints(const std::string filename)
{
  libconfig::Config config;
  libconfig::Setting& root = config.getRoot();
  libconfig::Setting& mapspace = root.add("mapspace", libconfig::Setting::TypeGroup);
  
  FormatAsConstraints(mapspace);

  config.writeFile(filename.c_str());
}

//
// FIXME: move to Constraints class.
//
void Mapping::FormatAsConstraints(libconfig::Setting& mapspace)
{
  mapspace.add("constraints", libconfig::Setting::TypeString) = "singlemapping";
  libconfig::Setting& constraints = mapspace.add("constraints_singlemapping", libconfig::Setting::TypeList);

  auto num_storage_levels = loop_nest.storage_tiling_boundaries.size();
  
  // Datatype Bypass.
  auto mask_nest = tiling::TransposeMasks(datatype_bypass_nest);

  for (unsigned level = 0; level < num_storage_levels; level++)
  {
    libconfig::Setting& constraint = constraints.add(libconfig::Setting::TypeGroup);
    constraint.add("target", libconfig::Setting::TypeInt) = static_cast<int>(level);
    constraint.add("type", libconfig::Setting::TypeString) = "datatype";
    
    libconfig::Setting& keep = constraint.add("keep", libconfig::Setting::TypeArray);
    libconfig::Setting& bypass = constraint.add("bypass", libconfig::Setting::TypeArray);

    auto& compound_mask = mask_nest.at(level);    
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      problem::Shape::DataSpaceID pv = problem::Shape::DataSpaceID(pvi);
      if (compound_mask.at(pv))
        keep.add(libconfig::Setting::TypeString) = problem::GetShape()->DataSpaceIDToName.at(pv);
      else
        bypass.add(libconfig::Setting::TypeString) = problem::GetShape()->DataSpaceIDToName.at(pv);
    }
  }

  // Factors and Permutations.
  unsigned loop_level = 0;
  for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
  {
    std::map<spacetime::Dimension, std::string> permutations;
    std::map<spacetime::Dimension, std::map<problem::Shape::DimensionID, unsigned>> factors;
    unsigned spatial_split;

    for (unsigned sdi = 0; sdi < unsigned(spacetime::Dimension::Num); sdi++)
    {
      auto sd = spacetime::Dimension(sdi);
      permutations[sd] = "";
      for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumDimensions); idim++)
        factors[sd][problem::Shape::DimensionID(idim)] = 1;
    }

    for (; loop_level <= loop_nest.storage_tiling_boundaries.at(storage_level); loop_level++)
    {
      auto& loop = loop_nest.loops.at(loop_level);
      if (loop.end > 1)
      {
        factors.at(loop.spacetime_dimension).at(loop.dimension) = loop.end;
        permutations.at(loop.spacetime_dimension) += problem::GetShape()->DimensionIDToName.at(loop.dimension);
      }
    }

    // Determine X-Y split.
    spatial_split = permutations.at(spacetime::Dimension::SpaceX).size();
    
    // Merge spatial X and Y factors and permutations.
    std::string spatial_permutation =
      permutations.at(spacetime::Dimension::SpaceX) +
      permutations.at(spacetime::Dimension::SpaceY);
    
    // Only print spatial constraints if there is a spatial permutation.
    if (spatial_permutation.size() > 0)
    {
      std::string spatial_factor_string = "";

      std::map<problem::Shape::DimensionID, unsigned> spatial_factors;
      for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumDimensions); idim++)
      {
        auto dim = problem::Shape::DimensionID(idim);
        spatial_factors[dim] =
          factors.at(spacetime::Dimension::SpaceX).at(dim) *
          factors.at(spacetime::Dimension::SpaceY).at(dim);

        spatial_factor_string += problem::GetShape()->DimensionIDToName.at(dim);
        char factor[8];
        sprintf(factor, "%d", spatial_factors.at(dim));
        spatial_factor_string += factor;
        if (idim != unsigned(problem::GetShape()->NumDimensions)-1)
          spatial_factor_string += " ";
        
        // If the factor is 1, concatenate it to the permutation.
        if (spatial_factors.at(dim) == 1)
          spatial_permutation += problem::GetShape()->DimensionIDToName.at(dim);
      }
      
      libconfig::Setting& constraint = constraints.add(libconfig::Setting::TypeGroup);
      constraint.add("target", libconfig::Setting::TypeInt) = static_cast<int>(storage_level);
      constraint.add("type", libconfig::Setting::TypeString) = "spatial";
      constraint.add("factors", libconfig::Setting::TypeString) = spatial_factor_string;
      constraint.add("permutation", libconfig::Setting::TypeString) = spatial_permutation;
      constraint.add("split", libconfig::Setting::TypeInt) = static_cast<int>(spatial_split);
    }

    auto& temporal_permutation = permutations.at(spacetime::Dimension::Time);
    auto& temporal_factors = factors.at(spacetime::Dimension::Time);
    std::string temporal_factor_string = "";
    
    // Temporal factors: if the factor is 1, concatenate it into the permutation.
    for (unsigned idim = 0; idim < unsigned(problem::GetShape()->NumDimensions); idim++)
    {
      auto dim = problem::Shape::DimensionID(idim);

      temporal_factor_string += problem::GetShape()->DimensionIDToName.at(dim);
      char factor[8];
      sprintf(factor, "%d", temporal_factors.at(dim));
      temporal_factor_string += factor;
      if (idim != unsigned(problem::GetShape()->NumDimensions)-1)
        temporal_factor_string += " ";
      
      if (temporal_factors.at(dim) == 1)
        temporal_permutation += problem::GetShape()->DimensionIDToName.at(dim);
    }

    libconfig::Setting& constraint = constraints.add(libconfig::Setting::TypeGroup);
    constraint.add("target", libconfig::Setting::TypeInt) = static_cast<int>(storage_level);
    constraint.add("type", libconfig::Setting::TypeString) = "temporal";
    constraint.add("factors", libconfig::Setting::TypeString) = temporal_factor_string;
    constraint.add("permutation", libconfig::Setting::TypeString) = temporal_permutation;
  }
}

void Mapping::PrettyPrint(std::ostream& out, const std::vector<std::string>& storage_level_names,
                          const std::vector<problem::PerDataSpace<std::uint64_t>>& utlized_capacities,
                          const std::vector<problem::PerDataSpace<std::uint64_t>>& tile_sizes,
                          const std::string _indent)
{
  loop_nest.PrettyPrint(out, storage_level_names, tiling::TransposeMasks(datatype_bypass_nest), utlized_capacities, tile_sizes, _indent);
}

void Mapping::PrintWhoopNest(std::ostream& out, const std::vector<std::string>& storage_level_names,
                             const std::vector<problem::PerDataSpace<std::uint64_t>>& tile_sizes,
                             const std::vector<problem::PerDataSpace<std::uint64_t>>& utilized_instances)
{
  loop_nest.PrintWhoopNest(out, storage_level_names, tiling::TransposeMasks(datatype_bypass_nest), tile_sizes, utilized_instances);
}

std::string Mapping::PrintCompact()
{
  return loop_nest.PrintCompact(tiling::TransposeMasks(datatype_bypass_nest));
}
