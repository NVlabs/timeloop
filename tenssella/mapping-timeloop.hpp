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

#pragma once

#include "mapping.hpp"

#include "architecture.hpp"
#include "problem.hpp"

class Mapping_Timeloop : public Mapping
{
 // private:
 //  struct MappingDirectives
 //  {
 //    // Parsed from file.
 //    std::map<std::string, int> temporal_factors;
 //    std::string temporal_permutation;

 //    bool has_spatial;
 //    std::map<std::string, int> spatial_factors;
 //    std::string spatial_permutation;
 //  };


 public:
  Mapping_Timeloop(isl_ctx* context,
                   // Architecture* arch,
                   unsigned num_spacetime_levels,
                   ProblemShape* prob,
                   std::string filename) :
      Mapping(context)
  {
    auto problem_dimensions = prob->IterationSpaceDimensions();
    // auto num_spacetime_levels = arch->NumLevels();

    std::map<unsigned, std::map<std::string, int>> all_temporal_factors;
    std::map<unsigned, std::map<std::string, int>> all_spatial_factors;
    
    std::map<unsigned, std::string> all_temporal_permutations;
    std::map<unsigned, std::string> all_spatial_permutations;

    std::map<unsigned, unsigned> all_spatial_splits;

    std::ifstream file(filename);

    std::string line;
    while (std::getline(file, line))
    {
      // Temporal or spatial?
      bool is_spatial;
      if (line.at(0) == 's')
        is_spatial = true;
      else if (line.at(0) == 't')
        is_spatial = false;
      else
        assert(false);

      // Parse level ID.
      unsigned spacetime_level = std::stoul(line.substr(1));
      
      // Read in factors.
      assert(std::getline(file, line));
      auto factors = ParseFactorLine(line);

      // Read in permutations.
      assert(std::getline(file, line));
      std::string permutation = line;

      // If spatial, read in split.
      unsigned split = 0;
      if (is_spatial)
      {
        assert(std::getline(file, line));
        split = std::stoul(line.substr(0));
      }

      // Populate the final input data structures.
      if (is_spatial)
      {
        all_spatial_factors[spacetime_level] = factors;
        all_spatial_permutations[spacetime_level] = permutation;
        all_spatial_splits[spacetime_level] = split;
      }
      else
      {
        all_temporal_factors[spacetime_level] = factors;
        all_temporal_permutations[spacetime_level] = permutation;

        // If spatial params haven't been set already, initialize them.
        if (all_spatial_factors.find(spacetime_level) == all_spatial_factors.end())
        {
          std::map<std::string, int> spatial_factors;
          std::string spatial_permutation = "";
          for (auto& dim: problem_dimensions)
          {
            spatial_factors[dim] = 1;
            spatial_permutation += dim;
          }
          all_spatial_factors[spacetime_level] = spatial_factors;
          all_spatial_permutations[spacetime_level] = spatial_permutation;
          all_spatial_splits[spacetime_level] = 0;
        }

        temporal_products_[spacetime_level] = ReduceFactors(factors);
        TRACE(1) << "Temporal product at st level " << spacetime_level << " = " << temporal_products_[spacetime_level] << std::endl;
      }
    }

    std::vector<std::vector<int>> all_factors_vector;

    for (unsigned spacetime_level = num_spacetime_levels-1; spacetime_level < num_spacetime_levels; spacetime_level--)
    {
      auto& temporal_factors = all_temporal_factors.at(spacetime_level);
      auto& spatial_factors = all_spatial_factors.at(spacetime_level);

      auto& temporal_permutation = all_temporal_permutations.at(spacetime_level);
      auto& spatial_permutation = all_spatial_permutations.at(spacetime_level);

      auto& split = all_spatial_splits.at(spacetime_level);

      // End inputs.

      // Begin creating a union of all factors at this level (flat, y, x and t)
      std::map<std::string, int> factors;

      // Flatten temporal and spatial factors end enter them into the factor
      // table using the original dimension name as key.
      PointwiseMultiplyUpdate<std::string, int>(temporal_factors, spatial_factors, factors);

      // Pack the map into an ordered vector that will be used later for GenerateTileMaps.
      // ** except ** if we are at level 0 (there's no additional partitioning at level 0).
      //if (spacetime_level != 0)
      //{
        std::vector<int> level_factors;
        for (auto& dim: problem_dimensions)
          level_factors.push_back(factors.at(dim));
        all_factors_vector.push_back(level_factors);
        //}

      // Prepare the temporal permutation, and update the factors table
      // with temporal factors using "i<dim>t" as key.
      std::vector<std::string> permutation_t;
      for (unsigned loc = 0; loc < temporal_permutation.length(); loc++)
      {
        char dim = temporal_permutation.at(loc);
        std::string dimstr(1, dim);
        auto factor = temporal_factors.at(dimstr);

        if (factor > 1)
        {
          std::string var_t = AddPrefixSuffix({ dimstr }, "i", "t").front();
          factors[var_t] = factor;
          permutation_t.push_back(var_t);
        }
      }    

      // Split spatial permutation into Y and X, and update the factors table
      // with Y and X factors using "i<dim>y/x" as key.
      std::vector<std::string> permutation_y, permutation_x;
      for (unsigned loc = 0; loc < spatial_permutation.length(); loc++)
      {
        char dim = spatial_permutation.at(loc);
        std::string dimstr(1, dim);
        auto factor = spatial_factors.at(dimstr);

        if (factor > 1)
        {
          std::string var_y = AddPrefixSuffix({ dimstr }, "i", "y").front();
          std::string var_x = AddPrefixSuffix({ dimstr }, "i", "x").front();
      
          if (loc < split)
          {
            factors[var_x] = factor;
            permutation_x.push_back(var_x);
          }
          else
          {
            factors[var_y] = factor;
            permutation_y.push_back(var_y);
          }
        }
      }
    
      // Reverse all permutations from little-endian to big-endian.
      std::reverse(permutation_y.begin(), permutation_y.end());
      std::reverse(permutation_x.begin(), permutation_x.end());
      std::reverse(permutation_t.begin(), permutation_t.end());

      // Create vectors of variable names. Index of each vector is the position
      // of that dimension in the original problem dimension order.
      std::vector<std::string> vars_flat = AddPrefixSuffix(problem_dimensions, "i", "");
      std::vector<std::string> vars_y = AddPrefixSuffix(problem_dimensions, "i", "y");
      std::vector<std::string> vars_x = AddPrefixSuffix(problem_dimensions, "i", "x");
      std::vector<std::string> vars_t = AddPrefixSuffix(problem_dimensions, "i", "t");

      std::string proj_str = "{ [";
      for (auto i = vars_flat.begin(); i != vars_flat.end(); i++)
      {
        proj_str += *i;
        if (std::next(i) != vars_flat.end())
          proj_str += ",";
      }

      proj_str += "] -> SpaceTime_" + std::to_string(spacetime_level) + "[y,x,t] :\n";

      std::vector<std::string> exists_vars;
      for (auto i = vars_y.begin(); i != vars_y.end(); i++)
      {
        if (factors.find(*i) != factors.end())
          exists_vars.push_back(*i);
      }
      for (auto i = vars_x.begin(); i != vars_x.end(); i++)
      {
        if (factors.find(*i) != factors.end())
          exists_vars.push_back(*i);
      }
      for (auto i = vars_t.begin(); i != vars_t.end(); i++)
      {
        if (factors.find(*i) != factors.end())
          exists_vars.push_back(*i);
      }

      if (!exists_vars.empty())
      {
        proj_str += "     exists ";
        for (auto i = exists_vars.begin(); i != exists_vars.end(); i++)
        {
          proj_str += *i;
          if (std::next(i) != exists_vars.end())
            proj_str += ",";
        }
        proj_str += " :\n";
      }

      for (unsigned i = 0; i < problem_dimensions.size(); i++)
      {
        std::string constraints;

        if (factors.find(vars_flat.at(i)) != factors.end())
          constraints += "0 <= " + vars_flat.at(i) + " < " + std::to_string(factors.at(vars_flat.at(i))) + " and ";
        if (factors.find(vars_y.at(i)) != factors.end())
          constraints += "0 <= " + vars_y.at(i) + " < " + std::to_string(factors.at(vars_y.at(i))) + " and ";
        if (factors.find(vars_x.at(i)) != factors.end())
          constraints += "0 <= " + vars_x.at(i) + " < " + std::to_string(factors.at(vars_x.at(i))) + " and ";
        if (factors.find(vars_t.at(i)) != factors.end())
          constraints += "0 <= " + vars_t.at(i) + " < " + std::to_string(factors.at(vars_t.at(i))) + " and ";

        if (!constraints.empty())
          proj_str += "       " + constraints + "\n";
      }

      // Flattening equations for each problem dimension.
      for (unsigned i = 0; i < problem_dimensions.size(); i++)
      {
        // Prepare the permutation that orders y->x->t (big-endian).
        // Only add the var if it has an entry in the factors table.
        std::vector<std::string> permutation;
        if (factors.find(vars_y.at(i)) != factors.end())
          permutation.push_back(vars_y.at(i));
        if (factors.find(vars_x.at(i)) != factors.end())
          permutation.push_back(vars_x.at(i));
        if (factors.find(vars_t.at(i)) != factors.end())
          permutation.push_back(vars_t.at(i));

        proj_str += "       ";
        proj_str += vars_flat.at(i) + " = ";
        if (!permutation.empty())
          proj_str += Flatten(permutation, factors);
        else
          proj_str += "0";
        proj_str += " and\n";
      }

      // Flattening equations for each hardware dimension (y, x, t).
      proj_str += "       y = " + Flatten(permutation_y, factors) + " and\n";
      proj_str += "       x = " + Flatten(permutation_x, factors) + " and\n";
      proj_str += "       t = " + Flatten(permutation_t, factors) + " }";

      TRACE(1) << proj_str << std::endl;
      
      skews_[spacetime_level] = isl_map_read_from_str(context_, proj_str.c_str());
    }

    GenerateTileMaps(all_factors_vector);

    //
    // Binding of data-spaces to hardware instances. Note that hardware instances are
    // indexed by *hardware* levels and not *tiling* levels (which are betweeen
    // hardware levels).
    //
    binding_[4]["Weights"] = { "DRAM", 0 };
    binding_[4]["Inputs"] = { "DRAM", 1 };
    binding_[4]["Outputs"] = { "DRAM", 2 };

    binding_[3]["Weights"] = { "GlobalBuffer", 0 };
    binding_[3]["Inputs"] = { "GlobalBuffer", 1 };
    binding_[3]["Outputs"] = { "GlobalBuffer", 2 };

    binding_[2]["Weights"] = { "Registers", 0 };
    binding_[2]["Inputs"] = { "Registers", 1 };
    binding_[2]["Outputs"] = { "Registers", 2 };

    binding_[1]["Weights"] = { "OperandA", 0 };
    binding_[1]["Inputs"] = { "OperandB", 1 };
    binding_[1]["Outputs"] = { "Result", 2 };

    binding_[0]["Multiply"] = { "Multiplier", 0 };
  }
};
