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

#include <sstream>
#include <unordered_set>

#include "nest.hpp"

namespace loop
{

// ----------
// NestConfig
// ----------

std::ostream& operator << (std::ostream& out, const NestConfig& nest)
{
  for (auto& loopblock : nest)
  {
    std::string indent = "";
    for (auto& loop : loopblock)
    {
      out << indent << loop << std::endl;
      indent = indent + "  ";
    }
  }
  return out;
}

// ---------
// Loop nest
// ---------

// All interface functions.
Nest::Nest()
{
}

bool Nest::operator == (const Nest& n) const
{
  return (loops == n.loops &&
          storage_tiling_boundaries == n.storage_tiling_boundaries);
}

void Nest::AddLoop(Descriptor descriptor)
{
  loops.push_back(descriptor);
}

void Nest::AddLoop(problem::Shape::DimensionID dimension, int start, int end, int stride,
                   spacetime::Dimension spacetime_dimension)
{
  AddLoop(loop::Descriptor(dimension, start, end, stride, spacetime_dimension));
}

bool Nest::AddStorageTilingBoundary()
{
  assert(loops.size() > 0);
  std::uint64_t level = loops.size() - 1;
  if (storage_tiling_boundaries.size() > 0)
  {
    if (storage_tiling_boundaries.back() == level)
    {
      std::cerr << "ERROR adding storage tiling boundary at level = " << level << std::endl;
      std::cerr << "ERROR failing this nest and proceeding, but THIS SHOULD NOT HAPPEN, FIXME!" << std::endl;
      return false;
    }
  }
  storage_tiling_boundaries.push_back(level);
  return true;
}

std::ostream& operator << (std::ostream& out, const Nest& nest)
{
  unsigned num_loops = nest.loops.size();
  unsigned inv_storage_level = nest.storage_tiling_boundaries.size()-2; // Skip printing the first boundary.

  std::string indent = "";
  for (unsigned loop_level = num_loops-1; loop_level != static_cast<unsigned>(-1); loop_level--)
  {
    if (inv_storage_level != static_cast<unsigned>(-1) &&
        nest.storage_tiling_boundaries.at(inv_storage_level) == loop_level)
    {
      out << "------------------------------------------" << std::endl;
      inv_storage_level--;
    }
    out << indent;
    indent += "  ";
    nest.loops.at(loop_level).Print(out, true);
    out << std::endl;
  }
  out << std::endl;
  return out;
}

void Nest::PrettyPrint(std::ostream& out, const std::vector<std::string>& storage_level_names,
                       const tiling::NestOfCompoundMasks& mask_nest,
                       const std::vector<problem::PerDataSpace<std::uint64_t>>& tile_sizes)
{
  unsigned num_loops = loops.size();
  unsigned inv_storage_level = storage_tiling_boundaries.size()-1; // Skip printing the first boundary.

  std::string indent = "";
  for (unsigned loop_level = num_loops-1; loop_level != static_cast<unsigned>(-1); loop_level--)
  {
    if (inv_storage_level != static_cast<unsigned>(-1) &&
        storage_tiling_boundaries.at(inv_storage_level) == loop_level)
    {
      out << "==========================================" << std::endl;
      out << storage_level_names.at(inv_storage_level) << std::endl;
      auto& mask = mask_nest.at(inv_storage_level);
      auto& tiles = tile_sizes.at(inv_storage_level);
      for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
      {
        if (mask.at(pvi))
        {
          out << std::setw(10) << problem::GetShape()->DataSpaceIDToName.at(pvi) << " tile: "
              << tiles.at(pvi) << std::endl;
        }
      }
      out << "------------------------------------------" << std::endl;
      inv_storage_level--;
    }
    out << indent;
    indent += "  ";
    loops.at(loop_level).Print(out, true);
    out << std::endl;
  }
  out << std::endl;
}

void Nest::PrintWhoopNest(std::ostream& out, const std::vector<std::string>& storage_level_names,
                          const tiling::NestOfCompoundMasks& mask_nest,
                          const std::vector<problem::PerDataSpace<std::uint64_t>>& tile_sizes,
                          const std::vector<problem::PerDataSpace<std::uint64_t>>& utilized_instances)
{
  unsigned num_loops = loops.size();
  unsigned num_storage_levels = tile_sizes.size();
  unsigned inv_storage_level = storage_tiling_boundaries.size()-1; // Skip printing the first boundary.

  // Prepare a set of "sensitive" dimensions for each tensor.
  std::vector<std::unordered_set<problem::Shape::DimensionID>> sensitive_dims(problem::GetShape()->NumDataSpaces);
  for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
  {
    auto d = problem::Shape::DataSpaceID(pvi);
    std::string tensor_name = problem::GetShape()->DataSpaceIDToName.at(d);

    // Prepare a set of problem-space dimensions that this data-space is sensitive to.
    for (unsigned data_space_dim = 0; data_space_dim < problem::GetShape()->DataSpaceOrder.at(d); data_space_dim++)
      for (auto& term: problem::GetShape()->Projections.at(d).at(data_space_dim))
        sensitive_dims.at(d).insert(term.second);
  }

  // Process each loop and generate var names, dim names etc.
  std::vector<problem::Shape::DimensionID> dimids;
  std::vector<std::string> dimnames;
  std::vector<int> dimbounds;
  std::vector<std::string> varnames;
  std::map<std::string, int> dimname_to_bound;

  std::vector<std::vector<std::string>> tiled_dimnames(problem::GetShape()->NumDataSpaces);
  std::vector<std::vector<std::string>> tiled_varnames(problem::GetShape()->NumDataSpaces);
  std::vector<problem::PerDataSpace<std::vector<std::string>>> tile_dimensions_algebraic(num_storage_levels);
  std::vector<problem::PerDataSpace<std::vector<std::string>>> spatial_instances_algebraic(num_storage_levels);

  for (unsigned loop_level = num_loops-1; loop_level != static_cast<unsigned>(-1); loop_level--)
  {
    if (inv_storage_level != static_cast<unsigned>(-1) &&
        storage_tiling_boundaries.at(inv_storage_level) == loop_level)
    {
      inv_storage_level--;
    }

    unsigned storage_level = inv_storage_level + 1;   // We decremented above.
    auto& loop = loops.at(loop_level);

    std::locale loc;
    std::string dimname = problem::GetShape()->DimensionIDToName.at(loop.dimension);
    std::string varname = dimname;

    for (unsigned i = 0; i < dimname.length(); i++)
      varname = tolower(dimname[i], loc);

    if (varname == dimname)
      varname = "cur_" + varname;

    varname += std::to_string(storage_level);
    dimname += std::to_string(storage_level);

    if (IsSpatial(loop.spacetime_dimension))
    {
      varname += "s";
      dimname += "s";
    }

    dimids.push_back(loop.dimension);
    dimnames.push_back(dimname);
    dimbounds.push_back(loop.end);
    varnames.push_back(varname);
    dimname_to_bound[dimname] = loop.end;

    // FIXME: the following is a hacky approach and does not work with sliding windows.
    // We need to maintain tile sizes in algebraic form through the nest analysis.
    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      auto d = problem::Shape::DataSpaceID(pvi);
      auto it = sensitive_dims.at(d).find(loop.dimension);
      if (it != sensitive_dims.at(d).end())
      {
        tiled_dimnames.at(d).push_back(dimname);
        tiled_varnames.at(d).push_back(varname);
        tile_dimensions_algebraic.at(storage_level).at(d).push_back(dimname);
      }
    }

    if (IsSpatial(loop.spacetime_dimension) && storage_level != 0)
    {
      for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
        spatial_instances_algebraic.at(storage_level-1).at(pvi).push_back(dimname);
    }    
    
  }

  // Tile dimensions are cumulative from inner to outer.
  for (unsigned storage_level = 1; storage_level < num_storage_levels; storage_level++)
  {
    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      tile_dimensions_algebraic.at(storage_level).at(pvi).insert(
        tile_dimensions_algebraic.at(storage_level).at(pvi).end(),
        tile_dimensions_algebraic.at(storage_level-1).at(pvi).begin(),
        tile_dimensions_algebraic.at(storage_level-1).at(pvi).end());
    }
  }

  // Spatial instances are cumulative from outer to inner.
  assert(num_storage_levels >= 2);
  for (unsigned storage_level = num_storage_levels-2; storage_level != static_cast<unsigned>(-1); storage_level--)
  {
    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      spatial_instances_algebraic.at(storage_level).at(pvi).insert(
        spatial_instances_algebraic.at(storage_level).at(pvi).begin(),
        spatial_instances_algebraic.at(storage_level+1).at(pvi).begin(),
        spatial_instances_algebraic.at(storage_level+1).at(pvi).end());
    }    
  }
  
  //
  // Start printing.
  //
  std::string indent = "  ";

  out << indent << "// =====================================================================" << std::endl
      << indent << "// WARNING: this is auto-generated, untested code and will probably need" << std::endl
      << indent << "// a good amount of massaging to work properly. In specific, please fix " << std::endl
      << indent << "// the following:" << std::endl
      << indent << "// (1) Tiled-tensor shapes will probably not work for sliding windows." << std::endl
      << indent << "// (2) Shrink sizes (the 2nd parameter in AddTileLevel()) are incorrect" << std::endl
      << indent << "//     for sliding windows." << std::endl
      << indent << "// (3) Tile-access granularities (3rd/1st parameter in Add/BypassTileLevel()" << std::endl
      << indent << "//     and multiplier for the 4th/2nd parameter) need to be massaged." << std::endl
      << indent << "// (4) Verify that the latency (kXXXLatency) variables are defined." << std::endl
      << indent << "// (5) Compute code only contains the tensors. An expression needs to be" << std::endl
      << indent << "//     filled in." << std::endl;
  out << std::endl;

  //
  // Print the tensors.
  //

  for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
  {
    std::string tensor_name = problem::GetShape()->DataSpaceIDToName.at(pvi);
    out << indent << "Tensor " << tensor_name << "(\"" << tensor_name << "\");" << std::endl;
  }
  out << std::endl;

  //
  // Print tiled dimension bounds.
  //
  for (unsigned i = 0; i < dimnames.size(); i++)
  {
    out << indent << "static const int " << dimnames[i] << " = " << dimbounds[i] << ";" << std::endl;
  }
  out << std::endl;  

  //
  // Print tiled tensor sizes.
  //
  for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
  {
    auto d = problem::Shape::DataSpaceID(pvi);
    std::string tensor_name = problem::GetShape()->DataSpaceIDToName.at(d);

    out << indent << tensor_name << ".Resize({ ";
    bool found_first = false;
    for (auto& dimname: tiled_dimnames.at(d))
    {
      if (found_first)
        out << ", ";
      found_first = true;
      out << dimname;
    }
    out << " });" << std::endl;
  }
  out << std::endl;

  //
  // Print iteration variable declarations.
  //
  for (auto& varname: varnames)
  {
    out << indent << "Var " << varname << "(\"" << varname << "\");" << std::endl;
  }
  // out << std::endl;

  //
  // Finally, print out the loop nest.
  //
  unsigned idx = 0;

  std::string prev_level_name = "BackingStore";

  inv_storage_level = storage_tiling_boundaries.size()-1; // Skip printing the first boundary.
  for (unsigned loop_level = num_loops-1; loop_level != static_cast<unsigned>(-1); loop_level--)
  {
    auto& loop = loops.at(loop_level);

    if (inv_storage_level != static_cast<unsigned>(-1) &&
        storage_tiling_boundaries.at(inv_storage_level) == loop_level)
    {
      out << std::endl;
      out << indent << "// " << storage_level_names.at(inv_storage_level) << " tiles " << std::endl;
      auto& mask = mask_nest.at(inv_storage_level);
      auto& tiles = tile_sizes.at(inv_storage_level);
      auto& instances = utilized_instances.at(inv_storage_level);

      std::string level_name = storage_level_names.at(inv_storage_level);
      std::string level_string = "\"" + level_name + "\"";

      for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
      {
        std::string tensor_name = problem::GetShape()->DataSpaceIDToName.at(pvi);
        if (mask.at(pvi))
        {
          out << indent << tensor_name << ".AddTileLevel(";
          for (auto& dimname: tile_dimensions_algebraic.at(inv_storage_level).at(pvi))
            out << dimname << "*";
          out << "1, ";
          for (auto& dimname: tile_dimensions_algebraic.at(inv_storage_level).at(pvi))
            out << dimname << "*";
          out << "1, 1 * k" << prev_level_name << "Latency);";

          // Verify that algebraic and real tile-sizes match.
          unsigned tile_size = 1;
          for (auto& dimname: tile_dimensions_algebraic.at(inv_storage_level).at(pvi))
          {
            tile_size *= dimname_to_bound.at(dimname);
          }
          if (tile_size != tiles.at(pvi))
            out << " // ERROR: mismatch algebraic = " << tile_size << " modeled = " << tiles.at(pvi);
          out << std::endl;

          // FIXME: utilized instances are directly mapped to expansion factor.
          // This will cause under-utilized instances to greedily consume physical
          // instances, which may be difficult/sub-optimal for the hardware to
          // route. Ideally we want the underutilized instances to be bound in
          // a pattern dictated by the loop nest.
          out << indent << tensor_name << ".BindCurrentTileLevel(" << level_string << ", ";
          for (auto& dimname: spatial_instances_algebraic.at(inv_storage_level).at(pvi))
            out << dimname << "*";
          out << "1);";

          // Verify that algebraic and real spatial instances match.
          unsigned spatial_instances = 1;
          for (auto& dimname: spatial_instances_algebraic.at(inv_storage_level).at(pvi))
          {
            spatial_instances *= dimname_to_bound.at(dimname);
          }
          if (spatial_instances != instances.at(pvi))
            out << " // ERROR: mismatch algebraic = " << spatial_instances << " modeled = " << instances.at(pvi);
          out << std::endl;
        }
        else
        {
          out << indent << tensor_name << ".BypassTileLevel(1, 1 * k"
              << prev_level_name << "Latency);" << std::endl;
        }        
      }
      out << std::endl;

      inv_storage_level--;
      prev_level_name = level_name;
    }

    out << indent;
    indent += "  ";

    if (IsSpatial(loop.spacetime_dimension))
      out << "s_for(";
    else
      out << "t_for(";
    out << varnames.at(idx) << ", " << loop.start << ", " << dimnames.at(idx) << "); {";
    out << std::endl;

    idx++;
  }
  out << std::endl;

  out << indent << "// === COMPUTE === fill in a compute expression using the following tensors:" << std::endl;
  for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
  {
    auto d = problem::Shape::DataSpaceID(pvi);
    std::string tensor_name = problem::GetShape()->DataSpaceIDToName.at(d);
    out << indent << tensor_name;
    for (auto& varname: tiled_varnames.at(d))
      out << "[" << varname << "]";
    out << ";";
    if (problem::GetShape()->IsReadWriteDataSpace.at(d))
      out << " // read-write";
    else
      out << " // read-only";
    out << std::endl;
  }
  out << std::endl;

  for (unsigned loop_level = num_loops-1; loop_level != static_cast<unsigned>(-1); loop_level--)
  {
    indent = "  ";
    for (unsigned i = 0; i < loop_level; i++)
      indent += "  ";
    out << indent << "} end();";
    out << std::endl;
  }

  out << std::endl;


}

}  // namespace loop
