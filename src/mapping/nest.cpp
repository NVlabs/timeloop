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

void Nest::PrettyPrint(std::ostream& out, const std::vector<std::string>& level_names,
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
      out << level_names.at(inv_storage_level+1) << std::endl; // +1 is to skip the arithmetic level FIXME.
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

void Nest::PrintWhoopNest(std::ostream& out, const std::vector<std::string>& level_names,
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
      out << std::endl;
      out << indent << "// " << level_names.at(inv_storage_level+1) << " tiles " << std::endl; // +1 is to skip the arithmetic level FIXME.
      auto& mask = mask_nest.at(inv_storage_level);
      auto& tiles = tile_sizes.at(inv_storage_level);

      std::string level_string = "\"" + level_names.at(inv_storage_level+1) + "\"";

      for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
      {
        std::string tensor_name = problem::GetShape()->DataSpaceIDToName.at(pvi);
        if (mask.at(pvi))
        {
          out << indent << tensor_name << ".AddTileLevel(" << tiles.at(pvi) << ");" << std::endl;
          out << indent << tensor_name << ".BindCurrentTileLevel(" << level_string << ");" << std::endl;
        }
        else
        {
          out << indent << tensor_name << ".BypassTileLevel();" << std::endl;
        }        
      }
      inv_storage_level--;
      out << std::endl;
    }
    out << indent;
    indent += "  ";
    loops.at(loop_level).PrintWhoop(out);
    out << std::endl;
  }

  out << std::endl;
  out << indent << "// === COMPUTE ===" << std::endl;
  out << std::endl;

  for (unsigned loop_level = num_loops-1; loop_level != static_cast<unsigned>(-1); loop_level--)
  {
    indent = "";
    for (unsigned i = 0; i < loop_level; i++)
      indent += "  ";
    out << indent << "} end();";
    out << std::endl;
  }

  out << std::endl;
}

}  // namespace loop
