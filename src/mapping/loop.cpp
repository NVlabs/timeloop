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


#include "workload/workload.hpp"
#include "loop.hpp"

namespace loop
{

// -----------------------------------
// Descriptor for a single loop level.
// -----------------------------------

Descriptor::Descriptor() {}

Descriptor::Descriptor(const problem::Shape::DimensionID _dimension, const int _start,
                       const int _end, const int _stride,
                       const spacetime::Dimension _spacetime_dimension,
                       const int _residual_end)
{
  assert(_stride >= 0);
  assert((_start + _stride) <= _end);
  
  dimension = _dimension;
  start = _start;
  end = _end;
  residual_end = _residual_end == 0 ? _end : _residual_end;
  stride = _stride;
  spacetime_dimension = _spacetime_dimension;
}

Descriptor::Descriptor(const problem::Shape::DimensionID _dimension,
                       const int _end,
                       const spacetime::Dimension _spacetime_dimension,
                       const int _residual_end)
{
  // Compact constructor: assume start = 0, stride = 1 and spacetime_dim
  // defaults to Time.
  assert(_end > 0);
  
  dimension = _dimension;
  start = 0;
  end = _end;
  residual_end = _residual_end == 0 ? _end : _residual_end;
  stride = 1;
  spacetime_dimension = _spacetime_dimension;
}

bool Descriptor::operator == (const Descriptor& d) const
{
  return (dimension == d.dimension &&
          start == d.start &&
          end == d.end &&
          residual_end == d.residual_end &&
          stride == d.stride &&
          spacetime_dimension == d.spacetime_dimension);
}

void Descriptor::Print(std::ostream& out, bool long_form) const
{
  if (long_form)
  {
    out << "for " << problem::GetShape()->DimensionIDToName.at(dimension) << " in [" << start << ":" << end;
    if (residual_end != end)
      out << "/" << residual_end;
    out << ")";
    if (IsSpatial(spacetime_dimension))
    {
      if (IsSpatialX(spacetime_dimension))
        out << " (Spatial-X)";
      else
        out << " (Spatial-Y)";
    }
  }
  else
  {
    out << "(" << dimension << "," << end;
    if (residual_end != end)
      out << "/" << residual_end;
    if (loop::IsSpatial(spacetime_dimension))
    {
      if (IsSpatialX(spacetime_dimension))
        out << ",spX";
      else
        out << ",spY";
    }
    out << ") ";
  }
}

void Descriptor::PrintWhoop(std::ostream& out, int storage_level,
                            std::vector<problem::Shape::DimensionID>& dimids,
                            std::vector<std::string>& dimnames,
                            std::vector<int>& dimbounds,
                            std::vector<std::string>& varnames) const
{
  if (residual_end != end)
  {
    std::cerr << "ERROR: residuals cannot be printed in Whoop format." << std::endl;
    exit(1);
  }

  // std::locale loc;
  std::string dimname = problem::GetShape()->DimensionIDToName.at(dimension);
  std::string varname = dimname;

  for (unsigned i = 0; i < dimname.length(); i++)
  {
    // varname[i] = tolower(dimname[i], loc);
    if (dimname[i] >= 'A' && dimname[i] <= 'Z')
      varname[i] = dimname[i] + 'a' - 'A';
  }

  if (varname == dimname)
    varname = "cur_" + varname;

  varname += std::to_string(storage_level);
  dimname += std::to_string(storage_level);

  if (IsSpatial(spacetime_dimension))
  {
    varname += "s";
    dimname += "s";
  }

  if (IsSpatial(spacetime_dimension))
    out << "s_for(";
  else
    out << "t_for(";
  out << varname << ", " << start << ", " << dimname << "); {";

  dimids.push_back(dimension);
  dimnames.push_back(dimname);
  dimbounds.push_back(end);
  varnames.push_back(varname);
}

std::string Descriptor::PrintCompact() const
{
  assert(start == 0);
  std::ostringstream str;
  str << problem::GetShape()->DimensionIDToName.at(dimension) << end;
  if (residual_end != end)
    str << "/" << residual_end;
  if (IsSpatial(spacetime_dimension))
  {
    if (IsSpatialX(spacetime_dimension))
      str << "X";
    else
      str << "Y";
  }
  return str.str();
}

std::ostream& operator << (std::ostream& out, const Descriptor& loop)
{
  loop.Print(out, true);
  return out;
}

bool IsSpatial(spacetime::Dimension dim)
{
  assert(dim < spacetime::Dimension::Num);
  return (dim != spacetime::Dimension::Time);
}

bool IsSpatialX(spacetime::Dimension dim)
{
  assert(IsSpatial(dim));
  return (dim == spacetime::Dimension::SpaceX);
}

}  // namespace loop
