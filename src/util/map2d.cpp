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

// ------------------------------------------------------------------------
// 2D Map type, with routines to read from a CSV file and dump into stdout.
//
// Primarily used for Energy/Area maps in the low-level energy and area
// models. The code below shows artifacts from having been extracted away
// from those codebases (e.g., width, height, std::size_t type, etc.).
// Could use some cleanup and templatization.
// ------------------------------------------------------------------------

#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>

#include "util/map2d.hpp"

//
// Generate map by reading a CSV file.
//
Map2D ReadCSV(std::string name, std::string prefix)
{
  Map2D map;
  
  std::ifstream csvfile(prefix + ".csv");
  if (csvfile.fail())
  { 
    std::cerr << "ERROR: failed to open file " << prefix + ".csv" << std::endl;
    exit(-1);
  }
  
  std::string line;
  std::string token;
  std::stringstream iss;

  // Read header line to figure out available widths.
  getline(csvfile, line);
  iss << line;

  // Make sure the first token is name and then discard it.
  getline(iss, token, ',');
  assert(token.compare(name) == 0);

  // Use the remaining tokens in the row as widths.
  while (getline(iss, token, ','))
  {
    std::size_t width = static_cast<std::size_t>(std::stoul(token));
    map[width] = Map1D();
  }
  iss.clear();

  // Each new line will give us entries for a different height.
  while (getline(csvfile, line))
  {
    iss << line;

    // The first token in the line gives us the height.
    getline(iss, token, ',');
    std::size_t height = static_cast<std::size_t>(std::stoul(token));

    // Now walk through all the widths in this line.
    auto width_it = map.begin();
    while (getline(iss, token, ','))
    {
      width_it->second[height] = std::stod(token);
      width_it++;
    }
    iss.clear();
  }

  csvfile.close();

  return map;
}

//
// Dump map in C++ syntax.
//
void WriteCPPHeader(const Map2D& map, std::string name, std::string prefix)
{
  std::ofstream hdrfile(prefix + ".hpp");

  hdrfile << "#pragma once" << std::endl;
  hdrfile << "Map2D " << name << " = {" << std::endl;
  
  for (auto& width : map)
  {
    hdrfile << "  { " << width.first << ", {" << std::endl;
    for (auto& height : width.second)
    {
      hdrfile << "    { " << height.first << ", " << height.second << " }," << std::endl;
    }
    hdrfile << "  }}," << std::endl;
  }
  hdrfile << "};" << std::endl;
  
  hdrfile.close();
}
