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

#include "problem-config.hpp"
#include "workload-config.hpp"

namespace problem
{

void ParseWorkload(libconfig::Setting& config, WorkloadConfig& workload)
{
  if (!config.exists("shape"))
  {
    std::cerr << "ERROR: problem shape not found. Please specify a complete problem shape or a string corresponding to a pre-existing shape." << std::endl;
    exit(1);
  }

  std::string shape_name = "";
  if (config.lookupValue("shape", shape_name))
  {
    const char* timeloopdir = std::getenv("TIMELOOPDIR");
    if (!timeloopdir)
    {
      timeloopdir = BUILD_BASE_DIR;
      std::cerr << "WARNING: environment variable TIMELOOPDIR not found, assuming it to be: " << timeloopdir << std::endl;
    }
    
    std::string shape_file_name = std::string(timeloopdir) + "/problem-shapes/" + shape_name + ".cfg";

    std::cout << "Attempting to read problem shape from file: " << shape_file_name << std::endl;
      
    libconfig::Config shape_config;
    shape_config.readFile(shape_file_name.c_str());
    libconfig::Setting& shape = shape_config.lookup("shape");
    ParseProblemShape(shape);    
  }
  else
  {
    libconfig::Setting& shape = config.lookup("shape");
    ParseProblemShape(shape);
  }
  
  if (!ShapeParsed)
  {
    std::cerr << "ERROR: cannot parse workload before problem shape "
              << "has been parsed and set up." << std::endl;
    exit(1);
  }
  
  // Loop bounds for each problem dimension.
  Bounds bounds;
  for (unsigned i = 0; i < NumDimensions; i++)
    assert(config.lookupValue(DimensionIDToName.at(i), bounds[i]));
  workload.setBounds(bounds);

  Coefficients coefficients;
  for (unsigned i = 0; i < NumCoefficients; i++)
  {
    coefficients[i] = DefaultCoefficients[i];
    config.lookupValue(CoefficientIDToName.at(i), coefficients[i]);
  }
  workload.setCoefficients(coefficients);
  
  Densities densities;
  double common_density;
  if (config.lookupValue("commonDensity", common_density))
  {
    for (unsigned i = 0; i < NumDataSpaces; i++)
      densities[i] = common_density;
  }
  else if (config.exists("densities"))
  {
    libconfig::Setting &config_densities = config.lookup("densities");
    for (unsigned i = 0; i < NumDataSpaces; i++)
      assert(config_densities.lookupValue(DataSpaceIDToName.at(i), densities[i]));
  }
  else
  {
    for (unsigned i = 0; i < NumDataSpaces; i++)
      densities[i] = 1.0;
  }
  workload.setDensities(densities);
}

} // namespace problem
