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

#include "problem-shape.hpp"
#include "workload.hpp"

namespace problem
{

// ======================================== //
//              Shape instance              //
// ======================================== //
// See comment in .hpp file.

Shape shape_;

const Shape* GetShape()
{
  return &shape_;
}

// ======================================== //
//                 Workload                 //
// ======================================== //

std::string ShapeFileName(const std::string shape_name)
{
  std::string shape_file_name;
  std::string shape_dir;
 
  const char* shape_dir_env = std::getenv("TIMELOOP_PROBLEM_SHAPE_DIR");
  if (shape_dir_env)
  {
    shape_dir = std::string(shape_dir_env) + "/";
  }
  else
  {  
    const char* timeloopdir = std::getenv("TIMELOOP_DIR");
    if (!timeloopdir)
    {
      timeloopdir = BUILD_BASE_DIR;
    }
    shape_dir = std::string(timeloopdir) + "/problem-shapes/";
  }
    
  shape_file_name = shape_dir + shape_name + ".cfg";
  
  std::cerr << "MESSAGE: reading problem shapes from: " << shape_dir << std::endl;
  std::cout << "MESSAGE: attempting to read problem shape from file: " << shape_file_name << std::endl;

  return shape_file_name;
}

void ParseWorkload(libconfig::Setting& config, Workload& workload)
{
  std::string shape_name;
  if (!config.exists("shape"))
  {
    std::cerr << "WARNING: found neither a problem shape description nor a string corresponding to a to a pre-existing shape description. Assuming shape: cnn-layer." << std::endl;
    libconfig::Config shape_config;
    shape_config.readFile(ShapeFileName("cnn-layer").c_str());
    libconfig::Setting& shape = shape_config.lookup("shape");
    shape_.Parse(shape);    
  }
  else if (config.lookupValue("shape", shape_name))
  {    
    libconfig::Config shape_config;
    shape_config.readFile(ShapeFileName(shape_name).c_str());
    libconfig::Setting& shape = shape_config.lookup("shape");
    shape_.Parse(shape);    
  }
  else
  {
    libconfig::Setting& shape = config.lookup("shape");
    shape_.Parse(shape);
  }
  
  // Loop bounds for each problem dimension.
  Workload::Bounds bounds;
  for (unsigned i = 0; i < GetShape()->NumDimensions; i++)
    assert(config.lookupValue(GetShape()->DimensionIDToName.at(i), bounds[i]));
  workload.SetBounds(bounds);

  Workload::Coefficients coefficients;
  for (unsigned i = 0; i < GetShape()->NumCoefficients; i++)
  {
    coefficients[i] = GetShape()->DefaultCoefficients.at(i);
    config.lookupValue(GetShape()->CoefficientIDToName.at(i), coefficients[i]);
  }
  workload.SetCoefficients(coefficients);
  
  Workload::Densities densities;
  double common_density;
  if (config.lookupValue("commonDensity", common_density))
  {
    for (unsigned i = 0; i < GetShape()->NumDataSpaces; i++)
      densities[i] = common_density;
  }
  else if (config.exists("densities"))
  {
    libconfig::Setting &config_densities = config.lookup("densities");
    for (unsigned i = 0; i < GetShape()->NumDataSpaces; i++)
      assert(config_densities.lookupValue(GetShape()->DataSpaceIDToName.at(i), densities[i]));
  }
  else
  {
    for (unsigned i = 0; i < GetShape()->NumDataSpaces; i++)
      densities[i] = 1.0;
  }
  workload.SetDensities(densities);
}

} // namespace problem
