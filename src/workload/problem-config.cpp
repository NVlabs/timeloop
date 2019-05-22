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

#include <vector>
#include <list>

#include "loop-analysis/point-set.hpp"

#include "problem-config.hpp"
#include "workload-config.hpp"
#include "operation-space.hpp"

namespace problem
{

// ======================================== //
//              Problem Shape               //
// ======================================== //

// Globals.
bool ShapeParsed = false;

unsigned NumDimensions;
std::map<DimensionID, std::string> DimensionIDToName;
std::map<std::string, DimensionID> DimensionNameToID;

unsigned NumCoefficients;
std::map<CoefficientID, std::string> CoefficientIDToName;
std::map<std::string, CoefficientID> CoefficientNameToID;
Coefficients DefaultCoefficients;

unsigned NumDataSpaces;
std::map<DataSpaceID, std::string> DataSpaceIDToName;
std::map<std::string, DataSpaceID> DataSpaceNameToID;
std::map<DataSpaceID, unsigned> DataSpaceOrder;
std::map<DataSpaceID, bool> IsReadWriteDataSpace;

std::vector<Projection> Projections;

// API.
void ParseProblemShape(libconfig::Setting& config)
{
  if (!config.exists("shape"))
  {
    std::cerr << "ERROR: problem shape not found. Please specify a problem shape, or @include a pre-existing shape in the .cfg file." << std::endl;
    exit(1);
  }
  libconfig::Setting& shape = config.lookup("shape");
  
  // Not sure what to do with the name, since we only ever
  // parse one shape per invocation.
  std::string name = "";
  shape.lookupValue("name", name);

  // Dimensions.
  libconfig::Setting& dimensions = shape.lookup("dimensions");
  assert(dimensions.isArray());

  NumDimensions = 0;
  for (const std::string& dim_name : dimensions)
  {
    if (dim_name.length() != 1)
    {
      std::cerr << "ERROR: unfortunately, dimension names can only be 1 character in length. To remove this limitation, improve the constraint-parsing code in ParseUserPermutations() and ParseUserFactors() in mapping/parser.cpp and mapspaces/uber.hpp." << std::endl;
      exit(1);
    }
    DimensionIDToName[NumDimensions] = dim_name;
    DimensionNameToID[dim_name] = NumDimensions;
    NumDimensions++;
  }

  // Coefficients.
  libconfig::Setting& coefficients = shape.lookup("coefficients");
  assert(coefficients.isList());

  NumCoefficients = 0;
  for (auto& coefficient : coefficients)
  {
    std::string name;
    assert(coefficient.lookupValue("name", name));
           
    Coefficient default_value;
    assert(coefficient.lookupValue("default", default_value));

    CoefficientIDToName[NumCoefficients] = name;
    CoefficientNameToID[name] = NumCoefficients;
    DefaultCoefficients[NumCoefficients] = default_value;
    NumCoefficients++;
  }
  
  // Data Spaces.
  libconfig::Setting& data_spaces = shape.lookup("data-spaces");
  assert(data_spaces.isList());

  NumDataSpaces = 0;
  for (auto& data_space : data_spaces)
  {
    std::string name;
    assert(data_space.lookupValue("name", name));

    DataSpaceIDToName[NumDataSpaces] = name;
    DataSpaceNameToID[name] = NumDataSpaces;

    bool read_write = false;
    data_space.lookupValue("read-write", read_write);
    IsReadWriteDataSpace[NumDataSpaces] = read_write;

    Projection projection;
    libconfig::Setting& projection_cfg = data_space.lookup("projection");
    if (projection_cfg.isArray())
    {
      DataSpaceOrder[NumDataSpaces] = 0;
      for (const std::string& dim_name : projection_cfg)
      {
        auto& dim_id = DimensionNameToID.at(dim_name);
        projection.push_back({{ NumCoefficients, dim_id }});        
        DataSpaceOrder[NumDataSpaces]++;
      }
    }
    else if (projection_cfg.isList())
    {
      DataSpaceOrder[NumDataSpaces] = 0;
      for (auto& dimension : projection_cfg)
      {
        // Process one data-space dimension.
        ProjectionExpression expression;

        // Each expression is a list of terms. Each term can be
        // a libconfig array or list.
        for (auto& term : dimension)
        {
          assert(term.isArray());
          
          // Each term may have exactly 1 or 2 items.
          if (term.getLength() == 1)
          {
            const std::string& dim_name = term[0];
            auto& dim_id = DimensionNameToID.at(dim_name);
            expression.push_back({ NumCoefficients, dim_id });
          }
          else if (term.getLength() == 2)
          {
            const std::string& dim_name = term[0];
            const std::string& coeff_name = term[1];
            auto& dim_id = DimensionNameToID.at(dim_name);
            auto& coeff_id = CoefficientNameToID.at(coeff_name);
            expression.push_back({ coeff_id, dim_id });
          }
          else
          {
            assert(false);
          }
        }
        
        projection.push_back(expression);
        DataSpaceOrder[NumDataSpaces]++;
      }
    }
    else
    {
      assert(false);
    }

    Projections.push_back(projection);
    NumDataSpaces++;
  }
  // FIXME: deal with Shared/Illegal

  ShapeParsed = true;
}

}  // namespace problem
