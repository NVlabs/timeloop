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

#include <algorithm>

#include "loop-analysis/point-set.hpp"

#include "problem-config.hpp"
#include "workload-config.hpp"
#include "operation-space.hpp"

namespace problem
{

// ======================================== //
//              Problem Shape               //
// ======================================== //

// Globals
unsigned NumDimensions;
std::map<Dimension, std::string> DimensionName;
std::map<char, Dimension> DimensionID;

unsigned NumDataSpaces;
std::map<DataSpaceID, std::string> DataSpaceIDToName;
std::map<std::string, DataSpaceID> DataSpaceNameToID;
std::vector<unsigned> DataSpaceOrder;

std::function<bool(const DataSpaceID d)> IsReadWriteDataSpace;

std::vector<std::function<Point(WorkloadConfig*, const OperationPoint&)>> Projectors;

void ParseProblemShape()
{
  NumDimensions = 7;
  NumDataSpaces = 3;
  
  enum class WeightDimension {
    R,
    S,
    C,
    K,
    Num
  };
  enum class InputDimension {
    W,
    H,
    C,
    N,
    Num
  };
  enum class OutputDimension {
    P,
    Q,
    K,
    N,
    Num
  };

  DataSpaceIDToName = {
    {0, "Weights"},
    {1, "Inputs"},
    {2, "Outputs"},
    {3, "Shared/Illegal"}};

  DataSpaceNameToID = {
    {"Weights", 0},
    {"Inputs", 1},
    {"Outputs", 2},
    {"Shared/Illegal", 3}};

  DataSpaceOrder = {
    unsigned(WeightDimension::Num),
    unsigned(InputDimension::Num),
    unsigned(OutputDimension::Num) };

  IsReadWriteDataSpace = [](const DataSpaceID d) -> bool
    {
      // ASSERT(d < DataSpaceID::Num);
      return d == 2; // DataSpaceID::Output;
    };
  
  DimensionName = {{0, "R"},
                   {1, "S"},
                   {2, "P"},
                   {3, "Q"},
                   {4, "C"},
                   {5, "K"},
                   {6, "N"}, };

  DimensionID = {{'R', 0 },
                 {'S', 1 },
                 {'P', 2 },
                 {'Q', 3 },
                 {'C', 4 },
                 {'K', 5 },
                 {'N', 6 }, };

  Projectors =
    {
      [](WorkloadConfig* wc, const OperationPoint& problem_point)
      {
        (void) wc;

        Point weight_point(int(WeightDimension::Num));

        weight_point[int(WeightDimension::R)] = problem_point[0]; // R
        weight_point[int(WeightDimension::S)] = problem_point[1]; // S
        weight_point[int(WeightDimension::C)] = problem_point[4]; // C
        weight_point[int(WeightDimension::K)] = problem_point[5]; // K

        return weight_point;
      },
      [](WorkloadConfig* wc, const OperationPoint& problem_point)
      {
        Point input_point(int(InputDimension::Num));
        
        input_point[int(InputDimension::W)] =
          wc->getWstride() * problem_point[2] +  // P
          wc->getWdilation() * problem_point[0]; // R
        input_point[int(InputDimension::H)] =
          wc->getHstride() * problem_point[3] +  // Q
          wc->getHdilation() * problem_point[1]; // S
        
        input_point[int(InputDimension::C)] = problem_point[4]; // C
        input_point[int(InputDimension::N)] = problem_point[6]; // N

        return input_point;
      },
      [](WorkloadConfig* wc, const OperationPoint& problem_point)
      {
        (void) wc;

        Point output_point(int(OutputDimension::Num));
        
        output_point[int(OutputDimension::P)] = problem_point[2]; // P
        output_point[int(OutputDimension::Q)] = problem_point[3]; // Q
        
        output_point[int(OutputDimension::K)] = problem_point[5]; // K
        output_point[int(OutputDimension::N)] = problem_point[6]; // N

        return output_point;
      }
    };  
}

PerDataSpace<std::size_t> GetMaxWorkingSetSizes(
    problem::PerProblemDimension<int> dimension_sizes)
{
  PerDataSpace<std::size_t> datatype_size;

  // Weight: R*S*C*K
  datatype_size[0] =
      dimension_sizes[0] * dimension_sizes[1] *
      dimension_sizes[4] * dimension_sizes[5];

  // Input: (P+R-1)*(Q+S-1)*C*N
  datatype_size[1] =
      (dimension_sizes[2] + dimension_sizes[0] - 1) *
      (dimension_sizes[3] + dimension_sizes[1] - 1) *
      dimension_sizes[4] * dimension_sizes[6];

  // Output: P*Q*K*N
  datatype_size[2] =
      dimension_sizes[2] * dimension_sizes[3] *
      dimension_sizes[5] * dimension_sizes[6];
  
  return datatype_size;
}

}  // namespace problem