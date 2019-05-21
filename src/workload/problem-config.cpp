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

// Globals.
unsigned NumDimensions;
std::map<DimensionID, std::string> DimensionIDToName;
std::map<char, DimensionID> DimensionNameToID;

unsigned NumCoefficients;
std::map<CoefficientID, std::string> CoefficientIDToName;
std::map<std::string, CoefficientID> CoefficientNameToID;
Coefficients DefaultCoefficients;

unsigned NumDataSpaces;
std::map<DataSpaceID, std::string> DataSpaceIDToName;
std::map<std::string, DataSpaceID> DataSpaceNameToID;
std::vector<unsigned> DataSpaceOrder;

std::function<bool(const DataSpaceID d)> IsReadWriteDataSpace;

std::vector<std::function<Point(const WorkloadConfig*, const OperationPoint&)>> Projectors;

// Projection AST: the projection function for each dataspace dimension is a
//                 Sum-Of-Products where each Product is the product of a
//                 Coefficient and a Dimension. This is fairly restrictive
//                 but efficient. We can generalize later if needed.

// typedef std::pair<CoefficientID, DimensionNameToID> 


// API.
void ParseProblemShape()
{
  // Dimensions.
  NumDimensions = 7;

  DimensionIDToName = {{0, "R"},
                   {1, "S"},
                   {2, "P"},
                   {3, "Q"},
                   {4, "C"},
                   {5, "K"},
                   {6, "N"}, };

  DimensionNameToID = {{'R', 0 },
                 {'S', 1 },
                 {'P', 2 },
                 {'Q', 3 },
                 {'C', 4 },
                 {'K', 5 },
                 {'N', 6 }, };

  // Coefficients.
  NumCoefficients = 4;

  CoefficientIDToName = {
    { 0, "Wstride" },
    { 1, "Hstride" },
    { 2, "Wdilation" },
    { 3, "Hdilation" }};

  CoefficientNameToID = {
    { "Wstride", 0 },
    { "Hstride", 1 },
    { "Wdilation", 2 },
    { "Hdilation", 3 }};

  DefaultCoefficients = {
    { 0, 1 },
    { 1, 1 },
    { 2, 1 },
    { 3, 1 }};

  // Data Spaces.
  NumDataSpaces = 3;
  
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

  DataSpaceOrder = { 4, // Weight
                     4, // Input
                     4 }; // Output
  
  IsReadWriteDataSpace = [](const DataSpaceID d) -> bool
    { return d == 2; }; // Output

  Projectors =
    {
      [](const WorkloadConfig* wc, const OperationPoint& problem_point)
      {
        (void) wc;

        Point weight_point(DataSpaceOrder[0]);

        // R,S,C,K
        weight_point[0] = problem_point[0]; // R
        weight_point[1] = problem_point[1]; // S
        weight_point[2] = problem_point[4]; // C
        weight_point[3] = problem_point[5]; // K

        return weight_point;
      },
      [](const WorkloadConfig* wc, const OperationPoint& problem_point)
      {
        Point input_point(DataSpaceOrder[1]);

        // W,H,C,N
        input_point[0] =
          wc->getCoefficient(0) * problem_point[2] +  // Wstride * P
          wc->getCoefficient(2) * problem_point[0];   // Wdilation * R
        input_point[1] =
          wc->getCoefficient(1) * problem_point[3] +  // Hstride * Q
          wc->getCoefficient(3) * problem_point[1];   // Hdilation * S
        
        input_point[2] = problem_point[4]; // C
        input_point[3] = problem_point[6]; // N

        return input_point;
      },
      [](const WorkloadConfig* wc, const OperationPoint& problem_point)
      {
        (void) wc;

        Point output_point(DataSpaceOrder[2]);

        // P,Q,K,N
        output_point[0] = problem_point[2]; // P
        output_point[1] = problem_point[3]; // Q
        
        output_point[2] = problem_point[5]; // K
        output_point[3] = problem_point[6]; // N

        return output_point;
      }
    };  
}

}  // namespace problem
