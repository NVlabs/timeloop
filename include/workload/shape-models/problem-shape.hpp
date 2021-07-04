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

#pragma once

#include <map>
#include <vector>
#include <list>

#include "compound-config/compound-config.hpp"

namespace problem
{

class Shape
{
 public:
  typedef unsigned DimensionID;
  
  unsigned NumDimensions;
  std::map<DimensionID, std::string> DimensionIDToName;
  std::map<std::string, DimensionID> DimensionNameToID;

  typedef int Coefficient;
  typedef unsigned CoefficientID;
  typedef std::map<CoefficientID, int> Coefficients;

  unsigned NumCoefficients;
  std::map<std::string, CoefficientID> CoefficientNameToID;
  std::map<CoefficientID, std::string> CoefficientIDToName;
  std::map<CoefficientID, int> DefaultCoefficients;

  typedef unsigned DataSpaceID;

  unsigned NumDataSpaces;
  std::map<std::string, DataSpaceID> DataSpaceNameToID;
  std::map<DataSpaceID, std::string> DataSpaceIDToName;
  std::map<DataSpaceID, unsigned> DataSpaceOrder;
  std::map<DataSpaceID, bool> IsReadWriteDataSpace;

  // Projection AST: the projection function for each dataspace dimension is a
  //                 Sum-Of-Products where each Product is the product of a
  //                 Coefficient and a Dimension. This is fairly restrictive
  //                 but efficient. We can generalize later if needed.
  typedef std::pair<CoefficientID, DimensionID> ProjectionTerm;
  typedef std::list<ProjectionTerm> ProjectionExpression;
  typedef std::vector<ProjectionExpression> Projection;

  std::vector<Projection> Projections;

  std::vector<std::set<DimensionID>> DataSpaceIDToDimensionIDVector;

 public: 
  void Parse(config::CompoundConfigNode config);
  std::set<DimensionID> GetCoIteratedDimensions(const std::vector<DataSpaceID> dataspace_pair) const;
  std::set<DimensionID> GetFullyContractedDimensions() const;
};

} // namespace problem
