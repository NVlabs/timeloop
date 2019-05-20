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

#include "cnn/problem-config.hpp"

namespace problem
{

unsigned NumDimensions;
unsigned NumDataSpaces;
std::map<DataSpaceID, std::string> DataSpaceIDToName;
std::map<std::string, DataSpaceID> DataSpaceNameToID;
std::vector<unsigned> DataSpaceOrder;

std::function<bool(const DataSpaceID d)> IsReadWriteDataSpace;
std::vector<std::function<Point(WorkloadConfig*, const OperationPoint&)>> projectors;

// std::ostream& operator << (std::ostream& out, const DataSpaceID& d)
// {
//   out << DataSpaceIDToName[d];
//   return out;
// }

std::map<Dimension, std::string> DimensionName;
std::map<char, Dimension> DimensionID;

// std::ostream& operator << (std::ostream& out, const Dimension& dim)
// {
//   out << DimensionName[dim];
//   return out;
// }

// ======================================== //
//              Problem Shape               //
// ======================================== //

void BuildProblemShape()
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

  // enum class DataSpaceID : unsigned int
  // {
  //   Weight,
  //   Input,
  //   Output,
  //   Num
  // };
  
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

  projectors =
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

// ======================================== //
//             OperationSpace               //
// ======================================== //

OperationSpace::OperationSpace(WorkloadConfig* wc) :
    workload_config_(wc)
{
  for (unsigned space_id = 0; space_id < NumDataSpaces; space_id++)
    data_spaces_.push_back(DataSpace(DataSpaceOrder.at(space_id)));
}

OperationSpace::OperationSpace() :
    OperationSpace(nullptr)
{ }

OperationSpace::OperationSpace(WorkloadConfig* wc, const OperationPoint& low, const OperationPoint& high) :
    workload_config_(wc)
{
  for (unsigned space_id = 0; space_id < NumDataSpaces; space_id++)
  {
    auto space_low = projectors.at(space_id)(workload_config_, low);
    auto space_high = projectors.at(space_id)(workload_config_, high);
    // Increment the high points by 1 because the AAHR constructor wants
    // an exclusive max point.
    space_high.IncrementAllDimensions();
    data_spaces_.push_back(DataSpace(DataSpaceOrder.at(space_id), space_low, space_high));
  }
}

void OperationSpace::Reset()
{
  for (auto& d : data_spaces_)
    d.Reset();
}

OperationSpace& OperationSpace::operator += (const OperationSpace& s)
{
  for (unsigned i = 0; i < data_spaces_.size(); i++)
    data_spaces_.at(i) += s.data_spaces_.at(i);

  return (*this);
}

OperationSpace& OperationSpace::operator += (const OperationPoint& p)
{
  for (unsigned i = 0; i < data_spaces_.size(); i++)
    data_spaces_.at(i) += projectors.at(i)(workload_config_, p);

  return (*this);
}

OperationSpace OperationSpace::operator - (const OperationSpace& p)
{
  OperationSpace retval(workload_config_);

  for (unsigned i = 0; i < data_spaces_.size(); i++)
    retval.data_spaces_.at(i) = data_spaces_.at(i) - p.data_spaces_.at(i);
  
  return retval;
}

PerDataSpace<std::size_t> OperationSpace::GetSizes() const
{
  PerDataSpace<std::size_t> retval;
  
  for (unsigned i = 0; i < data_spaces_.size(); i++)
    retval.at(i) = data_spaces_.at(i).size();

  return retval;
}

std::size_t OperationSpace::GetSize(const int t) const
{
  return data_spaces_.at(t).size();
}

bool OperationSpace::IsEmpty(const int t) const
{
  return data_spaces_.at(t).empty();
}

bool OperationSpace::CheckEquality(const OperationSpace& rhs, const int t) const
{
  return data_spaces_.at(t) == rhs.data_spaces_.at(t);
}

void OperationSpace::PrintSizes()
{
  for (unsigned i = 0; i < data_spaces_.size()-1; i++)
    std::cout << DataSpaceIDToName[i] << " = " << data_spaces_.at(i).size() << ", ";
  std::cout << DataSpaceIDToName[data_spaces_.size()-1] << " = " << data_spaces_.back().size() << std::endl;
}

void OperationSpace::Print() const
{
  for (auto& d : data_spaces_)
    d.Print();
}

void OperationSpace::Print(DataSpaceID pv) const
{
  auto& d = data_spaces_.at(unsigned(pv));
  d.Print();
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
