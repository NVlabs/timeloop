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

std::map<DataType, std::string> DataTypeName = {
    {DataType::Weight, "Weights"},
    {DataType::Input,  "Inputs"},
    {DataType::Output, "Outputs"},
    {DataType::Num,    "Shared/Illegal"}};

std::map<std::string, DataType> DataTypeID = {
  {"Weights", DataType::Weight},
  {"Inputs", DataType::Input},
  {"Outputs", DataType::Output},
  {"Shared/Illegal", DataType::Num}};

std::ostream& operator << (std::ostream& out, const DataType& d)
{
  out << DataTypeName[d];
  return out;
}

bool IsReadWriteDataType(const DataType d)
{
  // ASSERT(d < DataType::Num);
  return d == DataType::Output;
}

std::map<Dimension, std::string> DimensionName = {{Dimension::R, "R"},
                                                  {Dimension::S, "S"},
                                                  {Dimension::P, "P"},
                                                  {Dimension::Q, "Q"},
                                                  {Dimension::C, "C"},
                                                  {Dimension::K, "K"},
                                                  {Dimension::N, "N"}, };

std::map<char, Dimension> DimensionID = {{'R', Dimension::R },
                                         {'S', Dimension::S },
                                         {'P', Dimension::P },
                                         {'Q', Dimension::Q },
                                         {'C', Dimension::C },
                                         {'K', Dimension::K },
                                         {'N', Dimension::N }, };

std::ostream& operator << (std::ostream& out, const Dimension& dim)
{
  out << DimensionName[dim];
  return out;
}

// ======================================== //
//              WorkloadConfig              //
// ======================================== //

WeightPoint MakeWeightPoint(WorkloadConfig* wc, const OperationPoint& problem_point)
{
  (void) wc;

  WeightPoint weight_point(int(WeightDimension::Num));

  weight_point[int(WeightDimension::R)] = problem_point[int(Dimension::R)];
  weight_point[int(WeightDimension::S)] = problem_point[int(Dimension::S)];
  weight_point[int(WeightDimension::C)] = problem_point[int(Dimension::C)];
  weight_point[int(WeightDimension::K)] = problem_point[int(Dimension::K)];

  return weight_point;
}

InputPoint MakeInputPoint(WorkloadConfig* wc, const OperationPoint& problem_point)
{
  InputPoint input_point(int(InputDimension::Num));

  input_point[int(InputDimension::W)] =
    wc->getWstride() * problem_point[int(Dimension::P)] +
    wc->getWdilation() * problem_point[int(Dimension::R)];
  input_point[int(InputDimension::H)] =
    wc->getHstride() * problem_point[int(Dimension::Q)] +
    wc->getHdilation() * problem_point[int(Dimension::S)];

  input_point[int(InputDimension::C)] = problem_point[int(Dimension::C)];
  input_point[int(InputDimension::N)] = problem_point[int(Dimension::N)];

  return input_point;
}

OutputPoint MakeOutputPoint(WorkloadConfig* wc, const OperationPoint& problem_point)
{
  (void) wc;

  OutputPoint output_point(int(OutputDimension::Num));

  output_point[int(OutputDimension::P)] = problem_point[int(Dimension::P)];
  output_point[int(OutputDimension::Q)] = problem_point[int(Dimension::Q)];

  output_point[int(OutputDimension::K)] = problem_point[int(Dimension::K)];
  output_point[int(OutputDimension::N)] = problem_point[int(Dimension::N)];

  return output_point;
}

// ======================================== //
//             OperationSpace               //
// ======================================== //


OperationSpace::OperationSpace() :
    OperationSpace(nullptr)
{
}

OperationSpace::OperationSpace(const OperationSpace& s) :
    workload_config_(s.workload_config_),
    weights_(s.weights_),
    inputs_(s.inputs_),
    outputs_(s.outputs_)
{
}

OperationSpace::OperationSpace(WorkloadConfig* wc) :
    workload_config_(wc),
    weights_(int(WeightDimension::Num)),
    inputs_(int(InputDimension::Num)),
    outputs_(int(OutputDimension::Num))
{
}

OperationSpace::OperationSpace(WorkloadConfig* wc, const OperationPoint& low, const OperationPoint& high) :
    OperationSpace(wc)
{
  
  auto weights_low = MakeWeightPoint(workload_config_, low);
  auto inputs_low = MakeInputPoint(workload_config_, low);
  auto outputs_low = MakeOutputPoint(workload_config_, low);

  auto weights_high = MakeWeightPoint(workload_config_, high);
  auto inputs_high = MakeInputPoint(workload_config_, high);
  auto outputs_high = MakeOutputPoint(workload_config_, high);

  // Increment the high points by 1 because the AAHR constructor wants
  // an exclusive max point.
  for (unsigned i = 0; i < unsigned(problem::WeightDimension::Num); i++)
  {
    weights_high[i]++;
  }
  for (unsigned i = 0; i < unsigned(problem::InputDimension::Num); i++)
  {
    inputs_high[i]++;
  }
  for (unsigned i = 0; i < unsigned(problem::OutputDimension::Num); i++)
  {
    outputs_high[i]++;
  }
  
  weights_ = WeightPointSet(int(WeightDimension::Num), weights_low, weights_high);
  inputs_ = InputPointSet(int(InputDimension::Num), inputs_low, inputs_high);
  outputs_ = OutputPointSet(int(OutputDimension::Num), outputs_low, outputs_high);
}

void OperationSpace::Reset()
{
  weights_ = WeightPointSet(int(WeightDimension::Num));
  inputs_ = InputPointSet(int(InputDimension::Num));
  outputs_ = OutputPointSet(int(OutputDimension::Num));
}

OperationSpace& OperationSpace::operator+=(const OperationSpace& s)
{
  weights_ += s.weights_;
  inputs_ += s.inputs_;
  outputs_ += s.outputs_;
  return (*this);
}

OperationSpace& OperationSpace::operator+=(const OperationPoint& p)
{
  weights_ += MakeWeightPoint(workload_config_, p);
  inputs_ += MakeInputPoint(workload_config_, p);
  outputs_ += MakeOutputPoint(workload_config_, p);
  return (*this);
}

OperationSpace OperationSpace::operator-(const OperationSpace& p)
{
  OperationSpace retval;
  retval.weights_ = weights_ - p.weights_;
  retval.inputs_ = inputs_ - p.inputs_;
  retval.outputs_ = outputs_ - p.outputs_;
  return retval;
}

PerDataSpace<std::size_t> OperationSpace::GetSizes() const
{
  return { weights_.size(), inputs_.size(), outputs_.size() };
}

std::size_t OperationSpace::GetSize(const int t) const
{
  assert(t >= 0 && t < int(problem::DataType::Num));
  if (t == int(problem::DataType::Weight))
  {
    return weights_.size();
  }
  else if (t == int(problem::DataType::Input))
  {
    return inputs_.size();
  }
  else
  {
    return outputs_.size();
  }
}

bool OperationSpace::IsEmpty(const int t) const
{
  assert(t >= 0 && t < int(problem::DataType::Num));
  if (t == int(problem::DataType::Weight))
  {
    return weights_.empty();
  }
  else if (t == int(problem::DataType::Input))
  {
    return inputs_.empty();
  }
  else
  {
    return outputs_.empty();
  }
}

bool OperationSpace::CheckEquality(const OperationSpace& rhs, const int t) const
{
  assert(t >= 0 && t < int(problem::DataType::Num));
  if (t == int(problem::DataType::Weight))
  {
    return weights_ == rhs.weights_;
  }
  else if (t == int(problem::DataType::Input))
  {
    return inputs_ == rhs.inputs_;
  }
  else
  {
    return outputs_ == rhs.outputs_;
  }
}

void OperationSpace::PrintSizes()
{
  std::cout << "weights = " << weights_.size() << ", ";
  std::cout << "outputs = " << outputs_.size() << std::endl;
  std::cout << "inputs = " << inputs_.size() << ", ";
}

void OperationSpace::Print() const
{
  std::cout << "Weights[" << weights_.size() << "]: ";
  weights_.Print();
  std::cout << std::endl;
  std::cout << "Inputs[" << inputs_.size() << "]: ";
  inputs_.Print();
  std::cout << std::endl;
  std::cout << "Outputs[" << outputs_.size() << "]: ";
  outputs_.Print();
  std::cout << std::endl;
}

void OperationSpace::Print(DataType pv) const
{
  switch (pv)
  {
    case DataType::Weight: 
      std::cout << "Weights[" << weights_.size() << "]: ";
      weights_.Print();
      std::cout << std::endl;
      break;
    case DataType::Input:
      std::cout << "Inputs[" << inputs_.size() << "]: ";
      inputs_.Print();
      std::cout << std::endl;
      break;
    case DataType::Output:
      std::cout << "Outputs[" << outputs_.size() << "]: ";
      outputs_.Print();
      std::cout << std::endl;
      break;
    default:
      std::cout << "ILLEGAL DATATYPE, CAN'T PRINT" << std::endl;
      break;
  }
}


PerDataSpace<std::size_t> GetMaxWorkingSetSizes(
    problem::PerProblemDimension<int> dimension_sizes)
{
  PerDataSpace<std::size_t> datatype_size;

  datatype_size[DataType::Weight] =
      dimension_sizes[int(Dimension::R)] * dimension_sizes[int(Dimension::S)] *
      dimension_sizes[int(Dimension::C)] * dimension_sizes[int(Dimension::K)];

  datatype_size[DataType::Input] =
      (dimension_sizes[int(Dimension::P)] + dimension_sizes[int(Dimension::R)] - 1) *
      (dimension_sizes[int(Dimension::Q)] + dimension_sizes[int(Dimension::S)] - 1) *
      dimension_sizes[int(Dimension::C)] * dimension_sizes[int(Dimension::N)];

  datatype_size[DataType::Output] =
      dimension_sizes[int(Dimension::P)] * dimension_sizes[int(Dimension::Q)] *
      dimension_sizes[int(Dimension::K)] * dimension_sizes[int(Dimension::N)];
  
  return datatype_size;
}

}  // namespace problem
