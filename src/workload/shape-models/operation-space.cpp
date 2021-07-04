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

#include "workload/shape-models/operation-space.hpp"

namespace problem
{

// ======================================= //
//              OperationPoint             //
// ======================================= //

std::ostream& operator << (std::ostream& out, const OperationPoint& p)
{
  p.Print(out);
  return out;
}

// ======================================= //
//              OperationSpace             //
// ======================================= //

OperationSpace::OperationSpace(const Workload* wc) :
    workload_(wc)
{
  for (unsigned space_id = 0; space_id < wc->GetShape()->NumDataSpaces; space_id++)
    data_spaces_.push_back(DataSpace(wc->GetShape()->DataSpaceOrder.at(space_id)));
}

OperationSpace::OperationSpace() :
    OperationSpace(nullptr)
{ }

OperationSpace::OperationSpace(const Workload* wc, const OperationPoint& low, const OperationPoint& high) :
    workload_(wc)
{
  // Note: high *must* be inclusive. Projecting an exclusive high operation-point into
  // a data-space may not result in the exclusive high point in that data-space.
  for (unsigned space_id = 0; space_id < wc->GetShape()->NumDataSpaces; space_id++)
  {
    Point space_low(workload_->GetShape()->DataSpaceOrder.at(space_id));
    Point space_high(workload_->GetShape()->DataSpaceOrder.at(space_id));

    ProjectLowHigh(space_id, workload_, low, high, space_low, space_high);

    // Increment the high points by 1 because the AAHR constructor wants
    // an exclusive max point.
    space_high.IncrementAllDimensions();
    data_spaces_.push_back(DataSpace(wc->GetShape()->DataSpaceOrder.at(space_id), space_low, space_high));
  }
}

void OperationSpace::ProjectLowHigh(Shape::DataSpaceID d,
                                    const Workload* wc,
                                    const OperationPoint& problem_low,
                                    const OperationPoint& problem_high,
                                    Point& data_space_low,
                                    Point& data_space_high)
{
  for (unsigned data_space_dim = 0; data_space_dim < wc->GetShape()->DataSpaceOrder.at(d); data_space_dim++)
  {
    data_space_low[data_space_dim] = 0;
    data_space_high[data_space_dim] = 0;

    for (auto& term : wc->GetShape()->Projections.at(d).at(data_space_dim))
    {
      Coordinate low = problem_low[term.second];
      Coordinate high = problem_high[term.second];
      if (term.first != wc->GetShape()->NumCoefficients)
      {
        // If Coefficient is negative, flip high/low.
        auto coeff = wc->GetCoefficient(term.first);
        if (coeff < 0)
        {
          data_space_low[data_space_dim] += (high * coeff);
          data_space_high[data_space_dim] += (low * coeff);
        }
        else
        {
          data_space_low[data_space_dim] += (low * coeff);
          data_space_high[data_space_dim] += (high * coeff);
        }
      }
      else
      {
        data_space_low[data_space_dim] += low;
        data_space_high[data_space_dim] += high;
      }
    }
  }
}

Point OperationSpace::Project(Shape::DataSpaceID d,
                              const Workload* wc,
                              const OperationPoint& problem_point)
{
  Point data_space_point(wc->GetShape()->DataSpaceOrder.at(d));

  for (unsigned data_space_dim = 0; data_space_dim < wc->GetShape()->DataSpaceOrder.at(d); data_space_dim++)
  {
    data_space_point[data_space_dim] = 0;
    for (auto& term : wc->GetShape()->Projections.at(d).at(data_space_dim))
    {
      Coordinate x = problem_point[term.second];
      // FIXME: somehow "compile" the coefficients down for a given
      // workload config so that we avoid the branch and lookup below.
      if (term.first != wc->GetShape()->NumCoefficients)
        data_space_point[data_space_dim] += (x * wc->GetCoefficient(term.first));
      else
        data_space_point[data_space_dim] += x;
    }
  }

  return data_space_point;
}

void OperationSpace::Reset()
{
  for (auto& d : data_spaces_)
    d.Reset();
}

DataSpace& OperationSpace::GetDataSpace(Shape::DataSpaceID pv)
{
  return data_spaces_.at(pv);
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
    data_spaces_.at(i) += Project(i, workload_, p);

  return (*this);
}

OperationSpace& OperationSpace::ExtrudeAdd(const OperationSpace& s)
{
  for (unsigned i = 0; i < data_spaces_.size(); i++)
    data_spaces_.at(i).ExtrudeAdd(s.data_spaces_.at(i));

  return (*this);
}

OperationSpace OperationSpace::operator - (const OperationSpace& p)
{
  OperationSpace retval(workload_);

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
  {
    std::cout << workload_->GetShape()->DataSpaceIDToName.at(i) << " = " << data_spaces_.at(i).size() << ", ";
  }
  std::cout << workload_->GetShape()->DataSpaceIDToName.at(data_spaces_.size()-1) << " = " << data_spaces_.back().size() << std::endl;
}

void OperationSpace::Print(std::ostream& out) const
{
  for (unsigned i = 0; i < data_spaces_.size(); i++)
  {
    out << workload_->GetShape()->DataSpaceIDToName.at(i) << ": ";
    data_spaces_.at(i).Print(out);
    out << " ";
  }
  // for (auto& d : data_spaces_)
  // {
  //   d.Print(out);
  //   out << " ";
  // }
}

void OperationSpace::Print(Shape::DataSpaceID pv, std::ostream& out) const
{
  auto& d = data_spaces_.at(unsigned(pv));
  d.Print(out);
}

std::ostream& operator << (std::ostream& out, const OperationSpace& os)
{
  os.Print(out);
  return out;
}

} // namespace problem
