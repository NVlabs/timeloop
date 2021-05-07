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

#include "util/numeric.hpp"

#include "operation-space.hpp"

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

OperationSpace::OperationSpace(const Workload* wc,
                               const OperationPoint& flattened_low,
                               const OperationPoint& flattened_high) :
    workload_(wc)
{
  // Note: high *must* be inclusive. Projecting an exclusive high
  // either into the factorized space, or into a data-space may not result
  // in the exclusive high point in the projected space.

  auto shape = wc->GetShape();

  // Step 1: Factorize *each* rank in the flattened point into groups of
  // coordinates.
  std::vector<Point> factor_groups_low;
  std::vector<Point> factor_groups_high;
  std::vector<Point> factor_groups_bounds;
  
  FactorizeGrouped(wc, flattened_low, flattened_high,
                   factor_groups_low, factor_groups_high, factor_groups_bounds);

  ASSERT(factor_groups_low.size() == shape->NumFlattenedDimensions);
  ASSERT(factor_groups_high.size() == shape->NumFlattenedDimensions);
  ASSERT(factor_groups_bounds.size() == shape->NumFlattenedDimensions);

  // Step 2: Carve up the low->high region into problem-space AAHRs. We need to
  // perform the carving *before* projecting into data spaces because the
  // carving assumes that the region we're asking for (between low, high) is
  // a contiguous section within the flattened/factorized space. Once we
  // project into data-spaces that information is lost, and the carving
  // algorithm has no way to distinguish between contiguous regions (which may
  // need to be carved up) and clean AAHR regions. We assume that each AAHR in
  // factorized problem space will project onto an AAHR in each data-space.
  std::vector<std::vector<std::pair<Point, Point>>> region_aahrs;
  std::vector<std::size_t> num_region_aahrs;

  // Walk through each flattened rank.
  for (unsigned rank = 0; rank < shape->NumFlattenedDimensions; rank++)
  {
    // Carve up the local factorized region for this rank.
    auto region_carved = Carve(factor_groups_low.at(rank),
                               factor_groups_high.at(rank),
                               factor_groups_bounds.at(rank));
    // std::cout << "flat rank = " << rank << " carved into " << region_carved.size() << " aahrs\n";
    // for (auto& aahr: region_carved)
    // {
    //   std::cout << "  " << aahr.first << " - " << aahr.second << std::endl;
    // }
    region_aahrs.push_back(region_carved);
    num_region_aahrs.push_back(region_carved.size());
  }


  // The final set of AAHRs we are putting together.
  std::vector<std::pair<Point, Point>> carved_aahrs;

  // Walk through the AAHR sets using a Cartesian Counter. This flattens
  // an exponential space (e.g., that formed from a Cartesian product
  // into a linear space.
  // std::cout << "num_region_aahrs: ";
  // for (auto& n: num_region_aahrs)
  // {
  //   std::cout << n << " ";
  // }
  // std::cout << std::endl;

  CartesianCounterGeneric<std::size_t> counter(num_region_aahrs);
  // std::cout << "CC size = " << counter.EndInteger() << std::endl;

  do
  {
    // Obtain a per-flattened-rank AAHR id from the counter.
    std::vector<std::size_t> ids = counter.Read();
    ASSERT(ids.size() == shape->NumFlattenedDimensions);

    // This is the final AAHR we will be assembling across all ranks.
    Point final_low(shape->NumFactorizedDimensions);
    Point final_high(shape->NumFactorizedDimensions);

    // Walk through each flattened rank.
    for (unsigned flattened_dim = 0; flattened_dim < shape->NumFlattenedDimensions; flattened_dim++)
    {
      // std::cout << "flat rank = " << flattened_dim << " extracting aahr# " << ids.at(flattened_dim)
      //           << " of " << region_aahrs.at(flattened_dim).size() << std::endl;
      auto& region_aahr = region_aahrs.at(flattened_dim).at(ids.at(flattened_dim));
      auto& region_low = region_aahr.first;
      auto& region_high = region_aahr.second;
      
      // Walk though the compacted coordinates in the region aahrs, and
      // re-scatter them into the target flattened_dim in the final factorized AAHR. 
      for (unsigned i = 0; i < shape->FlattenedToFactorized.at(flattened_dim).size(); i++)
      {
        auto factorized_dim = shape->FlattenedToFactorized.at(flattened_dim).at(i);

        ASSERT(final_low[factorized_dim] == 0);
        ASSERT(final_high[factorized_dim] == 0);

        final_low[factorized_dim] = region_low[i];
        final_high[factorized_dim] = region_high[i];
      }
    }

    carved_aahrs.push_back(std::make_pair(final_low, final_high));
  }
  while (counter.Increment());
  
  // if (carved_aahrs.size() > 1)
  // {
  //   std::cout << "bounds: " << wc->GetFactorizedBounds() << std::endl;
  //   std::cout << "flattened: " << flattened_low << " - " << flattened_high << std::endl;

  //   std::cout << "carved:\n";
  //   for (auto& aahr: carved_aahrs)
  //   {
  //     std::cout << "  " << aahr.first << " - " << aahr.second << std::endl;
  //   }
  // }
  
  // Step 3: Project each of the factorized AAHRs onto data spaces.
  for (unsigned space_id = 0; space_id < wc->GetShape()->NumDataSpaces; space_id++)
  {
    auto dataspace_order = workload_->GetShape()->DataSpaceOrder.at(space_id);

    std::vector<std::pair<Point, Point>> dataspace_corners;
    for (auto& aahr: carved_aahrs)
    {
      Point space_low(dataspace_order);
      Point space_high(dataspace_order);

      // std::cout << "projecting ispace aahr  : " << aahr.first << " - " << aahr.second << std::endl;
      ProjectLowHigh(space_id, workload_, aahr.first, aahr.second, space_low, space_high);
      // std::cout << "projected to dspace aahr: " << space_low << " - " << space_high << std::endl;

      // Increment the high points by 1 because the AAHR constructor wants
      // an exclusive max point.
      space_high.IncrementAllDimensions();

      dataspace_corners.push_back(std::make_pair(space_low, space_high));

      // std::cout << "dataspace: " << space_low << " - " << space_high << ")" << std::endl;
    }

    data_spaces_.push_back(DataSpace(dataspace_order, dataspace_corners));
  }
}

// Project a point from flattened operation space into factorized problem space.
Point OperationSpace::Factorize(const Workload* wc, const OperationPoint& flattened)
{
  auto shape = wc->GetShape();

  Point factorized(shape->NumFactorizedDimensions);

  for (unsigned flattened_dim = 0; flattened_dim < shape->NumFlattenedDimensions; flattened_dim++)
  {
    // Assumption: un-flatten list is in low->high order.
    auto coordinate = flattened[flattened_dim];
    for (auto& factorized_dim: shape->FlattenedToFactorized.at(flattened_dim))
    {
      auto bound = wc->GetFactorizedBound(factorized_dim);
      factorized[factorized_dim] = coordinate % bound;
      coordinate = coordinate / bound;
    }
  }  

  return factorized;
}

// Factorize each rank and group the results into sub-vectors.
void OperationSpace::FactorizeGrouped(const Workload* wc,
                                      const OperationPoint& flattened_low,
                                      const OperationPoint& flattened_high,

                                      std::vector<Point>& factor_groups_low,
                                      std::vector<Point>& factor_groups_high,
                                      std::vector<Point>& factor_groups_bounds)
{
  auto shape = wc->GetShape();

  for (unsigned flattened_dim = 0; flattened_dim < shape->NumFlattenedDimensions; flattened_dim++)
  {
    // std::cout << "flat rank = " << flattened_dim << std::endl;

    // The "point" for this flattened dim is a group of factorized coordinates
    // that the flattened rank will project to.
    Point factors_low(shape->FlattenedToFactorized.at(flattened_dim).size());
    Point factors_high(shape->FlattenedToFactorized.at(flattened_dim).size());
    Point factors_bounds(shape->FlattenedToFactorized.at(flattened_dim).size());

    // Assumption: factorized list is in low->high order.
    auto coordinate_low = flattened_low[flattened_dim];
    auto coordinate_high = flattened_high[flattened_dim];

    for (unsigned i = 0; i < shape->FlattenedToFactorized.at(flattened_dim).size(); i++)
    {
      auto factorized_dim = shape->FlattenedToFactorized.at(flattened_dim).at(i);
      auto bound = wc->GetFactorizedBound(factorized_dim);

      // Note that we are shoving coordinates into the "point" serially,
      // ignoring the factorized_dim they came out of. We will perform
      // carving in this compressed/re-ordered space, and then re-scatter
      // the dimensions into the right place during re-assembly in a different
      // function.
      factors_low[i] = coordinate_low % bound;
      factors_high[i] = coordinate_high % bound;
      factors_bounds[i] = bound;
      // factorized[factorized_dim] = coordinate % bound;

      coordinate_low = coordinate_low / bound;
      coordinate_high = coordinate_high / bound;
    }

    // std::cout << "  factors low-high: " << factors_low << " - " << factors_high << std::endl;
    
    // Insertion index will be flattened_dim for all these vectors.
    factor_groups_low.push_back(factors_low);
    factor_groups_high.push_back(factors_high);
    factor_groups_bounds.push_back(factors_bounds);
  }  
}

void OperationSpace::ProjectLowHigh(Shape::DataSpaceID d,
                                    const Workload* wc,
                                    const Point& factorized_low,
                                    const Point& factorized_high,
                                    Point& data_space_low,
                                    Point& data_space_high)
{
  for (unsigned data_space_dim = 0; data_space_dim < wc->GetShape()->DataSpaceOrder.at(d); data_space_dim++)
  {
    data_space_low[data_space_dim] = 0;
    data_space_high[data_space_dim] = 0;

    for (auto& term : wc->GetShape()->Projections.at(d).at(data_space_dim))
    {
      Coordinate low = factorized_low[term.second];
      Coordinate high = factorized_high[term.second];
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
                              const Point& factorized_point)
{
  Point data_space_point(wc->GetShape()->DataSpaceOrder.at(d));

  for (unsigned data_space_dim = 0; data_space_dim < wc->GetShape()->DataSpaceOrder.at(d); data_space_dim++)
  {
    data_space_point[data_space_dim] = 0;
    for (auto& term : wc->GetShape()->Projections.at(d).at(data_space_dim))
    {
      Coordinate x = factorized_point[term.second];
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


// OperationSpace& OperationSpace::operator += (const OperationSpace& s)
// {
//   for (unsigned i = 0; i < data_spaces_.size(); i++)
//     data_spaces_.at(i) += s.data_spaces_.at(i);

//   return (*this);
// }

OperationSpace& OperationSpace::operator += (const OperationPoint& p)
{
  // Step 1: un-flattern the provided point.
  Point factorized = Factorize(workload_, p);

  // Step 2: project and add into all data-spaces.
  for (unsigned i = 0; i < data_spaces_.size(); i++)
  {
    data_spaces_.at(i) += Project(i, workload_, factorized);
    std::cout << "dataspace after add: " << data_spaces_.at(i) << std::endl;
  }

  return (*this);
}

// OperationSpace& OperationSpace::ExtrudeAdd(const OperationSpace& s)
// {
//   for (unsigned i = 0; i < data_spaces_.size(); i++)
//     data_spaces_.at(i).ExtrudeAdd(s.data_spaces_.at(i));

//   return (*this);
// }

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
    out << data_spaces_.at(i);
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
  out << d;
}

std::ostream& operator << (std::ostream& out, const OperationSpace& os)
{
  os.Print(out);
  return out;
}

} // namespace problem
