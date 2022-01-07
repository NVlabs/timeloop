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

#include <cmath>

#include "loop-analysis/point-set-multi-aahr.hpp"

MultiAAHR::MultiAAHR(std::uint32_t order) :
    order_(order),
    aahrs_()
{
}

MultiAAHR::MultiAAHR(std::uint32_t order, const Point unit) :
    order_(order)
{
  // Create a single AAHR.
  ASSERT(aahrs_.size() == 0);
  aahrs_.push_back(AxisAlignedHyperRectangle(order, unit));
}

MultiAAHR::MultiAAHR(std::uint32_t order, const Point min, const Point max) :
    order_(order)
{
  // Create a single AAHR.
  ASSERT(aahrs_.size() == 0);
  aahrs_.push_back(AxisAlignedHyperRectangle(order, min, max));
}

MultiAAHR::MultiAAHR(std::uint32_t order, const std::vector<std::pair<Point, Point>> corner_sets) :
    order_(order)
{
  // Create multiple AAHRs.
  ASSERT(aahrs_.size() == 0);

  for (auto& corners: corner_sets)
  {
    // Don't trust the user to give us non-overlapping AAHRs: perform a
    // union-insertion just to be sure.

    // FIXME: performance optimization: perform this on a single AAHR
    // candidate instead of a multiAAHR.
    MultiAAHR candidate(order, corners.first, corners.second);
    (*this) += candidate;
    //aahrs_.push_back(AxisAlignedHyperRectangle(order, corners.first, corners.second));
  }
}

MultiAAHR::MultiAAHR(const MultiAAHR& a) :
    order_(a.order_),
    aahrs_(a.aahrs_)
{
}

// Copy-and-swap idiom.
MultiAAHR& MultiAAHR::operator = (MultiAAHR other)
{
  swap(*this, other);
  return *this;
}

void swap(MultiAAHR& first, MultiAAHR& second)
{
  using std::swap;
  swap(first.order_, second.order_);
  swap(first.aahrs_, second.aahrs_);
}

std::size_t MultiAAHR::size() const
{
  size_t size = 0;
  for (auto& aahr: aahrs_)
  {
    size += aahr.size();
  }

  return size;
}

bool MultiAAHR::empty() const
{
  for (auto& aahr: aahrs_)
  {
    if (!aahr.empty())
    {
      return false;
    }
  }
  return true;
}

void MultiAAHR::Reset()
{
  aahrs_.clear();
}

MultiAAHR& MultiAAHR::operator += (const Point& p)
{
  bool found = false;

  // If this point is already a subset of one of the AAHRs, we are done.
  for (auto& aahr: aahrs_)
  {
    if (aahr.Contains(p))
    {
      found = true;
      break;
    }
  }

  if (found)
  {
    return *this;
  }

  // If this point is adjacent to one of the AAHRs, merge it.
  for (auto& aahr: aahrs_)
  {
    if (aahr.MergeIfAdjacent(p))
    {
      found = true;
      break;
    }
  }
    
  if (found)
  {
    return *this;
  }

  // None of the AAHRs could naturally merge the point, so we need to
  // create a new AAHR for this point.
  aahrs_.push_back(AxisAlignedHyperRectangle(order_, p));

  return *this;
}

void MultiAAHR::Subtract(const MultiAAHR& other)
{
  // For each AAHR in other, subtract that AAHR from each one of our AAHRs
  // and place all the splinters in a new vector. Swap that vector with our
  // AAHR vector and continue until we run out of other's AAHRs.
  std::vector<AxisAlignedHyperRectangle> deltas;
  for (auto& b: other.aahrs_)
  {
    for (auto& a: aahrs_)
    {
      auto diff = a.MultiSubtract(b);
      deltas.insert(deltas.end(), diff.begin(), diff.end());
    }
    std::swap(deltas, aahrs_);
    deltas.clear();
  }
}

MultiAAHR& MultiAAHR::operator += (const MultiAAHR& s)
{
  Subtract(s);
  for (auto& aahr: s.aahrs_)
  {
    if (!aahr.empty())
      aahrs_.push_back(aahr);
  }
  return *this;
}

MultiAAHR MultiAAHR::operator - (const MultiAAHR& other)
{
  MultiAAHR delta(*this);
  delta.Subtract(other);
  return delta;
}

bool MultiAAHR::operator == (const MultiAAHR& s) const
{
  // Do a superficial match: try to find an exact copy of each AAHR
  // in the other set. In reality the points may be re-distributed
  // in a different pattern in the other set, but we ignore this
  // at the moment.

  // Because the AAHRs in each set are guaranteed to be disjoint,
  // our match procedure is a little simpler.
  if (aahrs_.size() != s.aahrs_.size())
    return false;

  for (auto& a: aahrs_)
  {
    // Find this AAHR in the other set.
    bool found = false;
    for (auto& b: s.aahrs_)
    {
      if (a == b)
      {
        found = true;
        break;
      }
    }

    if (!found)
    {
      return false;
    }
  }

  return true;
}

Point MultiAAHR::GetTranslation(const MultiAAHR& s) const
{
  // We're computing translation from (this) -> (s).

  // Approximation: compute the weighted center-of-mass of each set and
  // compute the translation between the centers-of-mass.
  std::vector<double> weighted_centroid_a(order_, 0.0);
  std::size_t total_size_a = 0;

  for (auto& x: aahrs_)
  {      
    std::vector<double> centroid = x.Centroid();
    size_t size = x.size();
    for (unsigned rank = 0; rank < order_; rank++)
    {
      weighted_centroid_a[rank] += centroid[rank] * size;
    }
    total_size_a += size;
  }

  if (total_size_a != 0)
  {
    for (unsigned rank = 0; rank < order_; rank++)
    {
      weighted_centroid_a[rank] /= double(total_size_a);
    }
  }

  std::vector<double> weighted_centroid_b(order_, 0.0);
  std::size_t total_size_b = 0;

  for (auto& x: s.aahrs_)
  {      
    std::vector<double> centroid = x.Centroid();
    size_t size = x.size();
    for (unsigned rank = 0; rank < order_; rank++)
    {
      weighted_centroid_b[rank] += centroid[rank] * size;
    }
    total_size_b += size;
  }

  if (total_size_b != 0)
  {
    for (unsigned rank = 0; rank < order_; rank++)
    {
      weighted_centroid_b[rank] /= double(total_size_b);
    }
  }

  // If either *this or s are null, we return 0 as the translation.
  Point retval(order_);

  if (total_size_a > 0 && total_size_b > 0)
  {
    for (unsigned rank = 0; rank < order_; rank++)
    {
      double diff = weighted_centroid_b[rank] - weighted_centroid_a[rank];
      retval[rank] = static_cast<Coordinate>(round(diff));
    }
  }

  return retval;
}

void MultiAAHR::Translate(const Point& p)
{
  // Translate each AAHR.
  for (auto& x: aahrs_)
  {
    x.Translate(p);
  }
}

std::ostream& operator << (std::ostream& out, const MultiAAHR& m)
{
  out << "{ ";
  for (auto& x: m.aahrs_)
  {
    out << x << ", ";
  }
  out << "}";
  return out;
}
