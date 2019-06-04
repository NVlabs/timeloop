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

#pragma once

#include <set>
#include <unordered_set>
#include <vector>
#include <iostream>

#include "point-set-aahr.hpp"

// A FlexPoint is a variable-order point.
typedef std::vector<Coordinate> FlexPoint;

// The following function expands two order-dimensional endpoints into a vector
// of order-dimensional points that fill out the axis-aligned hyper-rectangle
// between the endpoints. This is used by a constructor in the point set
// implementation below. However, implementing the logic as a recursive template
// causes GCC to hang for an unknown reason. Therefore, we use a different point
// implementation as an intermediate to construct this space.

std::vector<FlexPoint> Expand(const FlexPoint min, const FlexPoint max);

// ---------------------------------------------
//     Generic-Slow Point Set implementation
// ---------------------------------------------

template <std::uint32_t order>
class PointSetGenericSlow
{
 private:
  std::set<Point<order>> points_;

  // We maintain an internal copy of the point set in AAHR representation.
  // This is *only* used to debugging AAHR, and should really be #ifdef'd
  // out. However, because AAHR is so much faster than this implementation,
  // it adds negligible overhead. We just need to comment out the Check()
  // calls for validation, since they are guaranteed to fail. Alternatively,
  // we could re-build our point space after each Subtract call from what
  // the AAHR thinks it should be (if we trust the AAHR implementation),
  // because that is more likely to be what hardware pattern-generation
  // state machines can capture. TODO.
  AxisAlignedHyperRectangle<order> aahr_;

 public:
  PointSetGenericSlow() : points_(), aahr_()
  {
  }

  // This is a very dangerous constructor. At construction-time,
  // it assumes that the point set volume is an axis-aligned
  // hyper-rectangle. Therefore, the entire set can be captured
  // within 2 order-dimensional min and max points. Once the
  // set is constructed, it is free to violate the AAHR shape.
  // It's also a very expensive operation.
  PointSetGenericSlow(const Point<order> min, const Point<order> max)
      : aahr_(min, max)
  {
    // Construct a flex-point version of min and max.
    FlexPoint flex_min(order), flex_max(order);
    for (unsigned dim = 0; dim < order; dim++)
    {
      flex_min[dim] = min[dim];
      flex_max[dim] = max[dim];
    }

    // Expand the space.
    auto flex_point_vector = Expand(flex_min, flex_max);

    // Convert into our internal space representation.
    for (auto& flex_point: flex_point_vector)
    {
      assert(flex_point.size() == order);
      Point<order> point;
      for (unsigned dim = 0; dim < order; dim++)
      {
        point[dim] = flex_point[dim];
      }
      points_.insert(point);
    }

    CheckAAHREquivalence();
  }

  std::size_t size() const
  {
    return points_.size();
  }

  bool empty() const
  {
    return points_.empty();
  }

  void Reset()
  {
    points_.clear();
    aahr_.Reset();
  }

  PointSetGenericSlow<order>& operator+=(const Point<order>& p)
  {
    points_.insert(p);
    aahr_ += p;
    CheckAAHREquivalence();
    return *this;
  }

  PointSetGenericSlow<order>& operator+=(const PointSetGenericSlow<order>& s)
  {
    points_.insert(s.points_.begin(), s.points_.end());
    aahr_ += s.aahr_;
    CheckAAHREquivalence();
    return *this;
  }

  PointSetGenericSlow<order> operator-(const PointSetGenericSlow<order>& s)
  {
    PointSetGenericSlow<order> r(*this);
    for (auto i = s.points_.begin(); i != s.points_.end(); i++)
    {
      r.points_.erase(*i);  // Note: MUST erase by value.
    }
    r.aahr_ = aahr_ - s.aahr_;
    r.CheckAAHREquivalence();
    return r;
  }

  bool operator==(const PointSetGenericSlow<order>& rhs) const
  {
    PointSetGenericSlow<order> lhs = *(this);
    if (lhs.points_.size() != rhs.points_.size())
    {
      return false;
    }
    else
    {
      for (auto i = rhs.points_.begin(); i != rhs.points_.end(); i++)
      {
        if (lhs.points_.count(*i) == 0) return false;
        lhs.points_.erase(*i);
      }
      return lhs.points_.empty();
    }
  };

  void CheckAAHREquivalence()
  {
#ifdef CHECK_AAHR_EQUIVALENCE
    assert(points_.size() == aahr_.size());

    // Construct a flex-point version of min and max.
    FlexPoint flex_min(order), flex_max(order);
    for (unsigned dim = 0; dim < order; dim++)
    {
      flex_min[dim] = aahr_.Min()[dim];
      flex_max[dim] = aahr_.Max()[dim];
    }

    // Expand the space.
    auto flex_point_vector = Expand(flex_min, flex_max);
    assert(flex_point_vector.size() == points_.size());

    // Convert into our internal space representation.
    for (auto& flex_point: flex_point_vector)
    {
      assert(flex_point.size() == order);
      Point<order> point;
      for (unsigned dim = 0; dim < order; dim++)
      {
        point[dim] = flex_point[dim];
      }
      assert(points_.find(point) != points_.end());
    }
#endif
  }

  void Print() const
  {
    for (auto point = points_.begin(); point != points_.end(); point++)
    {
      std::cout << "< ";
      for (std::uint32_t i = 0; i < order; i++)
      {
        std::cout << (*point)[i] << " ";
      }
      std::cout << "> ";
    }
  }
};
