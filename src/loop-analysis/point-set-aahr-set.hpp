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

#include "point-set-aahr.hpp"
#include "aahr-carve.hpp"

class AAHRSet
{
 protected:

  std::uint32_t order_;

  // All AAHRs in the set are guaranteed to be disjoint.
  // This property must be maintained at all times.
  std::vector<AxisAlignedHyperRectangle> aahrs_;

 public:

  AAHRSet() = delete;

  AAHRSet(std::uint32_t order) :
      order_(order)
  {
    // Create a single empty AAHR. **** FIXME **** remove.
    assert(aahrs_.size() == 0);
    aahrs_.push_back(AxisAlignedHyperRectangle(order));
  }

  AAHRSet(std::uint32_t order, const Point unit) :
      order_(order)
      //AAHRSet(order)
  {
    // Create a single AAHR.
    assert(aahrs_.size() == 0);
    aahrs_.push_back(AxisAlignedHyperRectangle(order, unit));
  }

  AAHRSet(std::uint32_t order, const Point min, const Point max) :
      order_(order)
      //AAHRSet(order)
  {
    // Create a single AAHR.
    assert(aahrs_.size() == 0);
    aahrs_.push_back(AxisAlignedHyperRectangle(order, min, max));
  }

  AAHRSet(const AAHRSet& a) :
      order_(a.order_),
      aahrs_(a.aahrs_)
  {
    assert(aahrs_.size() == 1);
  }

  // Copy-and-swap idiom.
  AAHRSet& operator = (AAHRSet other)
  {
    assert(other.aahrs_.size() == 1);
    swap(*this, other);
    return *this;
  }

  friend void swap(AAHRSet& first, AAHRSet& second)
  {
    using std::swap;
    swap(first.order_, second.order_);
    swap(first.aahrs_, second.aahrs_);
  }

  std::size_t size() const
  {
    size_t size = 0;
    for (auto& aahr: aahrs_)
    {
      size += aahr.size();
    }
    return size;
  }

  bool empty() const
  {
    for (auto& aahr: aahrs_)
    {
      if (!aahr.empty())
        return false;
    }
    return true;
  }

  void Reset()
  {
    aahrs_.clear();
    // Create a single empty AAHR. **** FIXME **** remove.
    assert(aahrs_.size() == 0);
    aahrs_.push_back(AxisAlignedHyperRectangle(order_));
  }

  AAHRSet& operator += (const Point& p)
  {
    // FIXME: this only works with a single AAHR.
    assert(aahrs_.size() == 1);
    for (auto& aahr: aahrs_)
    {
      aahr += p;
    }
    return *this;
  }

  AAHRSet operator - (const AAHRSet& s)
  {
    // FIXME: this only works with a single AAHR.
    assert(aahrs_.size() == 1);
    assert(s.aahrs_.size() == 1);

    AAHRSet retval(order_);
    retval.aahrs_.clear();

    for (auto& a: aahrs_)
    {
      for (auto& b: s.aahrs_)
      {
        retval.aahrs_.push_back(a-b);
        break;
      }
      break;
    }

    assert(retval.aahrs_.size() == 1);

    return retval;
  }

  bool operator == (const AAHRSet& s) const
  {
    // Do a superficial match: try to find an exact copy of each AAHR
    // in the other set. In reality the points may be re-distributed
    // in a different pattern in the other set, but we ignore this
    // at the moment.

    // Because the AAHRs in each set are guaranteed to be disjoint,
    // our match procedure is a little simpler.
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
        return false;
    }

    return true;
  }

  Point GetTranslation(const AAHRSet& s) const
  {
    // FIXME: placeholder implementation that just looks at the first element
    // in each set.
    assert(aahrs_.size() == 1);
    assert(s.aahrs_.size() == 1);

    return aahrs_.front().GetTranslation(s.aahrs_.front());
  }

  void Translate(const Point& p)
  {
    assert(aahrs_.size() == 1);
    // Translate each AAHR.
    for (auto& x: aahrs_)
    {
      x.Translate(p);
    }
  }

  void Print(std::ostream& out = std::cout) const
  {
    out << "{ ";
    for (auto& x: aahrs_)
    {
      x.Print(out); out << ", ";
    }
    out << "}";
  }
};

