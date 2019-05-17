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

#include <iostream>

#include "point-set.hpp"

// ---------------------------------------------
//        AAHR Point Set implementation
// ---------------------------------------------

struct Gradient
{
  std::uint32_t order;
  std::uint32_t dimension;
  Magnitude magnitude;

  Gradient() = delete;

  Gradient(std::uint32_t _order) :
      order(_order)
  {
    Reset();
  }

  void Reset()
  {
    dimension = 0;
    magnitude = 0;
  }
  
  Magnitude Sign() const
  {
    if (magnitude < 0)
    {
      return -1;
    }
    else if (magnitude == 0)
    {
      return 0;
    }
    else
    {
      return 1;
    }
  }
  
  void Print() const
  {
    std::cout << "< ";
    for (unsigned i = 0; i < order; i++)
    {
      if (i == dimension)
        std::cout << magnitude << " ";
      else
        std::cout << "0 ";
    }
    std::cout << ">";
  }
};

class AxisAlignedHyperRectangle
{
 private:
  std::uint32_t order_;
  Point min_, max_; // min inclusive, max: exclusive
  Gradient gradient_;

 public:

  AxisAlignedHyperRectangle() = delete;
  
  AxisAlignedHyperRectangle(std::uint32_t order) :
      order_(order),
      min_(order),
      max_(order),
      gradient_(order)
  {
    Reset();
  }

  AxisAlignedHyperRectangle(std::uint32_t order, const Point unit) :
      AxisAlignedHyperRectangle(order)
  {
    ASSERT(order_ == unit.Order());
    min_ = unit;
    for (unsigned dim = 0; dim < order_; dim++)
    {
      max_[dim] = min_[dim] + 1;
    }
  }

  AxisAlignedHyperRectangle(std::uint32_t order, const Point min, const Point max) :
      AxisAlignedHyperRectangle(order)
  {
    min_ = min;
    max_ = max;
  }

  AxisAlignedHyperRectangle(const AxisAlignedHyperRectangle& a) :
      order_(a.order_),
      min_(a.min_),
      max_(a.max_),
      gradient_(a.gradient_)
  {
  }

  Point Min() const
  {
    return min_;
  }

  Point Max() const
  {
    return max_;
  }

  std::size_t size() const
  {
    std::size_t size = max_[0] - min_[0];
    for (unsigned i = 1; i < order_; i++)
    {
      size *= (max_[i] - min_[i]);
    }
    return size;
  }

  bool empty() const
  {
    return (size() == 0);
  }

  void Reset()
  {
    min_.Reset();
    max_.Reset();
    gradient_.Reset();
  }

  void Add(const Point& p)
  {
    Add(AxisAlignedHyperRectangle(order_, p));
  }  

  void Add(const AxisAlignedHyperRectangle& s)
  {
    ASSERT(order_ == s.order_);
    
    // Special cases.
    if (size() == 0)
    {
      *this = s;
      return;
    }

    if (s.size() == 0)
    {
      return;
    }

    if (*this == s)
    {
      return;
    }

    auto orig = *this;
    
    // Both AAHRs should have identical min_, max_ along all-but-one axes, and
    // must be contiguous along the but-one axis.
    bool found = false;
    for (unsigned dim = 0; dim < order_; dim++)
    {
      if (s.max_[dim] >= min_[dim] && max_[dim] >= s.min_[dim])
      {
        auto u = *this;
        bool need_update = false;
        
        if (s.min_[dim] < min_[dim])
        {
          u.min_[dim] = s.min_[dim];
          need_update = true;
        }
        if (s.max_[dim] > max_[dim])
        {
          u.max_[dim] = s.max_[dim];
          need_update = true;
        }
        
        if (need_update)
        {
          if (found)
          {
            std::cout << "AAHR Add error: non-HR shape\n";
            orig.Print(); std::cout << std::endl;
            s.Print(); std::cout << std::endl;          
            assert(false);
          }
          else
          {
            *this = u;
            found = true;
          }
        }
      }
      else
      {
#undef DISALLOW_DISCONTIGUOUS_ADD
#ifdef DISALLOW_DISCONTIGUOUS_ADD
        std::cout << "AAHR Add error: discontiguous volumes\n";
        orig.Print(); std::cout << std::endl;
        s.Print(); std::cout << std::endl;          
        assert(false);
#else
        auto u = *this;
        bool need_update = false;
        
        if (s.max_[dim] < min_[dim])
        {
          u.min_[dim] = s.min_[dim];
          need_update = true;
        }
        else
        {
          u.max_[dim] = s.max_[dim];
          need_update = true;
        }
        
        if (need_update)
        {
          if (found)
          {
            std::cout << "AAHR Add error: non-HR shape\n";
            orig.Print(); std::cout << std::endl;
            s.Print(); std::cout << std::endl;          
            assert(false);
          }
          else
          {
            *this = u;
            found = true;
          }
        }
#endif
      }
    }
  }

  Gradient Subtract(const AxisAlignedHyperRectangle& s)
  {
    ASSERT(order_ == s.order_);
    
    // Special cases.
    if (size() == 0 || s.size() == 0)
    {
      return Gradient(order_);
    }

    if (*this == s)
    {
      Reset();
      return Gradient(order_);
    }

    for (unsigned dim = 0; dim < order_; dim++)
    {
      if (s.max_[dim] <= min_[dim] || s.min_[dim] >= max_[dim])
      {
        // No overlap along even a single dimension means there's
        // no intersection at all. Skip this function.
        return Gradient(order_);
      }
    }
 
    auto updated = *this;
    Gradient gradient(order_);
    
    // General case: Both AAHRs should have identical min_, max_ along
    // all-but-one axes, and be contiguous or overlapping along the but-one
    // axis. If this isn't true, then torpedo everything, keep the source
    // as the result, and set gradient to 0.
    bool found = false;
    for (unsigned dim = 0; dim < order_; dim++)
    {
      if (min_[dim] != s.min_[dim] || max_[dim] != s.max_[dim])
      {
        if (found)
        {
          // Torpedo everything, set delta to source, gradient to 0.
          // WARNING: this simply discards potential non-AAHR shapes,
          // which is something we do want to do occasionally. However,
          // there may be bugs causing non-AAHR shapes, which will be
          // masked by this step.
          return Gradient(order_);
        }
        
        found = true;

        if (s.min_[dim] <= min_[dim])
        {
          if (s.max_[dim] <= max_[dim])
          {
            gradient.dimension = dim;
            gradient.magnitude = s.max_[dim] - min_[dim];
            updated.min_[dim] = s.max_[dim];
          }
          else
          {
            gradient.Reset();
            updated.max_[dim] = min_[dim];
          }
        }
        else if (s.min_[dim] > min_[dim])
        {
          if (s.max_[dim] < max_[dim])
          {
            assert(false);

            // The accuracy of the following comment is questionable. Fractures
            // don't appear to be happening for the dataflows we've been looking
            // at so far. We are enabling the assertion.
            
            // Subtraction is causing a fracture. This can happen during
            // macro tile changes with sliding windows. Discard the operand,
            // and return a zero gradient.
            return Gradient(order_);
          }
          else
          {
            gradient.dimension = dim;
            gradient.magnitude = s.min_[dim] - max_[dim];
            updated.max_[dim] = s.min_[dim];
          }
        }
        else
        {
          assert(false);
        }

        // If we just shrunk the AAHR down to NULL, reset it into canonical form
        // and skip the remainder of this function.
        if (updated.min_[dim] == updated.max_[dim])
        {
          // Discard updated, we're going to Reset ourselves anyway.
          Reset();
          return Gradient(order_);
        }
      }
    }

    assert(found);

    *this = updated;
    
    return gradient;
  }

  AxisAlignedHyperRectangle& operator += (const Point& p)
  {
    Add(p);
    return *this;
  }

  AxisAlignedHyperRectangle& operator += (const AxisAlignedHyperRectangle& s)
  {
    Add(s);
    return *this;
  }

  AxisAlignedHyperRectangle operator - (const AxisAlignedHyperRectangle& s)
  {
    // Calculate the delta.
    AxisAlignedHyperRectangle delta(*this);

#define RESET_ON_GRADIENT_CHANGE
#ifdef RESET_ON_GRADIENT_CHANGE
    auto g = delta.Subtract(s);
    
    // Now check if the newly-calculated gradient is different from the gradient
    // of the operand. UGH, this is ugly. This code shouldn't be in the math
    // library, it should be outside.
    if (s.gradient_.magnitude == 0)
    {
      // Gradient was zero. Use newly-computed gradient.
      gradient_ = g;
    }
    else if (g.magnitude == 0 && delta.size() == 0)
    {
      // Note the delta size check. We need that because the gradient can
      // be zero in two cases:
      // - The set difference really yielded a 0 (the case we're capturing here).
      // - There was no intersection and therefore gradient was invalid (we'll
      //   default to the final else.
      // FIXME: UGH UGH UGH.
      gradient_ = g;
    }
    else if (s.gradient_.dimension == g.dimension &&
             s.gradient_.Sign() == g.Sign())
    {
      // New gradient is in the same direction as current gradient.
      gradient_ = g;
    }
    else
    {
      // New gradient is either in a different dimension, or a different
      // direction (+/-) in the same dimension. Discard my residual state,
      // and re-initialize gradient.
      delta = *this;
      gradient_ = Gradient(order_);
    }

#else

    delta.Subtract(s);
    
#endif
    
    // The delta itself doesn't carry a gradient.
    delta.gradient_ = Gradient(order_);

    return delta;
  }

  bool operator == (const AxisAlignedHyperRectangle& s) const
  {
    for (unsigned dim = 0; dim < order_; dim++)
    {
      if (min_[dim] != s.min_[dim] || max_[dim] != s.max_[dim])
      {
        return false;
      }
    }
    return true;
  }

  void Print() const
  {
    std::cout << "min = < "; 
    for (unsigned dim = 0; dim < order_; dim++)
    {
      std::cout << min_[dim] << " ";
    }
    std::cout << "> max = < "; 
    for (unsigned dim = 0; dim < order_; dim++)
    {
      std::cout << max_[dim] << " ";
    }
    std::cout << "> gradient = ";
    gradient_.Print();
  }
  
};
