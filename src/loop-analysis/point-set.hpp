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

#include <vector>
#include <cassert>
#include <iostream>

// We should have asserts turned on in this code.
// They aren't very costly and we aren't fully sure if we are doing
// the right thing in the code, in particular in new PointSet implementation.
// Will help us catch some serious bugs for 10% extra runtime.
#define ASSERT(args...) assert(args)
//#define ASSERT(args...)

#define POINT_SET_GENERIC_SLOW 1
#define POINT_SET_GENERIC_FAST 2
#define POINT_SET_4D           3
#define POINT_SET_AAHR         4

//#define POINT_SET_IMPL POINT_SET_GENERIC_SLOW
#define POINT_SET_IMPL POINT_SET_AAHR

typedef std::int32_t Coordinate;

class Point
{
 protected:
  std::uint32_t order_;
  std::vector<Coordinate> coordinates_;

 public:
  Point() = delete;

  Point(const Point& p) :
      order_(p.order_),
      coordinates_(p.coordinates_)
  {
  }

  Point(std::uint32_t order) :
      order_(order)
  {
    coordinates_.resize(order_);
    Reset();
  }
  
  void Reset()
  {
    std::fill(coordinates_.begin(), coordinates_.end(), 0);
  }

  std::uint32_t Order() const { return order_; }

  Coordinate& operator[] (std::uint32_t i)
  {
    return coordinates_[i];
  }

  const Coordinate& operator[] (std::uint32_t i) const
  {
    return coordinates_[i];
  }

  void IncrementAllDimensions(Coordinate m = 1)
  {
    for (auto& c : coordinates_)
      c += m;
  }

  std::ostream& Print(std::ostream& out = std::cout) const
  {
    out << "[" << order_ << "]: ";
    for (auto& c : coordinates_)
      out << c << " ";
    return out;
  }
};

#include "point-set-aahr.hpp"
//#include "point-set-generic-slow.hpp"
//#include "point-set-4d.hpp"
//#include "point-set-generic-fast.hpp"

#if POINT_SET_IMPL == POINT_SET_AAHR
typedef AxisAlignedHyperRectangle PointSet;
#elif POINT_SET_IMPL == POINT_SET_GENERIC_SLOW
#error fix API error with PointSetGenericSlow
// typedef PointSetGenericSlow PointSet;
#elif POINT_SET_IMPL == POINT_SET_4D
#error fix API error with PointSet4D
#elif POINT_SET_IMPL == POINT_SET_GENERIC_FAST
#error fix API error with PointSetGenericFast
#else
#error illegal point set implementation
#endif
