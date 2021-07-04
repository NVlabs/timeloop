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

#include <iostream>

#include "loop-analysis/point.hpp"

// ---------------------------------------------
//                   Gradient
// ---------------------------------------------

struct Gradient
{
  std::uint32_t order;
  std::uint32_t dimension;
  std::int32_t value;

  Gradient() = delete;
  Gradient(std::uint32_t _order);

  void Reset();
  
  std::int32_t Sign() const;
  
  void Print(std::ostream& out = std::cout) const;
};

// ---------------------------------------------
//        AAHR Point Set implementation
// ---------------------------------------------

class AxisAlignedHyperRectangle
{
 protected:
  
  std::uint32_t order_;
  Point min_, max_; // min inclusive, max: exclusive
  Gradient gradient_;

 public:

  AxisAlignedHyperRectangle() = delete;  
  AxisAlignedHyperRectangle(std::uint32_t order);
  AxisAlignedHyperRectangle(std::uint32_t order, const Point unit);
  AxisAlignedHyperRectangle(std::uint32_t order, const Point min, const Point max);
  AxisAlignedHyperRectangle(const AxisAlignedHyperRectangle& a);

  // Copy-and-swap idiom.
  AxisAlignedHyperRectangle& operator = (AxisAlignedHyperRectangle other);
  friend void swap(AxisAlignedHyperRectangle& first, AxisAlignedHyperRectangle& second);

  Point Min() const;
  Point Max() const;

  std::size_t size() const;
  bool empty() const;

  void Reset();

  void Add(const Point& p, bool extrude_if_discontiguous = false);
  void ExtrudeAdd(const AxisAlignedHyperRectangle& s);
  void Add(const AxisAlignedHyperRectangle& s, bool extrude_if_discontiguous = false);

  Gradient Subtract(const AxisAlignedHyperRectangle& s);

  AxisAlignedHyperRectangle& operator += (const Point& p);
  AxisAlignedHyperRectangle& operator += (const AxisAlignedHyperRectangle& s);

  AxisAlignedHyperRectangle operator - (const AxisAlignedHyperRectangle& s);

  bool operator == (const AxisAlignedHyperRectangle& s) const;

  Point GetTranslation(const AxisAlignedHyperRectangle& s) const;
  void Translate(const Point& p);

  void Print(std::ostream& out = std::cout) const;
};
