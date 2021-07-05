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

#include "util/misc.hpp"
#include "point-set-aahr.hpp"

class MultiAAHR
{
 protected:

  std::uint32_t order_;

  // All AAHRs in the set are guaranteed to be disjoint.
  // This property must be maintained at all times.
  std::vector<AxisAlignedHyperRectangle> aahrs_;

 public:

  MultiAAHR() = delete;
  MultiAAHR(std::uint32_t order);
  MultiAAHR(std::uint32_t order, const Point unit);
  MultiAAHR(std::uint32_t order, const Point min, const Point max);
  MultiAAHR(std::uint32_t order, const std::vector<std::pair<Point, Point>> corner_sets);
  MultiAAHR(const MultiAAHR& a);

  // Copy-and-swap idiom.
  MultiAAHR& operator = (MultiAAHR other);
  friend void swap(MultiAAHR& first, MultiAAHR& second);

  std::size_t size() const;
  bool empty() const;

  void Reset();

  void Subtract(const MultiAAHR& other);
  MultiAAHR& operator += (const Point& p);
  MultiAAHR& operator += (const MultiAAHR& s);
  MultiAAHR operator - (const MultiAAHR& other);
  bool operator == (const MultiAAHR& s) const;

  Point GetTranslation(const MultiAAHR& s) const;
  void Translate(const Point& p);

  friend std::ostream& operator << (std::ostream& out, const MultiAAHR& m);
};

