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

#include <cstdint>
#include <iostream>
#include <vector>

typedef std::int32_t Coordinate;

class Point
{
 protected:
  std::uint32_t order_;
  std::vector<Coordinate> coordinates_;

 public:
  // We really wanted to delete this constructor, but that would mean we can't
  // use DynamicArray<Point> (and consequently PerDataSpace<Point>).
  Point();
  Point(const Point& p);
  Point(std::uint32_t order);
  Point(std::vector<Coordinate> coordinates);
  
  // Copy-and-swap idiom.
  Point& operator = (Point other);
  friend void swap(Point& first, Point& second);

  bool operator == (const Point& other);

  Point DiscardTopRank() const;
  void AddTopRank(Coordinate x);

  void Reset();

  std::uint32_t Order() const;
  std::vector<Coordinate> GetCoordinates() const;

  Coordinate& operator[] (std::uint32_t i);
  const Coordinate& operator[] (std::uint32_t i) const;

  void IncrementAllDimensions(Coordinate m = 1);

  // Translation operator.
  Point operator + (Point& other);

  void Scale(unsigned factor);

  std::ostream& Print(std::ostream& out = std::cout) const;
  friend std::ostream& operator << (std::ostream& out, const Point& p);
};
