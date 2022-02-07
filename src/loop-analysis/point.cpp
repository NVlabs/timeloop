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

#include "loop-analysis/point.hpp"

// We really wanted to delete this constructor, but that would mean we can't
// use DynamicArray<Point> (and consequently PerDataSpace<Point>).
Point::Point() :
      order_(0)
{
}

Point::Point(const Point& p) :
    order_(p.order_),
    coordinates_(p.coordinates_)
{
}

Point::Point(std::uint32_t order) :
    order_(order)
{
  coordinates_.resize(order_);
  Reset();
}
  
// Copy-and-swap idiom.
Point& Point::operator = (Point other)
{
  swap(*this, other);
  return *this;
}

void swap(Point& first, Point& second)
{
  using std::swap;
  swap(first.order_, second.order_);
  swap(first.coordinates_, second.coordinates_);
}

bool Point::operator == (const Point& other)
{
  if (order_ != other.order_)
    return false;

  for (unsigned rank = 0; rank < order_; rank++)
  {
    if (coordinates_.at(rank) != other.coordinates_.at(rank))
      return false;
  }

  return true;
}

Point Point::DiscardTopRank() const
{
  Point p = *this;
  p.coordinates_.pop_back();
  p.order_--;
  return p;
}

void Point::AddTopRank(Coordinate x)
{
  order_++;
  coordinates_.push_back(x);
}

void Point::Reset()
{
  std::fill(coordinates_.begin(), coordinates_.end(), 0);
}

std::uint32_t Point::Order() const
{
  return order_;
}

Coordinate& Point::operator[] (std::uint32_t i)
{
  return coordinates_[i];
}

const Coordinate& Point::operator[] (std::uint32_t i) const
{
  return coordinates_[i];
}

void Point::IncrementAllDimensions(Coordinate m)
{
  for (auto& c : coordinates_)
    c += m;
}

// Translation operator.
Point Point::operator + (Point& other)
{
  Point retval(order_);
  for (unsigned i = 0; i < order_; i++)
    retval.coordinates_.at(i) = coordinates_.at(i) += other.coordinates_.at(i);
  return retval;
}

void Point::Scale(unsigned factor)
{
  for (auto& c : coordinates_)
    c *= factor;
}

std::ostream& Point::Print(std::ostream& out) const
{
  out << "[" << order_ << "]: ";
  for (auto& c : coordinates_)
    out << c << " ";
  return out;
}

std::ostream& operator << (std::ostream& out, const Point& p)
{
  out << "[" << p.order_ << "]: ";
  for (auto& c : p.coordinates_)
    out << c << " ";
  return out;
}
