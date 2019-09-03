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

#error fix API incompatibility before using this point set implementation 

// ---------------------------------------------
//          4D Point Set implementation
// ---------------------------------------------

class PointSet4D
{
 private:
  std::set<std::uint64_t> points_;

  std::uint64_t Flatten_(const Point<4>& point) const
  {
    return
      (std::uint64_t(point[3]) << 48) |
      (std::uint64_t(point[2]) << 32) |
      (std::uint64_t(point[1]) << 16) |
      (std::uint64_t(point[0]));
  }

 public:
  PointSet4D() : points_() {}

  std::size_t size() const { return points_.size(); }

  bool empty() const { return points_.empty(); }

  void Reset() { points_.clear(); }

  PointSet4D operator+(const Point<4>& p) const
  {
    PointSet4D r(*this);
    r.points_.insert(Flatten_(p));
    return r;
  }

  PointSet4D operator+(const PointSet4D& s) const
  {
    PointSet4D r(*this);
    r.points_.insert(s.points_.begin(), s.points_.end());
    return r;
  }

  PointSet4D& operator+=(const Point<4>& p)
  {
    points_.insert(Flatten_(p));
    return *this;
  }

  PointSet4D& operator+=(const PointSet4D& s)
  {
    points_.insert(s.points_.begin(), s.points_.end());
    return *this;
  }

  PointSet4D operator-(const PointSet4D& s) const
  {
    PointSet4D r(*this);
    for (auto i = s.points_.begin(); i != s.points_.end(); i++)
    {
      r.points_.erase(*i);  // Note: MUST erase by value.
    }
    return r;
  }

  PointSet4D& operator-=(const PointSet4D& s)
  {
    for (auto i = s.points_.begin(); i != s.points_.end(); i++)
    {
      points_.erase(*i);  // Note: MUST erase by value.
    }
    return *this;
  }

  bool operator==(const PointSet4D& rhs) const
  {
    PointSet4D lhs = *(this);
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

  void Print() const
  {
    for (auto& point : points_)
    {
      std::cout << "< ";
      std::cout << ((point) & 0xFFFF) << " ";
      std::cout << ((point >> 16) & 0xFFFF) << " ";
      std::cout << ((point >> 32) & 0xFFFF) << " ";
      std::cout << ((point >> 48) & 0xFFFF) << " ";
      std::cout << "> ";
    }
  }
};
