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

#include <list>
#include <vector>
#include <queue>

#include <boost/serialization/vector.hpp>

#include "workload/problem-shape.hpp"
#include "spacetime.hpp"

namespace loop
{

// -----------------------------------
// Descriptor for a single loop level.
// -----------------------------------
class Descriptor
{
 public:
  problem::Shape::DimensionID dimension;
  int start;
  int end;
  int stride;
  spacetime::Dimension spacetime_dimension;

  Descriptor();

  Descriptor(const problem::Shape::DimensionID _dimension, const int _start,
             const int _end, const int _stride,
             const spacetime::Dimension _spacetime_dimension);

  Descriptor(const problem::Shape::DimensionID _dimension,
             const int _end,
             const spacetime::Dimension _spacetime_dimension = spacetime::Dimension::Time);

  bool operator == (const Descriptor& d) const;
  
  void Print(std::ostream& out, bool long_form = true) const;

  void PrintWhoop(std::ostream& out) const;

  // Serialization
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0)
  {
    if(version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(dimension);
      ar& BOOST_SERIALIZATION_NVP(start);
      ar& BOOST_SERIALIZATION_NVP(end);
      ar& BOOST_SERIALIZATION_NVP(stride);
      ar& BOOST_SERIALIZATION_NVP(spacetime_dimension);
    }
  }
};

std::ostream& operator<<(std::ostream& out, const Descriptor& loop);

bool IsSpatial(spacetime::Dimension dim);
bool IsSpatialX(spacetime::Dimension dim);

} // namespace loop
