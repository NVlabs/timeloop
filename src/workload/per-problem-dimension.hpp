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

#include "util/dynamic-array.hpp"

namespace problem
{

// Think of this as std::array<T, NumFlattenedDimensions>, except that the goal is
// to support dynamic values of NumFlattenedDimensions determined by reading user input.
template<class T>
class PerFlattenedDimension : public DynamicArray<T>
{
 public:
  PerFlattenedDimension() :
      DynamicArray<T>(GetShape()->NumFlattenedDimensions)
  {
  }

  PerFlattenedDimension(std::initializer_list<T> l) :
    DynamicArray<T>(l)
  {
    assert(this->size() == GetShape()->NumFlattenedDimensions);
  }

  friend std::ostream& operator << (std::ostream& out, const PerFlattenedDimension<T>& x)
  {
    for (unsigned i = 0; i < x.size(); i++)
    {
      out << GetShape()->FlattenedDimensionIDToName.at(i) << ": " << x[i] << std::endl;
    }
    return out;
  }

  // Serialization
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      ar << boost::serialization::make_nvp(
        "PerFlattenedDimension",
        boost::serialization::make_array(this->begin(), this->size()));
    }
  }
};

// template<class T>
// std::ostream& operator<<(std::ostream& out, const PerFlattenedDimension<T>& x);

}
