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

#include "util/dynamic-array.hpp"

namespace problem
{

// Think of this as std::array<T, DataSpaceID::Num>, except that the goal is
// to support dynamic values of DataSpaceID::Num determined by reading user input.
template<class T>
class PerDataSpace : public DynamicArray<T>
{
 public:
  PerDataSpace() :
    DynamicArray<T>(NumDataSpaces)
  {
  }

  PerDataSpace(const T & val) :
    DynamicArray<T>(NumDataSpaces)
  {
    this->fill(val);
  }

  PerDataSpace(std::initializer_list<T> l) :
    DynamicArray<T>(l)
  {
    assert(this->size() == NumDataSpaces);
  }

  T & operator [] (unsigned pv)
  {
    assert(pv < NumDataSpaces);
    return DynamicArray<T>::at(pv);
  }
  const T & operator [] (unsigned pv) const
  {
    assert(pv < NumDataSpaces);
    return DynamicArray<T>::at(pv);
  }

  // T & operator [] (DataSpaceID pv)
  // {
  //   return (*this)[unsigned(pv)];
  // }
  // const T & operator [] (DataSpaceID pv) const
  // {
  //   return (*this)[unsigned(pv)];
  // }

  // T & at(DataSpaceID pv)
  // {
  //   return (*this)[pv];
  // }
  // const T & at(DataSpaceID pv) const
  // {
  //   return (*this)[pv];
  // }

  T & at(unsigned pv)
  {
    return (*this)[pv];
  }
  const T & at(unsigned pv) const
  {
    return (*this)[pv];
  }

  void clear()
  {
    DynamicArray<T>::clear();
  }

  T Max() const
  {
    return *std::max_element(this->begin(), this->end());
  }

  friend std::ostream& operator << (std::ostream& out, const PerDataSpace<T>& x)
  {
    for (unsigned pvi = 0; pvi < NumDataSpaces; pvi++)
    {
      out << std::setw(10) << DataSpaceIDToName[pvi] << ": " << x[pvi] << std::endl;
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
      ar& boost::serialization::make_nvp(
        "PerDataSpace",
        boost::serialization::make_array(this->begin(), this->size()));
    }
  }
};

template<class T>
std::ostream& operator<<(std::ostream& out, const PerDataSpace<T>& px)
{
  for (unsigned i = 0; i < NumDataSpaces; i++)
  {
    out << px[i] << " ";
  }
  return out;
}


} // namespace problem
