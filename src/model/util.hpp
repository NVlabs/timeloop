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

namespace model
{

// DataSpaceID sharing.
enum class DataSpaceIDSharing
{
  Partitioned,
  Shared
};

// A special-purpose class with a std::map-like interface used to hold
// *either* a collection of values of type T, one for each data space,
// *or* a single value of type T accessed with the key DataSpaceID::Num.
template<class T>
class PerDataSpaceOrShared
{
 private:
  problem::PerDataSpace<T> per_data_space;
  T shared;
  bool is_per_data_space = false;
  bool is_shared = false;

 public:
  PerDataSpaceOrShared()
  {
  }

  void SetPerDataSpace() {
    assert(!is_shared);
    is_per_data_space = true;
    // Construct a separate T value for each element.
    for (T& val : per_data_space)
    {
      val = T();
    }
  }

  void SetShared(T val=T()) {
    assert(!is_per_data_space);
    is_shared = true;
    shared = val;
  }

  T & operator [] (problem::Shape::DataSpaceID pv)
  {
    if (pv == problem::GetShape()->NumDataSpaces)
    {
      assert(is_shared);
      return shared;
    }
    else
    {
      assert(pv < problem::GetShape()->NumDataSpaces);
      assert(is_per_data_space);
      return per_data_space[pv];
    }
  }

  T & at(problem::Shape::DataSpaceID pv)
  {
    if (pv == problem::GetShape()->NumDataSpaces)
    {
      assert(is_shared);
      return shared;
    }
    else
    {
      assert(pv < problem::GetShape()->NumDataSpaces);
      assert(is_per_data_space);
      return per_data_space[pv];
    }
  }

  const T & at(problem::Shape::DataSpaceID pv) const
  {
    if (pv == problem::GetShape()->NumDataSpaces)
    {
      assert(is_shared);
      return shared;
    }
    else
    {
      assert(pv < problem::GetShape()->NumDataSpaces);
      assert(is_per_data_space);
      return per_data_space[pv];
    }
  }

  T Max() const
  {
    if (is_shared)
    {
      return shared;
    }
    else
    {
      assert(is_per_data_space);
      return per_data_space.Max();
    }
  }

  friend std::ostream& operator << (std::ostream& out, const PerDataSpaceOrShared<T>& x)
  {
    if (x.is_per_data_space)
    {
      out << "PerDataSpace:" << std::endl;
      out << x.per_data_space;
    }
    else
    {
      assert(x.is_shared);
      out << "Shared: " << x.shared << std::endl;
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
      if (is_per_data_space)
      {
        ar& BOOST_SERIALIZATION_NVP(per_data_space);
      }
      else
      {
        assert(is_shared);
        ar& BOOST_SERIALIZATION_NVP(shared);
      }
    }
  }

};

// template<class T>
// std::ostream& operator<<(std::ostream& out, const PerDataSpaceOrShared<T>& px);

// ----- Macro to add Spec Accessors -----
#define ADD_ACCESSORS(FuncName, MemberName, Type)                          \
    Attribute<Type> & FuncName(problem::Shape::DataSpaceID pv)             \
    {                                                                      \
      return (sharing_type == DataSpaceIDSharing::Partitioned)             \
             ? MemberName[pv]                                              \
             : MemberName[problem::GetShape()->NumDataSpaces];             \
    }                                                                      \
                                                                           \
    const Attribute<Type> & FuncName(problem::Shape::DataSpaceID pv) const \
    {                                                                      \
      return (sharing_type == DataSpaceIDSharing::Partitioned)             \
             ? MemberName.at(pv)                                           \
             : MemberName.at(problem::GetShape()->NumDataSpaces);          \
    }                                                                      \
                                                                           \
    Attribute<Type> & FuncName()                                           \
    {                                                                      \
      assert(sharing_type == DataSpaceIDSharing::Shared);                  \
      return MemberName[problem::GetShape()->NumDataSpaces];               \
    }                                                                      \
                                                                           \
    const Attribute<Type> & FuncName() const                               \
    {                                                                      \
      assert(sharing_type == DataSpaceIDSharing::Shared);                  \
      return MemberName.at(problem::GetShape()->NumDataSpaces);            \
    }                                                               
// ----- End Macro -----

// ----- Macro to add Stat Accessors -----
#define STAT_ACCESSOR(Type, FuncName, Expression)                                     \
Type BufferLevel::FuncName(problem::Shape::DataSpaceID pv) const                      \
{                                                                                     \
  if (pv != problem::GetShape()->NumDataSpaces)                                       \
  {                                                                                   \
    return Expression;                                                                \
  }                                                                                   \
  else                                                                                \
  {                                                                                   \
    Type stat = 0;                                                                    \
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++) \
    {                                                                                 \
      stat += FuncName(problem::Shape::DataSpaceID(pvi));                             \
    }                                                                                 \
    return stat;                                                                      \
  }                                                                                   \
}



} // namespace model
