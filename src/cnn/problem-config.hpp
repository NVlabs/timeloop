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

#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <iomanip>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "loop-analysis/point-set.hpp"

namespace problem
{

enum class DataType : unsigned int
{
  Weight,
  Input,
  Output,
  Num
};

extern std::map<std::string, DataType> DataTypeID;
extern std::map<DataType, std::string> DataTypeName;

std::ostream& operator<<(std::ostream& out, const DataType& d);
bool IsReadWriteDataType(const DataType d);

} // namespace problem

#include "per-data-space.hpp"

namespace problem
{

enum class WeightDimension {
  R,
  S,
  C,
  K,
  Num
};
enum class InputDimension {
  W,
  H,
  C,
  N,
  Num
};
enum class OutputDimension {
  P,
  Q,
  K,
  N,
  Num
};
enum class Dimension {
  R,
  S,
  P,
  Q,
  C,
  K,
  N,
  Num
};

extern std::map<Dimension, std::string> DimensionName;
extern std::map<char, Dimension> DimensionID;

std::ostream& operator << (std::ostream& out, const Dimension& dim);

// Think of this as std::array<T, Dimension::Num>, except that the goal is
// to support dynamic values of Dimension::Num determined by reading user input.
template<class T>
class PerProblemDimension : public DynamicArray<T>
{
 public:
  PerProblemDimension() :
    DynamicArray<T>(unsigned(Dimension::Num))
  {
  }

  PerProblemDimension(std::initializer_list<T> l) :
    DynamicArray<T>(l)
  {
    assert(this->size() == unsigned(Dimension::Num));
  }

  friend std::ostream& operator << (std::ostream& out, const PerProblemDimension<T>& x)
  {
    for (unsigned i = 0; i < x.size(); i++)
    {
      out << Dimension(i) << ": " << x[i] << std::endl;
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
        "PerProblemDimension",
        boost::serialization::make_array(this->begin(), this->size()));
    }
  }
};

template<class T>
std::ostream& operator<<(std::ostream& out, const PerProblemDimension<T>& x);

class OperationPoint : public Point
{
 public:
  OperationPoint() :
      Point(int(Dimension::Num))
  {
  }
};

typedef std::map<problem::Dimension, int> Bounds;
typedef std::map<problem::DataType, double> Densities;

// ======================================== //
//              WorkloadConfig              //
// ======================================== //

class WorkloadConfig
{
  Bounds bounds_;
  Densities densities_;

  // Stride and dilation. FIXME: ugly.
  int Wstride, Hstride;
  int Wdilation, Hdilation;
  
 public:
  WorkloadConfig() {}

  int getBound(problem::Dimension dim) const
  {
    return bounds_.at(dim);
  }
  
  double getDensity(problem::DataType pv) const
  {
    return densities_.at(pv);
  }

  int getWstride() const { return Wstride; }
  void setWstride(const int s) { Wstride = s; }
  
  int getHstride() const { return Hstride; }
  void setHstride(const int s) { Hstride = s; }

  int getWdilation() const { return Wdilation; }
  void setWdilation(const int s) { Wdilation = s; }

  int getHdilation() const { return Hdilation; }
  void setHdilation(const int s) { Hdilation = s; }

  void setBounds(const std::map<problem::Dimension, int> &bounds)
  {
    bounds_ = bounds;
  }
  
  void setDensities(const std::map<problem::DataType, double> &densities)
  {
    densities_ = densities;
  }

 private:
  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(bounds_);
      ar& BOOST_SERIALIZATION_NVP(densities_);
    }
  }
};

//Point MakeWeightPoint(WorkloadConfig* wc, const OperationPoint& problem_point);
//Point MakeInputPoint(WorkloadConfig* wc, const OperationPoint& problem_point);
//Point MakeOutputPoint(WorkloadConfig* wc, const OperationPoint& problem_point);

// ======================================== //
//                 DataSpace                //
// ======================================== //
// class DataSpace
// {
//  private:
//   std::string name_;
//   PointSet point_set_;

//  public:
//   DataSpace() = delete;

//   DataSpace(std::string name, std::uint32_t order) :
//       name_(name),
//       point_set_(order)
//   { }
  
//   DataSpace(std::string name, std::uint32_t order, Point base, Point bound) :
//       name_(name),
//       point_set_(order, base, bound)
//   { }

//   std::string Name() const
//   {
//     return name_;
//   }

//   void Print() const
//   {
//     std::cout << Name() << "[" << size() << "]: ";
//     point_set_.Print();
//     std::cout << std::endl;
//   }

//   // Pass-throughs to PointSet class.
//   std::size_t size() const { return point_set_.size(); }
//   bool empty() const { return point_set_.empty(); }
//   void Reset() { point_set_.Reset(); }
//   DataSpace& operator += (const Point& p) { point_set_ += p; return *this; }
//   DataSpace& operator += (const DataSpace& d) { point_set_ += d.point_set_; return *this; }
//   DataSpace operator - (const DataSpace& d)
//   {
//     DataSpace retval(name_, 0);
//     retval.point_set_ = point_set_ - d.point_set_;
//     return retval;
//   }
//   bool operator == (const DataSpace& d) const { return (point_set_ == d.point_set_); }

// };

typedef PointSet DataSpace;

// ======================================== //
//              OperationSpace              //
// ======================================== //

class OperationSpace
{
 private:
  WorkloadConfig* workload_config_;

  std::vector<DataSpace> data_spaces_;
  std::vector<std::string> data_space_names_;
  
 public:
  OperationSpace();
  OperationSpace(WorkloadConfig* wc);
  OperationSpace(WorkloadConfig* wc, const OperationPoint& low, const OperationPoint& high);

  void Reset();
  OperationSpace& operator+=(const OperationSpace& s);
  OperationSpace& operator+=(const OperationPoint& p);
  OperationSpace operator-(const OperationSpace& p);
  PerDataSpace<std::size_t> GetSizes() const;
  std::size_t GetSize(const int t) const;
  bool IsEmpty(const int t) const;
  bool CheckEquality(const OperationSpace& rhs, const int t) const;
  void PrintSizes();
  void Print() const;
  void Print(DataType pv) const;
};

PerDataSpace<std::size_t> GetMaxWorkingSetSizes(
    PerProblemDimension<int> dimension_sizes);

} // namespace problem
