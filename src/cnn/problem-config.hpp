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

void BuildProblemShape();

typedef unsigned DataSpaceID;

extern unsigned NumDataSpaces;
extern std::map<std::string, DataSpaceID> DataSpaceNameToID;
extern std::map<DataSpaceID, std::string> DataSpaceIDToName;

// std::ostream& operator<<(std::ostream& out, const DataSpaceID& d);

extern std::function<bool(const DataSpaceID d)> IsReadWriteDataSpace;

} // namespace problem

#include "per-data-space.hpp"

namespace problem
{

typedef unsigned Dimension;

extern unsigned NumDimensions;
extern std::map<Dimension, std::string> DimensionName;
extern std::map<char, Dimension> DimensionID;

// std::ostream& operator << (std::ostream& out, const Dimension& dim);

} // namespace problem

#include "per-problem-dimension.hpp"

namespace problem
{

class OperationPoint : public Point
{
 public:
  OperationPoint() :
      Point(int(NumDimensions))
  {
  }
};

typedef std::map<problem::Dimension, int> Bounds;
typedef std::map<problem::DataSpaceID, double> Densities;

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
  
  double getDensity(problem::DataSpaceID pv) const
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
  
  void setDensities(const std::map<problem::DataSpaceID, double> &densities)
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

// ======================================== //
//                 DataSpace                //
// ======================================== //

typedef PointSet DataSpace;

// class DataSpace : public PointSet
// {
//  private:
//   std::string name_;

//  public:
//   DataSpace() = delete;

//   DataSpace(std::string name, std::uint32_t order) :
//       PointSet(order),
//       name_(name)
//   { }
  
//   DataSpace(std::string name, std::uint32_t order, Point base, Point bound) :
//       PointSet(order, base, bound),
//       name_(name)
//   { }

//   DataSpace(const PointSet& p) :
//       PointSet(p),
//       name_("__UNNAMED__")
//   { }

//   std::string Name() const
//   {
//     return name_;
//   }

//   DataSpace operator - (const DataSpace& d)
//   {
//     PointSet delta = PointSet::operator - (d);
//     DataSpace retval(delta);
//     retval.name_ = name_;
//     return retval;
//   }

//   void Print() const
//   {
//     std::cout << Name() << "[" << size() << "]: ";
//     PointSet::Print();
//     std::cout << std::endl;
//   }
// };

// ======================================== //
//              OperationSpace              //
// ======================================== //

class OperationSpace
{
 private:
  WorkloadConfig* workload_config_;

  std::vector<DataSpace> data_spaces_;
  
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
  void Print(DataSpaceID pv) const;
};

PerDataSpace<std::size_t> GetMaxWorkingSetSizes(
    PerProblemDimension<int> dimension_sizes);

} // namespace problem
