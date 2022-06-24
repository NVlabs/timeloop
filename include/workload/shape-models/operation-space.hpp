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

#include "workload/workload.hpp"
#include "data-space.hpp"
#include "workload/util/per-data-space.hpp"

namespace problem
{

// ======================================== //
//              OperationPoint              //
// ======================================== //

class OperationPoint : public Point
{
 public:
  OperationPoint() :
      Point(GetShape()->NumFlattenedDimensions) // note! flattened, not factorized.
  {
  }
};

std::ostream& operator << (std::ostream& out, const OperationPoint& p);

// ======================================== //
//              OperationSpace              //
// ======================================== //

class OperationSpace
{
 private:
  const Workload* workload_;

  std::vector<DataSpace> data_spaces_;

 private:
  void FactorizeCarveMultiply(const Workload* wc,
                              const OperationPoint& flattened_low,
                              const OperationPoint& flattened_high,
                              std::vector<std::pair<Point, Point>>& carved_aahrs);

  Point Factorize(const Workload* wc, const OperationPoint& flattened);

  void FactorizeGrouped(const Workload* wc,
                        const OperationPoint& flattened_low,
                        const OperationPoint& flattened_high,
                        
                        std::vector<Point>& factor_groups_low,
                        std::vector<Point>& factor_groups_high,
                        std::vector<Point>& factor_groups_bounds);

  Point Project(Shape::DataSpaceID d, const Workload* wc,
                const Point& factorized_point);
  
  void ProjectLowHigh(Shape::DataSpaceID d,
                      const Workload* wc,
                      const Point& factorized_low,
                      const Point& factorized_high,
                      Point& data_space_low,
                      Point& data_space_high);
  
 public:
  
  // API always uses flattened "Operation" points and spaces.
  // Factorized points and spaces are used internally.

  OperationSpace() = delete;
  OperationSpace(const Workload* wc);
  OperationSpace(const Workload* wc,
                 const OperationPoint& flattened_low,
                 const OperationPoint& flattened_high);

  void Reset();

  // OperationSpace& operator+=(const OperationSpace& s);
  OperationSpace& operator+=(const OperationPoint& p);
  // OperationSpace& ExtrudeAdd(const OperationSpace& s);
  OperationSpace operator-(const OperationSpace& p);

  void SaveAndSubtract(OperationSpace& prev, problem::PerDataSpace<bool> no_temporal_reuse);
  void SaveAndSubtractIfSameStride(OperationSpace& prev, problem::PerDataSpace<Point>& prev_translation, problem::PerDataSpace<bool> no_temporal_reuse);

  DataSpace& GetDataSpace(Shape::DataSpaceID pv);
  PerDataSpace<std::size_t> GetSizes() const;
  std::size_t GetSize(const int t) const;
  bool IsEmpty(const int t) const;
  bool CheckEquality(const OperationSpace& rhs, const int t) const;
  void PrintSizes();
  void Print(std::ostream& out = std::cerr) const;
  void Print(Shape::DataSpaceID pv, std::ostream& out = std::cerr) const;
};

std::ostream& operator << (std::ostream& out, const OperationSpace& os);

} // namespace problem
