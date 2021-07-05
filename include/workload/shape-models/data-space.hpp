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

#include "loop-analysis/point-set.hpp"

namespace problem
{

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

} // namespace problem
