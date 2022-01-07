/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <isl/space.h>
#include <isl/set.h>
#include <barvinok/isl.h>
#include <unordered_map>
#include <mutex>

#include "loop-analysis/point.hpp"

class ISLPointSet
{
 protected:
  static std::mutex mutex;
  static std::unordered_map<pthread_t, isl_ctx*> contexts;
  static std::unordered_map<pthread_t, isl_printer*> consoles;

  std::uint32_t order_;
  isl_set* set_;

  // Convert point to isl_point.
  isl_point* ToISL(const Point p);

  // Get the thread-local ISL context.
  isl_ctx* Context();

 public:

  ISLPointSet() = delete;
  ISLPointSet(std::uint32_t order);
  ISLPointSet(std::uint32_t order, isl_set* set);
  ISLPointSet(std::uint32_t order, const Point unit);
  ISLPointSet(std::uint32_t order, const Point min, const Point max);
  ISLPointSet(const ISLPointSet& a);

  ~ISLPointSet();

  // Copy-and-swap idiom.
  ISLPointSet& operator = (ISLPointSet other);
  friend void swap(ISLPointSet& first, ISLPointSet& second);

  std::size_t size() const;
  bool empty() const;

  void Reset();

  ISLPointSet& operator += (const Point& p);
  ISLPointSet operator - (const ISLPointSet& s);
  bool operator == (const ISLPointSet& s) const;

  Point GetTranslation(const ISLPointSet& s) const;
  void Translate(const Point& p);

  void Print(std::ostream& out = std::cout) const;
};
