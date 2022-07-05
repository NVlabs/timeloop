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
#include <iostream>

namespace model
{

// ----- Macro to add Stat Accessors -----

#define STAT_ACCESSOR_HEADER_QUALIFIED(Type, Class, FuncName)                         \
Type Class::FuncName(problem::Shape::DataSpaceID pv) const

#define STAT_ACCESSOR_HEADER(Type, FuncName)                                          \
Type FuncName(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const

#define STAT_ACCESSOR_BODY(Type, FuncName, Expression)                                \
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

#define STAT_ACCESSOR(Type, Class, FuncName, Expression)                              \
  STAT_ACCESSOR_HEADER_QUALIFIED(Type, Class, FuncName)                               \
  STAT_ACCESSOR_BODY(Type, FuncName, Expression)

#define STAT_ACCESSOR_INLINE(Type, FuncName, Expression)                              \
  STAT_ACCESSOR_HEADER(Type, FuncName)                                                \
  STAT_ACCESSOR_BODY(Type, FuncName, Expression)

extern bool enableScientificStatOutput;
extern bool enableDefaultFloatStatOutput;


#define PRINTFLOAT_PRECISION std::setprecision(3)
#define LOG_FLOAT_PRECISION std::setprecision(2)

#define OUT_FLOAT_FORMAT (                                                          \
  model::enableScientificStatOutput ? std::scientific :                             \
  model::enableDefaultFloatStatOutput ? std::defaultfloat : std::fixed              \
)

} // namespace model
