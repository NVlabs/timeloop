/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <sstream>

#include "problem.hpp"
#include "printers.hpp"
#include "data.hpp"


class ProblemShape_CONV_1D: public ProblemShape
{
 public:
  ProblemShape_CONV_1D() {}
  ProblemShape_CONV_1D(isl_ctx* context,
          std::string compute_space_name, std::string input_buf, std::string output_buf,
          bool bounds_set=false, unsigned P=0, unsigned R=0) :
      ProblemShape(context)
  {
    // -----------------------------------
    // Problem shape: CONV_1D
    //    for p = [0:P)
    //      for r = [0:R)
    //        O[p] += W[r] * I[p+r]
    // -----------------------------------

    // -- Iteration space.
    if (!bounds_set)
    {
      iteration_space_ =
        isl_set_read_from_str(context_, "[P,R] -> { [p,r] :  0 <= p < P and 0 <= r < R }");
    }
    else
    {
      // FIXME: find a cleaner way to apply these bounds.
      char buf[256];
      sprintf(buf, "{ [p,r] : 0 <= p < %u and 0 <= r < %u }", P, R);
      iteration_space_ = isl_set_read_from_str(context_, buf);
    }
    iteration_space_subscripts_ = { "p", "r" };

    // -- Compute space.
    ComputeSpace Sum;
    Sum.name = compute_space_name;
    Sum.num_ranks = 2;
    Sum.subscripts = iteration_space_subscripts_;
    Sum.transform_txt = std::string("") +
      "    auto a = operands.at(0);\n" +
      "    auto b = operands.at(1);\n" +
      "    auto x = a + b;\n" +
      "    results.push_back(x);\n";

    compute_space_ = Sum;


    data_spaces_.push_back(input_buf);
    is_read.insert(input_buf);
    data_spaces_.push_back(output_buf);
    is_write.insert(output_buf);
    is_read.insert(output_buf);
  }
};

void construct_problem(isl_ctx* context_,
        std::map<std::string, DataPtr> & data_map,
        std::map<std::string, ProblemPtr> & problem, int P, int R);

void construct_problem_fusion(isl_ctx* context_,
        std::map<std::string, DataPtr> & data_map,
        std::map<std::string, ProblemPtr> & problem, int P, int R);
