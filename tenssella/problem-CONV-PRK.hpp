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

class ProblemShape_CONV_PRK : public ProblemShape
{
 public:
  ProblemShape_CONV_PRK(isl_ctx* context, bool bounds_set=false, unsigned K=0, unsigned P=0, unsigned R=0) :
      ProblemShape(context)
  {
    // -----------------------------------
    // Problem shape: CONV_PRK
    //  for k = [0:K)
    //    for p = [0:P)
    //      for r = [0:R)
    //        O[k][p] += W[k][r] * I[p+r]
    // -----------------------------------

    // -- Iteration space.
    if (!bounds_set)
    {
      iteration_space_ =
        isl_set_read_from_str(context_, "[K,P,R] -> { [k,p,r] : 0 <= k < K and 0 <= p < P and 0 <= r < R }");
    }
    else
    {
      // FIXME: find a cleaner way to apply these bounds.
      char buf[256];
      sprintf(buf, "{ [k,p,r] : 0 <= k < %u and 0 <= p < %u and 0 <= r < %u }", K, P, R);
      iteration_space_ = isl_set_read_from_str(context_, buf);
    }
    iteration_space_subscripts_ = { "k", "p", "r" };

    // -- Compute space.
    ComputeSpace Multiply;
    Multiply.name = "Multiply";
    Multiply.num_ranks = 3;
    Multiply.subscripts = iteration_space_subscripts_;
    Multiply.transform_txt = std::string("") +
      "    auto a = operands.at(0);\n" +
      "    auto b = operands.at(1);\n" +
      "    auto z = operands.at(2);\n" +
      "    auto x = z + a*b;\n" +
      "    results.push_back(x);\n";

    compute_space_ = Multiply;

    // -- Data spaces.
    DataSpace W, I, O;

    W.name = "Weights";
    I.name = "Inputs";
    O.name = "Outputs";

    W.num_ranks = 2;
    I.num_ranks = 1;
    O.num_ranks = 2;

    // We need the following only for generating human-readable emulation code for transfer blocks.
    W.subscripts = { "k", "r" };
    I.subscripts = { "w" };
    O.subscripts = { "k", "p" };

    // -- Tensor accesses.
    W.read_projection = isl_map_read_from_str(context_, "{ [k,p,r] -> Weights[k,r] }");
    I.read_projection = isl_map_read_from_str(context_, "{ [k,p,r] -> Inputs[p+r] }");
    O.read_projection = isl_map_read_from_str(context_, "{ [k,p,r] -> Outputs[k,p] }");

    W.write_projection = nullptr;
    I.write_projection = nullptr;
    O.write_projection = isl_map_read_from_str(context_, "{ [k,p,r] -> Outputs[k,p] }");

    W.read_projection_txt = { "k", "r" };
    I.read_projection_txt = { "p+r" };
    O.read_projection_txt = { "k", "p" };

    O.write_projection_txt = { "k", "p" };

    // ---- Limit the tensor accesses to the same domains as iteration space.
    //      e.g., A_read := A_read * iteration_space;
    W.read_projection = isl_map_intersect_domain(W.read_projection, isl_set_copy(iteration_space_));
    I.read_projection = isl_map_intersect_domain(I.read_projection, isl_set_copy(iteration_space_));
    O.read_projection = isl_map_intersect_domain(O.read_projection, isl_set_copy(iteration_space_));
    O.write_projection = isl_map_intersect_domain(O.write_projection, isl_set_copy(iteration_space_));

    // -- Compute dependencies.
    //    all_reads := A_read + B_read + Z_read;
    //    all_writes := Z_write;
    //    closure := all_writes.(all_reads^-1);
    // 
    //    less_than := { [i] -> [j] : i < j };
    //    schedule_reference := { MAC[m,n,k] -> [m,n,k] };
    //    program_order := schedule_reference.less_than.(schedule_reference^-1);
    // 
    //    dependencies := closure * program_order;

    data_spaces_.push_back(W);
    data_spaces_.push_back(I);
    data_spaces_.push_back(O);
  }
};
