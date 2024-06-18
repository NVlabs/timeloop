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

class ProblemShape_CONV7D : public ProblemShape
{
 public:
  ProblemShape_CONV7D(isl_ctx* context, std::string prob_filename) :
      ProblemShape(context)
  {
    // -----------------------------------
    // Problem shape: CONV7D
    // -----------------------------------
    std::ifstream prob_file(prob_filename.c_str());
    assert(prob_file);

    std::string line;
    std::getline(prob_file, line);
    auto factors = ParseFactorLine(line);

    // -- Iteration space.
    char buf[256];
    sprintf(buf, "{ [c,k,r,s,p,q,n] : 0 <= c < %d and 0 <= k < %d and 0 <= r < %d and 0 <= s < %d and 0 <= p < %d and 0 <= q < %d and 0 <= n < %d }",
            factors.at("C"),
            factors.at("K"),
            factors.at("R"),
            factors.at("S"),
            factors.at("P"),
            factors.at("Q"),
            factors.at("N"));

    iteration_space_ = isl_set_read_from_str(context_, buf);

    iteration_space_subscripts_ = { "c", "k", "r", "s", "p", "q", "n" };
    iteration_space_dimensions_ = { "C", "K", "R", "S", "P", "Q", "N" };

    // -- Compute space.
    ComputeSpace Multiply;
    Multiply.name = "Multiply";
    Multiply.num_ranks = 7;
    Multiply.subscripts = iteration_space_subscripts_;
    Multiply.transform_txt = std::string("") +
      "      auto a = operands.at(0);\n" +
      "      auto b = operands.at(1);\n" +
      "      auto z = operands.at(2);\n" +
      "      auto x = z + a*b;\n" +
      "      results.push_back(x);\n";

    compute_space_ = Multiply;

    // -- Data spaces.
    DataSpace W, I, O;

    W.name = "Weights";
    I.name = "Inputs";
    O.name = "Outputs";

    W.num_ranks = 4;
    I.num_ranks = 4;
    O.num_ranks = 4;

    // We need the following only for generating human-readable emulation code for transfer blocks.
    W.subscripts = { "k", "c", "s", "r" };
    I.subscripts = { "n", "c", "h", "w" };
    O.subscripts = { "k", "c", "q", "p" };

    // -- Tensor accesses.
    W.read_projection = isl_map_read_from_str(context_, "{ [c,k,r,s,p,q,n] -> Weights[k,c,s,r] }");
    I.read_projection = isl_map_read_from_str(context_, "{ [c,k,r,s,p,q,n] -> Inputs[n,c,q+s,p+r] }");
    O.read_projection = isl_map_read_from_str(context_, "{ [c,k,r,s,p,q,n] -> Outputs[n,k,q,p] }");

    W.write_projection = nullptr;
    I.write_projection = nullptr;
    O.write_projection = isl_map_read_from_str(context_, "{ [c,k,r,s,p,q,n] -> Outputs[n,k,q,p] }");

    W.read_projection_txt = { "k", "c", "s", "r" };
    I.read_projection_txt = { "n", "c", "q+s", "p+r" };
    O.read_projection_txt = { "n", "k", "q", "p" };

    O.write_projection_txt = { "n", "k", "q", "p" };

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
