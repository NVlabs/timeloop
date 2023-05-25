/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "problem.hpp"

class ProblemShape_CONV2D: public ProblemShape
{
 public:
  ProblemShape_CONV2D(isl_ctx* context, string cs_name,
          string I, string W, string O, unsigned P, unsigned Q, unsigned C) :
      ProblemShape(context)
  {
    // -----------------------------------
    // Problem shape: GEMM
    //  for m = [0:M)
    //    for n = [0:N)
    //      for k = [0:K)
    //        Z[m][n] += A[m][k] * B[k][n]
    // -----------------------------------

      char buf[256];
      sprintf(buf,
              "{ [p,q,r,s,c] :0 <= p < %u and 0 <= q < %u and 0 <= c < %u \
              and 0 <= r <= 2 and 0 <= s <= 2}", P, Q, C);
    // -- Iteration space.
    iteration_space_ =
      isl_set_read_from_str(context_, buf);
    iteration_space_subscripts_ = { "p", "q", "r", "s", "c" };

    // -- Compute space.
    ComputeSpace Multiply;
    Multiply.name = cs_name;
    Multiply.num_ranks = 5;
    Multiply.subscripts = iteration_space_subscripts_;
    Multiply.transform_txt = std::string("") +
      "    float i = operands.at(0);\n" +
      "    float w = operands.at(1);\n" +
      "    float z = operands.at(2);\n" +
      "    float x = z + i*w;\n" +
      "    results.push_back(x);\n";

    compute_space_ = Multiply;


    data_spaces_.push_back(I);
    is_read.insert(I);
    data_spaces_.push_back(W);
    is_read.insert(W);
    data_spaces_.push_back(O);
    is_write.insert(O);
    is_read.insert(O);
  }

};

void construct_conv2d_problem(isl_ctx* ctx,
        map<string, DataPtr> & data_map,
        map<string, ProblemPtr> & einsum_map,
        int P, int Q, int C) {
    string cs_name = "CONV2D0";
    string a_name = "I";
    string b_name = "W";
    string z_name = "O";
    shared_ptr<ProblemShape_CONV2D> problem_test =
        make_shared<ProblemShape_CONV2D>(ctx, cs_name,
                a_name, b_name, z_name, P, Q, C);

    einsum_map.insert({cs_name, problem_test});

    // -- Data spaces.
    DataPtr A = make_shared<DataSpace>(a_name, 3);
    DataPtr B = make_shared<DataSpace>(b_name, 3);
    DataPtr Z = make_shared<DataSpace>(z_name, 3);
    A->setInput();
    B->setInput();
    Z->setOutput();


    // We need the following only for generating human-readable emulation code for transfer blocks.
    A->subscripts = { "w", "h", "c" };
    B->subscripts = { "r", "s", "c" };
    Z->subscripts = { "p", "q", "c" };

    // -- Tensor accesses.
    auto A_read_projection = isl_map_read_from_str(ctx,
            "{ [p, q, r, s, c] -> I[p+r,q+s,c] }");
    auto B_read_projection = isl_map_read_from_str(ctx,
            "{ [p, q, r, s, c] -> W[r,s,c] }");
    auto Z_read_projection = isl_map_read_from_str(ctx,
            "{ [p, q, r, s, c] -> O[p,q,c] }");


    // ---- Limit the tensor accesses to the same domains as iteration space.
    //      e.g., A_read := A_read * iteration_space;
    isl_set* iteration_space_ = problem_test->IterationSpace();
    A->read_projection[cs_name] =
        isl_map_intersect_domain(A_read_projection, isl_set_copy(iteration_space_));
    B->read_projection[cs_name] =
        isl_map_intersect_domain(B_read_projection, isl_set_copy(iteration_space_));
    Z->read_projection[cs_name] =
        isl_map_intersect_domain(cpy(Z_read_projection), isl_set_copy(iteration_space_));
    Z->write_projection[cs_name] =
        isl_map_intersect_domain(cpy(Z_read_projection), isl_set_copy(iteration_space_));

    data_map.insert({a_name, A});
    data_map.insert({b_name, B});
    data_map.insert({z_name, Z});

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

}
