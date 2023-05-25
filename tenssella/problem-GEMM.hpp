/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

//Compute space of gemm
class ProblemShape_GEMM_old : public ProblemShape
{
 public:
  ProblemShape_GEMM_old(isl_ctx* context, string cs_name,
          string A, string B, string Z,
          unsigned M, unsigned N, unsigned K) :
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
              "{ [m,n,k] :0 <= m < %u and 0 <= n < %u and 0 <= k < %u }", M, N, K);
    // -- Iteration space.
    iteration_space_ =
      isl_set_read_from_str(context_, buf);
    iteration_space_subscripts_ = { "m", "n", "k" };

    // -- Compute space.
    ComputeSpace Multiply;
    Multiply.name = cs_name;
    Multiply.num_ranks = 3;
    Multiply.subscripts = iteration_space_subscripts_;
    Multiply.transform_txt = std::string("") +
      "    float a = operands.at(0);\n" +
      "    float b = operands.at(1);\n" +
      "    float z = operands.at(2);\n" +
      "    float x = z + a*b;\n" +
      "    results.push_back(x);\n";

    compute_space_ = Multiply;


    data_spaces_.push_back(A);
    is_read.insert(A);
    data_spaces_.push_back(B);
    is_read.insert(B);
    data_spaces_.push_back(Z);
    is_write.insert(Z);
    is_read.insert(Z);
  }

};

//One stage Einsum
void construct_gemm_problem(isl_ctx* ctx,
        map<string, DataPtr> & data_map,
        map<string, ProblemPtr> & einsum_map,
        int M, int N, int K) {

    // -- Compute space
    string cs_name = "GEMM0";
    string a_name = "A";
    string b_name = "B";
    string z_name = "Z";
    shared_ptr<ProblemShape_GEMM_old> problem_test =
        make_shared<ProblemShape_GEMM_old>(ctx, cs_name, a_name, b_name, z_name, M, N, K);
    einsum_map.insert({cs_name, problem_test});

    // -- Data spaces.
    DataPtr A = make_shared<DataSpace>(a_name, 2);
    DataPtr B = make_shared<DataSpace>(b_name, 2);
    DataPtr Z = make_shared<DataSpace>(z_name, 2);
    A->setInput();
    B->setInput();
    Z->setOutput();


    // We need the following only for generating human-readable emulation code for transfer blocks.
    A->subscripts = { "m", "k" };
    B->subscripts = { "k", "n" };
    Z->subscripts = { "m", "n" };

    // -- Tensor accesses.
    auto A_read_projection = isl_map_read_from_str(ctx, "{ [m,n,k] -> A[m,k] }");
    auto B_read_projection = isl_map_read_from_str(ctx, "{ [m,n,k] -> B[k,n] }");
    auto Z_read_projection = isl_map_read_from_str(ctx, "{ [m,n,k] -> Z[m,n] }");


    // ---- Limit the tensor accesses to the same domains as iteration space.
    //      itersect with the iteration domain
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

}

