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
#include "data.hpp"

//Compute space of gemm
class ProblemShape_GEMM : public ProblemShape
{
 public:
  ProblemShape_GEMM(isl_ctx* context, string cs_name,
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

//2D elementwise compute space
class ProblemShape_Elemwise: public ProblemShape
{
 public:
  ProblemShape_Elemwise(isl_ctx* context, string cs_name,
          string I, string O, unsigned M, unsigned N) :
      ProblemShape(context)
  {
    // -----------------------------------
    // Problem shape: Elementwise
    //  for m = [0:M)
    //    for n = [0:N)
    //        O[m][n] = A[m][n] * 2

      char buf[256];
      sprintf(buf,
              "{ [m,n] :0 <= m < %u and 0 <= n < %u }", M, N);
    // -- Iteration space.
    iteration_space_ =
      isl_set_read_from_str(context_, buf);
    iteration_space_subscripts_ = { "m", "n" };

    // -- Compute space.
    ComputeSpace Multiply;
    Multiply.name = cs_name;
    Multiply.num_ranks = 2;
    Multiply.subscripts = iteration_space_subscripts_;
    Multiply.transform_txt = std::string("") +
      "    float a = operands.at(0);\n" +
      "    float z = operands.at(1);\n" +
      "    float x = z+a*2;\n" +
      "    results.push_back(x);\n";

    compute_space_ = Multiply;

    //Add the data space pointer
    data_spaces_.push_back(I);
    is_read.insert(I);

    //A caveat is that Tenssella assume all outputs are update statement
    //if it's a non reduction operand, we just assume it reduce with zero
    data_spaces_.push_back(O);
    is_write.insert(O);
    is_read.insert(O);
  }

};

void construct_gemm_elemwise_problem(isl_ctx* ctx,
        map<string, DataPtr> & data_map,
        map<string, ProblemPtr> & einsum_map,
        int M, int N, int K);
