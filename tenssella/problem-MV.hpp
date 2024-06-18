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

#include "problem.hpp"

class ProblemShape_MV: public ProblemShape
{
 public:
  ProblemShape_MV(isl_ctx* context, string cs_name,
          string A, string B, string Z, unsigned M, unsigned K) :
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
              "{ [m, k] :0 <= m < %u and 0 <= k < %u }", M, K);
    // -- Iteration space.
    iteration_space_ =
      isl_set_read_from_str(context_, buf);
    iteration_space_subscripts_ = { "m", "k" };

    // -- Compute space.
    ComputeSpace Multiply;
    Multiply.name = cs_name;
    Multiply.num_ranks = 2;
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

void construct_mv_problem_2stages(isl_ctx* ctx,
        map<string, DataPtr> & data_map,
        map<string, ProblemPtr> & einsum_map,
        int Mo, int M, int K) {
    string in_tensor = "A0";
    string w0_tensor = "B0";
    string w1_tensor = "B1";
    string tmp_tensor = "A1";
    string out_tensor = "Z";

    string cs_name = "MV0";
    shared_ptr<ProblemShape_MV> mv0 =
        make_shared<ProblemShape_MV>(ctx, cs_name,
                in_tensor, w0_tensor, tmp_tensor, M, K);
    einsum_map.insert({cs_name, mv0});

    cs_name = "MV1";
    shared_ptr<ProblemShape_MV> mv1 =
        make_shared<ProblemShape_MV>(ctx, cs_name,
                tmp_tensor, w1_tensor, out_tensor, Mo, M);
    einsum_map.insert({cs_name, mv1});

    // -- Data spaces.
    DataPtr W0 = make_shared<DataSpace>(w0_tensor, 2);
    DataPtr W1 = make_shared<DataSpace>(w1_tensor, 2);
    DataPtr In = make_shared<DataSpace>(in_tensor, 1);
    DataPtr Interm = make_shared<DataSpace>(tmp_tensor, 1);
    DataPtr Out = make_shared<DataSpace>(out_tensor, 1);
    W0->setInput();
    W1->setInput();
    In->setInput();
    Out->setOutput();


    // -- Tensor accesses.
    auto W0_read_projection = isl_map_read_from_str(ctx, "{ [m,k] -> B0[m,k] }");
    auto W1_read_projection = isl_map_read_from_str(ctx, "{ [m,k] -> B1[m,k] }");
    auto In_read_projection = isl_map_read_from_str(ctx, "{ [m,k] -> A0[k] }");
    auto Interm_read_projection = isl_map_read_from_str(ctx, "{ [m,k] -> A1[m] }");
    auto Interm_read_projection_1 = isl_map_read_from_str(ctx, "{ [m,k] -> A1[k] }");
    auto Out_read_projection = isl_map_read_from_str(ctx, "{ [m,k] -> Z[m] }");


    // ---- Limit the tensor accesses to the same domains as iteration space.
    //      e.g., A_read := A_read * iteration_space;
    isl_set* iteration_space_0 = mv0->IterationSpace();
    isl_set* iteration_space_1 = mv1->IterationSpace();
    W0->read_projection["MV0"] =
        isl_map_intersect_domain(W0_read_projection, isl_set_copy(iteration_space_0));
    W1->read_projection["MV1"] =
        isl_map_intersect_domain(W1_read_projection, isl_set_copy(iteration_space_1));
    In->read_projection["MV0"] =
        isl_map_intersect_domain(In_read_projection, isl_set_copy(iteration_space_0));
    Interm->write_projection["MV0"] =
        isl_map_intersect_domain(cpy(Interm_read_projection), isl_set_copy(iteration_space_0));
    Interm->read_projection["MV0"] =
        isl_map_intersect_domain(cpy(Interm_read_projection), isl_set_copy(iteration_space_0));
    Interm->read_projection["MV1"] =
        isl_map_intersect_domain(Interm_read_projection_1, isl_set_copy(iteration_space_1));
    Out->write_projection["MV1"] =
        isl_map_intersect_domain(cpy(Out_read_projection), isl_set_copy(iteration_space_1));
    Out->read_projection["MV1"] =
        isl_map_intersect_domain(cpy(Out_read_projection), isl_set_copy(iteration_space_1));

    data_map.insert({in_tensor, In});
    data_map.insert({tmp_tensor, Interm});
    data_map.insert({out_tensor, Out});
    data_map.insert({w0_tensor, W0});
    data_map.insert({w1_tensor, W1});

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

void construct_mv_problem(isl_ctx* ctx,
        map<string, DataPtr> & data_map,
        map<string, ProblemPtr> & einsum_map,
        int M, int K) {
    string cs_name = "MV0";
    string a_name = "A";
    string b_name = "B";
    string z_name = "Z";
    shared_ptr<ProblemShape_MV> problem_test =
        make_shared<ProblemShape_MV>(ctx, cs_name, a_name, b_name, z_name, M, K);

    einsum_map.insert({cs_name, problem_test});

    // -- Data spaces.
    DataPtr A = make_shared<DataSpace>(a_name, 2);
    DataPtr B = make_shared<DataSpace>(b_name, 1);
    DataPtr Z = make_shared<DataSpace>(z_name, 1);
    A->setInput();
    B->setInput();
    Z->setOutput();


    // We need the following only for generating human-readable emulation code for transfer blocks.
    A->subscripts = { "m", "k" };
    B->subscripts = { "k" };
    Z->subscripts = { "m" };

    // -- Tensor accesses.
    auto A_read_projection = isl_map_read_from_str(ctx, "{ [m,k] -> A[m,k] }");
    auto B_read_projection = isl_map_read_from_str(ctx, "{ [m,k] -> B[k] }");
    auto Z_read_projection = isl_map_read_from_str(ctx, "{ [m,k] -> Z[m] }");


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
