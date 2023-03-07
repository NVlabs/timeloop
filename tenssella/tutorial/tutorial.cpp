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

#include "tutorial.hpp"

//This is just a test for gemm with elemwise
void test_GEMM_EW() {

  string name = "gemm_elemwise";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  int M = 16, N = 16, K = 16;

  //The largest thred I can assign is 4x8
  int M1 = 4, M2 = 4;
  int N1 = 4, N2 = 4;
  int K1 = 16, K2 = 1;

  assert(M == M2*M1);
  assert(N == N2*N1);
  assert(K == K2*K1);

  vector<int> t_m = {1, M1, M2};
  vector<int> t_n = {1, N1, N2};
  vector<int> t_k = {1, K1, K2};

  auto arch_test = make_shared<Arch_2D_3L_GEMM>(context);
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_gemm_elemwise_problem(context, data_space_map, einsum_map, M, N, K);

  auto mapping_test = ConstructMappingGEMM_Elemwise(context, t_m, t_n, t_k);
  auto binding_test = CreateBindingGEMM_Elemwise();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);

  std::cout << "Congratulations! Tutorial test passes." << std::endl
      << "You can check the generated EDDO code in" << "\t"
      << "./test_collaterals/gemm_elemwise/out.cpp" << std::endl;
}

//Multi-Stage Einsum, a GEMM followed by elementwise
void construct_gemm_elemwise_problem(isl_ctx* ctx,
        map<string, DataPtr> & data_map,
        map<string, ProblemPtr> & einsum_map,
        int M, int N, int K) {

    //Define the compute space
    string cs_name = "GEMM0";
    string a_name = "A";
    string b_name = "B";
    string z_name = "Z";
    shared_ptr<ProblemShape_GEMM> e0 =
        make_shared<ProblemShape_GEMM>(ctx, cs_name, a_name, b_name, z_name, M, N, K);
    einsum_map.insert({"GEMM0", e0});

    //The compute space for second einsum
    cs_name = "Elemwise";
    string o_name = "O";
    shared_ptr<ProblemShape_Elemwise> e1=
        make_shared<ProblemShape_Elemwise>(ctx, cs_name, z_name, o_name, M, N);
    einsum_map.insert({"Elemwise", e1});

    // -- Data spaces.
    DataPtr A = make_shared<DataSpace>(a_name, 2);
    DataPtr B = make_shared<DataSpace>(b_name, 2);
    DataPtr Z = make_shared<DataSpace>(z_name, 2);
    DataPtr O = make_shared<DataSpace>(o_name, 2);
    A->setInput();
    B->setInput();
    O->setOutput();


    // We need the following only for generating human-readable emulation code for transfer blocks.
    A->subscripts = { "m", "k" };
    B->subscripts = { "k", "n" };
    Z->subscripts = { "m", "n" };
    O->subscripts = { "m", "n" };

    // -- Tensor accesses.
    auto A_read_projection = isl_map_read_from_str(ctx, "{ [m,n,k] -> A[m,k] }");
    auto B_read_projection = isl_map_read_from_str(ctx, "{ [m,n,k] -> B[k,n] }");
    auto Z_read_projection = isl_map_read_from_str(ctx, "{ [m,n,k] -> Z[m,n] }");

    isl_set* iteration_space_ = e0->IterationSpace();
    A->read_projection["GEMM0"] =
        isl_map_intersect_domain(A_read_projection, isl_set_copy(iteration_space_));
    B->read_projection["GEMM0"] =
        isl_map_intersect_domain(B_read_projection, isl_set_copy(iteration_space_));
    Z->read_projection["GEMM0"] =
        isl_map_intersect_domain(cpy(Z_read_projection), isl_set_copy(iteration_space_));
    Z->write_projection["GEMM0"] =
        isl_map_intersect_domain(cpy(Z_read_projection), isl_set_copy(iteration_space_));

    auto I_read_projection = isl_map_read_from_str(ctx, "{ [m,n] -> Z[m,n] }");
    auto O_read_projection = isl_map_read_from_str(ctx, "{ [m,n] -> O[m,n] }");

    isl_set* iteration_space_elem = e1->IterationSpace();
    Z->read_projection["Elemwise"] =
        isl_map_intersect_domain(I_read_projection, isl_set_copy(iteration_space_elem));
    O->write_projection["Elemwise"] =
        isl_map_intersect_domain(cpy(O_read_projection), isl_set_copy(iteration_space_elem));
    O->read_projection["Elemwise"] =
        isl_map_intersect_domain(cpy(O_read_projection), isl_set_copy(iteration_space_elem));


    data_map.insert({a_name, A});
    data_map.insert({b_name, B});
    data_map.insert({z_name, Z});
    data_map.insert({o_name, O});

}

shared_ptr<Binding> CreateBindingGEMM_Elemwise() {

  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //
  // The indexing logic is
  //    for which tensor,
  //    at which hardware level,
  //    access by which Einsum
  //
  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["A"][3]["GEMM0"] = {"GlobalBuffer", 0};
  b->memory_binding_["B"][3]["GEMM0"]=  {"GlobalBuffer", 1};
  b->memory_binding_["O"][3]["Elemwise"] = {"GlobalBuffer", 2};

  b->memory_binding_["A"][2]["GEMM0"] = {"RegFile", 0};
  b->memory_binding_["B"][2]["GEMM0"] = {"RegFile", 1};
  b->memory_binding_["Z"][2]["GEMM0"] = {"RegFile", 2};
  b->memory_binding_["Z"][2]["Elemwise"] = {"RegFile", 2};
  b->memory_binding_["O"][2]["Elemwise"] = {"RegFile", 3};

  b->memory_binding_["A"][1]["GEMM0"] = {"OperandA", 0};
  b->memory_binding_["B"][1]["GEMM0"] = {"OperandB", 0};
  b->memory_binding_["Z"][1]["GEMM0"] = {"Result", 0};
  b->memory_binding_["Z"][1]["Elemwise"] = {"OperandA", 1};
  b->memory_binding_["O"][1]["Elemwise"] = {"Result", 1};

  b->compute_binding_["GEMM0"] = {"MAC", 0};
  b->compute_binding_["Elemwise"] = {"MAC", 1};
  return b;
}

map<string, MappingPtr> ConstructMappingGEMM_Elemwise(isl_ctx* c,
        vector<int> & ms, vector<int> & ns, vector<int> & ks) {
  map<string, MappingPtr> app_mapping;
  shared_ptr<Mapping_GEMM_3L_Fuse> g = make_shared<Mapping_GEMM_3L_Fuse>(c, ms, ns, ks, 2, 0);
  app_mapping.insert({"GEMM0", g});
  shared_ptr<Mapping_Elemwise_3L_Fuse> e = make_shared<Mapping_Elemwise_3L_Fuse>(c, ms, ns, 2, 1);
  app_mapping.insert({"Elemwise", e});
  return app_mapping;
}
