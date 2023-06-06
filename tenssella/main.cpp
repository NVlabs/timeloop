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

#include <iostream>
#include <filesystem>

#include <cassert>
#include <cstring>

#include <isl/ctx.h>

#include "tenssella.hpp"

#include "printers.hpp"
#include "data.hpp"

#include "problem-GEMM.hpp"
#include "problem-MV.hpp"
#include "problem-CONV-1D.hpp"
#include "problem-CONV-2D.hpp"
//#include "problem-CONV7D.hpp"
//#include "problem-CONV-PRK.hpp"

#include "arch-2D-3L.hpp"
#include "arch-2D-4L.hpp"
#include "arch-fuse.hpp"
//#include "arch-timeloop-3L.hpp"

#include "mapping-GEMM-3L.hpp"
#include "mapping-MV.hpp"
#include "mapping-CONV-1D.hpp"
#include "mapping-CONV-2D.hpp"
//#include "mapping-GEMM-4L.hpp"
//#include "mapping-CONV-PRK-4L.hpp"
//#include "mapping-CONV-PRK-ref.hpp"
//#include "mapping-timeloop.hpp"
//#include "mapping-CONV7D-timeloop-3L.hpp"
//#include "mapping-CONV7D-ref.hpp"

#include "tutorial/tutorial.hpp"

#define WORKLOAD_GEMM 0
#define WORKLOAD_CONV 1
#define WORKLOAD_GEMM_4L 2
#define WORKLOAD_CONV7D 3
#define WORKLOAD_CONV1D 4
#define WORKLOAD WORKLOAD_CONV1D

void test_fusion_conv() {

  string name = "fuse";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  //define arch
  auto arch_test = make_shared<Arch_Fuse>(context);

  //Define problem
  int P = 128, R = 3;
  map<string, ProblemPtr> einsum_map;
  map<string, DataPtr> data_space_map;
  construct_problem_fusion(context, data_space_map, einsum_map, P, R);
  char str_problem_instance[256];
  sprintf(str_problem_instance, "  int P=%d, R=%d;\n", P, R);

  //Define mapping
  map<string, MappingPtr> mapping_test =
      ConstructMappingMulti(context,  {1,34,4}, {1,3,1}, {1,32,4}, {1,3,1});
  auto binding_test = Create2StageBinding();


  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);

  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_sequential_fusion_3L() {

  string name = "seq_fuse_3L";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  //define arch
  //3Level sequential
  auto arch_test = make_shared<Arch_Fuse_Seq>(context);

  //Define problem
  int P = 128, R = 3;
  map<string, ProblemPtr> einsum_map;
  map<string, DataPtr> data_space_map;
  construct_problem_fusion(context, data_space_map, einsum_map, P, R);
  char str_problem_instance[256];
  sprintf(str_problem_instance, "  int P=%d, R=%d;\n", P, R);

  //Define mapping
  map<string, MappingPtr> mapping_test =
      ConstructMappingMulti(context,  {1,34,4}, {1,3,1}, {1,32,4}, {1,3,1});
  auto binding_test = Create2Stage3LevelBindingSeq();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);

  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_sequential_fusion() {

  string name = "seq_fuse";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  //define arch
  //2level sequential fusion
  auto arch_test = make_shared<Arch_Fuse_2L>(context);

  //Define problem
  int P = 128, R = 3;
  map<string, ProblemPtr> einsum_map;
  map<string, DataPtr> data_space_map;
  construct_problem_fusion(context, data_space_map, einsum_map, P, R);
  char str_problem_instance[256];
  sprintf(str_problem_instance, "  int P=%d, R=%d;\n", P, R);

  //Define mapping
  map<string, MappingPtr> mapping_test =
      ConstructMappingMulti2Level(context, P, R);
  auto binding_test = Create2Stage2LevelBindingSeq();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);

  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}


void test_fusion_pw() {

  string name = "pw_fuse";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  //define arch
  auto arch_test = make_shared<Arch_Fuse>(context);

  //Define problem
  int P = 128, R = 1;
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_problem_fusion(context, data_space_map, einsum_map, P, R);
  char str_problem_instance[256];
  sprintf(str_problem_instance, "  int P=%d, R=%d;\n", P, R);

  //Define mapping
  map<string, MappingPtr> mapping_test =
      ConstructMappingMulti(context,  {1,32,4}, {1,1,1}, {1,32,4}, {1,1,1});
  auto binding_test = Create2StageBinding();


  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);

  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_conv2d() {

  string name = "conv2d";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  int P = 16, Q = 16, C = 16;

  //The largest thred I can assign is 4x8
  int P1 = 4, P2 = 4;
  int Q1 = 4, Q2 = 4;
  int C1 = 16, C2 = 1;

  assert(P == P2*P1);
  assert(Q == Q2*Q1);
  assert(C == C2*C1);

  auto arch_test = make_shared<Arch_2D_3L_CONV2D>(context);
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_conv2d_problem(context, data_space_map, einsum_map, P, Q, C);

  auto mapping_test = ConstructMappingCONV2D(context,
          vector<int>({P2,Q2,C2,3,3}), vector<int>({P1,Q1,C1,1,1}));
  auto binding_test = CreateBindingCONV2D();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_MV_Fusion_Seq() {

  string name = "matrix_vector_fuse_seq";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  int M = 64, K = 64;

  //The largest thred I can assign is 4x8
  int M1 = 8, M2 = 8;
  int K1 = 64, K2 = 1;

  assert(M == M2*M1);
  assert(K == K2*K1);

  auto arch_test = make_shared<Arch_2D_3L_MV_Fuse_Seq>(context);
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_mv_problem_2stages(context,
          data_space_map, einsum_map, M, M, K);
  vector<int> t_m = {1, M1, M2};
  vector<int> t_k = {1, K1, K2};

  auto mapping_test = ConstructMappingMV2StagesSeq(context, t_m, t_k, t_k, t_m);
  auto binding_test = CreateBindingMV2StagesSeq();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_MV_Fusion_GB() {

  string name = "matrix_vector_fuse_GB";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  int M = 64, K = 64;

  //The largest thred I can assign is 4x8
  int M1 = 8, M2 = 8;
  int K1 = 64, K2 = 1;

  assert(M == M2*M1);
  assert(K == K2*K1);

  auto arch_test = make_shared<Arch_2D_3L_MV_Fuse_GB>(context);
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_mv_problem_2stages(context,
          data_space_map, einsum_map, M, M, K);
  vector<int> t_m = {1, M1, M2};
  vector<int> t_k = {1, K1, K2};

  auto mapping_test = ConstructMappingMV2StagesGB(context, t_m, t_k, t_k, t_m);
  auto binding_test = CreateBindingMV2StagesGB();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_MV_4L() {

  string name = "matrix_vector_4L";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  int M = 64, K = 16;

  //The largest thred I can assign is 4x8
  int M1 = 4, M2 = 4, M3 = 4;
  int K1 = 16, K2 = 1, K3 = 1;

  assert(M == M3*M2*M1);
  assert(K == K3*K2*K1);

  auto arch_test = make_shared<Arch_2D_4L_MV>(context);
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_mv_problem(context,
          data_space_map, einsum_map, M, K);
  vector<int> t_m = {1, M1, M2, M3};
  vector<int> t_k = {1, K1, K2, K3};

  auto mapping_test = ConstructMappingMV4L(context, t_m, t_k);
  auto binding_test = CreateBindingMV4L();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_MV_Fusion_4L_Finer_Grained() {

  string name = "matrix_vector_FG_fuse_4L";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  int N = 64, M = 64, K = 64;

  int N1 = 64, N2 = 1, N3 = 1;
  int M1 = 4,  M2 = 4, M3 = 4;
  int K1 = 64, K2 = 1, K3 = 1;

  assert(M == M3*M2*M1);
  assert(K == K3*K2*K1);
  assert(N == N3*N2*N1);

  auto arch_test = make_shared<Arch_2D_4L_MV_Fuse_Finer>(context);
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_mv_problem_2stages(context,
          data_space_map, einsum_map, N, M, K);
  vector<int> t_m0 = {1, M1, M2, M3};
  vector<int> t_k0 = {1, K1, K2, K3};

  vector<int> t_m1 = {1, N1, N2, N3};
  vector<int> t_k1 = {1, M1, M2, M3};

  auto mapping_test = ConstructMappingMV2Stages4LFiner(context, t_m0, t_k0, t_m1, t_k1);
  auto binding_test = CreateBindingMV2Stages4LFiner();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}


void test_MV_Fusion_4L() {

  string name = "matrix_vector_fuse_4L";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  int N = 64, M = 64, K = 64;

  int N1 = 64, N2 = 1, N3 = 1;
  int M1 = 4,  M2 = 4, M3 = 4;
  int K1 = 64, K2 = 1, K3 = 1;

  assert(M == M3*M2*M1);
  assert(K == K3*K2*K1);
  assert(N == N3*N2*N1);

  auto arch_test = make_shared<Arch_2D_4L_MV_Fuse>(context);
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_mv_problem_2stages(context,
          data_space_map, einsum_map, N, M, K);
  vector<int> t_m0 = {1, M1, M2, M3};
  vector<int> t_k0 = {1, K1, K2, K3};

  vector<int> t_m1 = {1, N1, N2, N3};
  vector<int> t_k1 = {1, M1, M2, M3};

  auto mapping_test = ConstructMappingMV2Stages4L(context, t_m0, t_k0, t_m1, t_k1);
  auto binding_test = CreateBindingMV2Stages4L();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_MV_Fusion() {

  string name = "matrix_vector_fuse";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  int M = 64, K = 64;

  //The largest thred I can assign is 4x8
  int M1 = 8, M2 = 8;
  int K1 = 64, K2 = 1;

  assert(M == M2*M1);
  assert(K == K2*K1);

  auto arch_test = make_shared<Arch_2D_3L_MV_Fuse>(context);
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_mv_problem_2stages(context,
          data_space_map, einsum_map, M, M, K);
  vector<int> t_m = {1, M1, M2};
  vector<int> t_k = {1, K1, K2};

  auto mapping_test = ConstructMappingMV2Stages(context, t_m, t_k, t_k, t_m);
  auto binding_test = CreateBindingMV2Stages();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_MV_reduction() {

  string name = "matrix_vector_reduction";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  int M = 16, K = 32;

  //The largest thred I can assign is 4x8
  int M1 = 16, M2 = 1;
  int K1 = 8, K2 = 4;

  assert(M == M2*M1);
  assert(K == K2*K1);

  vector<int> tm = {1, M1, M2};
  vector<int> tk = {1, K1, K2};

  auto arch_test = make_shared<Arch_2D_3L_MV>(context);
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_mv_problem(context, data_space_map, einsum_map, M, K);

  auto mapping_test = ConstructMappingMVParallel(context, tm, tk);
  auto binding_test = CreateBindingMV();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_MV() {

  string name = "matrix_vector";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  int M = 16, K = 32;

  //The largest thred I can assign is 4x8
  int M1 = 4, M2 = 4;
  int K1 = 32, K2 = 1;

  assert(M == M2*M1);
  assert(K == K2*K1);

  auto arch_test = make_shared<Arch_2D_3L_MV>(context);
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_mv_problem(context, data_space_map, einsum_map, M, K);

  auto mapping_test = ConstructMappingMV(context, M1, M2, K1, K2);
  auto binding_test = CreateBindingMV();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_GEMM() {

  string name = "gemm";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  int M = 16, N = 28, K = 15;

  //The largest thred I can assign is 4x8
  int M1 = 4, M2 = 4;
  int N1 = 7, N2 = 4;
  int K1 = 3, K2 = 5;

  assert(M == M2*M1);
  assert(N == N2*N1);
  assert(K == K2*K1);

  auto arch_test = make_shared<Arch_2D_3L>(context);
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_gemm_problem(context, data_space_map, einsum_map, M, N, K);

  auto mapping_test = ConstructMappingGEMM(context, M1, M2, N1, N2, K1, K2);
  auto binding_test = CreateBindingGEMM();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

void test_single_einsum() {

  string name = "single_einsum";

  isl_ctx* context = isl_ctx_alloc();
  InitPrinters(context);

  //define arch
  auto arch_test = make_shared<Arch_3L>(context);

  //Define problem
  int P = 128, R = 3;
  std::map<std::string, ProblemPtr> einsum_map;
  std::map<std::string, DataPtr> data_space_map;
  construct_problem(context, data_space_map, einsum_map, P, R);
  char str_problem_instance[256];
  sprintf(str_problem_instance, "  int P=%d, R=%d;\n", P, R);

  //Define mapping
  map<string, MappingPtr> mapping_test =
      ConstructMapping(context,  {1,32,4}, {1,3,1});
  auto binding_test = CreateBinding();

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;


  //Main Codegen Driver Function
  TenssellaCompile(context, name, einsum_map, data_space_map, mapping_test, arch_test, binding_test);

  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);

  int res = cmd("bash run_emulator.sh " + name);
  assert(res == 0);
}

//The old main function
#if 0
  isl_ctx* context = isl_ctx_alloc();

  InitPrinters(context);

  // == Instantiate hardware.
#if WORKLOAD == WORKLOAD_GEMM
  Arch_2D_3L arch_test(context);
  Arch_2D_3L arch_ref(context);
#elif WORKLOAD == WORKLOAD_CONV7D
  //Arch_Timeloop_3L arch_test(context);
  Arch_Reference arch_ref(context);
#elif WORKLOAD == WORKLOAD_CONV1D
  Arch_3L arch_test(context);
  Arch_Reference arch_ref(context);
#else
  Arch_RowDiagCol_4L arch_test(context);
  Arch_Reference arch_ref(context);
#endif

  TRACE(1) << "Architecture instantiated." << std::endl;

  // == Instantiate problem shape and instance.

  // FIXME: we are instantiating 2 copies of the ProblemShape object, one for
  // the test code and another for the reference code. We should be able to
  // use the same object, but the Generate() code currently destroys some
  // ISL objects in the Problem object. If we comb through the code carefully
  // and place some isl_*_copy() calls we should be able to use the same
  // Problem object.

  char str_problem_instance[256];

#if WORKLOAD == WORKLOAD_GEMM

  int M = 7, N = 13, K = 15;

  int M1 = 1, M2 = 7;
  int N1 = 1, N2 = 13;
  int K1 = 3, K2 = 5;

  assert(M == M2*M1);
  assert(N == N2*N1);
  assert(K == K2*K1);

  sprintf(str_problem_instance, "  int M=%d, N=%d, K=%d;\n", M, N, K);

  ProblemShape_GEMM problem_test(context);
  ProblemShape_GEMM problem_ref(context);

#elif WORKLOAD == WORKLOAD_GEMM_4L

  int M = 3, N = 4, K = 7;

  int M3 = 3, M2 = 1, M1 = 1;
  int N3 = 4, N2 = 1, N1 = 1;
  int K3 = 1, K2 = 7, K1 = 1;

  assert(M == M3*M2*M1);
  assert(N == N3*N2*N1);
  assert(K == K3*K2*K1);

  sprintf(str_problem_instance, "  int M=%d, N=%d, K=%d;\n", M, N, K);

  ProblemShape_GEMM problem_test(context);
  ProblemShape_GEMM problem_ref(context);

#elif WORKLOAD == WORKLOAD_CONV7D

  (void) str_problem_instance;

  ProblemShape_CONV7D problem_test(context, input_dir / "conv7d.txt");
  ProblemShape_CONV7D problem_ref(context, input_dir / "conv7d.txt");

#elif WORKLOAD == WORKLOAD_CONV1D

  int P = 128, R = 3;
  std::map<std::string, ProblemShape*> einsum_map;
  std::map<std::string, DataSpace*> data_space_map;
  construct_problem(context, data_space_map, einsum_map, P, R);
  //ProblemShape_CONV_1D problem_ref(context, true, P, R);
  sprintf(str_problem_instance, "  int P=%d, R=%d;\n", P, R);

#else // WORKLOAD_CONV

  int K = 16, P = 5, R = 3; // P <= Hardware X.
  //int K = 16, P = 14, R = 3;

  int K3 = 1, K2 = K, K1 = 1;
  int P3 = P, P2 = 1, P1 = 1;
  int R3 = R, R2 = 1, R1 = 1;

  assert(K == K3*K2*K1);
  assert(P == P3*P2*P1);
  assert(R == R3*R2*R1);

  sprintf(str_problem_instance, "  int K=%d, P=%d, R=%d;\n", K, P, R);

  ProblemShape_CONV_PRK problem_test(context, true, K, P, R);
  ProblemShape_CONV_PRK problem_ref(context, true, K, P, R);

#endif

  TRACE(1) << "Problem instantiated." << std::endl;

  // == Instantiate mapping.
#if WORKLOAD == WORKLOAD_GEMM
  Mapping_GEMM_3L mapping_test(context, M1, M2, N1, N2, K1, K2);
  Mapping_GEMM_3L mapping_ref(context, M1, M2, N1, N2, K1, K2);
#elif WORKLOAD == WORKLOAD_GEMM_4L
  Mapping_GEMM_4L mapping_test(context, M1, M2, M3, N1, N2, N3, K1, K2, K3);
  Mapping_GEMM_4L mapping_ref(context, M1, M2, M3, N1, N2, N3, K1, K2, K3);
#elif WORKLOAD == WORKLOAD_CONV7D
  //Mapping_Timeloop mapping_test(context, &arch_test, &problem_test, input_dir / "mapping.txt");
  Mapping_Timeloop mapping_test(context, 4, &problem_test, input_dir / "mapping.txt");
  Arch_Timeloop_3L arch_test(context, mapping_test.TemporalProducts());
  // Mapping_CONV7D_Timeloop_3L mapping_test(context, "inputs/mapping-1l.txt");
  Mapping_CONV7D_Reference mapping_ref(context, input_dir / "conv7d.txt");
#elif WORKLOAD == WORKLOAD_CONV1D
  map<string, Mapping*> mapping_test = ConstructMapping(context,  {1,32,4}, {1,3,1});
  auto binding_test = CreateBinding();
  map<string, Mapping*> mapping_ref = ConstructMappingRef(context, P, R);
  auto binding_ref = CreateBindingRef();

#else
  Mapping_CONV_PRK_4L mapping_test(context, K1, K2, K3, P1, P2, P3, R1, R2, R3);
  Mapping_CONV_PRK_Reference mapping_ref(context, K, P, R);
#endif

  TRACE(1) << "Mapping instantiated." << std::endl;
  TRACE(1) << std::endl;

  // == Print prelude.

  Printer p(context, "out/out.cpp");
  Printer q(context, "out/out.ast", true);

  p << str_prelude;
  p << str_begin_main;

  // We don't need to emit the problem instance since we baked the bounds into the Problem
  // shape itself.
  // p << str_problem_instance << "\n";

  // == Run the code generation.

#define RUN_TEST
#define RUN_REF

#ifdef RUN_TEST
  Tenssella test(context, &arch_test, einsum_map, mapping_test, data_space_map, binding_test);
  test.Generate(p, q);

  p << str_run_arch << "\n";
  p << "  std::cerr << std::endl << \"=== TEST RUN COMPLETE ===\" << std::endl << std::endl;\n\n";

  test.PrintValidationSavePrograms(p);
#endif

#ifdef RUN_REF
  p << str_reset_arch << "\n";

  Tenssella ref(context, &arch_ref, einsum_map, mapping_ref, data_space_map, binding_ref);
  ref.Generate(p, q, true, false);

  p << str_run_arch << "\n";
  p << "  std::cerr << std::endl << \"=== REF RUN COMPLETE ===\" << std::endl << std::endl;\n\n";
#endif

#if defined RUN_TEST && defined RUN_REF
  ref.PrintValidationCheckPrograms(p);
  p << str_print_validation_result;
  p << "  std::cerr << std::endl << \"=== VALIDATION COMPLETE ===\" << std::endl << std::endl;\n\n";
#endif

  p << end_main;

  //release all the heap memory
  //TODO: change to share_ptr
  for (auto it : einsum_map) {
    delete it.second;
    //delete problem_ref.at(it.first);
    delete mapping_test.at(it.first);
    delete mapping_ref.at(it.first);
  }

  for (auto it: data_space_map) {
    delete it.second;
  }


  // == Cleanup.
  UninitPrinters();

  isl_ctx_free(context);
#endif

void playground() {
  isl_ctx* ctx = isl_ctx_alloc();


  auto ms = isl_map_read_from_str(ctx, "{s[k,i]->[i+1]}");
  auto mp = isl_map_read_from_str(ctx, "{p[k,i]->[i]}");
  auto mt = isl_map_read_from_str(ctx, "{t[k,i, j]->[i+1, j]}");
  auto aff_s = pick(get_aff_vec(ms));
  auto aff_t = isl_multi_union_pw_aff_from_union_map(to_umap(mt));
  auto aff_p = isl_multi_union_pw_aff_from_union_map(to_umap(mp));
  auto domain_s = isl_set_read_from_str(ctx, "{s[k,i]: 0<=i<=32 and 0<=k<=3}");
  auto domain_p = isl_set_read_from_str(ctx, "{p[k,i]: 0<=i<=32 and 0<=k<=3}");
  auto domain_t = isl_set_read_from_str(ctx, "{t[k,i, j]: 0<=i<=32 and 0<=j<=8 and 0<=k<=3}");

  //auto sched0 = isl_schedule_from_domain(to_uset(domain_s));
  //auto m0 = rdmap(ctx, "{s[k,i]->[i+1]}");
  //sched0 = isl_schedule_insert_partial_schedule(sched0, isl_multi_union_pw_aff_from_union_map(m0));
  //auto m1 = rdmap(ctx, "{s[k,i]->[k]}");
  //sched0 = isl_schedule_insert_partial_schedule(sched0, isl_multi_union_pw_aff_from_union_map(m1));
  //auto sched_map0 = isl_schedule_get_map(sched0);
  //std::cout << "sched map: " << str(sched_map0) << endl;
  //assert(false);

  auto sched_s = isl_schedule_from_domain(to_uset(domain_s));
  auto maff_s = isl_multi_union_pw_aff_from_multi_aff(isl_multi_aff_from_aff(aff_s));
  sched_s = isl_schedule_insert_partial_schedule(sched_s, maff_s);

  auto sched_t = isl_schedule_from_domain(to_uset(domain_t));
  sched_t = isl_schedule_insert_partial_schedule(sched_t, aff_t);

  auto sched_p = isl_schedule_from_domain(to_uset(domain_p));
  sched_p = isl_schedule_insert_partial_schedule(sched_p, aff_p);
  auto sched_merge = isl_schedule_sequence(sched_s, cpy(sched_t));
  sched_merge = isl_schedule_sequence(sched_merge, cpy(sched_p));
  auto mu1 = rdmap(ctx, "{p[k, i]->[k];s[k, i]->[k]; t[k,i,j]->[k+1]}");
  sched_merge = isl_schedule_insert_partial_schedule(sched_merge, isl_multi_union_pw_aff_from_union_map(mu1));

  auto sched_map = isl_schedule_get_map(sched_merge);

  std::cout  << "schedule s: " << str(sched_s) << std::endl;
  std::cout  << "schedule t: " << str(sched_t) << std::endl;
  std::cout  << "\t\nschedule merge: " << str(sched_merge) << std::endl;
  std::cout  << "schedule map: " << str(sched_map) << std::endl;

  //std::cout << codegen_c(sched_map) << std::endl;

  assert(false);

}

//
// Main.
//
//

int main(int argc, char* argv[])
{
  string test;
  if (argc == 2)
    test = argv[1];
  else {
    std::cout << "tenssella need an input argument." << std::endl;
    exit(0);
  }

  if (test == "regression") {
    //Single Einsum
    test_single_einsum();
    test_MV();
    test_MV_4L();
    test_conv2d();
    test_GEMM();

    //Two Einsum
    test_GEMM_EW();
    test_MV_Fusion_4L();
    test_MV_Fusion_4L_Finer_Grained();
    test_MV_Fusion_GB();
    test_MV_Fusion_Seq();
    test_MV_Fusion();
    test_fusion_conv();
    test_fusion_pw();
    test_sequential_fusion();
    test_sequential_fusion_3L();

    //Did not work because of no collector implemented
    //test_MV_reduction();
  } else if (test == "tutorial") {
    test_GEMM_EW();
  } else if (test == "playground") {
    playground();
  } else {
    std::cout << "Does not find test: [" << test << "]" << endl;
    exit(0);
  }
  return 0;
}
