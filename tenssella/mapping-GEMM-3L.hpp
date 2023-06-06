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

#include "mapping.hpp"

class Mapping_GEMM_3L : public Mapping
{
 private:
  // Parameters.
  int M1, M2;
  int N1, N2;
  int K1, K2;

 public:
  Mapping_GEMM_3L(isl_ctx* context, int m1, int m2, int n1, int n2, int k1, int k2) :
      Mapping(context),
      M1(m1), M2(m2), N1(n1), N2(n2), K1(k1), K2(k2)
  {
    char buf[1024];

    //
    // Tile ID -> Global coordinates.
    //

    // Fixme: add m0,n0,k0
    sprintf(buf,
            "{[[] -> [m2,n2,k2]] -> [m,n,k] : exists m1,n1,k1 : m = m2*%d + m1 and 0 <= m1 < %d and "
            "                                            n = n2*%d + n1 and 0 <= n1 < %d and "
            "                                            k = k2*%d + k1 and 0 <= k1 < %d }",
            M1, M1,
            N1, N1,
            K1, K1);
    tile_id_to_set_[2] = isl_map_read_from_str(context_, buf);

    // Fixme: add m0,n0,k0
    sprintf(buf,
            "{ [[[] -> [m2,n2,k2]] -> [m1,n1,k1]] -> [m,n,k] : m = m2*%d + m1 and 0 <= m1 < %d and "
            "                                          n = n2*%d + n1 and 0 <= n1 < %d and "
            "                                          k = k2*%d + k1 and 0 <= k1 < %d }",
            M1, M1,
            N1, N1,
            K1, K1);
    tile_id_to_set_[1] = isl_map_read_from_str(context_, buf);

    // Fixme: add constraints on m0,n0,k0
    sprintf(buf,
            "{ [[[[] -> [m2,n2,k2] ] -> [m1,n1,k1]] -> [m0,n0,k0]]-> [m,n,k] : m = m2*%d + m1 and 0 <= m1 < %d and "
            "                                                         n = n2*%d + n1 and 0 <= n1 < %d and "
            "                                                         k = k2*%d + k1 and 0 <= k1 < %d }",
            M1, M1,
            N1, N1,
            K1, K1);
    tile_id_to_set_[0] = isl_map_read_from_str(context_, buf);

    //
    // Tile ID domains.
    //

    sprintf(buf,
            "{ [m2,n2,k2] : 0 <= m2 < %d and "
            "               0 <= n2 < %d and "
            "               0 <= k2 < %d }",
            M2, N2, K2);
    tile_id_domains_[2] = isl_set_read_from_str(context_, buf);

    sprintf(buf,
            "{ [m1,n1,k1] : 0 <= m1 < %d and "
            "               0 <= n1 < %d and "
            "               0 <= k1 < %d }",
            M1, N1, K1);
    tile_id_domains_[1] = isl_set_read_from_str(context_, buf);

    sprintf(buf,
            "{ [m0,n0,k0] : 0 <= m0 < %d and "
            "               0 <= n0 < %d and "
            "               0 <= k0 < %d }",
            1, 1, 1);
    tile_id_domains_[0] = isl_set_read_from_str(context_, buf);

    //
    // Skews.
    //
    sprintf(buf,
            "{ [] -> SpaceTime_3[0]}" // Hardware spacetime is <y,x>.
            );
    skews_[3] = isl_map_read_from_str(context_, buf);


    sprintf(buf,
            "{ [m2,n2,k2] -> SpaceTime_2[m2*%d + n2, k2] : 0 <= n2 < %d }",
            N2, N2);
    skews_[2] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [m1,n1,k1] -> SpaceTime_1[0, %d*%d*m1 + %d*n1 + k1] : 0 <= k1 < %d and 0 <= n1 < %d }",
            N1, K1, K1, K1, N1);
    skews_[1] = isl_map_read_from_str(context, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [m0,n0,k0] -> SpaceTime_0[] : m0 = 0 and n0 = 0 and n0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

map<string, MappingPtr> ConstructMappingGEMM(isl_ctx* c,
        int M1, int M2, int N1, int N2, int K1, int K2) {
  shared_ptr<Mapping_GEMM_3L> g = make_shared<Mapping_GEMM_3L>(c, M1, M2, N1, N2, K1, K2);
  return {{"GEMM0", g}};
}

shared_ptr<Binding> CreateBindingGEMM() {

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
  b->memory_binding_["Z"][3]["GEMM0"] = {"GlobalBuffer", 2};

  b->memory_binding_["A"][2]["GEMM0"] = {"RegFile", 0};
  b->memory_binding_["B"][2]["GEMM0"] = {"RegFile", 1};
  b->memory_binding_["Z"][2]["GEMM0"] = {"RegFile", 2};

  b->memory_binding_["A"][1]["GEMM0"] = {"OperandA", 0};
  b->memory_binding_["B"][1]["GEMM0"] = {"OperandB", 0};
  b->memory_binding_["Z"][1]["GEMM0"] = {"Result", 0};

  b->compute_binding_["GEMM0"] = {"MAC", 0};
  return b;
}

