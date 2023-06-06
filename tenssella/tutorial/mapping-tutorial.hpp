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

class Mapping_Elemwise_3L_Fuse: public Mapping
{
 private:
  // Parameters.
  int M1, M2, M3;
  int N1, N2, N3;
  int K1, K2, K3;

 public:
  Mapping_Elemwise_3L_Fuse(isl_ctx* context,
                      vector<int>& ms,
                      vector<int>& ns,
                      //Least level of storage, interleaving granularity of fusion
                      int lls,
                      int einsum_id
                    ) :
      Mapping(context),
      M1(ms.at(0)), M2(ms.at(1)), M3(ms.at(2)),
      N1(ns.at(0)), N2(ns.at(1)), N3(ns.at(2))
  {

    //A helper function to create tiling
    GenerateTileMaps(
            {{ M3, N3}, {  M2, N2}, {  M1, N1}},
            {{ M2, N2}, {  M1, N2}}
            );

    char buf[1024];

    //
    // Skews.
    //

    //Root level there is no tiling factor, so LHS is an dummy set
    sprintf(buf, "{ [] -> SpaceTime_3[0] }");
    skews_[3] = isl_map_read_from_str(context_, buf);

    assert(lls == 2);

    sprintf(buf,
            "{ [m2,n2] -> SpaceTime_2[m2, n2, %d] }", einsum_id// Hardware spacetime is <y,x>.
            );
    skews_[2] = isl_map_read_from_str(context_, buf);


    sprintf(buf,
            "{ [m1,n1] -> SpaceTime_1[0,m1*%d +n1]: }", N1// Hardware spacetime is <t>.
            );
    skews_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [m0,n0] -> SpaceTime_0[] : m0 = 0 and n0 = 0}" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};


class Mapping_GEMM_3L_Fuse: public Mapping
{
 private:
  // Parameters.
  int M1, M2, M3;
  int N1, N2, N3;
  int K1, K2, K3;

 public:
  Mapping_GEMM_3L_Fuse(isl_ctx* context,
                      vector<int>& ms,
                      vector<int>& ns,
                      vector<int>& ks,
                      int lls, int einsum_id
                    ) :
      Mapping(context),
      M1(ms.at(0)), M2(ms.at(1)), M3(ms.at(2)),
      N1(ns.at(0)), N2(ns.at(1)), N3(ns.at(2)),
      K1(ks.at(0)), K2(ks.at(1)), K3(ks.at(2))
  {

    //A helper function to create tiling
    GenerateTileMaps(
            {{ M3, N3, K3 }, {  M2, N2, K2 }, {  M1, N1, K1 }},
            {{ M2, N2, K2 }, {  M1, N2, K1 }}
            );

    char buf[1024];

    //
    // Skews.
    //

    //Root level there is no tiling factor, so LHS is an dummy set
    sprintf(buf, "{ [] -> SpaceTime_3[0] }");
    skews_[3] = isl_map_read_from_str(context_, buf);

    assert(lls == 2);

    sprintf(buf,
            "{ [m2,n2,k2] -> SpaceTime_2[m2, n2*%d+k2, %d] }", K3, einsum_id// Hardware spacetime is <y,x>.
            );
    skews_[2] = isl_map_read_from_str(context_, buf);


    sprintf(buf,
            "{ [m1,n1,k1] -> SpaceTime_1[0,m1*%d +n1*%d + k1]: }", N1*K1, K2 // Hardware spacetime is <t>.
            );
    skews_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [m0,n0,k0] -> SpaceTime_0[] : m0 = 0 and n0 = 0 and k0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

//Function implemented in tutorial.cpp
map<string, MappingPtr> ConstructMappingGEMM_Elemwise(isl_ctx* c,
        vector<int> & ms, vector<int> & ns, vector<int> & ks);
