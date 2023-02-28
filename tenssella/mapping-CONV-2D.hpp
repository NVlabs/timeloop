
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

#include "mapping.hpp"

class Mapping_CONV2D: public Mapping
{
 private:
  // Parameters.
  int P1, P2;
  int Q1, Q2;
  int R1, R2;
  int S1, S2;
  int C1, C2;

 public:
  Mapping_CONV2D(isl_ctx* context, vector<int>& t1, vector<int>& t2) :
      Mapping(context),
      P1(t1.at(0)), P2(t2.at(0)),
      Q1(t1.at(1)), Q2(t2.at(1)),
      R1(t1.at(2)), R2(t2.at(2)),
      S1(t1.at(3)), S2(t2.at(3)),
      C1(t1.at(4)), C2(t2.at(4))
  {
    char buf[1024];

    //
    // Tile ID -> Global coordinates.
    //

    // Fixme: add m0,n0,k0
    sprintf(buf,
            "{[[] -> [p2,q2,r2,s2,c2]] -> [p,q,s,r,c] : "
            "   exists p1,q1,r1,s1,c1 : p = p2*%d + p1 and 0 <= p1 < %d and "
            "                           q = q2*%d + q1 and 0 <= q1 < %d and "
            "                           r = r2*%d + r1 and 0 <= r1 < %d and "
            "                           s = s2*%d + s1 and 0 <= s1 < %d and "
            "                           c = c2*%d + c1 and 0 <= c1 < %d }",
            P1, P1,
            Q1, Q1,
            R1, R1,
            S1, S1,
            C1, C1);
    tile_id_to_set_[2] = isl_map_read_from_str(context_, buf);

    // Fixme: add m0,n0,k0
    sprintf(buf,
            "{[[[] -> [p2,q2,r2,s2,c2]] -> [p1, q1, r1, s1, c1]] -> [p,q,s,r,c] : "
            "                           p = p2*%d + p1 and 0 <= p1 < %d and "
            "                           q = q2*%d + q1 and 0 <= q1 < %d and "
            "                           r = r2*%d + r1 and 0 <= r1 < %d and "
            "                           s = s2*%d + s1 and 0 <= s1 < %d and "
            "                           c = c2*%d + c1 and 0 <= c1 < %d }",
            P1, P1,
            Q1, Q1,
            R1, R1,
            S1, S1,
            C1, C1);
    tile_id_to_set_[1] = isl_map_read_from_str(context_, buf);

    // Fixme: add constraints on m0,n0,k0
    sprintf(buf,
            "{[[[[] -> [p2,q2,r2,s2,c2]] -> [p1, q1, r1, s1, c1]] -> [p0, q0, r0, s0, c0]] -> [p,q,s,r,c] : "
            "                           p = p2*%d + p1 and 0 <= p1 < %d and "
            "                           q = q2*%d + q1 and 0 <= q1 < %d and "
            "                           r = r2*%d + r1 and 0 <= r1 < %d and "
            "                           s = s2*%d + s1 and 0 <= s1 < %d and "
            "                           c = c2*%d + c1 and 0 <= c1 < %d }",
            P1, P1,
            Q1, Q1,
            R1, R1,
            S1, S1,
            C1, C1);
    tile_id_to_set_[0] = isl_map_read_from_str(context_, buf);

    //
    // Tile ID domains.
    //

    sprintf(buf,
            "{ [p1,q1,r1,s1,c1] : "
            "               0 <= p1 < %d and "
            "               0 <= q1 < %d and "
            "               0 <= r1 < %d and "
            "               0 <= s1 < %d and "
            "               0 <= c1 < %d }",
            P2, Q2, R2, S2, C2);
    tile_id_domains_[2] = isl_set_read_from_str(context_, buf);

    sprintf(buf,
            "{ [p1,q1,r1,s1,c1] : "
            "               0 <= p1 < %d and "
            "               0 <= q1 < %d and "
            "               0 <= r1 < %d and "
            "               0 <= s1 < %d and "
            "               0 <= c1 < %d }",
            P1, Q1, R1, S1, C1);
    tile_id_domains_[1] = isl_set_read_from_str(context_, buf);

    sprintf(buf,
            "{ [p1,q1,r1,s1,c1] : "
            "               0 <= p1 < %d and "
            "               0 <= q1 < %d and "
            "               0 <= r1 < %d and "
            "               0 <= s1 < %d and "
            "               0 <= c1 < %d }",
            1, 1, 1, 1, 1);
    tile_id_domains_[0] = isl_set_read_from_str(context_, buf);

    //
    // Skews.
    //
    //TODO: skew does not need bound?
    sprintf(buf,
            "{ [] -> SpaceTime_3[0]}" // Hardware spacetime is <y,x>.
            );
    skews_[3] = isl_map_read_from_str(context_, buf);


    sprintf(buf,
            "{ [p2,q2,r2,s2,c2] -> SpaceTime_2[0, p2*%d + q2] }",
            Q2);
    skews_[2] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [p1,q1,r1,s1,c1] -> SpaceTime_1[0, %d*p1 + %d*q1 + %d*c1 + %d*r1 + s1] }",
            S1*R1*C1*Q1, S1*R1*C1, S1*R1, S1);
    skews_[1] = isl_map_read_from_str(context, buf);

    sprintf(buf,
            "{ [p0,q0,r0,s0,c0] -> SpaceTime_0[] : p0 = 0 and q0 = 0 and r0 = 0 and s0=0 and c0=0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

map<string, MappingPtr> ConstructMappingCONV2D(isl_ctx* c,
        vector<int> t2, vector<int> t1) {
  shared_ptr<Mapping_CONV2D> g = make_shared<Mapping_CONV2D>(c, t2, t1);
  return {{"CONV2D0", g}};
}

shared_ptr<Binding> CreateBindingCONV2D() {

  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //
  //
  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["I"][3]["CONV2D0"] = {"GlobalBuffer", 0};
  b->memory_binding_["W"][3]["CONV2D0"]=  {"GlobalBuffer", 1};
  b->memory_binding_["O"][3]["CONV2D0"] = {"GlobalBuffer", 2};

  b->memory_binding_["I"][2]["CONV2D0"] = {"RegFile", 0};
  b->memory_binding_["W"][2]["CONV2D0"] = {"RegFile", 1};
  b->memory_binding_["O"][2]["CONV2D0"] = {"RegFile", 2};

  b->memory_binding_["I"][1]["CONV2D0"] = {"OperandA", 0};
  b->memory_binding_["W"][1]["CONV2D0"] = {"OperandB", 0};
  b->memory_binding_["O"][1]["CONV2D0"] = {"Result", 0};

  b->compute_binding_["CONV2D0"] = {"MAC", 0};
  return b;
}
