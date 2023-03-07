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

class Mapping_MV_4L: public Mapping
{
 private:
  // Parameters.
  int M1, M2, M3, M4;
  int K1, K2, K3, K4;

 public:
  Mapping_MV_4L(isl_ctx* context,
                      vector<int>& ms,
                      vector<int>& ks,
                      int lls, int child_id) :
      Mapping(context),
      M1(ms.at(0)), M2(ms.at(1)), M3(ms.at(2)), M4(ms.at(3)),
      K1(ks.at(0)), K2(ks.at(1)), K3(ks.at(2)), K4(ks.at(3))
  {
    GenerateTileMaps(
            {{M4, K4}, { M3, K3 }, {  M2, K2 }, {  M1, K1 }},
            {{M3, K3}, { M2, K2 }, {  M1, K1 }}
            );

    char buf[1024];

    //
    // Skews.
    //

    //Root level there is no tiling factor, so LHS is an dummy set
    sprintf(buf, "{ [] -> SpaceTime_4[0] }");
    skews_[4] = isl_map_read_from_str(context_, buf);

    if (lls == 3) {
      sprintf(buf, "{ [m3,k3] -> SpaceTime_3[0, m3*%d + k3, %d ]}", K4, child_id);
      skews_[3] = isl_map_read_from_str(context_, buf);

      sprintf(buf, "{ [m2,k2] -> SpaceTime_2[0, m2*%d + k2] }", K3);
      skews_[2] = isl_map_read_from_str(context_, buf);
    } else if(lls == 2) {
      sprintf(buf, "{ [m3,k3] -> SpaceTime_3[0, m3*%d + k3] }", K4);
      skews_[3] = isl_map_read_from_str(context_, buf);

      sprintf(buf, "{ [m2,k2] -> SpaceTime_2[0, m2*%d + k2, %d ]}", K3, child_id);
      skews_[2] = isl_map_read_from_str(context_, buf);

    } else {
      TRACE(1) << "NOT IMPLEMENT THIS FUSE LEVEL" << endl;
      exit(0);
    }

    sprintf(buf,
            "{ [m1,k1] -> SpaceTime_1[0,m1*%d + k1]: }", K2 // Hardware spacetime is <t>.
            );
    skews_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [m0,k0] -> SpaceTime_0[] : m0 = 0 and k0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

class Mapping_MV: public Mapping
{
 private:
  // Parameters.
  int M1, M2;
  int K1, K2;

 public:
  Mapping_MV(isl_ctx* context, int m1, int m2, int k1, int k2) :
      Mapping(context),
      M1(m1), M2(m2), K1(k1), K2(k2)
  {
    char buf[1024];

    //
    // Tile ID -> Global coordinates.
    //

    // Fixme: add m0,n0,k0
    sprintf(buf,
            "{[[] -> [m2,k2]] -> [m,k] : exists m1,k1 : m = m2*%d + m1 and 0 <= m1 < %d and "
            "                                            k = k2*%d + k1 and 0 <= k1 < %d }",
            M1, M1,
            K1, K1);
    tile_id_to_set_[2] = isl_map_read_from_str(context_, buf);

    // Fixme: add m0,n0,k0
    sprintf(buf,
            "{ [[[] -> [m2,k2]] -> [m1,k1]] -> [m,k] : m = m2*%d + m1 and 0 <= m1 < %d and "
            "                                          k = k2*%d + k1 and 0 <= k1 < %d }",
            M1, M1,
            K1, K1);
    tile_id_to_set_[1] = isl_map_read_from_str(context_, buf);

    // Fixme: add constraints on m0,n0,k0
    sprintf(buf,
            "{ [[[[] -> [m2,k2] ] -> [m1,k1]] -> [m0,k0]]-> [m,k] : m = m2*%d + m1 and 0 <= m1 < %d and "
            "                                                         k = k2*%d + k1 and 0 <= k1 < %d }",
            M1, M1,
            K1, K1);
    tile_id_to_set_[0] = isl_map_read_from_str(context_, buf);

    //
    // Tile ID domains.
    //

    sprintf(buf,
            "{ [m2,k2] : 0 <= m2 < %d and "
            "               0 <= k2 < %d }",
            M2, K2);
    tile_id_domains_[2] = isl_set_read_from_str(context_, buf);

    sprintf(buf,
            "{ [m1,k1] : 0 <= m1 < %d and "
            "               0 <= k1 < %d }",
            M1, K1);
    tile_id_domains_[1] = isl_set_read_from_str(context_, buf);

    sprintf(buf,
            "{ [m0,k0] : 0 <= m0 < %d and "
            "               0 <= k0 < %d }",
            1, 1);
    tile_id_domains_[0] = isl_set_read_from_str(context_, buf);

    //
    // Skews.
    //
    sprintf(buf,
            "{ [] -> SpaceTime_3[0]}" // Hardware spacetime is <y,x>.
            );
    skews_[3] = isl_map_read_from_str(context_, buf);


    sprintf(buf,
            "{ [m2,k2] -> SpaceTime_2[m2, k2] }");
    skews_[2] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [m1,k1] -> SpaceTime_1[0, %d*m1 + k1] : 0 <= k1 < %d and 0 <= m1 < %d }",
            K1, K1, M1);
    skews_[1] = isl_map_read_from_str(context, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [m0,k0] -> SpaceTime_0[] : m0 = 0 and k0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

class Mapping_MV_Spatial: public Mapping
{
 private:
  // Parameters.
  int M1, M2, M3;
  int K1, K2, K3;

 public:
  Mapping_MV_Spatial(isl_ctx* context,
                      vector<int>& ms,
                      vector<int>& ks,
                      bool k_unroll
                      ) :
      Mapping(context),
      M1(ms.at(0)), M2(ms.at(1)), M3(ms.at(2)),
      K1(ks.at(0)), K2(ks.at(1)), K3(ks.at(2))
  {
    GenerateTileMaps(
            {{ M3, K3 }, {  M2, K2 }, {  M1, K1 }},
            {{ M2, K2 }, {  M1, K1 }}
            );

    char buf[1024];

    //
    // Skews.
    //

    //Root level there is no tiling factor, so LHS is an dummy set
    sprintf(buf, "{ [] -> SpaceTime_3[0] }");
    skews_[3] = isl_map_read_from_str(context_, buf);

    if (k_unroll) {
      sprintf(buf,
              "{ [m2,k2] -> SpaceTime_2[k2, m2] }"// Hardware spacetime is <y,x>.
              );
      skews_[2] = isl_map_read_from_str(context_, buf);
    } else {
      sprintf(buf,
              "{ [m2,k2] -> SpaceTime_2[m2, k2] }"// Hardware spacetime is <y,x>.
              );
      skews_[2] = isl_map_read_from_str(context_, buf);

    }


    sprintf(buf,
            "{ [m1,k1] -> SpaceTime_1[0,m1*%d + k1]: }", K2 // Hardware spacetime is <t>.
            );
    skews_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [m0,k0] -> SpaceTime_0[] : m0 = 0 and k0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

class Mapping_MV_Fusion_GB: public Mapping
{
 private:
  // Parameters.
  int M1, M2, M3;
  int K1, K2, K3;

 public:
  Mapping_MV_Fusion_GB(isl_ctx* context,
                      vector<int>& ms,
                      vector<int>& ks,
                      int lls, int lls_child_id
                      ) :
      Mapping(context),
      M1(ms.at(0)), M2(ms.at(1)), M3(ms.at(2)),
      K1(ks.at(0)), K2(ks.at(1)), K3(ks.at(2))
  {
    GenerateTileMaps(
            {{ M3, K3 }, {  M2, K2 }, {  M1, K1 }},
            {{ M2, K2 }, {  M1, K1 }}
            );

    char buf[1024];

    //
    // Skews.
    //
    assert(lls == 3);

    //Root level there is no tiling factor, so LHS is an dummy set
    sprintf(buf, "{ [] -> SpaceTime_3[%d] }", lls_child_id );
    skews_[3] = isl_map_read_from_str(context_, buf);

      sprintf(buf,
              "{ [m2,k2] -> SpaceTime_2[0,m2*%d+k2] }", K3// Hardware spacetime is <y,x>.
              );
      skews_[2] = isl_map_read_from_str(context_, buf);


    sprintf(buf,
            "{ [m1,k1] -> SpaceTime_1[0,m1*%d + k1]: }", K2 // Hardware spacetime is <t>.
            );
    skews_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [m0,k0] -> SpaceTime_0[] : m0 = 0 and k0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

class Mapping_MV_Fusion_Spatial: public Mapping
{
 private:
  // Parameters.
  int M1, M2, M3;
  int K1, K2, K3;

 public:
  Mapping_MV_Fusion_Spatial(isl_ctx* context,
                      vector<int>& ms,
                      vector<int>& ks,
                      bool k_unroll,
                      int lls, int lls_child_id
                      ) :
      Mapping(context),
      M1(ms.at(0)), M2(ms.at(1)), M3(ms.at(2)),
      K1(ks.at(0)), K2(ks.at(1)), K3(ks.at(2))
  {
    GenerateTileMaps(
            {{ M3, K3 }, {  M2, K2 }, {  M1, K1 }},
            {{ M2, K2 }, {  M1, K1 }}
            );

    char buf[1024];

    //
    // Skews.
    //
    assert(lls == 2);

    //Root level there is no tiling factor, so LHS is an dummy set
    sprintf(buf, "{ [] -> SpaceTime_3[0] }");
    skews_[3] = isl_map_read_from_str(context_, buf);

    if (k_unroll) {
      sprintf(buf,
              "{ [m2,k2] -> SpaceTime_2[, m2,%d] }", lls_child_id // Hardware spacetime is <y,x>.
              );
      skews_[2] = isl_map_read_from_str(context_, buf);
    } else {
      sprintf(buf,
              "{ [m2,k2] -> SpaceTime_2[m2, k2,%d] }", lls_child_id // Hardware spacetime is <y,x>.
              );
      skews_[2] = isl_map_read_from_str(context_, buf);

    }


    sprintf(buf,
            "{ [m1,k1] -> SpaceTime_1[0,m1*%d + k1]: }", K2 // Hardware spacetime is <t>.
            );
    skews_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [m0,k0] -> SpaceTime_0[] : m0 = 0 and k0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

class Mapping_MV_Fusion_Seq_Spatial: public Mapping
{
 private:
  // Parameters.
  int M1, M2, M3;
  int K1, K2, K3;

 public:
  Mapping_MV_Fusion_Seq_Spatial(isl_ctx* context,
                      vector<int>& ms,
                      vector<int>& ks,
                      int lls, int lls_child_id
                      ) :
      Mapping(context),
      M1(ms.at(0)), M2(ms.at(1)), M3(ms.at(2)),
      K1(ks.at(0)), K2(ks.at(1)), K3(ks.at(2))
  {
    GenerateTileMaps(
            {{ M3, K3 }, {  M2, K2 }, {  M1, K1 }},
            {{ M2, K2 }, {  M1, K1 }}
            );

    char buf[1024];

    //
    // Skews.
    //
    assert(lls == 2);

    //Root level there is no tiling factor, so LHS is an dummy set
    sprintf(buf, "{ [] -> SpaceTime_3[0] }");
    skews_[3] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
              "{ [m2,k2] -> SpaceTime_2[0, m2*%d+k2,%d] }", K3, lls_child_id // Hardware spacetime is <y,x>.
              );
    skews_[2] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [m1,k1] -> SpaceTime_1[m1%%2, floord(m1,2)*%d + k1]: }", K2 // Hardware spacetime is <t>.
            );
    skews_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [m0,k0] -> SpaceTime_0[] : m0 = 0 and k0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

class Mapping_MV_Fusion: public Mapping
{
 private:
  // Parameters.
  int M1, M2, M3;
  int K1, K2, K3;

 public:
  Mapping_MV_Fusion(isl_ctx* context,
                      vector<int>& ms,
                      vector<int>& ks,
                      int lls, int lls_child_id
                      ) :
      Mapping(context),
      M1(ms.at(0)), M2(ms.at(1)), M3(ms.at(2)),
      K1(ks.at(0)), K2(ks.at(1)), K3(ks.at(2))
  {
    GenerateTileMaps(
            {{ M3, K3 }, {  M2, K2 }, {  M1, K1 }},
            {{ M2, K2 }, {  M1, K1 }}
            );

    char buf[1024];

    //
    // Skews.
    //
    assert(lls == 2);

    //Root level there is no tiling factor, so LHS is an dummy set
    sprintf(buf, "{ [] -> SpaceTime_3[0] }");
    skews_[3] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [m2,k2] -> SpaceTime_2[0,m2*%d+k2,%d] }", K3, lls_child_id // Hardware spacetime is <y,x>.
            );
    skews_[2] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [m1,k1] -> SpaceTime_1[0,m1*%d + k1]: }", K2 // Hardware spacetime is <t>.
            );
    skews_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [m0,k0] -> SpaceTime_0[] : m0 = 0 and k0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};


map<string, MappingPtr> ConstructMappingMV(isl_ctx* c,
        int M1, int M2, int K1, int K2) {
  shared_ptr<Mapping_MV> g = make_shared<Mapping_MV>(c, M1, M2, K1, K2);
  return {{"MV0", g}};
}

map<string, MappingPtr> ConstructMappingMV4L(isl_ctx* c,
        vector<int> tm, vector<int> tk) {
  shared_ptr<Mapping_MV_4L> g = make_shared<Mapping_MV_4L>(c, tm, tk, 3, 0);
  return {{"MV0", g}};
}

map<string, MappingPtr> ConstructMappingMV2Stages4L(isl_ctx* c,
        vector<int> tm0, vector<int> tk0,
        vector<int> tm1, vector<int> tk1) {
    assert(tm0.size() == 4);
    assert(tk0.size() == 4);
    assert(tm1.size() == 4);
    assert(tk1.size() == 4);
  shared_ptr<Mapping_MV_4L> g0 = make_shared<Mapping_MV_4L>(c, tm0, tk0, 3, 0);
  shared_ptr<Mapping_MV_4L> g1 = make_shared<Mapping_MV_4L>(c, tm1, tk1, 3, 1);
  return {{"MV0", g0}, {"MV1", g1}};
}

//This mapping fuse at L1 buffer
map<string, MappingPtr> ConstructMappingMV2Stages4LFiner(isl_ctx* c,
        vector<int> tm0, vector<int> tk0,
        vector<int> tm1, vector<int> tk1) {
    assert(tm0.size() == 4);
    assert(tk0.size() == 4);
    assert(tm1.size() == 4);
    assert(tk1.size() == 4);
  shared_ptr<Mapping_MV_4L> g0 = make_shared<Mapping_MV_4L>(c, tm0, tk0, 2, 0);
  shared_ptr<Mapping_MV_4L> g1 = make_shared<Mapping_MV_4L>(c, tm1, tk1, 2, 1);
  return {{"MV0", g0}, {"MV1", g1}};
}


map<string, MappingPtr> ConstructMappingMVParallel(isl_ctx* c,
        vector<int> tm, vector<int> tk) {
  shared_ptr<Mapping_MV_Spatial> g = make_shared<Mapping_MV_Spatial>(c, tm, tk, true);
  return {{"MV0", g}};
}

map<string, MappingPtr> ConstructMappingMV2Stages(isl_ctx* c,
        vector<int> m0_tile_factors, vector<int> k0_tile_factors,
        vector<int> m1_tile_factors, vector<int> k1_tile_factors) {
    map<string, MappingPtr> app_mapping;
    assert(m0_tile_factors.size() == 3);
    assert(m1_tile_factors.size() == 3);
    assert(k0_tile_factors.size() == 3);
    assert(k1_tile_factors.size() == 3);
    auto mapping_ispace_0 =
        make_shared<Mapping_MV_Fusion>(c, m0_tile_factors, k0_tile_factors, 2, 0);
    app_mapping.insert({"MV0", mapping_ispace_0});
    auto mapping_ispace_1 =
        make_shared<Mapping_MV_Fusion>(c, m1_tile_factors, k1_tile_factors, 2, 1);
    app_mapping.insert({"MV1", mapping_ispace_1});
    return app_mapping;
}

map<string, MappingPtr> ConstructMappingMV2StagesGB(isl_ctx* c,
        vector<int> m0_tile_factors, vector<int> k0_tile_factors,
        vector<int> m1_tile_factors, vector<int> k1_tile_factors) {
    map<string, MappingPtr> app_mapping;
    assert(m0_tile_factors.size() == 3);
    assert(m1_tile_factors.size() == 3);
    assert(k0_tile_factors.size() == 3);
    assert(k1_tile_factors.size() == 3);
    auto mapping_ispace_0 =
        make_shared<Mapping_MV_Fusion_GB>(c, m0_tile_factors, k0_tile_factors, 3, 0);
    app_mapping.insert({"MV0", mapping_ispace_0});
    auto mapping_ispace_1 =
        make_shared<Mapping_MV_Fusion_GB>(c, m1_tile_factors, k1_tile_factors, 3, 1);
    app_mapping.insert({"MV1", mapping_ispace_1});
    return app_mapping;
}

map<string, MappingPtr> ConstructMappingMV2StagesParallel(isl_ctx* c,
        vector<int> m0_tile_factors, vector<int> k0_tile_factors,
        vector<int> m1_tile_factors, vector<int> k1_tile_factors) {
    map<string, MappingPtr> app_mapping;
    assert(m0_tile_factors.size() == 3);
    assert(m1_tile_factors.size() == 3);
    assert(k0_tile_factors.size() == 3);
    assert(k1_tile_factors.size() == 3);
    auto mapping_ispace_0 =
        make_shared<Mapping_MV_Fusion_Spatial>(c,
                m0_tile_factors, k0_tile_factors,
                false/*parallel along m*/, 2, 0);
    app_mapping.insert({"MV0", mapping_ispace_0});
    auto mapping_ispace_1 =
        make_shared<Mapping_MV_Fusion_Spatial>(c,
                m1_tile_factors, k1_tile_factors,
                true/*parallel along k*/, 2, 1);
    app_mapping.insert({"MV1", mapping_ispace_1});
    return app_mapping;
}

map<string, MappingPtr> ConstructMappingMV2StagesSeq(isl_ctx* c,
        vector<int> m0_tile_factors, vector<int> k0_tile_factors,
        vector<int> m1_tile_factors, vector<int> k1_tile_factors) {
    map<string, MappingPtr> app_mapping;
    assert(m0_tile_factors.size() == 3);
    assert(m1_tile_factors.size() == 3);
    assert(k0_tile_factors.size() == 3);
    assert(k1_tile_factors.size() == 3);
    auto mapping_ispace_0 =
        make_shared<Mapping_MV_Fusion_Seq_Spatial>(c,
                m0_tile_factors, k0_tile_factors, 2, 0);
    app_mapping.insert({"MV0", mapping_ispace_0});
    auto mapping_ispace_1 =
        make_shared<Mapping_MV_Fusion_Seq_Spatial>(c,
                m1_tile_factors, k1_tile_factors, 2, 1);
    app_mapping.insert({"MV1", mapping_ispace_1});
    return app_mapping;
}


shared_ptr<Binding> CreateBindingMV() {

  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //
  //
  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["A"][3]["MV0"] = {"GlobalBuffer", 0};
  b->memory_binding_["B"][3]["MV0"]=  {"GlobalBuffer", 1};
  b->memory_binding_["Z"][3]["MV0"] = {"GlobalBuffer", 2};

  b->memory_binding_["A"][2]["MV0"] = {"RegFile", 0};
  b->memory_binding_["B"][2]["MV0"] = {"RegFile", 1};
  b->memory_binding_["Z"][2]["MV0"] = {"RegFile", 2};

  b->memory_binding_["A"][1]["MV0"] = {"OperandA", 0};
  b->memory_binding_["B"][1]["MV0"] = {"OperandB", 0};
  b->memory_binding_["Z"][1]["MV0"] = {"Result", 0};

  b->compute_binding_["MV0"] = {"MAC", 0};
  return b;
}

shared_ptr<Binding> CreateBindingMV4L() {

  shared_ptr<Binding> b = make_shared<Binding>();

  b->memory_binding_["A"][4]["MV0"] = {"DRAM", 0};
  b->memory_binding_["B"][4]["MV0"]=  {"DRAM", 1};
  b->memory_binding_["Z"][4]["MV0"] = {"DRAM", 2};

  b->memory_binding_["A"][3]["MV0"] = {"GlobalBuffer", 0};
  b->memory_binding_["B"][3]["MV0"]=  {"GlobalBuffer", 1};
  b->memory_binding_["Z"][3]["MV0"] = {"GlobalBuffer", 2};

  b->memory_binding_["A"][2]["MV0"] = {"RegFile", 0};
  b->memory_binding_["B"][2]["MV0"] = {"RegFile", 1};
  b->memory_binding_["Z"][2]["MV0"] = {"RegFile", 2};

  b->memory_binding_["A"][1]["MV0"] = {"OperandA", 0};
  b->memory_binding_["B"][1]["MV0"] = {"OperandB", 0};
  b->memory_binding_["Z"][1]["MV0"] = {"Result", 0};

  b->compute_binding_["MV0"] = {"MAC", 0};
  return b;
}

shared_ptr<Binding> CreateBindingMV2Stages() {

  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //
  //
  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["A0"][3]["MV0"] = {"GlobalBuffer", 0};
  b->memory_binding_["B0"][3]["MV0"]=  {"GlobalBuffer", 1};
  b->memory_binding_["B1"][3]["MV1"]=  {"GlobalBuffer", 2};
  b->memory_binding_["Z"][3]["MV1"] = {"GlobalBuffer", 3};

  b->memory_binding_["A0"][2]["MV0"] = {"InRegFile", 0};
  b->memory_binding_["B0"][2]["MV0"] = {"InRegFile", 1};
  b->memory_binding_["B1"][2]["MV1"] = {"InRegFile", 2};
  b->memory_binding_["Z"][2]["MV1"] = {"OutRegFile", 0};

  //FIXME: intermediate tensor appeared at both producers and consumers?
  b->memory_binding_["A1"][2]["MV0"] = {"TmpRegFile", 0};
  b->memory_binding_["A1"][2]["MV1"] = {"TmpRegFile", 0};

  b->memory_binding_["A0"][1]["MV0"] = {"OperandA", 0};
  b->memory_binding_["B0"][1]["MV0"] = {"OperandB", 0};
  b->memory_binding_["A1"][1]["MV0"] = {"Result", 0};
  b->memory_binding_["A1"][1]["MV1"] = {"OperandA", 1};
  b->memory_binding_["B1"][1]["MV1"] = {"OperandB", 1};
  b->memory_binding_["Z"][1]["MV1"] = {"Result", 1};

  b->compute_binding_["MV0"] = {"MAC", 0};
  b->compute_binding_["MV1"] = {"MAC", 1};
  return b;
}

shared_ptr<Binding> CreateBindingMV2Stages4L() {

  shared_ptr<Binding> b = make_shared<Binding>();

  b->memory_binding_["A0"][4]["MV0"] = {"DRAM", 0};
  b->memory_binding_["B0"][4]["MV0"]=  {"DRAM", 1};
  b->memory_binding_["B1"][4]["MV1"]=  {"DRAM", 2};
  b->memory_binding_["Z"][4]["MV1"] = {"DRAM", 3};

  b->memory_binding_["A0"][3]["MV0"] = {"GlobalBuffer", 0};
  b->memory_binding_["B0"][3]["MV0"]=  {"GlobalBuffer", 1};
  b->memory_binding_["A1"][3]["MV0"] = {"GlobalBuffer", 2};

  b->memory_binding_["A1"][3]["MV1"] = {"GlobalBuffer", 2};
  b->memory_binding_["B1"][3]["MV1"]=  {"GlobalBuffer", 3};
  b->memory_binding_["Z"][3]["MV1"] = {"GlobalBuffer", 4};

  b->memory_binding_["A0"][2]["MV0"] = {"RegFile", 0};
  b->memory_binding_["B0"][2]["MV0"] = {"RegFile", 1};
  b->memory_binding_["A1"][2]["MV0"] = {"RegFile", 4};

  b->memory_binding_["B1"][2]["MV1"] = {"RegFile", 2};
  b->memory_binding_["Z"][2]["MV1"] = {"RegFile", 3};
  b->memory_binding_["A1"][2]["MV1"] = {"RegFile", 5};


  b->memory_binding_["A0"][1]["MV0"] = {"OperandA", 0};
  b->memory_binding_["B0"][1]["MV0"] = {"OperandB", 0};
  b->memory_binding_["A1"][1]["MV0"] = {"Result", 0};

  b->memory_binding_["A1"][1]["MV1"] = {"OperandA", 1};
  b->memory_binding_["B1"][1]["MV1"] = {"OperandB", 1};
  b->memory_binding_["Z"][1]["MV1"] = {"Result", 1};

  b->compute_binding_["MV0"] = {"MAC", 0};
  b->compute_binding_["MV1"] = {"MAC", 1};
  return b;
}

shared_ptr<Binding> CreateBindingMV2Stages4LFiner() {

  shared_ptr<Binding> b = make_shared<Binding>();

  b->memory_binding_["A0"][4]["MV0"] = {"DRAM", 0};
  b->memory_binding_["B0"][4]["MV0"]=  {"DRAM", 1};
  b->memory_binding_["B1"][4]["MV1"]=  {"DRAM", 2};
  b->memory_binding_["Z"][4]["MV1"] = {"DRAM", 3};

  b->memory_binding_["A0"][3]["MV0"] = {"GlobalBuffer", 0};
  b->memory_binding_["B0"][3]["MV0"]=  {"GlobalBuffer", 1};
  b->memory_binding_["B1"][3]["MV1"]=  {"GlobalBuffer", 2};
  b->memory_binding_["Z"][3]["MV1"] = {"GlobalBuffer", 3};

  b->memory_binding_["A0"][2]["MV0"] = {"RegFile", 0};
  b->memory_binding_["B0"][2]["MV0"] = {"RegFile", 1};
  b->memory_binding_["A1"][2]["MV0"] = {"RegFile", 2};

  b->memory_binding_["A1"][2]["MV1"] = {"RegFile", 2};
  b->memory_binding_["B1"][2]["MV1"] = {"RegFile", 3};
  b->memory_binding_["Z"][2]["MV1"] = {"RegFile", 4};


  b->memory_binding_["A0"][1]["MV0"] = {"OperandA", 0};
  b->memory_binding_["B0"][1]["MV0"] = {"OperandB", 0};
  b->memory_binding_["A1"][1]["MV0"] = {"Result", 0};

  b->memory_binding_["A1"][1]["MV1"] = {"OperandA", 1};
  b->memory_binding_["B1"][1]["MV1"] = {"OperandB", 1};
  b->memory_binding_["Z"][1]["MV1"] = {"Result", 1};

  b->compute_binding_["MV0"] = {"MAC", 0};
  b->compute_binding_["MV1"] = {"MAC", 1};
  return b;
}

shared_ptr<Binding> CreateBindingMV2StagesGB() {

  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //
  //
  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["A0"][3]["MV0"] = {"GlobalBuffer", 0};
  b->memory_binding_["B0"][3]["MV0"]=  {"GlobalBuffer", 1};
  b->memory_binding_["A1"][3]["MV0"] = {"GlobalBuffer", 2};

  b->memory_binding_["A1"][3]["MV1"] = {"GlobalBuffer", 2};
  b->memory_binding_["B1"][3]["MV1"]=  {"GlobalBuffer", 3};
  b->memory_binding_["Z"][3]["MV1"] = {"GlobalBuffer", 4};

  b->memory_binding_["A0"][2]["MV0"] = {"RegFile", 0};
  b->memory_binding_["B0"][2]["MV0"] = {"RegFile", 1};
  b->memory_binding_["A1"][2]["MV0"] = {"RegFile", 4};

  b->memory_binding_["B1"][2]["MV1"] = {"RegFile", 2};
  b->memory_binding_["Z"][2]["MV1"] = {"RegFile", 3};
  b->memory_binding_["A1"][2]["MV1"] = {"RegFile", 5};


  b->memory_binding_["A0"][1]["MV0"] = {"OperandA", 0};
  b->memory_binding_["B0"][1]["MV0"] = {"OperandB", 0};
  b->memory_binding_["A1"][1]["MV0"] = {"Result", 0};

  b->memory_binding_["A1"][1]["MV1"] = {"OperandA", 1};
  b->memory_binding_["B1"][1]["MV1"] = {"OperandB", 1};
  b->memory_binding_["Z"][1]["MV1"] = {"Result", 1};

  b->compute_binding_["MV0"] = {"MAC", 0};
  b->compute_binding_["MV1"] = {"MAC", 1};
  return b;
}

shared_ptr<Binding> CreateBindingMV2StagesSeq() {

  //
  // Binding of data-spaces to hardware instances. Note that hardware instances are
  // indexed by *hardware* levels and not *tiling* levels (which are betweeen
  // hardware levels).
  //
  //
  shared_ptr<Binding> b = make_shared<Binding>();
  b->memory_binding_["A0"][3]["MV0"] = {"GlobalBuffer", 0};
  b->memory_binding_["B0"][3]["MV0"]=  {"GlobalBuffer", 1};
  b->memory_binding_["B1"][3]["MV1"]=  {"GlobalBuffer", 2};
  b->memory_binding_["Z"][3]["MV1"] = {"GlobalBuffer", 0};

  b->memory_binding_["A0"][2]["MV0"] = {"RegFile", 0};
  b->memory_binding_["B0"][2]["MV0"] = {"RegFile", 1};
  b->memory_binding_["B1"][2]["MV1"] = {"RegFile", 2};
  b->memory_binding_["Z"][2]["MV1"] = {"RegFile", 0};

  //FIXME: intermediate tensor appeared at both producers and consumers?
  b->memory_binding_["A1"][2]["MV0"] = {"RegFile", 0};
  b->memory_binding_["A1"][2]["MV1"] = {"RegFile", 0};

  b->memory_binding_["A0"][1]["MV0"] = {"OperandA", 0};
  b->memory_binding_["B0"][1]["MV0"] = {"OperandB", 0};
  b->memory_binding_["A1"][1]["MV0"] = {"Result", 0};
  b->memory_binding_["A1"][1]["MV1"] = {"OperandA", 0};
  b->memory_binding_["B1"][1]["MV1"] = {"OperandB", 0};
  b->memory_binding_["Z"][1]["MV1"] = {"Result", 0};

  b->compute_binding_["MV0"] = {"MAC", 0};
  b->compute_binding_["MV1"] = {"MAC", 0};
  return b;
}
