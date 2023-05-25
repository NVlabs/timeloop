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

class Mapping_CONV_1D: public Mapping
{
 private:
  // Parameters.
  int P1, P2, P3;
  int R1, R2, R3;

 public:
  Mapping_CONV_1D(isl_ctx* context,
                      vector<int>& ps,
                      vector<int>& rs
                      ) :
      Mapping(context),
      P1(ps.at(0)), P2(ps.at(1)), P3(ps.at(2)),
      R1(rs.at(0)), R2(rs.at(1)), R3(rs.at(2))
  {
    GenerateTileMaps(
        {{  P3, R3 },
        {  P2, R2 },
        {  P1, R1 }},
        {{  P2, R2 },
        {  P1, R1 }
        }
        );

    char buf[1024];

    //
    // Skews.
    //
    //
    sprintf(buf,
            "{ [] -> SpaceTime_3[0]}" // Hardware spacetime is <y,x>.
            );
    skews_[3] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [p2,r2] -> SpaceTime_2[p2] : r2 = 0 }" // Hardware spacetime is <y,x>.
            );
    skews_[2] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [p1,r1] -> SpaceTime_1[p1*%d + r1]: }", R2 // Hardware spacetime is <t>.
            );
    skews_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [p0,r0] -> SpaceTime_0[] : p0 = 0 and r0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

class Mapping_CONV_1D_3Level: public Mapping
{
 private:
  // Parameters.
  int P1, P2, P3;
  int R1, R2, R3;

 public:
  Mapping_CONV_1D_3Level(isl_ctx* context,
                      vector<int>& ps,
                      vector<int>& rs,
                      int lls, int lls_child_id
                      ) :
      Mapping(context),
      P1(ps.at(0)), P2(ps.at(1)), P3(ps.at(2)),
      R1(rs.at(0)), R2(rs.at(1)), R3(rs.at(2))
  {
    GenerateTileMaps(
            {{  P3, R3 }, {  P2, R2 },{  P1, R1 }},
            //FIXME: need a more general way to generate the tiling
            {{  P2 + (lls_child_id-1)*2*(R2!=1) , R2 }, {  P1, R1 }}
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
            "{ [p2,r2] -> SpaceTime_2[p2, %d] : r2 = 0 }", lls_child_id // Hardware spacetime is <y,x>.
            );
    skews_[2] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [p1,r1] -> SpaceTime_1[p1*%d + r1]: }", R2 // Hardware spacetime is <t>.
            );
    skews_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [p0,r0] -> SpaceTime_0[] : p0 = 0 and r0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

class Mapping_CONV_1D_2Level: public Mapping
{
 private:
  // Parameters.
  int P1, R1;

 public:
  Mapping_CONV_1D_2Level(isl_ctx* context,
                             int p1, int r1,
                             int lls, int lls_child_id
                            ) :
      Mapping(context),
      P1(p1), R1(r1)
  {
    char buf[1024];

    //
    // Tile ID -> Global coordinates.
    //
    // Fixme: add k0, p0, r0
    sprintf(buf,
            "{ [[]->[p1,r1]] -> [p,r] : "
            "                          p1 = p and 0 <= p1 < %d and "
            "                          r1 = r and 0 <= r1 < %d }",
            P1, R1);
    tile_id_to_set_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... would be great if we didn't need this.
    // Fixme: set constraints on k0,p0,r0
    sprintf(buf,
            "{ [[[]->[p1,r1]] -> [p0,r0]] -> [p,r] : "
            "                                          p = p1 and 0 <= p1 < %d and "
            "                                          r = r1 and 0 <= r1 < %d }",
            P1, R1);
    tile_id_to_set_[0] = isl_map_read_from_str(context_, buf);

    //
    // Tile ID domains.
    //
    sprintf(buf,
            "{ [p1,r1] : "
            "               0 <= p1 < %d and "
            "               0 <= r1 < %d }",
            P1, R1);
    tile_id_domains_[1] = isl_set_read_from_str(context_, buf);

    // Ugh... would be great if we didn't need this.
    sprintf(buf,
            "{ [p0,r0] : "
            "               0 <= p0 < %d and "
            "               0 <= r0 < %d }",
            1, 1);
    tile_id_domains_[0] = isl_set_read_from_str(context_, buf);

    //
    // Skews.
    //
    if (lls > 1) {
      sprintf(buf,
              "{ []->SpaceTime_2[%d] }", // Hardware spacetime is <t>.
              lls_child_id);
      skews_[2] = isl_map_read_from_str(context, buf);
    } else {
      sprintf(buf, "{ [] -> SpaceTime_2[0] }");
      skews_[2] = isl_map_read_from_str(context, buf);
    }
      sprintf(buf,
              "{ [p1,r1] -> SpaceTime_1[ p1*%d + r1] }", // Hardware spacetime is <t>.
              R1);
      skews_[1] = isl_map_read_from_str(context, buf);


    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [p0,r0] -> SpaceTime_0[] : p0 = 0 and r0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);

  }
};

std::map<std::string, MappingPtr> ConstructMapping2Level(isl_ctx* c, int P, int R);

std::map<std::string, MappingPtr> ConstructMappingMulti2Level(isl_ctx* c, int P, int R);

map<string, MappingPtr> ConstructMapping(isl_ctx* context,
        vector<int>  p_tile_factors,
        vector<int>  r_tile_factors);
map<string, MappingPtr> ConstructMappingMulti(isl_ctx* context,
        vector<int>  p_tile_factors_0,
        vector<int>  r_tile_factors_0,
        vector<int>  p_tile_factors_1,
        vector<int>  r_tile_factors_1);

shared_ptr<Binding> CreateBinding();
shared_ptr<Binding> CreateBindingSeq();
shared_ptr<Binding> Create2StageBinding();
shared_ptr<Binding> Create2Stage2LevelBindingSeq();
shared_ptr<Binding> Create2Stage3LevelBindingSeq();
