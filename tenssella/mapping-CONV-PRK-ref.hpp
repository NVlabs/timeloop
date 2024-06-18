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

class Mapping_CONV_PRK_Reference : public Mapping
{
 private:
  // Parameters.
  int K1, P1, R1;

 public:
  Mapping_CONV_PRK_Reference(isl_ctx* context,
                             int k1, int p1, int r1
                            ) :
      Mapping(context),
      K1(k1), P1(p1), R1(r1)
  {
    char buf[1024];

    //
    // Tile ID -> Global coordinates.
    //
    // Fixme: add k0, p0, r0
    sprintf(buf, 
            "{ [k1,p1,r1] -> [k,p,r] : k1 = k and 0 <= k1 < %d and "
            "                          p1 = p and 0 <= p1 < %d and "
            "                          r1 = r and 0 <= r1 < %d }",
            K1, P1, R1);
    tile_id_to_set_[1] = isl_map_read_from_str(context_, buf);

    // Ugh... would be great if we didn't need this.
    // Fixme: set constraints on k0,p0,r0
    sprintf(buf, 
            "{ [[k1,p1,r1] -> [k0,p0,r0]] -> [k,p,r] : k = k1 and 0 <= k1 < %d and "
            "                                          p = p1 and 0 <= p1 < %d and "
            "                                          r = r1 and 0 <= r1 < %d }",
            K1, P1, R1);
    tile_id_to_set_[0] = isl_map_read_from_str(context_, buf);

    //
    // Tile ID domains.
    //
    sprintf(buf,
            "{ [k1,p1,r1] : 0 <= k1 < %d and "
            "               0 <= p1 < %d and "
            "               0 <= r1 < %d }",
            K1, P1, R1);
    tile_id_domains_[1] = isl_set_read_from_str(context_, buf);

    // Ugh... would be great if we didn't need this.
    sprintf(buf,
            "{ [k0,p0,r0] : 0 <= k0 < %d and "
            "               0 <= p0 < %d and "
            "               0 <= r0 < %d }",
            1, 1, 1);
    tile_id_domains_[0] = isl_set_read_from_str(context_, buf);

    //
    // Skews.
    //
    sprintf(buf,
            "{ [k1,p1,r1] -> SpaceTime_1[k1*%d*%d + p1*%d + r1] }", // Hardware spacetime is <t>.
            P1, R1, R1);
    skews_[1] = isl_map_read_from_str(context, buf);

    // Ugh... what does this even mean?
    sprintf(buf,
            "{ [k0,p0,r0] -> SpaceTime_0[] : k0 = 0 and p0 = 0 and r0 = 0 }" // Hardware spacetime is <>.
            );
    skews_[0] = isl_map_read_from_str(context, buf);    

    //
    // Binding of data-spaces to hardware instances. Note that hardware instances are
    // indexed by *hardware* levels and not *tiling* levels (which are betweeen
    // hardware levels).
    //
    binding_[2]["Weights"] = { "Memory", 0 };
    binding_[2]["Inputs"] = { "Memory", 1 };
    binding_[2]["Outputs"] = { "Memory", 2 };

    binding_[1]["Weights"] = { "OperandA", 0 };
    binding_[1]["Inputs"] = { "OperandB", 0 };
    binding_[1]["Outputs"] = { "Result", 0 };

    binding_[0]["Multiply"] = { "ALU", 0 };
  }
};
