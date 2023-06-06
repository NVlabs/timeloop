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

class Mapping_CONV_PRK_4L : public Mapping
{
 private:
  // Parameters.
  int K1, K2, K3;
  int P1, P2, P3;
  int R1, R2, R3;

 public:
  Mapping_CONV_PRK_4L(isl_ctx* context,
                      int k1, int k2, int k3,
                      int p1, int p2, int p3,
                      int r1, int r2, int r3
                      ) :
      Mapping(context),
      K1(k1), K2(k2), K3(k3),
      P1(p1), P2(p2), P3(p3),
      R1(r1), R2(r2), R3(r3)
  {
    GenerateTileMaps({
        { K3, P3, R3 },
        { K2, P2, R2 },
        { K1, P1, R1 }
      });

    char buf[1024];

    //
    // Skews.
    //
    sprintf(buf,
            "{ [k3,p3,r3] -> SpaceTime_3[r3,p3] : k3 = 0 }" // Hardware spacetime is <y,x>.
            );
    skews_[3] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [k2,p2,r2] -> SpaceTime_2[k2] : p2 = 0 and r2 = 0 }" // Hardware spacetime is <t>.
            );
    skews_[2] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [k1,p1,r1] -> SpaceTime_1[] : k1 = 0 and p1 = 0 and r1 = 0 }" // Hardware spacetime is <>.
            );
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
    binding_[4]["Weights"] = { "DRAM", 0 };
    binding_[4]["Inputs"] = { "DRAM", 1 };
    binding_[4]["Outputs"] = { "DRAM", 2 };

    binding_[3]["Weights"] = { "RowBuffer", 0 };
    binding_[3]["Inputs"] = { "DiagBuffer", 0 };
    binding_[3]["Outputs"] = { "ColBuffer", 0 };

    binding_[2]["Weights"] = { "RowBroadcaster", 0 };
    binding_[2]["Inputs"] = { "DiagBroadcaster", 0 };
    binding_[2]["Outputs"] = { "ColSpatialReducer", 0 };

    binding_[1]["Weights"] = { "OperandA", 0 };
    binding_[1]["Inputs"] = { "OperandB", 0 };
    binding_[1]["Outputs"] = { "Result", 0 };

    binding_[0]["Multiply"] = { "Multiplier", 0 };
  }
};
