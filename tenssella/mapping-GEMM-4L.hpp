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

class Mapping_GEMM_4L : public Mapping
{
 private:
  // Parameters.
  int M1, M2, M3;
  int N1, N2, N3;
  int K1, K2, K3;

 public:
  Mapping_GEMM_4L(isl_ctx* context,
                  int m1, int m2, int m3,
                  int n1, int n2, int n3,
                  int k1, int k2, int k3) :
      Mapping(context),
      M1(m1), M2(m2), M3(m3),
      N1(n1), N2(n2), N3(n3),
      K1(k1), K2(k2), K3(k3)
  {
    //
    // Tile ID -> Global coordinates.
    //

    char buf[1024];
    sprintf(buf, 
            "{ [m3,n3,k3] -> [m,n,k] : exists m2,n2,k2,m1,n1,k1 : m = (m3*%d + m2)*%d + m1 and 0 <= m2 < %d and 0 <= m1 < %d and "
            "                                                     n = (n3*%d + n2)*%d + n1 and 0 <= n2 < %d and 0 <= n1 < %d and "
            "                                                     k = (k3*%d + k2)*%d + k1 and 0 <= k2 < %d and 0 <= k1 < %d }",
            M2, M1, M2, M1,
            N2, N1, N2, N1,
            K2, K1, K2, K1);
    tile_id_to_set_[3] = isl_map_read_from_str(context_, buf);

    sprintf(buf, 
            "{ [[m3,n3,k3] -> [m2,n2,k2]] -> [m,n,k] : exists m1,n1,k1 : m = (m3*%d + m2)*%d + m1 and 0 <= m2 < %d and 0 <= m1 < %d and "
            "                                                            n = (n3*%d + n2)*%d + n1 and 0 <= n2 < %d and 0 <= n1 < %d and "
            "                                                            k = (k3*%d + k2)*%d + k1 and 0 <= k2 < %d and 0 <= k1 < %d }",
            M2, M1, M2, M1,
            N2, N1, N2, N1,
            K2, K1, K2, K1);
    tile_id_to_set_[2] = isl_map_read_from_str(context_, buf);

    sprintf(buf, 
            "{ [[[m3,n3,k3] -> [m2,n2,k2] -> [m1,n1,k1]] -> [m,n,k] : m = (m3*%d + m2)*%d + m1 and 0 <= m2 < %d and 0 <= m1 < %d and "
            "                                                         n = (n3*%d + n2)*%d + n1 and 0 <= n2 < %d and 0 <= n1 < %d and "
            "                                                         k = (k3*%d + k2)*%d + k1 and 0 <= k2 < %d and 0 <= k1 < %d }",
            M2, M1, M2, M1,
            N2, N1, N2, N1,
            K2, K1, K2, K1);
    tile_id_to_set_[1] = isl_map_read_from_str(context_, buf);

    //
    // Tile ID domains.
    //

    sprintf(buf,
            "{ [m3,n3,k3] : 0 <= m3 < %d and "
            "               0 <= n3 < %d and "
            "               0 <= k3 < %d }",
            M3, N3, K3);
    tile_id_domains_[3] = isl_set_read_from_str(context_, buf);

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

    //
    // Skews.
    //
    sprintf(buf,
            "{ [m3,n3,k3] -> SpaceTime_3[m3,n3] : k3 = 0 }" // Hardware spacetime is <y,x>.
            );
    skews_[3] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [m2,n2,k2] -> SpaceTime_2[k2] : n2 = 0 and k2 = 0 }" // Hardware spacetime is <t>.
            );
    skews_[2] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [m1,n1,k1] -> SpaceTime_1[] : m1 = 0 and n1 = 0 and k1 = 0 }" // Hardware spacetime is <>.
            );
    skews_[1] = isl_map_read_from_str(context, buf);

    //
    // Binding of data-spaces to hardware instances. Note that hardware instances are
    // indexed by *hardware* levels and not *tiling* levels (which are betweeen
    // hardware levels).
    //
    binding_[3]["MatrixA"] = "DRAM";
    binding_[3]["MatrixB"] = "DRAM";
    binding_[3]["MatrixZ"] = "DRAM";

    binding_[2]["MatrixA"] = "RowBuffer";
    binding_[2]["MatrixB"] = "DiagBuffer";
    binding_[2]["MatrixZ"] = "ColBuffer";

    binding_[1]["MatrixA"] = "RowBroadcaster";
    binding_[1]["MatrixB"] = "DiagBroadcaster";
    binding_[1]["MatrixZ"] = "ColSpatialReducer";

    binding_[0]["MatrixA"] = "OperandA";
    binding_[0]["MatrixB"] = "OperandB";
    binding_[0]["MatrixZ"] = "Result";
  }
};
