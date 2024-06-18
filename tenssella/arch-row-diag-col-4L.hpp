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

#include "architecture.hpp"

class Arch_RowDiagCol_4L : public Architecture
{
 private:

 public:
  Arch_RowDiagCol_4L(isl_ctx* context) :
      Architecture(4, context)
  {
    //
    // The indices appear to be off-by-1 because we wanted to be consistent
    // with the SpaceTime hierarchy used in the T functions.
    //

    char buf[256];

    // Level 4: find the DRAM.
    sprintf(buf,
            "{ [] -> DRAM[0,0] }"
            );
    instance_map_[4]["DRAM"] = isl_map_read_from_str(context_, buf);

    // Level 3: find a Row/Diag/Col Buffer.
    sprintf(buf,
            "{ [ [] -> SpaceTime_3[y,x]] -> RowBuffer[y,0] }"
            );
    instance_map_[3]["RowBuffer"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [ [] -> SpaceTime_3[y,x]] -> DiagBuffer[y+x,0] }"
            );
    instance_map_[3]["DiagBuffer"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [ [] -> SpaceTime_3[y,x]] -> ColBuffer[x,0] }"
            );
    instance_map_[3]["ColBuffer"] = isl_map_read_from_str(context_, buf);

    // Level 2: find the Broadcasters/Reducers.
    sprintf(buf,
            "{ [[ [] -> SpaceTime_3[y,x]] -> SpaceTime_2[t]] -> RowBroadcaster[y,t] }"
            );
    instance_map_[2]["RowBroadcaster"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ [] -> SpaceTime_3[y,x]] -> SpaceTime_2[t]] -> DiagBroadcaster[y+x,t] }"
            );
    instance_map_[2]["DiagBroadcaster"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ [] -> SpaceTime_3[y,x]] -> SpaceTime_2[t]] -> ColSpatialReducer[x,t] }"
            );
    instance_map_[2]["ColSpatialReducer"] = isl_map_read_from_str(context_, buf);

    // Level 1: find the Operand/Result ports.
    sprintf(buf,
            //"{ [[[ [] -> SpaceTime_3[y,x,0]] -> SpaceTime_2[0,t]] -> SpaceTime_1[0,0]] -> OperandA[y,x,t] }"
            "{ [[[ [] -> SpaceTime_3[y,x]] -> SpaceTime_2[t]] -> SpaceTime_1[]] -> OperandA[y*5+x,t] : 0 <= x < 5 }"
            );
    instance_map_[1]["OperandA"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            //"{ [[[ [] -> SpaceTime_3[y,x,0]] -> SpaceTime_2[0,t]] -> SpaceTime_1[0,0]] -> OperandB[y,x,t] }"
            "{ [[[ [] -> SpaceTime_3[y,x]] -> SpaceTime_2[t]] -> SpaceTime_1[]] -> OperandB[y*5+x,t] : 0 <= x < 5 }"
            );
    instance_map_[1]["OperandB"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[[ [] -> SpaceTime_3[y,x]] -> SpaceTime_2[t]] -> SpaceTime_1[]] -> Result[y*5+x,t] : 0 <= x < 5 }"
            );
    instance_map_[1]["Result"] = isl_map_read_from_str(context_, buf);

    // Arithmetic level: connect the Operand ports to the Result ports.
    sprintf(buf,
            "{ [[[[ [] -> SpaceTime_3[y,x]] -> SpaceTime_2[t]] -> SpaceTime_1[]] -> SpaceTime_0[]] -> Multiplier[y*5+x,t] : 0 <= x < 5 }"
            );
    instance_map_[0]["Multiplier"] = isl_map_read_from_str(context_, buf);
  }

};
