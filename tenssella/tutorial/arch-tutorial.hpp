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

#include "architecture.hpp"

class Arch_2D_3L_GEMM : public Architecture
{
 private:

 public:
  Arch_2D_3L_GEMM(isl_ctx* context) :
      Architecture(3, context)
  {
    //
    // The indices appear to be off-by-1 because we wanted to be consistent
    // with the SpaceTime hierarchy used in the T functions.
    //

    char buf[256];

    // Level 3: find the GlobalBuffer.
    sprintf(buf,
            "{ SpaceTime_3[0] -> GlobalBuffer[0,0] }"
            );
    mem_instance_map_[3]["GlobalBuffer"]["GEMM0"] = isl_map_read_from_str(context_, buf);
    mem_instance_map_[3]["GlobalBuffer"]["Elemwise"] = isl_map_read_from_str(context_, buf);

    // Level 2: find a PE Buffer.
    sprintf(buf,
            "{ [ SpaceTime_3[0] -> SpaceTime_2[s,t,0]] -> RegFile[s,2*t] }"
            );
    mem_instance_map_[2]["RegFile"]["GEMM0"] = isl_map_read_from_str(context_, buf);

    // Level 2: find a PE Buffer.
    sprintf(buf,
            "{ [ SpaceTime_3[0] -> SpaceTime_2[s,t,1]] -> RegFile[s,2*t+1] }"
            );
    mem_instance_map_[2]["RegFile"]["Elemwise"] = isl_map_read_from_str(context_, buf);

    //FIXME:Proposed, we should remove the einsum index in architecture
    //sprintf(buf,
    //        "{ [ SpaceTime_3[0] -> SpaceTime_2[s,t,e]] -> RegFile[s,t,t2=e] }"
    //        );
    //mem_instance_map_[2]["RegFile"] = isl_map_read_from_str(context_, buf);

    // Level 1: find the Operand/Result ports.
    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[s,t2,0]] -> SpaceTime_1[0,t1]] -> OperandA[s,t2,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["GEMM0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[s,t2,0]] -> SpaceTime_1[0,t1]] -> OperandB[s,t2,t1] }"
            );
    mem_instance_map_[1]["OperandB"]["GEMM0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[s,t2,0]] -> SpaceTime_1[0,t1]] -> Result[s,t2,t1] }"
            );
    mem_instance_map_[1]["Result"]["GEMM0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{  [[[ SpaceTime_3[0] -> SpaceTime_2[s,t2,0]] -> SpaceTime_1[0,t1]] -> SpaceTime_0[]] -> MAC[s,t2,t1] }"
            );
    comp_instance_map_[0]["GEMM0"] = isl_map_read_from_str(context_, buf);

    // Level 1: find the Operand/Result ports.
    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[s,t2,1]] -> SpaceTime_1[0,t1]] -> OperandA[s,t2,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["Elemwise"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[s,t2,1]] -> SpaceTime_1[0,t1]] -> Result[s,t2,t1] }"
            );
    mem_instance_map_[1]["Result"]["Elemwise"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{  [[[ SpaceTime_3[0] -> SpaceTime_2[s,t2,1]] -> SpaceTime_1[0,t1]] -> SpaceTime_0[]] -> MAC[s,t2,t1] }"
            );
    comp_instance_map_[0]["Elemwise"] = isl_map_read_from_str(context_, buf);

  }
};
