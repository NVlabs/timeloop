/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

class Arch_2D_4L_MV: public Architecture
{
 private:

 public:
  Arch_2D_4L_MV(isl_ctx* context) :
      Architecture(4, context)
  {
    //
    // The indices appear to be off-by-1 because we wanted to be consistent
    // with the SpaceTime hierarchy used in the T functions.
    //

    char buf[256];

    // Level 4: DRAM level
    sprintf(buf,
            "{ SpaceTime_4[0] -> DRAM[0,0] }"
            );
    mem_instance_map_[4]["DRAM"]["MV0"] = isl_map_read_from_str(context_, buf);

    // Level 3: find the GlobalBuffer.
    sprintf(buf,
            "{ [SpaceTime_4[0] -> SpaceTime_3[0,t,0]] -> GlobalBuffer[0,t] }"
            );
    mem_instance_map_[3]["GlobalBuffer"]["MV0"] = isl_map_read_from_str(context_, buf);

    // Level 2: find a PE Buffer.
    sprintf(buf,
            "{ [[SpaceTime_4[0] -> SpaceTime_3[0,t2,0]] -> SpaceTime_2[s,t1]] -> RegFile[s,t2,t1] }"
            );
    mem_instance_map_[2]["RegFile"]["MV0"] = isl_map_read_from_str(context_, buf);

    // Level 1: find the Operand/Result ports.
    sprintf(buf,
            "{ [[[SpaceTime_4[0] -> SpaceTime_3[0,t3,0]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> OperandA[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[[SpaceTime_4[0] -> SpaceTime_3[0,t3,0]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> OperandB[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["OperandB"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ [SpaceTime_4[0] -> SpaceTime_3[0,t3,0]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> Result[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["Result"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{  [[[ [SpaceTime_4[0] -> SpaceTime_3[0,t3,0]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> SpaceTime_0[]] -> MAC[s,t3,t2,t1] }"
            );
    comp_instance_map_[0]["MV0"] = isl_map_read_from_str(context_, buf);
  }
};


class Arch_2D_4L_MV_Fuse_Finer: public Architecture
{
 private:

 public:
  Arch_2D_4L_MV_Fuse_Finer(isl_ctx* context) :
      Architecture(4, context)
  {
    //
    // The indices appear to be off-by-1 because we wanted to be consistent
    // with the SpaceTime hierarchy used in the T functions.
    //

    char buf[256];

    // Level 4: DRAM level
    sprintf(buf,
            "{ SpaceTime_4[0] -> DRAM[0,0] }"
            );
    mem_instance_map_[4]["DRAM"]["MV0"] = isl_map_read_from_str(context_, buf);
    mem_instance_map_[4]["DRAM"]["MV1"] = isl_map_read_from_str(context_, buf);

    // Level 3: find the GlobalBuffer.
    sprintf(buf,
            "{ [SpaceTime_4[0] -> SpaceTime_3[0,t]] -> GlobalBuffer[0,t] }"
            );
    mem_instance_map_[3]["GlobalBuffer"]["MV0"] = isl_map_read_from_str(context_, buf);
    mem_instance_map_[3]["GlobalBuffer"]["MV1"] = isl_map_read_from_str(context_, buf);

    // Level 2: find a PE Buffer.
    sprintf(buf,
            "{ [[SpaceTime_4[0] -> SpaceTime_3[0,t2]] -> SpaceTime_2[s,t1,0]] -> RegFile[s,t2,2*t1] }"
            );
    mem_instance_map_[2]["RegFile"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[SpaceTime_4[0] -> SpaceTime_3[0,t2]] -> SpaceTime_2[s,t1,1]] -> RegFile[s,t2,2*t1+1] }"
            );
    mem_instance_map_[2]["RegFile"]["MV1"] = isl_map_read_from_str(context_, buf);

    // Level 1: find the Operand/Result ports.
    sprintf(buf,
            "{ [[[SpaceTime_4[0] -> SpaceTime_3[0,t3]] -> SpaceTime_2[s,t2,0]] -> SpaceTime_1[0,t1]] -> OperandA[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[[SpaceTime_4[0] -> SpaceTime_3[0,t3]] -> SpaceTime_2[s,t2,0]] -> SpaceTime_1[0,t1]] -> OperandB[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["OperandB"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ [SpaceTime_4[0] -> SpaceTime_3[0,t3]] -> SpaceTime_2[s,t2,0]] -> SpaceTime_1[0,t1]] -> Result[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["Result"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[[SpaceTime_4[0] -> SpaceTime_3[0,t3]] -> SpaceTime_2[s,t2,1]] -> SpaceTime_1[0,t1]] -> OperandA[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["MV1"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[[SpaceTime_4[0] -> SpaceTime_3[0,t3]] -> SpaceTime_2[s,t2,1]] -> SpaceTime_1[0,t1]] -> OperandB[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["OperandB"]["MV1"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ [SpaceTime_4[0] -> SpaceTime_3[0,t3]] -> SpaceTime_2[s,t2,1]] -> SpaceTime_1[0,t1]] -> Result[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["Result"]["MV1"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{  [[[ [SpaceTime_4[0] -> SpaceTime_3[0,t3]] -> SpaceTime_2[s,t2,0]] -> SpaceTime_1[0,t1]] -> SpaceTime_0[]] -> MAC[s,t3,t2,t1] }"
            );
    comp_instance_map_[0]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{  [[[ [SpaceTime_4[0] -> SpaceTime_3[0,t3]] -> SpaceTime_2[s,t2,1]] -> SpaceTime_1[0,t1]] -> SpaceTime_0[]] -> MAC[s,t3,t2,t1] }"
            );
    comp_instance_map_[0]["MV1"] = isl_map_read_from_str(context_, buf);
  }
};


class Arch_2D_4L_MV_Fuse: public Architecture
{
 private:

 public:
  Arch_2D_4L_MV_Fuse(isl_ctx* context) :
      Architecture(4, context)
  {
    //
    // The indices appear to be off-by-1 because we wanted to be consistent
    // with the SpaceTime hierarchy used in the T functions.
    //

    char buf[256];

    // Level 4: DRAM level
    sprintf(buf,
            "{ SpaceTime_4[0] -> DRAM[0,0] }"
            );
    mem_instance_map_[4]["DRAM"]["MV0"] = isl_map_read_from_str(context_, buf);
    mem_instance_map_[4]["DRAM"]["MV1"] = isl_map_read_from_str(context_, buf);

    // Level 3: find the GlobalBuffer.
    sprintf(buf,
            "{ [SpaceTime_4[0] -> SpaceTime_3[0,t,0]] -> GlobalBuffer[0,2*t] }"
            );
    mem_instance_map_[3]["GlobalBuffer"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [SpaceTime_4[0] -> SpaceTime_3[0,t,1]] -> GlobalBuffer[0,2*t+1] }"
            );
    mem_instance_map_[3]["GlobalBuffer"]["MV1"] = isl_map_read_from_str(context_, buf);

    // Level 2: find a PE Buffer.
    sprintf(buf,
            "{ [[SpaceTime_4[0] -> SpaceTime_3[0,t2,0]] -> SpaceTime_2[s,t1]] -> RegFile[s,t2,t1] }"
            );
    mem_instance_map_[2]["RegFile"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[SpaceTime_4[0] -> SpaceTime_3[0,t2,1]] -> SpaceTime_2[s,t1]] -> RegFile[s,t2,t1] }"
            );
    mem_instance_map_[2]["RegFile"]["MV1"] = isl_map_read_from_str(context_, buf);

    // Level 1: find the Operand/Result ports.
    sprintf(buf,
            "{ [[[SpaceTime_4[0] -> SpaceTime_3[0,t3,0]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> OperandA[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[[SpaceTime_4[0] -> SpaceTime_3[0,t3,0]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> OperandB[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["OperandB"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ [SpaceTime_4[0] -> SpaceTime_3[0,t3,0]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> Result[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["Result"]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[[SpaceTime_4[0] -> SpaceTime_3[0,t3,1]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> OperandA[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["MV1"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[[SpaceTime_4[0] -> SpaceTime_3[0,t3,1]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> OperandB[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["OperandB"]["MV1"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ [SpaceTime_4[0] -> SpaceTime_3[0,t3,1]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> Result[s,t3,t2,t1] }"
            );
    mem_instance_map_[1]["Result"]["MV1"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{  [[[ [SpaceTime_4[0] -> SpaceTime_3[0,t3,0]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> SpaceTime_0[]] -> MAC[s,t3,t2,t1] }"
            );
    comp_instance_map_[0]["MV0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{  [[[ [SpaceTime_4[0] -> SpaceTime_3[0,t3,1]] -> SpaceTime_2[s,t2]] -> SpaceTime_1[0,t1]] -> SpaceTime_0[]] -> MAC[s,t3,t2,t1] }"
            );
    comp_instance_map_[0]["MV1"] = isl_map_read_from_str(context_, buf);
  }
};
