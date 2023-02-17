
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

#include "architecture.hpp"

class Arch_2L: public Architecture
{
 private:

 public:
  Arch_2L(isl_ctx* context) :
      Architecture(2, context)
  {
    //
    // The indices appear to be off-by-1 because we wanted to be consistent
    // with the SpaceTime hierarchy used in the T functions.
    //

    char buf[256];

    // Level 2: Memory.
    sprintf(buf,
            "{ SpaceTime_2[0] -> Memory[0,0] }"
            );
    mem_instance_map_[2]["Memory"]["Add0"] = isl_map_read_from_str(context_, buf);

    // Level 1: Operand/Result ports.
    sprintf(buf,
            "{ [ SpaceTime_2[0] -> SpaceTime_1[t] ] -> OperandA[0,t] }"
            );
    mem_instance_map_[1]["OperandA"]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [ SpaceTime_2[0] -> SpaceTime_1[t] ] -> OperandB[0,t] }"
            );
    mem_instance_map_[1]["OperandB"]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [ SpaceTime_2[0] -> SpaceTime_1[t] ] -> Result[0,t] }"
            );
    mem_instance_map_[1]["Result"]["Add0"] = isl_map_read_from_str(context_, buf);

    // Level 0: ALU.
    sprintf(buf,
            "{ [[ SpaceTime_2[0] -> SpaceTime_1[t]] -> SpaceTime_0[]] -> ALU[0,t] }"
            );
    comp_instance_map_[0]["Add0"] = isl_map_read_from_str(context_, buf);
  }
};

class Arch_Fuse_2L: public Architecture
{
 private:

 public:
  Arch_Fuse_2L(isl_ctx* context) :
      Architecture(2, context)
  {
    //
    // The indices appear to be off-by-1 because we wanted to be consistent
    // with the SpaceTime hierarchy used in the T functions.
    //

    char buf[256];

    // Level 2: Memory.
    sprintf(buf,
            "{ SpaceTime_2[0] -> DRAM[0,0] }"
            );
    mem_instance_map_[2]["DRAM"]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ SpaceTime_2[1] -> DRAM[0,1] }"
            );
    mem_instance_map_[2]["DRAM"]["Add1"] = isl_map_read_from_str(context_, buf);

    // Level 1: Operand/Result ports.
    sprintf(buf,
            "{ [ SpaceTime_2[0] -> SpaceTime_1[t] ] -> OperandA[0,0,t] }"
            );
    mem_instance_map_[1]["OperandA"]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [ SpaceTime_2[1] -> SpaceTime_1[t] ] -> OperandA[0,1,t] }"
            );
    mem_instance_map_[1]["OperandA"]["Add1"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [ SpaceTime_2[0] -> SpaceTime_1[t] ] -> Result[0,0,t] }"
            );
    mem_instance_map_[1]["Result"]["Add0"] = isl_map_read_from_str(context_, buf);


    sprintf(buf,
            "{ [ SpaceTime_2[1] -> SpaceTime_1[t] ] -> Result[0,1,t] }"
            );
    mem_instance_map_[1]["Result"]["Add1"] = isl_map_read_from_str(context_, buf);

    // Level 0: ALU.
    sprintf(buf,
            "{ [[ SpaceTime_2[0] -> SpaceTime_1[t]] -> SpaceTime_0[]] -> Adder[0,0,t] }"
            );
    comp_instance_map_[0]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ SpaceTime_2[1] -> SpaceTime_1[t]] -> SpaceTime_0[]] -> Adder[0,1,t] }"
            );
    comp_instance_map_[0]["Add1"] = isl_map_read_from_str(context_, buf);
  }
};

class Arch_Fuse_Seq: public Architecture {
  public:
  Arch_Fuse_Seq(isl_ctx* context) : Architecture(3, context) {
    char buf[256];
    // Level 3: find the DRAM.
    //Add DRAM to each einsum that access DRAM
    sprintf(buf,
            "{ SpaceTime_3[0] -> DRAM[0,0] }"
            );
    mem_instance_map_[3]["DRAM"]["Add0"] = isl_map_read_from_str(context_, buf);
    mem_instance_map_[3]["DRAM"]["Add1"] = isl_map_read_from_str(context_, buf);

    // Level 2: regfile.
    sprintf(buf,
            "{ [ SpaceTime_3[0] -> SpaceTime_2[t, 0]] -> Regfile[0,2*t] }"
            );
    mem_instance_map_[2]["Regfile"]["Add0"] = isl_map_read_from_str(context_, buf);
    sprintf(buf,
            "{ [ SpaceTime_3[0] -> SpaceTime_2[t, 1]] -> Regfile[0,2*t+1] }"
            );
    mem_instance_map_[2]["Regfile"]["Add1"] = isl_map_read_from_str(context_, buf);

    // Level 1: find the Operand/Result ports.
    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[t2, 0]] -> SpaceTime_1[t1]]  -> OperandA[0, 2*t2,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[t2, 0]] -> SpaceTime_1[t1]]  -> Result[0, 2*t2,t1] }"
            );
    mem_instance_map_[1]["Result"]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[t2, 1]] -> SpaceTime_1[t1]]  -> OperandA[0, 2*t2+1,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["Add1"] = isl_map_read_from_str(context_, buf);
    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[t2, 1]] -> SpaceTime_1[t1]] -> Result[0, 2*t2+1,t1] }"
            );
    mem_instance_map_[1]["Result"]["Add1"] = isl_map_read_from_str(context_, buf);

    // Arithmetic level: connect the Operand ports to the Result ports.
    sprintf(buf,
            "{ [[[ SpaceTime_3[0]  -> SpaceTime_2[t2, 0]] -> SpaceTime_1[t1]] -> SpaceTime_0[]] -> Adder[0, 2*t2,t1]}"
            );
    comp_instance_map_[0]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[[ SpaceTime_3[0]  -> SpaceTime_2[t2, 1]] -> SpaceTime_1[t1]] -> SpaceTime_0[]] -> Adder[0, 2*t2+1,t1]}"
            );
    comp_instance_map_[0]["Add1"] = isl_map_read_from_str(context_, buf);

  }
};

class Arch_Fuse: public Architecture {
  public:
  Arch_Fuse(isl_ctx* context) : Architecture(3, context) {
    char buf[256];
    // Level 3: find the DRAM.
    //Add DRAM to each einsum that access DRAM
    sprintf(buf,
            "{ SpaceTime_3[0] -> DRAM[0,0] }"
            );
    mem_instance_map_[3]["DRAM"]["Add0"] = isl_map_read_from_str(context_, buf);
    mem_instance_map_[3]["DRAM"]["Add1"] = isl_map_read_from_str(context_, buf);

    // Level 2: find a input output regfile.
    sprintf(buf,
            "{ [ SpaceTime_3[0] -> SpaceTime_2[t, 0]] -> IRegfile[0,t] }"
            );
    mem_instance_map_[2]["IRegfile"]["Add0"] = isl_map_read_from_str(context_, buf);

    //TODO: check if this is correct, one may to the tmp reg update
    //the other map to read/drain
    sprintf(buf,
            "{ [ SpaceTime_3[0] -> SpaceTime_2[t, 0]] -> TmpRegfile[0,2*t] }"
            );
    mem_instance_map_[2]["TmpRegfile"]["Add0"] = isl_map_read_from_str(context_, buf);
    sprintf(buf,
            "{ [ SpaceTime_3[0] -> SpaceTime_2[t, 1]] -> TmpRegfile[0,2*t+1] }"
            );
    mem_instance_map_[2]["TmpRegfile"]["Add1"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [ SpaceTime_3[0] -> SpaceTime_2[t, 1]] -> ORegfile[0,t] }"
            );
    mem_instance_map_[2]["ORegfile"]["Add1"] = isl_map_read_from_str(context_, buf);

    // Level 1: find the Operand/Result ports.
    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[t2, 0]] -> SpaceTime_1[t1]]  -> OperandA[0, t2,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[t2, 0]] -> SpaceTime_1[t1]]  -> Result[0, t2,t1] }"
            );
    mem_instance_map_[1]["Result"]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[t2, 1]] -> SpaceTime_1[t1]]  -> OperandA[1, t2,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["Add1"] = isl_map_read_from_str(context_, buf);
    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[t2, 1]] -> SpaceTime_1[t1]] -> Result[1, t2,t1] }"
            );
    mem_instance_map_[1]["Result"]["Add1"] = isl_map_read_from_str(context_, buf);

    // Arithmetic level: connect the Operand ports to the Result ports.
    sprintf(buf,
            "{ [[[ SpaceTime_3[0]  -> SpaceTime_2[t2, 0]] -> SpaceTime_1[t1]] -> SpaceTime_0[]] -> Adder[0, t2,t1]}"
            );
    comp_instance_map_[0]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[[ SpaceTime_3[0]  -> SpaceTime_2[t2, 1]] -> SpaceTime_1[t1]] -> SpaceTime_0[]] -> Adder[1, t2,t1]}"
            );
    comp_instance_map_[0]["Add1"] = isl_map_read_from_str(context_, buf);

  }

};

class Arch_3L: public Architecture {
  public:
  Arch_3L(isl_ctx* context) : Architecture(3, context) {
    char buf[256];
    // Level 3: find the DRAM.
    sprintf(buf,
            "{ SpaceTime_3[0] -> DRAM[0,0] }"
            );
    mem_instance_map_[3]["DRAM"]["Add0"] = isl_map_read_from_str(context_, buf);

    // Level 2: find a input output regfile.
    sprintf(buf,
            "{ [ SpaceTime_3[0] -> SpaceTime_2[t]] -> IRegfile[0,t] }"
            );
    mem_instance_map_[2]["IRegfile"]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [ SpaceTime_3[0] -> SpaceTime_2[t]] -> ORegfile[0,t] }"
            );
    mem_instance_map_[2]["ORegfile"]["Add0"] = isl_map_read_from_str(context_, buf);

    // Level 1: find the Operand/Result ports.
    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[t2]] -> SpaceTime_1[t1]]  -> OperandA[0, t2,t1] }"
            );
    mem_instance_map_[1]["OperandA"]["Add0"] = isl_map_read_from_str(context_, buf);

    sprintf(buf,
            "{ [[ SpaceTime_3[0] -> SpaceTime_2[t2]] -> SpaceTime_1[t1]] -> Result[0, t2,t1] }"
            );
    mem_instance_map_[1]["Result"]["Add0"] = isl_map_read_from_str(context_, buf);

    // Arithmetic level: connect the Operand ports to the Result ports.
    sprintf(buf,
            "{ [[[ SpaceTime_3[0]  -> SpaceTime_2[t2]] -> SpaceTime_1[t1]] -> SpaceTime_0[]] -> Adder[0, t2,t1]}"
            );
    comp_instance_map_[0]["Add0"] = isl_map_read_from_str(context_, buf);

  }

};
