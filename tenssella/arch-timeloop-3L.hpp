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

class Arch_Timeloop_3L : public Architecture
{
 private:

 public:
  Arch_Timeloop_3L(isl_ctx* context, const std::map<unsigned, int>& time_ranges) :
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

    // Level 3: find the Global Buffer
    sprintf(buf,
            "{ [ [] -> SpaceTime_3[0,0,t3]] -> GlobalBuffer[0,t3] }"
            );
    instance_map_[3]["GlobalBuffer"] = isl_map_read_from_str(context_, buf);
    
    // Level 2: find the Register File.
    sprintf(buf,
            "{ [[ [] -> SpaceTime_3[0,0,t3]] -> SpaceTime_2[y,x,t2]] -> Registers[y*64+x,t3*%d+t2] : 0 <= x < 64 and 0 <= y < 16 and 0 <= t2 < %d }",
            time_ranges.at(2),
            time_ranges.at(2));
    instance_map_[2]["Registers"] = isl_map_read_from_str(context_, buf);
    
    // Level 1: find the Operand/Result latches.
    sprintf(buf,
            "{ [[[ [] -> SpaceTime_3[0,0,t3]] -> SpaceTime_2[y,x,t2]] -> SpaceTime_1[0,0,t1]] -> OperandA[y*64+x,(t3*%d+t2)*%d+t1] : 0 <= x < 64 and 0 <= y < 16 and 0 <= t2 < %d and 0 <= t1 < %d }",
            time_ranges.at(2), time_ranges.at(1),
            time_ranges.at(2), time_ranges.at(1));
    instance_map_[1]["OperandA"] = isl_map_read_from_str(context_, buf); 
    
    sprintf(buf,
            "{ [[[ [] -> SpaceTime_3[0,0,t3]] -> SpaceTime_2[y,x,t2]] -> SpaceTime_1[0,0,t1]] -> OperandB[y*64+x,(t3*%d+t2)*%d+t1] : 0 <= x < 64 and 0 <= y < 16 and 0 <= t2 < %d and 0 <= t1 < %d }",
            time_ranges.at(2), time_ranges.at(1),
            time_ranges.at(2), time_ranges.at(1));
    instance_map_[1]["OperandB"] = isl_map_read_from_str(context_, buf); 

    sprintf(buf,
            "{ [[[ [] -> SpaceTime_3[0,0,t3]] -> SpaceTime_2[y,x,t2]] -> SpaceTime_1[0,0,t1]] -> Result[y*64+x,(t3*%d+t2)*%d+t1] : 0 <= x < 64 and 0 <= y < 16 and 0 <= t2 < %d and 0 <= t1 < %d }",
            time_ranges.at(2), time_ranges.at(1),
            time_ranges.at(2), time_ranges.at(1));
    instance_map_[1]["Result"] = isl_map_read_from_str(context_, buf); 

    // Arithmetic level: connect the Operand ports to the Result ports.
    sprintf(buf,
            "{ [[[[ [] -> SpaceTime_3[0,0,t3]] -> SpaceTime_2[y,x,t2]] -> SpaceTime_1[0,0,t1]] -> SpaceTime_0[0,0,0]] -> Multiplier[y*64+x,(t3*%d+t2)*%d+t1] : 0 <= x < 64 and 0 <= y < 16 and 0 <= t2 < %d and 0 <= t1 < %d }",
            time_ranges.at(2), time_ranges.at(1),
            time_ranges.at(2), time_ranges.at(1));
    instance_map_[0]["Multiplier"] = isl_map_read_from_str(context_, buf); 
  }

};
