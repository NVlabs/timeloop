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

class Mapping_CONV7D_Reference : public Mapping
{
 public:
  Mapping_CONV7D_Reference(isl_ctx* context,
                           std::string prob_filename
    ) :
      Mapping(context)
  {
    std::ifstream prob_file(prob_filename.c_str());
    std::string line;
    std::getline(prob_file, line);
    auto factors = ParseFactorLine(line);

    char buf[1024];

    GenerateTileMaps({
        { factors.at("C"), factors.at("K"), factors.at("R"), factors.at("S"), factors.at("P"), factors.at("Q"), factors.at("N") },
        { 1, 1, 1, 1, 1, 1, 1 }
      });

    //
    // Skews.
    //
    sprintf(buf,
            "{ [c,k,r,s,p,q,n] -> SpaceTime_1[(((((n*%d+r)*%d+s)*%d+q)*%d+p)*%d+c)*%d+k] : 0 <= c < %d and 0 <= k < %d and 0 <= r < %d and 0 <= s < %d and 0 <= p < %d and 0 <= q < %d and 0 <= n < %d }", // Hardware spacetime is <t>.
            factors.at("R"),
            factors.at("S"),
            factors.at("Q"),
            factors.at("P"),
            factors.at("C"),
            factors.at("K"),

            factors.at("C"),
            factors.at("K"),
            factors.at("R"),
            factors.at("S"),
            factors.at("P"),
            factors.at("Q"),
            factors.at("N"));

    skews_[1] = isl_map_read_from_str(context, buf);

    sprintf(buf,
            "{ [c,k,r,s,p,q,n] -> SpaceTime_0[] : c=0 and k=0 and r=0 and s=0 and p=0 and q=0 and n=0 }" // Hardware spacetime is <>.
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
