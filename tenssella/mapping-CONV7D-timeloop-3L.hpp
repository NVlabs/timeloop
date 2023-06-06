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

class Mapping_CONV7D_Timeloop_3L : public Mapping
{
 private:

 public:
  Mapping_CONV7D_Timeloop_3L(isl_ctx* context,
                             std::string filename
    ) :
      Mapping(context)
  {
    std::ifstream file(filename.c_str());
    assert(file);

    std::string line;

    // std::vector<std::vector<int>> all_factors;
    // while (std::getline(factors_file, line))
    // {
    //   auto factors = ParseFactorLine(line);
    //   std::vector<int> level_factors = {
    //         factors.at("C"),
    //         factors.at("K"),
    //         factors.at("R"),
    //         factors.at("S"),
    //         factors.at("P"),
    //         factors.at("Q"),
    //         factors.at("N") };
    //   all_factors.push_back(level_factors);
    // }
    // all_factors.reverse(); // Timeloop emits levels in little-endian order.

    // Only 1 factors line for now.
    std::getline(file, line);
    auto factors = ParseFactorLine(line);
    std::vector<int> level_factors = {
      factors.at("C"),
      factors.at("K"),
      factors.at("R"),
      factors.at("S"),
      factors.at("P"),
      factors.at("Q"),
      factors.at("N") };

    GenerateTileMaps({
        { 1, 1, 1, 1, 1, 1, 1 },
        level_factors
      });

    char buf[1024];

    //
    // Skews.
    //
    sprintf(buf,
            "{ [c,k,r,s,p,q,n] -> SpaceTime_2[] : c=0 and k=0 and r=0 and s=0 and p=0 and q=0 and n=0 }" // Hardware spacetime is <>.
            );
    skews_[2] = isl_map_read_from_str(context_, buf);


    // Only the SpaceTime_1 skew is specified. Read the temporal permutation from the file.
    std::getline(file, line);

    // Reverse Timeloop's little-endian to big-endian.
    std::reverse(line.begin(), line.end());

    std::string flattening;
    for (char& dim: line)
    {
      char dim_uc, dim_lc;
      if (dim < 97)
      {
        dim_uc = dim;
        dim_lc = dim + 32;
      }
      else
      {
        dim_uc = dim - 32;
        dim_lc = dim;
      }

      std::string dimstr_uc(1, dim_uc);
      std::string dimstr_lc(1, dim_lc);

      int factor = factors.at(dimstr_uc);

      if (dim_lc == 'c')
      {
        dimstr_lc += 't';
        
        if (factor < 64)
        {
          factors["Cs"] = factor;
          factors["Ct"] = 1;
        }
        else
        {
          assert(factor % 64 == 0);
          factors["Cs"] = 64;
          factors["Ct"] = factor / 64;
        }

        factor = factors.at("Ct");
      }

      if (dim_lc == 'k')
      {
        dimstr_lc += 't';

        if (factor < 16)
        {
          factors["Ks"] = factor;
          factors["Kt"] = 1;
        }
        else
        {
          assert(factor % 16 == 0);
          factors["Ks"] = 16;
          factors["Kt"] = factor / 16;
        }

        factor = factors.at("Kt");
      }

      if (flattening.empty())
      {
        flattening += dimstr_lc;
      }
      else
      {
        flattening = "(" + flattening + ")";
        flattening += "*" + std::to_string(factor) + " + " + dimstr_lc;
      }
    }

    sprintf(buf,
            "{ [c,k,r,s,p,q,n] -> SpaceTime_1[ks,cs,t] : exists kt,ct : t = %s and c = ct*%d+cs and k = kt*%d+ks and 0 <= cs < %d and 0 <= ks < %d and 0 <= ct < %d and 0 <= kt < %d and 0 <= r < %d and 0 <= s < %d and 0 <= p < %d and 0 <= q < %d and 0 <= n < %d }", // Hardware spacetime is <y,x,t>.
            flattening.c_str(),
            factors.at("Cs"),
            factors.at("Ks"),

            factors.at("Cs"),
            factors.at("Ks"),

            factors.at("Ct"),
            factors.at("Kt"),
            factors.at("R"),
            factors.at("S"),
            factors.at("P"),
            factors.at("Q"),
            factors.at("N")     
            );

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
    binding_[3]["Weights"] = { "DRAM", 0 };
    binding_[3]["Inputs"] = { "DRAM", 1 };
    binding_[3]["Outputs"] = { "DRAM", 2 };

    binding_[2]["Weights"] = { "GlobalBuffer", 0 };
    binding_[2]["Inputs"] = { "GlobalBuffer", 1 };
    binding_[2]["Outputs"] = { "GlobalBuffer", 2 };

    binding_[1]["Weights"] = { "Registers", 0 };
    binding_[1]["Inputs"] = { "Registers", 1 };
    binding_[1]["Outputs"] = { "Registers", 2 };

    binding_[0]["Multiply"] = { "Multiplier", 0 };
  }
};
