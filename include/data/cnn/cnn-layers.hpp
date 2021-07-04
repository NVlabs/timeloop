/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include <tuple>

#include "workload/shape-models/problem-shape.hpp"
#include "workload/workload.hpp"
#include "compound-config/compound-config.hpp"

namespace problem
{

extern const unsigned kDimensionR;
extern const unsigned kDimensionS;
extern const unsigned kDimensionP;
extern const unsigned kDimensionQ;
extern const unsigned kDimensionC;
extern const unsigned kDimensionK;
extern const unsigned kDimensionN;

extern const unsigned kDataSpaceWeight;
extern const unsigned kDataSpaceInput;
extern const unsigned kDataSpaceOutput;

Workload::Bounds GetLayerBounds(std::string layer_name, bool pad_primes=true);
Workload::Densities GetLayerDensities(std::string layer_name);
void ReadDensities(std::string filename);
void DumpDensities(std::string filename);
void DumpDensities_CPP(std::string filename);
void ParseConfig(config::CompoundConfigNode problem, Workload& workload);

}
