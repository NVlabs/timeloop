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

#include <bitset>

#include "mapping/loop.hpp"
#include "util/numeric.hpp"
#include "workload/problem-shape.hpp"
#include "workload/per-data-space.hpp"
#include "workload/workload.hpp"
#include "operation-type.hpp"
#include "nest-analysis-tile-info.hpp"
#include "tiling-tile-info.hpp"

namespace tiling
{

bool operator < (const DataMovementInfo& a, const DataMovementInfo& b);
std::ostream& operator << (std::ostream& out, const DataMovementInfo& info);

CompoundTileNest CollapseTiles(analysis::CompoundTileNest& tiles, 
                               int num_tiling_levels,
                               const CompoundMaskNest& tile_mask,
                               const CompoundMaskNest& distribution_supported,
                               problem::Workload* workload);
CompoundDataMovementNest CollapseDataMovementNest(analysis::CompoundDataMovementNest& tiles, 
                                                  int num_tiling_levels,
                                                  const CompoundMaskNest& tile_mask,
                                                  const CompoundMaskNest& distribution_supported,
                                                  problem::Workload* workload);
ComputeNest CollapseComputeNest(analysis::CompoundComputeNest& tiles, int num_tiling_levels);


NestOfCompoundTiles TransposeTiles(const CompoundTileNest& tiles);
NestOfCompoundMasks TransposeMasks(const CompoundMaskNest& masks);

}  // namespace tiling
