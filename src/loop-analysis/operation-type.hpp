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

#include <math.h>

#include "workload/data-density.hpp"
#include "loop-analysis/tiling-tile-info.hpp"
#include "model/sparse.hpp"


namespace tiling
{

// define the (data-dependent fine-grained) operation types for each type of components
static std::string storageOperationTypes[17] = {"random_read",
                                                "random_fill",
                                                "random_update",
                                                "gated_read",
                                                "gated_fill",
                                                "gated_update",
                                                "skipped_read",
                                                "skipped_fill",
                                                "skipped_update",
                                                "metadata_read",
                                                "gated_metadata_read",
                                                "metadata_fill",
                                                "gated_metadata_fill",
                                                "metadata_update",
                                                "gated_metadata_update",
                                                "decompression_count",
                                                "compression_count"};   // we don't update a metadata

static std::string arithmeticOperationTypes[3] = {"random_compute",
                                                  "skipped_compute",
                                                  "gated_compute"};

static std::string networkOperationTypes[1] = {"random_transfer"};
										                             

int GetNumOpTypes();
int GetNumOpTypes(std::string component_type);
//void ComputeFineGrainComputeAccesses(tiling::ComputeInfo& compute_info,
//                                     tiling::CompoundDataMovementInfo& data_movement,
//                                     sparse::ComputeActionOptimizationInfo& compute_gating_info,
//                                     sparse::ComputeActionOptimizationInfo& compute_skipping_info);

  void ComputeFineGrainComputeAccesses(tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                       unsigned level,
                                       sparse::ComputeActionOptimizationInfo& compute_gating_info,
                                       sparse::ComputeActionOptimizationInfo& compute_skipping_info);

void ComputeFineGrainDataMovementAccesses(tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                          unsigned level,
                                          sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_gating,
                                          sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_skipping);

void ComputeFineGrainMetaDataAccesses(tiling::NestOfCompoundTiles& nest_of_compound_tiles,
                                      unsigned level,
                                      sparse::PerStorageLevelCompressionInfo& per_level_compression_info,
                                      sparse::PerStorageLevelActionOptimizationInfo& per_level_sparse_gating);


} // namespace problem