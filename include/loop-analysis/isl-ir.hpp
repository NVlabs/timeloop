/* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

/**
 * @file isl-ir.hpp
 * @author Michael Gilbert (gilbertm@mit.edu)
 * @brief This header describes the IR that forms the interface between the
 *        mapping and the nest analysis.
 */
#pragma once

#include <map>
#include <isl/cpp.h>
#include <variant>

#include "mapping/fused-mapping.hpp"
#include "mapping/mapping.hpp"
#include "workload/fused-workload.hpp"
#include "workload/workload.hpp"

namespace analysis
{

using DataSpaceId = problem::DataSpaceId;
using FactorizedDimensionID = problem::Shape::FactorizedDimensionID;
using EinsumID = problem::EinsumId;
using BufferId = mapping::BufferId;

struct Temporal {};
struct Spatial
{
  int spatial_dim;

  Spatial(int spatial_dim=0);
};
struct Sequential {};
struct PipelineTemporal {};
struct PipelineSpatial {};

using SpaceTime =
  std::variant<
    Temporal,
    Spatial,
    Sequential,
    PipelineTemporal,
    PipelineSpatial
  >;

std::ostream& operator<<(std::ostream& os, const Temporal& t);
std::ostream& operator<<(std::ostream& os, const Spatial& t);
std::ostream& operator<<(std::ostream& os, const Sequential& t);
std::ostream& operator<<(std::ostream& os, const PipelineTemporal& t);
std::ostream& operator<<(std::ostream& os, const PipelineSpatial& t);
std::ostream& operator<<(std::ostream& os, const SpaceTime& t);

bool IsTemporal(const SpaceTime& st);

struct LogicalComputeUnit
{
  BufferId buffer_id;
  mapping::NodeID branch_leaf_id;

  LogicalComputeUnit(BufferId buffer_id, mapping::NodeID branch_leaf_id);

  bool operator<(const LogicalComputeUnit& other) const;
  bool operator==(const LogicalComputeUnit& other) const;
};

std::ostream& operator<<(std::ostream& os, const LogicalComputeUnit& buf);

struct LogicalBuffer
{
  BufferId buffer_id;
  DataSpaceId dspace_id;
  mapping::NodeID branch_leaf_id;

  LogicalBuffer() = default;
  LogicalBuffer(
    BufferId buffer_id,
    DataSpaceId dspace_id,
    mapping::NodeID branch_leaf_id
  );

  bool operator<(const LogicalBuffer& other) const;
  bool operator==(const LogicalBuffer& other) const;
};

std::ostream& operator<<(std::ostream& os, const LogicalBuffer& buf);

/**
 * @brief Describes how time in different branches relate to each other.
 */
struct PipelineSchedule
{
  size_t top;
  size_t bot;
};
struct SequentialSchedule
{
  size_t bot;
};
using BranchSchedule = std::variant<PipelineSchedule, SequentialSchedule>;

/**
 * @brief Iteration -> Operation relation that specifies the tiling.
 * 
 * The tiling relation allows us to distribute data and operations using the
 * skew and data distribution relations.
 * 
 * The tiling relation may have unspecified bounds which will be inferred by
 * LoopTree. The tiling relation that goes to the nest analysis is guaranteed
 * to be fully specified.
 */
using Tiling = isl::map;
using BranchTilings = std::map<mapping::NodeID, Tiling>;
using LogicalBufTiling = std::map<LogicalBuffer, Tiling>;

/**
 * @brief Iteration -> Data relation that specifies distribution of data among
 *        spatial instances within a storage level.
 * 
 * The default is inferred from the mapping by assuming simple ops -> data
 * relationship even if spatial instances duplicate data.
 * 
 */
using DataDistribution = isl::map;
using LogicalBufDataDistributions = std::map<LogicalBuffer, DataDistribution>;

/**
 * @brief Space-Time -> Iteration relation that specifies distribution of tiles
 *        in time and space.
 * 
 * The default is inferred from the mapping.
 */
struct Skew
{
  std::vector<SpaceTime> dim_in_tags;
  isl::map map;

  Skew();
  Skew(const std::vector<SpaceTime>& dim_in_tags, isl::map map);
};

std::ostream& operator<<(std::ostream& os, const Skew& s);

/**
 * @brief Space-Time -> Data of a logical buffer. A complete description of
 *        data held in a logical buffer.
 */
struct Occupancy
{
  std::vector<SpaceTime> dim_in_tags;
  isl::map map;

  Occupancy();
  Occupancy(const std::vector<SpaceTime>& dim_in_tags, isl::map map);
};

std::ostream& operator<<(std::ostream& os, const Occupancy& s);

/**
 * @brief Space-Time -> Operations of a compute unit.
 */
struct OpOccupancy
{
  std::vector<SpaceTime> dim_in_tags;
  isl::map map;

  OpOccupancy();
  OpOccupancy(const std::vector<SpaceTime>& dim_in_tags, isl::map map);
};

std::ostream& operator<<(std::ostream& os, const OpOccupancy& s);

/**
 * @brief TARDIS-style X-relation.
 */
struct Transfers
{
  std::vector<SpaceTime> dim_in_tags;
  isl::map map;

  Transfers();
  Transfers(const std::vector<SpaceTime>& dim_in_tags, isl::map map);
};

/**
 * @brief Space-Time -> Fill of a logical buffer.
 */
struct Fill
{
  std::vector<SpaceTime> dim_in_tags;
  isl::map map;

  Fill();
  Fill(const std::vector<SpaceTime>& dim_in_tags, isl::map map);
};

std::ostream& operator<<(std::ostream& os, const Fill& s);

/**
 * @brief Space-Time -> Reads to a logical buffer.
 */
struct Reads
{
  std::vector<SpaceTime> dim_in_tags;
  isl::map map;

  Reads();
  Reads(const std::vector<SpaceTime>& dim_in_tags, isl::map map);
};

};  // namespace analysis