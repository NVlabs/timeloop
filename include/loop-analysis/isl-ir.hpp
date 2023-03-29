/**
 * @file isl-ir.hpp
 * @author your name (you@domain.com)
 * @brief This header describes the IR that forms the interface between the
 *        mapping and the nest analysis.
 * @version 0.1
 * @date 2023-02-07
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

#include <map>
#include <isl/cpp.h>
#include <variant>

#include "isl-wrapper/tagged.hpp"
#include "mapping/fused-mapping.hpp"
#include "mapping/mapping.hpp"
#include "workload/workload.hpp"

namespace analysis
{
/******************************************************************************
 * Intermediate representation for workload
 *****************************************************************************/
using DataSpaceID = problem::Shape::DataSpaceID;
using FactorizedDimensionID = problem::Shape::FactorizedDimensionID;
using EinsumID = size_t;

class WorkloadIR
{
 public:
  using ConstIterator =
    std::map<std::pair<EinsumID, DataSpaceID>, isl::map>::const_iterator;

 public:
  WorkloadIR();

  EinsumID NewEinsum();
  DataSpaceID NewDataSpace();

  void AddReadDependency(EinsumID einsum_id, DataSpaceID dspace_id,
                         const std::string& map_str);
  void AddWriteDependency(EinsumID einsum_id, DataSpaceID dspace_id,
                          const std::string& map_str);

  void AddOperationSpaceBounds(EinsumID einsum_id, const std::string& set_str);
  void AddDataSpaceBounds(DataSpaceID dspace_id, const std::string& set_str);

  ConstIterator
  GetReadDependency(EinsumID einsum_id, DataSpaceID dspace_id) const;
  ConstIterator
  GetWriteDependency(EinsumID einsum_id, DataSpaceID dspace_id) const;

 private:
  std::map<std::pair<EinsumID, DataSpaceID>, isl::map> reads_;
  std::map<std::pair<EinsumID, DataSpaceID>, isl::map> writes_;
  std::map<EinsumID, isl::set> operation_spaces_;
  std::map<DataSpaceID, isl::set> data_spaces_;

  size_t next_einsum_id_;
  size_t next_dspace_id_;
};

/******************************************************************************
 * Intermediate representation between mapping and analysis
 *****************************************************************************/
using BufferID = size_t;

struct LogicalBuffer
{
  BufferID buffer_id;
  DataSpaceID dspace_id;
  mapping::NodeID branch_leaf_id;

  LogicalBuffer(BufferID buffer_id,
                DataSpaceID dspace_id,
                mapping::NodeID branch_leaf_id) :
    buffer_id(buffer_id), dspace_id(dspace_id), branch_leaf_id(branch_leaf_id)
  {
  }

  LogicalBuffer(const LogicalBuffer& other) :
    buffer_id(other.buffer_id), dspace_id(other.dspace_id),
    branch_leaf_id(other.branch_leaf_id)
  {
  }

  bool operator<(const LogicalBuffer& other) const
  {
    if (buffer_id < other.buffer_id)
    {
      return true;
    }
    else if (buffer_id == other.buffer_id && dspace_id < other.dspace_id)
    {
      return true;
    }
    else if (buffer_id == other.buffer_id && dspace_id == other.dspace_id)
    {
      return branch_leaf_id < other.branch_leaf_id;
    }
    return false;
  }

  bool operator==(const LogicalBuffer& other) const
  {
    return buffer_id == other.buffer_id && dspace_id == other.dspace_id
           && branch_leaf_id == other.branch_leaf_id;
  }
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
 * 
 */
using Skew = TaggedMap<isl::map, spacetime::Dimension>;
using LogicalBufSkews = std::map<LogicalBuffer, Skew>;

/**
 * @brief Space-Time -> Data of a logical buffer. A complete description of
 *        data held in a logical buffer.
 * 
 */
using Occupancy = TaggedMap<isl::map, spacetime::Dimension>;
using LogicalBufOccupancies = std::map<LogicalBuffer, Occupancy>;

/**
 * @brief TARDIS-style X-relation.
 * 
 * This is used:
 *   - To calculate access counts which is passed to the uarch model
 *   - As input to TARDIS for code-gen
 */
using Transfers = TaggedMap<isl::map, spacetime::Dimension>;
using LogicalBufTransfers = std::map<std::pair<LogicalBuffer, LogicalBuffer>,
                                     Transfers>;

using Fill = TaggedMap<isl::map, spacetime::Dimension>;
using LogicalBufFills = std::map<LogicalBuffer, Fill>;

/******************************************************************************
 * Converter from mapping to intermediate representation
 *****************************************************************************/

/**
 * @brief Infer logical buffer occupancies from fused mapping
 * 
 * @param mapping 
 * @return LogicalBufOccupancies 
 */
// LogicalBufOccupancies
// OccupanciesFromMapping(const mapping::FusedMapping mapping,
//                        const problem::Workload& workload);

/**
 * @brief Infer logical buffer occupancies from loop nest mapping
 * 
 * @param nest 
 * @return LogicalBufOccupancies 
 */
LogicalBufOccupancies
OccupanciesFromMapping(const loop::Nest& nest,
                       const problem::Workload& workload);

};