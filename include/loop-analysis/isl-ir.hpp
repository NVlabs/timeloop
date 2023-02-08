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

#include "isl-wrapper/tagged.hpp"
#include "mapping/fused-mapping.hpp"
#include "mapping/mapping.hpp"
#include "workload/workload.hpp"

namespace analysis
{

/******************************************************************************
 * Intermediate representation between mapping and analysis
 *****************************************************************************/
using DataSpaceID = problem::Shape::DataSpaceID;
using FactorizedDimensionID = problem::Shape::FactorizedDimensionID;
using BufferID = unsigned;

struct LogicalBuffer
{
  BufferID buffer_id;
  DataSpaceID dspace_id;
  mapping::NodeID branch_leaf_id;

  bool operator<(const LogicalBuffer& other) const
  {
    return buffer_id < other.buffer_id;
  }
};

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
using Tiling = IslMap;
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
using DataDistribution = IslMap;
using LogicalBufDataDistributions = std::map<LogicalBuffer, DataDistribution>;

/**
 * @brief Space-Time -> Iteration relation that specifies distribution of tiles
 *        in time and space.
 * 
 * The default is inferred from the mapping.
 * 
 */
using Skew = IslMap;
using LogicalBufSkews = std::map<LogicalBuffer, Skew>;

/**
 * @brief Space-Time -> Data of a logical buffer. A complete description of
 *        data held in a logical buffer.
 * 
 */
using Occupancy = TaggedMap<IslMap, spacetime::Dimension>;
using LogicalBufOccupancies = std::map<LogicalBuffer, Occupancy>;

/**
 * @brief TARDIS-style X-relation.
 * 
 * This is used:
 *   - To calculate access counts which is passed to the uarch model
 *   - As input to TARDIS for code-gen
 */
using Transfers = IslMap;
using LogicalBufTransfers = std::map<std::pair<LogicalBuffer, LogicalBuffer>,
                                     Transfers>;

/******************************************************************************
 * Converter from mapping to intermediate representation
 *****************************************************************************/

/**
 * @brief Infer logical buffer occupancies from fused mapping
 * 
 * @param mapping 
 * @return LogicalBufOccupancies 
 */
LogicalBufOccupancies
OccupanciesFromMapping(const mapping::FusedMapping mapping);

/**
 * @brief Infer logical buffer occupancies from loop nest mapping
 * 
 * @param nest 
 * @return LogicalBufOccupancies 
 */
LogicalBufOccupancies OccupanciesFromMapping(const Mapping& nest);

};