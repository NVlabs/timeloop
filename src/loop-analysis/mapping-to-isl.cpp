/**
 * @file mapping-to-isl.cpp
 * @author Michael Gilbert (gilbertm@mit.edu)
 * @brief Implements conversion between mapping and analysis IR
 * @version 0.1
 * @date 2023-02-07
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "loop-analysis/isl-ir.hpp"

namespace analysis
{

/******************************************************************************
 * Local function declarations
 *****************************************************************************/

std::map<DataSpaceID, IslMap>
OpsToDSpaceFromEinsum(const problem::Workload& workload);

/**
 * @brief Compute iteration -> operation map for every branch in mapping
 * 
 * @param nest 
 * @return std::map<mapping::NodeID, IslMap> 
 */
std::map<mapping::NodeID, IslMap>
TilingFromMapping(const mapping::FusedMapping& mapping);

/**
 * @brief Compute iteration -> operation map. NodeID is always 0.
 * 
 * @param nest 
 * @return std::map<mapping::NodeID, IslMap> 
 */
std::map<mapping::NodeID, IslMap>
TilingFromMapping(const Mapping& nest);

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

LogicalBufOccupancies OccupanciesFromMapping(const Mapping& nest,
                                             const problem::Workload& workload)
{
  auto ops_to_dspace = OpsToDSpaceFromEinsum(workload);
  auto branch_tiling = TilingFromMapping(nest);
}

/******************************************************************************
 * Local function implementations
 *****************************************************************************/

};
