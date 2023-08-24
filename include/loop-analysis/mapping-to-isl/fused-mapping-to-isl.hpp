#pragma once

#include <isl/polynomial.h>

#include "loop-analysis/isl-ir.hpp"
#include "loop-analysis/isl-analysis/latency-analysis.hpp"

namespace analysis
{

/**
 * @brief Results of mapping analysis that will become input into reuse
 *   analysis.
 */
struct MappingAnalysisResult
{
  /**
   * @brief Whether a logical buffer is right above a sequential node.
   * 
   * This is used when calculating capacity since some data can be dropped
   * earlier than usual when using sequential mapping without tiling.
   */
  std::map<LogicalBuffer, bool> buf_right_above_sequential;
  /**
   * @brief The occupancy of every logical buffer as defined in the mapping.
   */
  std::map<LogicalBuffer, Occupancy> lbuf_to_occupancy;
  /**
   * @brief Logical buffers found between the current root/branch node and the
   *   next one.
   */
  std::map<mapping::NodeID, std::vector<LogicalBuffer>> node_to_lbufs;
  /**
   * @brief Tiling of each branch. The tiling is a relation between tiling
   *   variables and operations.
   * 
   * An uncompletely tiled branch will have multiple-valued isl::map.
   */
  std::map<mapping::NodeID, isl::map> branch_tiling;
  /**
   * @brief We can assume an amount of parallelism to quickly calculate approx.
   *   compute latency by simply dividing number of operations with assumed
   *   parallelism.
   */
  std::map<mapping::NodeID, double> compute_to_assumed_parallelism;
  /**
   * @brief An aggregator to calculate compute latency given branch latencies.
   */
  LatencyAggregator compute_latency_aggregator;
};

/**
 * @brief Compute occupancy, tiling, and other intermediate representations
 *   from the mapping.
 * 
 * @see analysis::MappingAnalysisResult
 */
MappingAnalysisResult
OccupanciesFromMapping(mapping::FusedMapping& mapping,
                       const problem::FusedWorkload& workload);

}; // namespace analysis
