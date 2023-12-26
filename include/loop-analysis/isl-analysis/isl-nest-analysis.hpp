#pragma once

#include <loop-analysis/isl-ir.hpp>
#include <loop-analysis/spatial-analysis.hpp>

namespace analysis
{

struct LogicalBufferStats
{
  const LogicalBuffer& buf;

  /// @brief The occupancy of the buffer at a point in space-time.
  Occupancy occupancy;

  /**
   * @brief The occupancy of the buffer at a point in space-time where reuse
   *   temporal dimensions have been pruned.
   */
  Occupancy effective_occupancy;

  /// @brief The amount of fills the buffer requires at a point in space-time.
  Fill fill;

  /// @brief The amount of link transfers going in at a point in space-time.
  Transfers link_transfer;

  /// @brief Reads going out to parent buffer
  Reads parent_reads;

  /***************** Compatibility with Timeloop v2.0 ************************/

  /**
   * @see analysis::DataMovementInfo
   */
  std::map<std::pair<uint64_t, uint64_t>, TransferInfo::AccessStats>
  compat_access_stats;

  /***************************************************************************/

  LogicalBufferStats(const LogicalBuffer& buf);
};

struct ReuseAnalysisOutput
{
  std::map<LogicalBuffer, LogicalBufferStats> buf_to_stats;
};

struct ReuseAnalysisOptions
{
  bool count_hops;

  ReuseAnalysisOptions();
};

/**
 * @brief Computes fills of buffers as well as the transfers to fulfill those
 *   fills.
 * 
 * @see analysis::ReuseAnalysisOutput
 */
ReuseAnalysisOutput ReuseAnalysis(
  const std::map<LogicalBuffer, Occupancy>& buf_to_occ,
  const ReuseAnalysisOptions& options = ReuseAnalysisOptions()
);

}; // namespace analysis