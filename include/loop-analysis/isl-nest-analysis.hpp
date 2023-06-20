#pragma once

#include <loop-analysis/isl-ir.hpp>
#include <loop-analysis/spatial-analysis.hpp>

namespace analysis
{

struct LogicalBufferStats
{
  const LogicalBuffer& buf;
  Occupancy occupancy;
  Occupancy effective_occupancy;
  Fill fill;
  Transfers link_transfer;
  Reads parent_reads;

  /***************** Compatibility with Timeloop v2.0 ************************/

  std::map<std::pair<uint64_t, uint64_t>, MulticastInfo::AccessStats>
  compat_access_stats;

  /***************************************************************************/

  LogicalBufferStats(const LogicalBuffer& buf);
};

struct ReuseAnalysisOutput
{
  std::map<LogicalBuffer, LogicalBufferStats> buf_to_stats;
};

struct ReuseAnalysisInput
{
  const std::map<LogicalBuffer, Occupancy>& buf_to_occupancy;

  ReuseAnalysisInput(const std::map<LogicalBuffer, Occupancy>& buf_to_occ);
};

/**
 * @brief Computes fills of buffers as well as the transfers to fulfill those
 *   fills.
 * 
 * @see analysis::ReuseAnalysisOutput
 */
ReuseAnalysisOutput ReuseAnalysis(ReuseAnalysisInput input);

}; // namespace analysis