#pragma once

#include "loop-analysis/isl-ir.hpp"

namespace analysis
{

struct BufTemporalReuseOpts
{
  bool exploit_temporal_reuse;
};

struct TemporalReuseAnalysisInput
{
  const Occupancy& occupancy;
  BufTemporalReuseOpts reuse_opts;

  TemporalReuseAnalysisInput(const Occupancy& occupancy,
                             BufTemporalReuseOpts reuse_opts);
};

struct TemporalReuseAnalysisOutput
{
  Occupancy effective_occupancy;
  Fill fill;
};

/**
 * @brief Computes the required fill to satisfy the buffer occupancy.
 * 
 * If the buffer can `exploit_temporal_reuse`, then the fill will only consist
 * of data not currently resident in buffer.
 * 
 * @see analysis::TemporalReuseAnalysisInput
 * @see analysis::TemporalReuseAnalysisOutput
 */
TemporalReuseAnalysisOutput
TemporalReuseAnalysis(TemporalReuseAnalysisInput input);

};  // namespace analysis
