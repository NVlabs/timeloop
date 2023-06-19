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
  const LogicalBuffer& logical_buffer;
  const Occupancy& occupancy;
  BufTemporalReuseOpts reuse_opts;

  TemporalReuseAnalysisInput(const LogicalBuffer& logical_buffer,
                             const Occupancy& occupancy,
                             BufTemporalReuseOpts reuse_opts);
};

struct TemporalReuseAnalysisOutput
{
  Occupancy effective_occupancy;
  Fill fill;
};

TemporalReuseAnalysisOutput
TemporalReuseAnalysis(TemporalReuseAnalysisInput input);

};  // namespace analysis
