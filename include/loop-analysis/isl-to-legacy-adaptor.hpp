#pragma once

#include <utility>

#include "loop-analysis/nest-analysis-tile-info.hpp"
#include "loop-analysis/isl-nest-analysis.hpp"

namespace analysis
{

std::pair<CompoundComputeNest, CompoundDataMovementNest>
GenerateLegacyNestAnalysisOutput(
  const ReuseAnalysisOutput& isl_analysis_output,
  const std::vector<analysis::LoopState>& nest_state,
  const std::vector<uint64_t>& storage_tiling_boundaries,
  const std::vector<bool>& master_spatial_level,
  const std::vector<bool>& storage_boundary_level,
  const std::vector<uint64_t>& num_spatial_elems,
  const std::vector<uint64_t>& logical_fanouts,
  const problem::Workload& workload
);

}; // namespace analysis