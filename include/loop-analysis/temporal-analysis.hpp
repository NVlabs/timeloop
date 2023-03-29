#pragma once

#include "loop-analysis/isl-ir.hpp"

namespace analysis
{

std::pair<LogicalBufOccupancies, LogicalBufFills>
TemporalReuseAnalysis(const LogicalBufOccupancies& occupancies);

} // namespace analysis
