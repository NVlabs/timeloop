#pragma once

#include "loop-analysis/isl-ir.hpp"
#include "mapping/mapping.hpp"

namespace analysis
{

std::map<LogicalBuffer, Occupancy>
OccupanciesFromMapping(const loop::Nest& mapping,
                       const problem::Workload& workload);

}; // namespace analysis
