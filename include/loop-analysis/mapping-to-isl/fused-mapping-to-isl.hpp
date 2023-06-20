#pragma once

#include "loop-analysis/isl-ir.hpp"

namespace analysis
{

std::map<LogicalBuffer, Occupancy>
OccupanciesFromMapping(mapping::FusedMapping& mapping,
                       const problem::FusedWorkload& workload);

}; // namespace analysis
