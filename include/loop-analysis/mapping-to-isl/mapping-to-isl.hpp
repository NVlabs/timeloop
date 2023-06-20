#pragma once

namespace analysis
{

std::map<LogicalBuffer, Occupancy>
OccupanciesFromMapping(const loop::Nest& mapping,
                       const problem::Workload& workload);

}; // namespace analysis
