#pragma once

#include "loop-analysis/isl-ir.hpp"

LogicalBufOccupancies
OccupanciesFromMapping(const mapping::FusedMapping& mapping,
                       const WorkloadIR& workload_ir);
