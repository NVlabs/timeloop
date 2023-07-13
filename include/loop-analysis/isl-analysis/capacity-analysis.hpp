#include <vector>

#include <barvinok/isl.h>

#include "loop-analysis/isl-ir.hpp"
#include "mapping/fused-mapping.hpp"
#include "workload/fused-workload.hpp"

namespace analysis 
{

std::map<mapping::BufferId, isl_pw_qpolynomial*>
ComputeCapacityFromMapping(
  mapping::FusedMapping& mapping,
  const std::map<LogicalBuffer, Occupancy>& occupancies,
  const problem::FusedWorkload& workload
);

};