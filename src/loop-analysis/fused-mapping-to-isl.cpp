#include "loop-analysis/fused-mapping-to-ir.hpp"
#include "loop-analysis/temporal-analysis.hpp"

namespace analysis
{

/******************************************************************************
 * Local declarations
 *****************************************************************************/
void ComputeBranchOccupancy(LogicalBufOccupancies& occupancies,
                            const LogicalBuffer& buf,
                            const LogicalBufTiling& tilings,
                            const LogicalBufSkews& buf_skew,
                            const WorkloadIR& workload_ir);

void BoundTilingsWithConsumerFills(LogicalBufTiling& tilings,
                                   const LogicalBufFills& fills,
                                   const LogicalBuffer& consumer_buf,
                                   const WorkloadIR& workload_ir);

/******************************************************************************
 * Global function implementations
 *****************************************************************************/
LogicalBufOccupancies
OccupanciesFromMapping(const mapping::FusedMapping& mapping,
                       const WorkloadIR& workload_ir)
{
  LogicalBufOccupancies occupancies;
  auto tilings = LogicalBufTilingFromMapping(mapping);
  auto buf_skew = LogicalBufSkewsFromMapping(maping);

  // STEP 1: get occupancy of consumer
  // TODO: get consumer branch id, get intermediate data id, get imtermediate data buf
  EinsumID consumer_branch_id;
  LogicalBuffer consumer_buf(intermediate_data_buf,
                             intermediate_data_id,
                             consumer_branch_id);
  ComputeBranchOccupancy(occupancies, consumer_buf, tilings, buf_skew,
                         workload_ir);

  // STEP 2: get fills of consumer input buffer
  auto [_, fills] = TemporalReuseAnalysis(occupancies);

  // STEP 3: fills of consumer input buffer determines operations of producer
  BoundTilingsWithConsumerFills(tilings, fills, consumer_buf, workload_ir);

  // STEP 4: use the completed tiling to get the rest of the occupancies
  for (auto& [buf, skew] : buf_skew)
  {
    if (buf.branch_leaf_id == consumer_branch_id)
    {
      continue;
    }

    LogicalBuffer buf()
    ComputeBranchOccupancy(occupancies, buf, tilings, buf_skew, workload_ir);
  }

  return occupancies;
}

/******************************************************************************
 * Local function implementations
 *****************************************************************************/

} // namespace analysis