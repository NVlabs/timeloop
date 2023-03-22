#include "loop-analysis/isl-ir.hpp"

#include "isl-wrapper/ctx-manager.hpp"

namespace analysis
{

WorkloadIR::WorkloadIR() : next_einsum_id_(0), next_dspace_id_(0) {}

EinsumID WorkloadIR::NewEinsum()
{
  return next_einsum_id_++;
}

DataSpaceID WorkloadIR::NewDataSpace()
{
  return next_dspace_id_++;
}

void WorkloadIR::AddReadDependency(EinsumID einsum_id, DataSpaceID dspace_id,
                                   const std::string& map_str)
{
  reads_.emplace(std::pair(std::pair(einsum_id, dspace_id),
                           isl::map(GetIslCtx(), map_str)));
}
void WorkloadIR::AddWriteDependency(EinsumID einsum_id, DataSpaceID dspace_id,
                                    const std::string& map_str)
{
  writes_.emplace(std::pair(std::pair(einsum_id, dspace_id),
                            isl::map(GetIslCtx(), map_str)));
}

void WorkloadIR::AddOperationSpaceBounds(EinsumID einsum_id,
                                         const std::string& set_str)
{
  operation_spaces_.emplace(std::make_pair(einsum_id,
                                           isl::set(GetIslCtx(), set_str)));
}
void WorkloadIR::AddDataSpaceBounds(DataSpaceID dspace_id,
                                    const std::string& set_str)
{
  data_spaces_.emplace(std::make_pair(dspace_id,
                                      isl::set(GetIslCtx(), set_str)));
}

WorkloadIR::ConstIterator
WorkloadIR::GetReadDependency(EinsumID einsum_id, DataSpaceID dspace_id) const
{
  return reads_.find(std::make_pair(einsum_id, dspace_id));
}
WorkloadIR::ConstIterator 
WorkloadIR::GetWriteDependency(EinsumID einsum_id, DataSpaceID dspace_id) const
{
  return writes_.find(std::make_pair(einsum_id, dspace_id));
}

} // namespace analysis