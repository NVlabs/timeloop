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
  read_einsums_[dspace_id].emplace(einsum_id);
  read_tensors_[einsum_id].emplace(dspace_id);
  reads_.emplace(std::pair(std::pair(einsum_id, dspace_id),
                           isl::map(GetIslCtx(), map_str)));
}
void WorkloadIR::AddWriteDependency(EinsumID einsum_id, DataSpaceID dspace_id,
                                    const std::string& map_str)
{
  write_einsums_[dspace_id] = einsum_id;
  write_tensors_[einsum_id].emplace(dspace_id);
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

const std::set<DataSpaceID>& WorkloadIR::ReadTensors(EinsumID einsum_id) const
{
  return read_tensors_.at(einsum_id);
}
const std::set<DataSpaceID>& WorkloadIR::WriteTensors(EinsumID einsum_id) const
{
  return write_tensors_.at(einsum_id);
}

const std::set<EinsumID>& WorkloadIR::ReadEinsums(DataSpaceID dspace_id) const
{
  return read_einsums_.at(dspace_id);
}
const std::optional<EinsumID>
WorkloadIR::WriteEinsum(DataSpaceID dspace_id) const
{
  auto it = write_einsums_.find(dspace_id);
  if (it == write_einsums_.end())
  {
    return std::nullopt;
  }
  else
  {
    return it->second;
  }
}

const isl::map&
WorkloadIR::GetReadDependency(EinsumID einsum_id, DataSpaceID dspace_id) const
{
  return reads_.at(std::make_pair(einsum_id, dspace_id));
}
const isl::map&
WorkloadIR::GetWriteDependency(EinsumID einsum_id, DataSpaceID dspace_id) const
{
  return writes_.at(std::make_pair(einsum_id, dspace_id));
}

} // namespace analysis