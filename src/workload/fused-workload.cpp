#include <iostream>
#include "workload/fused-workload.hpp"

#include "isl-wrapper/ctx-manager.hpp"

namespace problem
{

FusedWorkload::FusedWorkload() {}

EinsumId FusedWorkload::NewEinsum(std::string name)
{
  auto next_einsum_id = einsum_names_.size();
  if (name == "")
  {
    name = "einsum" + next_einsum_id;
  }
  einsum_names_.emplace_back(name);
  einsum_name_to_id_[name] = next_einsum_id;
  return next_einsum_id;
}

DataSpaceId FusedWorkload::NewDataSpace(std::string name)
{
  auto next_dspace_id = dspace_names_.size();
  if (name == "")
  {
    name = "dspace" + next_dspace_id;
  }
  dspace_names_.emplace_back(name);
  dspace_name_to_id_[name] = next_dspace_id;
  return next_dspace_id;
}

DimensionId FusedWorkload::NewDimension(std::string name)
{
  auto next_dim_id = dim_names_.size();
  if (name == "")
  {
    name = "dim" + next_dim_id;
  }
  dim_names_.emplace_back(name);
  dim_name_to_id_[name] = next_dim_id;
  return next_dim_id;
}

const std::map<std::string, EinsumId>& FusedWorkload::EinsumNameToId() const
{
  return einsum_name_to_id_;
}

const std::map<std::string, DataSpaceId>&
FusedWorkload::DataSpaceNameToId() const
{
  return dspace_name_to_id_;
}

const std::map<std::string, DimensionId>&
FusedWorkload::DimensionNameToId() const
{
  return dim_name_to_id_;
}

void FusedWorkload::AddDimToDspace(DataSpaceId dspace, DimensionId dim)
{
  auto& dim_to_idx = dspace_dim_to_idx_[dspace];
  auto next_idx = dim_to_idx.size();
  dim_to_idx[dim] = next_idx;
  dspace_dims_[dspace].push_back(dim);
}

void FusedWorkload::AddDimToEinsumOspace(EinsumId einsum, DimensionId dim)
{
  auto& dim_to_idx = einsum_dim_to_idx_[einsum];
  auto next_idx = dim_to_idx.size();
  dim_to_idx[dim] = next_idx;
  einsum_dims_[einsum].push_back(dim);
}

const std::vector<DimensionId>&
FusedWorkload::DataSpaceDimensions(DataSpaceId dspace) const
{
  return dspace_dims_.at(dspace);
}

const std::vector<DimensionId>&
FusedWorkload::EinsumOspaceDimensions(EinsumId einsum) const
{
  return einsum_dims_.at(einsum);
}

const std::map<DimensionId, size_t>&
FusedWorkload::DspaceDimToIdx(DataSpaceId dspace) const
{
  return dspace_dim_to_idx_.at(dspace);
}

const std::map<DimensionId, size_t>&
FusedWorkload::EinsumDimToIdx(EinsumId einsum) const
{
  return einsum_dim_to_idx_.at(einsum);
}

void FusedWorkload::SetEinsumProjection(EinsumId einsum, DataSpaceId dspace,
                                        bool is_rw, const std::string& expr)
{
  if (is_rw)
  {
    write_einsums_[dspace] = einsum;
    write_tensors_[einsum].emplace(dspace);
    write_exprs_.emplace(std::pair(std::pair(einsum, dspace),
                                   isl::multi_aff(GetIslCtx(), expr)));
    writes_.emplace(std::pair(std::pair(einsum, dspace),
                              isl::map(GetIslCtx(), expr)));
  }
  else
  {
    read_einsums_[dspace].emplace(einsum);
    read_tensors_[einsum].emplace(dspace);
    read_exprs_.emplace(std::pair(std::pair(einsum, dspace),
                                  isl::multi_aff(GetIslCtx(), expr)));
    reads_.emplace(std::pair(std::pair(einsum, dspace),
                             isl::map(GetIslCtx(), expr)));
  }
}

void FusedWorkload::SetEinsumOspaceBound(EinsumId einsum,
                                         const std::string& expr)
{
  operation_spaces_.emplace(std::make_pair(einsum,
                                           isl::set(GetIslCtx(), expr)));
}

void FusedWorkload::SetDataSpaceBound(DataSpaceId dspace,
                                      const std::string& expr)
{
  data_spaces_.emplace(std::make_pair(dspace, isl::set(GetIslCtx(), expr)));
}

const std::set<DataSpaceId>&
FusedWorkload::TensorsReadByEinsum(EinsumId einsum) const
{
  return read_tensors_.at(einsum);
}

const std::set<DataSpaceId>&
FusedWorkload::TensorsWrittenByEinsum(EinsumId einsum) const
{
  return write_tensors_.at(einsum);
}

const std::set<EinsumId>&
FusedWorkload::ReaderEinsums(DataSpaceId dspace) const
{
  return read_einsums_.at(dspace);
}

std::optional<EinsumId>
FusedWorkload::WriterEinsum(DataSpaceId dspace) const
{
  auto it = write_einsums_.find(dspace);
  if (it == write_einsums_.end())
  {
    return std::nullopt;
  }
  else
  {
    return it->second;
  }
}

const isl::multi_aff&
FusedWorkload::ReadAccessesAff(EinsumId einsum, DataSpaceId dspace) const
{
  return read_exprs_.at(std::make_pair(einsum, dspace));
}

const isl::multi_aff&
FusedWorkload::WriteAccessesAff(EinsumId einsum, DataSpaceId dspace) const
{
  return write_exprs_.at(std::make_pair(einsum, dspace));
}

const isl::map&
FusedWorkload::ReadAccesses(EinsumId einsum, DataSpaceId dspace) const
{
  return reads_.at(std::make_pair(einsum, dspace));
}

const isl::map&
FusedWorkload::WriteAccesses(EinsumId einsum, DataSpaceId dspace) const
{
  return writes_.at(std::make_pair(einsum, dspace));
}

FusedWorkload ParseFusedWorkload(const config::CompoundConfigNode& cfg)
{
  (void) cfg;
  FusedWorkload workload;

  std::string type;

  cfg.lookupValue("type", type);
  if (type != "fused")
  {
    throw std::logic_error("Not a fused problem.");
  }

  if (!cfg.exists("dspaces"))
  {
    throw std::logic_error("Problem does not have dspaces");
  }
  auto dspaces_cfg = cfg.lookup("dspaces");
  for (int i = 0; i < dspaces_cfg.getLength(); ++i)
  {
    auto dspace_cfg = dspaces_cfg[i];

    std::string dspace_name;
    dspace_cfg.lookupValue("name", dspace_name);
    std::cout << dspace_name << std::endl;
    auto dspace = workload.NewDataSpace(dspace_name);

    std::vector<std::string> dim_names;
    dspace_cfg.lookupArrayValue("dimensions", dim_names);
    for (const auto& dim_name : dim_names)
    {
      auto dim = workload.NewDimension(dim_name);
      workload.AddDimToDspace(dspace, dim);
    }

    std::string size_str;
    dspace_cfg.lookupValue("size", size_str);
    workload.SetDataSpaceBound(dspace, size_str);
  }

  if (!cfg.exists("einsums"))
  {
    throw std::logic_error("Problem does not have einsums");
  }
  auto einsums_cfg = cfg.lookup("einsums");
  for (int i = 0; i < einsums_cfg.getLength(); ++i)
  {
    auto einsum_cfg = einsums_cfg[i];

    std::string einsum_name;
    einsum_cfg.lookupValue("name", einsum_name);
    auto einsum = workload.NewEinsum(einsum_name);

    std::vector<std::string> dim_names;
    einsum_cfg.lookupArrayValue("dimensions", dim_names);
    for (const auto& dim_name : dim_names)
    {
      auto dim = workload.NewDimension(dim_name);
      workload.AddDimToEinsumOspace(einsum, dim);
    }

    auto projections_cfg = einsum_cfg.lookup("projections");
    for (int proj_idx = 0; proj_idx < projections_cfg.getLength(); ++proj_idx)
    {
      auto proj_cfg = projections_cfg[proj_idx];

      std::string dspace_name;
      proj_cfg.lookupValue("dspace", dspace_name);

      std::string proj_str;
      proj_cfg.lookupValue("projection", proj_str);

      bool read_write;
      proj_cfg.lookupValue("read_write", read_write);

      workload.SetEinsumProjection(
        einsum,
        workload.DataSpaceNameToId().at(dspace_name),
        read_write,
        proj_str
      );
    }
  }

  return workload;
}

}; // namespace problem