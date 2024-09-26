#include "workload/fused-workload.hpp"

#include <boost/algorithm/string/join.hpp>
#include <iostream>
#include <barvinok/isl.h>

#include "isl-wrapper/ctx-manager.hpp"
#include "isl-wrapper/isl-functions.hpp"

namespace problem
{

FusedWorkload::FusedWorkload() {}

EinsumId FusedWorkload::NewEinsum(std::string name)
{
  auto next_einsum_id = einsum_name_to_id_.size();
  if (name == "")
  {
    name = "einsum" + std::to_string(next_einsum_id);
  }

  if (einsum_name_to_id_.find(name) != einsum_name_to_id_.end())
  {
    throw std::logic_error("there is already an einsum with name " + name);
  }
  einsum_name_to_id_[name] = next_einsum_id;
  einsum_id_to_name_[next_einsum_id] = name;

  return next_einsum_id;
}

DataSpaceId FusedWorkload::NewDataSpace(std::string name)
{
  auto next_dspace_id = dspace_name_to_id_.size();
  if (name == "")
  {
    name = "dspace" + std::to_string(next_dspace_id);
  }

  if (dspace_name_to_id_.find(name) != dspace_name_to_id_.end())
  {
    throw std::logic_error("there is already a dspace with name " + name);
  }
  dspace_name_to_id_[name] = next_dspace_id;
  dspace_id_to_name_[next_dspace_id] = name;

  return next_dspace_id;
}

DimensionId FusedWorkload::NewDimension(std::string name)
{
  auto next_dim_id = dim_name_to_id_.size();
  if (name == "")
  {
    name = "dim" + std::to_string(next_dim_id);
  }

  if (dim_name_to_id_.find(name) != dim_name_to_id_.end())
  {
    throw std::logic_error("there is already a dimension with name " + name);
  }

  dim_name_to_id_[name] = next_dim_id;
  dim_id_to_name_[next_dim_id] = name;

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

const std::map<EinsumId, std::string>& FusedWorkload::EinsumIdToName() const
{
  return einsum_id_to_name_;
}

const std::map<DataSpaceId, std::string>&
FusedWorkload::DataSpaceIdToName() const
{
  return dspace_id_to_name_;
}

const std::map<DimensionId, std::string>&
FusedWorkload::DimensionIdToName() const
{
  return dim_id_to_name_;
}

void FusedWorkload::AddDimToDspace(DataSpaceId dspace, DimensionId dim)
{
  auto& dim_to_idx = dspace_dim_to_idx_[dspace];
  auto next_idx = dim_to_idx.size();
  dim_to_idx[dim] = next_idx;
  dspace_dims_[dspace].push_back(dim);
  dim_to_dspace_[dim] = dspace;
}

void FusedWorkload::AddDimToEinsumOspace(EinsumId einsum, DimensionId dim)
{
  auto& dim_to_idx = einsum_dim_to_idx_[einsum];
  auto next_idx = dim_to_idx.size();
  dim_to_idx[dim] = next_idx;
  auto& idx_to_dim = einsum_idx_to_dim_[einsum];
  idx_to_dim[next_idx] = dim;
  einsum_dims_[einsum].push_back(dim);
  dim_to_einsum_[dim] = einsum;
}

std::optional<DataSpaceId>
FusedWorkload::GetDataSpaceWithDim(DimensionId dim) const
{
  auto it = dim_to_dspace_.find(dim);
  if (it == dim_to_dspace_.end())
  {
    return std::nullopt;
  }
  else
  {
    return it->second;
  }
}

std::optional<EinsumId> FusedWorkload::GetEinsumWithDim(DimensionId dim) const
{
  auto it = dim_to_einsum_.find(dim);
  if (it == dim_to_einsum_.end())
  {
    return std::nullopt;
  }
  else
  {
    return it->second;
  }
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

const std::map<size_t, DimensionId>&
FusedWorkload::EinsumIdxToDim(EinsumId einsum) const
{
  return einsum_idx_to_dim_.at(einsum);
}

void FusedWorkload::SetEinsumProjection(EinsumId einsum, DataSpaceId dspace,
                                        bool is_rw, isl::multi_aff proj)
{
  if (is_rw)
  {
    write_tensors_[einsum].insert(dspace);
    write_einsums_[dspace] = einsum;
    writes_.emplace(std::pair(std::pair(einsum, dspace), proj.as_map()));
    write_affs_.emplace(std::pair(std::pair(einsum, dspace), std::move(proj)));
  }
  else
  {
    read_tensors_[einsum].insert(dspace);
    read_einsums_[dspace].insert(einsum);
    reads_.emplace(std::pair(std::pair(einsum, dspace), proj.as_map()));
    read_affs_.emplace(std::pair(std::pair(einsum, dspace), std::move(proj)));
  }
}

void FusedWorkload::SetEinsumOspaceBound(EinsumId einsum, isl::set set)
{
  auto it = operation_spaces_.find(einsum);
  if (it == operation_spaces_.end())
  {
    operation_spaces_.emplace_hint(it, std::make_pair(einsum, std::move(set)));
  }
  else
  {
    it->second = std::move(set);
  }
}

void FusedWorkload::SetDataSpaceBound(DataSpaceId dspace, isl::set set)
{
  auto it = data_spaces_.find(dspace);
  if (it == data_spaces_.end())
  {
    data_spaces_.emplace_hint(it, std::make_pair(dspace, std::move(set)));
  }
  else
  {
    it->second = std::move(set);
  }
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
  return read_einsums_[dspace];
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
  return read_affs_.at(std::make_pair(einsum, dspace));
}

const isl::multi_aff&
FusedWorkload::WriteAccessesAff(EinsumId einsum, DataSpaceId dspace) const
{
  return write_affs_.at(std::make_pair(einsum, dspace));
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


const isl::map&
FusedWorkload::Accesses(EinsumId einsum, DataSpaceId dspace) const
{
  const auto& read_tensors = TensorsReadByEinsum(einsum);
  if (read_tensors.find(dspace) != read_tensors.end())
  {
    return ReadAccesses(einsum, dspace);
  }
  else
  {
    return WriteAccesses(einsum, dspace);
  }
}


const isl::set& FusedWorkload::EinsumOspaceBound(EinsumId einsum) const
{
  return operation_spaces_.at(einsum);
}


std::tuple<int, int> FusedWorkload::GetRankShape(DimensionId einsum_rank) const
{
  auto einsum_opt = GetEinsumWithDim(einsum_rank);
  if (!einsum_opt)
  {
    throw std::runtime_error("not an Einsum rank");
  }
  auto& einsum_bound = EinsumOspaceBound(*einsum_opt);
  auto rank_dim_idx = EinsumDimToIdx(*einsum_opt).at(einsum_rank);
  int max = isl::val_to_double(einsum_bound.dim_max_val(rank_dim_idx).release());
  int min = isl::val_to_double(einsum_bound.dim_min_val(rank_dim_idx).release());
  return std::make_tuple(min, max);
}


const isl::set& FusedWorkload::DataSpaceBound(DataSpaceId dspace) const
{
  auto it = data_spaces_.find(dspace);
  if (it == data_spaces_.end())
  {
    auto n_dims = dspace_dims_.at(dspace).size();
    data_spaces_.emplace_hint(
      it,
      std::make_pair(
        dspace,
        isl::set::universe(isl::manage(isl_space_set_tuple_name(
            isl_space_set_alloc(
            GetIslCtx().get(),
            0,
            n_dims
          ),
          isl_dim_set,
          dspace_id_to_name_.at(dspace).c_str()
        )))
      )
    );
  }
  return data_spaces_.at(dspace);
}


int FusedWorkload::GetTensorSize(DataSpaceId dspace) const
{
  return isl::val_to_double(
    isl::get_val_from_singular(isl_set_card((DataSpaceBound(dspace).copy())))
  );
}


FusedWorkload ParseFusedWorkload(const config::CompoundConfigNode& cfg)
{
  FusedWorkload workload;
  if (!cfg.isList())
  {
    throw std::logic_error("not a fused problem");
  }

  for (auto i = 0; i < cfg.getLength(); ++i)
  {
    auto prob_cfg = cfg[i];

    auto shape_cfg = prob_cfg.lookup("shape");

    auto einsum_name = std::string();
    shape_cfg.lookupValue("name", einsum_name);
    auto einsum = workload.NewEinsum(einsum_name);

    auto dim_names = std::vector<std::string>();
    shape_cfg.lookupArrayValue("dimensions", dim_names);
    for (const auto& dim_name : dim_names)
    {
      auto dim = workload.NewDimension(dim_name);
      workload.AddDimToEinsumOspace(einsum, dim);
    }

    auto prologue = "{ " + einsum_name + "["
      + boost::algorithm::join(dim_names, ", ") + "] ";
    auto epilogue = " }";

    auto instance = std::string();
    prob_cfg.lookupValue("instance", instance);

    try
    {
      workload.SetEinsumOspaceBound(
        einsum,
        isl::set(
          GetIslCtx(),
          prologue + " : " + instance + epilogue
        )
      );
    }
    catch (isl::exception_invalid& e)
    {
      std::cout << "Error parsing problem instance:" << std::endl
                << "  " << instance << std::endl;
      throw e;
    }
    
    auto dspaces_cfg = shape_cfg.lookup("data_spaces");
    if (!dspaces_cfg.isList())
    {
      throw std::logic_error("data_spaces key should be an array");
    }
    for (auto d = 0; d < dspaces_cfg.getLength(); ++d)
    {
      auto dspace_cfg = dspaces_cfg[d];

      auto dspace_name = std::string();
      dspace_cfg.lookupValue("name", dspace_name);

      auto is_rw = false;
      if (dspace_cfg.exists("read_write"))
      {
        dspace_cfg.lookupValue("read_write", is_rw);
      }

      auto projection_str = std::string();
      dspace_cfg.lookupValue("projection", projection_str);
      projection_str = 
        prologue + " -> " + dspace_name + projection_str + epilogue;
      auto proj = isl::multi_aff();
      try
      {
        proj = isl::multi_aff(GetIslCtx(), projection_str);
      }
      catch (...)
      {
        throw std::logic_error("malformed projection: " + projection_str);
      }

      auto dspace = DataSpaceId();
      if (workload.DataSpaceNameToId().find(dspace_name)
          == workload.DataSpaceNameToId().end())
      {
        dspace = workload.NewDataSpace(dspace_name);

        auto dspace_dim_names = std::vector<std::string>();
        dspace_cfg.lookupArrayValue("dimensions", dspace_dim_names);
        for (const auto& dspace_dim_name : dspace_dim_names)
        {
          auto dspace_dim = workload.NewDimension(dspace_dim_name);
          workload.AddDimToDspace(dspace, dspace_dim);
        }
      }
      else
      {
        dspace = workload.DataSpaceNameToId().at(dspace_name);
      }

      workload.SetEinsumProjection(einsum, dspace, is_rw, std::move(proj));
    }
  }

  // Prune data space bound based on Einsum bound
  for (const auto& [_, dspace] : workload.DataSpaceNameToId())
  {
    auto bound = workload.DataSpaceBound(dspace);

    auto writer_einsum_opt = workload.WriterEinsum(dspace);
    if (writer_einsum_opt)
    {
      const auto& einsum_bound = workload.EinsumOspaceBound(*writer_einsum_opt);
      const auto& write_access = workload.WriteAccesses(*writer_einsum_opt,
                                                        dspace);
      bound = bound.intersect(einsum_bound.apply(write_access));
    }

    if (workload.ReaderEinsums(dspace).size() > 0)
    {
      auto bound_by_reader_einsums = isl::set::empty(bound.space());
      for (const auto& reader_einsum : workload.ReaderEinsums(dspace))
      {
        const auto& einsum_bound = workload.EinsumOspaceBound(reader_einsum);
        const auto& read_access = workload.ReadAccesses(reader_einsum, dspace);
        bound_by_reader_einsums = bound_by_reader_einsums.unite(
          einsum_bound.apply(read_access)
        );
      }
      bound = bound.intersect(bound_by_reader_einsums);
    }

    workload.SetDataSpaceBound(dspace, bound);
  }

  return workload;
}

}; // namespace problem
