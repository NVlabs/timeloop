#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <set>

#include <isl/cpp.h>

#include "compound-config/compound-config.hpp"

namespace problem
{

// TODO: this Einsum class is the old Workload class
using EinsumId = size_t;
using DataSpaceId = size_t;
using DimensionId = size_t;

// TODO: this should be the new Workload class
class FusedWorkload
{
 public:
  FusedWorkload();

  EinsumId NewEinsum(std::string name="");
  DataSpaceId NewDataSpace(std::string name="");
  DimensionId NewDimension(std::string name="");

  const std::map<std::string, EinsumId>& EinsumNameToId() const;
  const std::map<std::string, DataSpaceId>& DataSpaceNameToId() const;
  const std::map<std::string, DimensionId>& DimensionNameToId() const;

  void AddDimToDspace(DataSpaceId dspace, DimensionId dspace_dim);
  void AddDimToEinsumOspace(EinsumId einsum, DimensionId dim);

  const std::vector<DimensionId>&
  DataSpaceDimensions(DataSpaceId dspace) const;
  const std::vector<DimensionId>&
  EinsumOspaceDimensions(EinsumId einsum) const;

  size_t DspaceDimToIdx(DataSpaceId dspace, DimensionId dim) const;
  size_t EinsumDimToIdx(EinsumId einsum, DimensionId dim) const;

  void SetEinsumProjection(EinsumId einsum, DataSpaceId dspace, bool is_rw,
                           const std::string& expr);
  void SetEinsumOspaceBound(EinsumId einsum, const std::string& expr);
  void SetDataSpaceBound(DataSpaceId dspace, const std::string& expr);

  const std::set<DataSpaceId>& TensorsReadByEinsum(EinsumId einsum) const;
  const std::set<DataSpaceId>& TensorsWrittenByEinsum(EinsumId einsum) const;
  const std::set<EinsumId>& ReaderEinsums(DataSpaceId dspace) const;
  std::optional<EinsumId> WriterEinsum(DataSpaceId dspace) const;

  const isl::map& ReadAccesses(EinsumId einsum, DataSpaceId dspace) const;
  const isl::map& WriteAccesses(EinsumId einsum, DataSpaceId dspace) const;

 private:
  std::vector<std::string> einsum_names_;
  std::vector<std::string> dspace_names_;
  std::vector<std::string> dim_names_;
  std::map<std::string, EinsumId> einsum_name_to_id_;
  std::map<std::string, DataSpaceId> dspace_name_to_id_;
  std::map<std::string, DimensionId> dim_name_to_id_;

  std::map<EinsumId, std::set<DataSpaceId>> read_tensors_;
  std::map<EinsumId, std::set<DataSpaceId>> write_tensors_;
  std::map<DataSpaceId, std::set<EinsumId>> read_einsums_;
  std::map<DataSpaceId, EinsumId> write_einsums_;

  std::map<DataSpaceId, std::vector<DimensionId>> dspace_dims_;
  std::map<DataSpaceId, std::map<DimensionId, size_t>> dspace_dim_to_idx_;
  std::map<DataSpaceId, std::vector<DimensionId>> einsum_dims_;
  std::map<EinsumId, std::map<DimensionId, size_t>> einsum_dim_to_idx_;

  std::map<std::pair<EinsumId, DataSpaceId>, isl::map> reads_;
  std::map<std::pair<EinsumId, DataSpaceId>, isl::map> writes_;

  std::map<EinsumId, isl::set> operation_spaces_;
  std::map<DataSpaceId, isl::set> data_spaces_;
};

FusedWorkload ParseFusedWorkload(const config::CompoundConfigNode& cfg);

};