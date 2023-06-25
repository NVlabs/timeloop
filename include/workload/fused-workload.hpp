#pragma once

#include <boost/graph/adjacency_list.hpp>
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

  /**
   * @brief Returns dataspace id given `name`
   * 
   * @throws std::out_of_range if name could not be found.
   */
  DataSpaceId GetDspaceId(const std::string& name) const;

  /**
   * @brief Returns dimension id given `name`
   * 
   * @throws std::out_of_range if name could not be found.
   */
  DimensionId GetDimensionId(const std::string& name) const;

  void AddDimToDspace(DataSpaceId dspace, DimensionId dspace_dim);
  void AddDimToEinsumOspace(EinsumId einsum, DimensionId dim);

  const std::vector<DimensionId>&
  DataSpaceDimensions(DataSpaceId dspace) const;
  const std::vector<DimensionId>&
  EinsumOspaceDimensions(EinsumId einsum) const;

  const std::map<DimensionId, size_t>&
  DspaceDimToIdx(DataSpaceId dspace) const;
  const std::map<DimensionId, size_t>& EinsumDimToIdx(EinsumId einsum) const;

  void SetEinsumProjection(EinsumId einsum, DataSpaceId dspace, bool is_rw,
                           isl::multi_aff projection);
  void SetEinsumOspaceBound(EinsumId einsum, isl::set set);
  void SetDataSpaceBound(DataSpaceId dspace, isl::set set);

  const std::set<DataSpaceId>& TensorsReadByEinsum(EinsumId einsum) const;
  const std::set<DataSpaceId>& TensorsWrittenByEinsum(EinsumId einsum) const;
  const std::set<EinsumId>& ReaderEinsums(DataSpaceId dspace) const;
  std::optional<EinsumId> WriterEinsum(DataSpaceId dspace) const;

  const isl::multi_aff&
  ReadAccessesAff(EinsumId einsum, DataSpaceId dspace) const;
  const isl::multi_aff&
  WriteAccessesAff(EinsumId einsum, DataSpaceId dspace) const;
  const isl::map& ReadAccesses(EinsumId einsum, DataSpaceId dspace) const;
  const isl::map& WriteAccesses(EinsumId einsum, DataSpaceId dspace) const;

  const isl::set& EinsumOspaceBound(EinsumId einsum) const;
  const isl::set& DataSpaceBound(DataSpaceId dspace) const;

 private:
  std::map<std::string, EinsumId> einsum_name_to_id_;
  std::map<EinsumId, std::string> einsum_id_to_name_;
  std::map<std::string, DataSpaceId> dspace_name_to_id_;
  std::map<DataSpaceId, std::string> dspace_id_to_name_;
  std::map<std::string, DimensionId> dim_name_to_id_;

  enum class EinsumOrDspace { EINSUM, DATASPACE };
  struct VertexProperty
  {
    EinsumOrDspace einsum_or_dspace;
  };
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                                VertexProperty>
    EinsumGraph;
  typedef EinsumGraph::vertex_descriptor EinsumGraphVertex;
  typedef EinsumGraph::edge_descriptor EinsumGraphEdge;

  EinsumGraph einsum_graph_;
  std::map<EinsumId, EinsumGraphVertex> einsum_id_to_vertex_;
  std::map<DataSpaceId, EinsumGraphVertex> dspace_id_to_vertex_;

  std::map<EinsumId, std::set<DataSpaceId>> read_tensors_;
  std::map<EinsumId, std::set<DataSpaceId>> write_tensors_;
  mutable std::map<DataSpaceId, std::set<EinsumId>> read_einsums_;
  std::map<DataSpaceId, EinsumId> write_einsums_;

  std::map<DataSpaceId, std::vector<DimensionId>> dspace_dims_;
  std::map<DataSpaceId, std::map<DimensionId, size_t>> dspace_dim_to_idx_;
  std::map<DataSpaceId, std::vector<DimensionId>> einsum_dims_;
  std::map<EinsumId, std::map<DimensionId, size_t>> einsum_dim_to_idx_;

  std::map<std::pair<EinsumId, DataSpaceId>, isl::map> reads_;
  std::map<std::pair<EinsumId, DataSpaceId>, isl::multi_aff> read_affs_;
  std::map<std::pair<EinsumId, DataSpaceId>, isl::map> writes_;
  std::map<std::pair<EinsumId, DataSpaceId>, isl::multi_aff> write_affs_;

  std::map<EinsumId, isl::set> operation_spaces_;
  // mutable because FusedWorkload::DataSpaceBound will create a new universe
  // set and cache it in data_spaces_ if a set cannot be found
  // FIX: how this should work is that calling NewDataSpace and NewEinsum
  //      should require the dimensions so that the bounds can be created at
  //      at call instead of lazily at FusedWorkload::DataSpaceBound
  mutable std::map<DataSpaceId, isl::set> data_spaces_;
};

FusedWorkload ParseFusedWorkload(const config::CompoundConfigNode& cfg);

};