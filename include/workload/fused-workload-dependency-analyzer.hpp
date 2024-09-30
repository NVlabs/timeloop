#pragma once

#include "workload/fused-workload.hpp"


namespace problem
{

class FusedWorkloadDependencyAnalyzer
{
 public:
  FusedWorkloadDependencyAnalyzer(const FusedWorkload& workload);

  std::vector<std::vector<EinsumId>>
  FindEinsumDependencyChain(EinsumId src, EinsumId dst) const;

  /**
   * Returns whether `einsum_dim` is *directly* relevant to `dspace`, i.e.,
   * only returns true if the dspace is in `einsum` and `einsum_dim` is
   * relevant.
   */
  bool EinsumDimIsDirectlyRelevantToTensor(EinsumId einsum,
                                           DimensionId einsum_dim,
                                           DataSpaceId dspace) const;

  /**
   * Returns whether `einsum_dim` is relevant to `dspace`, i.e., if
   * partitioning `einsum_dim` creates tiles of `dspace` smaller than `dspace`.
   */
  bool EinsumDimIsRelevantToTensor(EinsumId einsum,
                                   DimensionId einsum_dim,
                                   DataSpaceId dspace) const;

  const std::set<DimensionId>&
  EinsumDimsDirectlyRelevantToTensor(EinsumId einsum, DataSpaceId dspace) const;

  const std::set<DimensionId>&
  EinsumDimsRelevantToTensor(EinsumId einsum, DataSpaceId dspace) const;

 private:
  const FusedWorkload& workload_;

  mutable std::map<std::pair<EinsumId, DataSpaceId>, std::set<DimensionId>>
    directly_relevant_einsum_dim_memo_;
  mutable std::map<std::pair<EinsumId, DataSpaceId>, std::set<DimensionId>>
    relevant_einsum_dim_memo_;
};

};