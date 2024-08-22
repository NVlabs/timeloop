#include "workload/fused-workload-dependency-analyzer.hpp"


namespace problem
{

FusedWorkloadDependencyAnalyzer::FusedWorkloadDependencyAnalyzer(
  const FusedWorkload& workload
) : workload_(workload)
{
}

std::vector<std::vector<EinsumId>>
FusedWorkloadDependencyAnalyzer::FindEinsumDependencyChain(
  EinsumId src,
  EinsumId dst
) const
{
  // The strategy used here is to use depth-first search to find all paths.
  // The stack keeps track of (node, all paths to node) pairs.
  std::vector<std::pair<EinsumId, std::vector<EinsumId>>>
  dfs_stack = {{src, {src}}};
  std::vector<std::vector<EinsumId>> result;

  if (src == dst)
  {
    result.emplace_back(std::vector({src}));
  }

  while (dfs_stack.size() > 0)
  {
    const auto [cur_einsum, cur_einsum_path] = std::move(dfs_stack.back());
    dfs_stack.pop_back();

    const auto cur_einsum_outputs = workload_.TensorsWrittenByEinsum(cur_einsum);
    for (const auto& cur_einsum_output : cur_einsum_outputs)
    {
      const auto readers = workload_.ReaderEinsums(cur_einsum_output);
      for (const auto neighbor_einsum : readers)
      {
        auto path_to_neighbor = cur_einsum_path;
        path_to_neighbor.emplace_back(neighbor_einsum);

        if (neighbor_einsum == dst)
        {
          result.emplace_back(path_to_neighbor);
        }
        else
        {
          const auto it = std::find(cur_einsum_path.begin(),
                                    cur_einsum_path.end(),
                                    neighbor_einsum);
          if (it == cur_einsum_path.end())  // Only keep searching if not acyclic
          {
            dfs_stack.emplace_back(
              std::make_pair(neighbor_einsum, path_to_neighbor)
            );
          }
        }

      }
    }
  }

  return result;
}

bool FusedWorkloadDependencyAnalyzer::EinsumDimIsDirectlyRelevantToTensor(
  EinsumId einsum,
  DimensionId einsum_dim,
  DataSpaceId dspace
) const
{
  const auto& read_tensors = workload_.TensorsReadByEinsum(einsum);
  const auto is_read_tensor = read_tensors.find(dspace) != read_tensors.end();

  const auto& written_tensors = workload_.TensorsWrittenByEinsum(einsum);
  const auto is_written_tensor =
      written_tensors.find(dspace) != written_tensors.end();

  if (!is_read_tensor && !is_written_tensor)
  {
    return false;
  }

  const auto dim_idx = workload_.EinsumDimToIdx(einsum).at(einsum_dim);
  const auto& projection = workload_.Accesses(einsum, dspace);

  const auto dim_involved = isl_map_involves_dims(projection.get(),
                                                  isl_dim_in,
                                                  dim_idx,
                                                  1);
  return dim_involved == isl_bool_true;
}

bool FusedWorkloadDependencyAnalyzer::EinsumDimIsRelevantToTensor(
  EinsumId einsum,
  DimensionId einsum_dim,
  DataSpaceId dspace
) const
{
  const auto dim_idx = workload_.EinsumDimToIdx(einsum).at(einsum_dim);

  // Priorotize `einsum` to check relevancy of `dspace`
  std::set<EinsumId> src_einsums;
  const auto& cur_einsum_read_tensors = workload_.TensorsReadByEinsum(einsum);
  const auto it = cur_einsum_read_tensors.find(dspace);
  if (it != cur_einsum_read_tensors.end())
  {
    src_einsums.emplace(einsum);
  }
  if (src_einsums.size() == 0)
  {
    const auto& cur_einsum_written_tensors = workload_.TensorsWrittenByEinsum(einsum);
    const auto it = cur_einsum_written_tensors.find(dspace);
    if (it != cur_einsum_written_tensors.end())
    {
      src_einsums.emplace(einsum);
    }
  }

  if (src_einsums.size() == 0)
  {
    const auto& reader_einsums = workload_.ReaderEinsums(dspace);
    src_einsums.insert(reader_einsums.begin(), reader_einsums.end());
  }
  if (src_einsums.size() == 0)
  {
    auto writer_einsum = workload_.WriterEinsum(dspace);
    src_einsums.insert(*writer_einsum);
  }

  for (const auto src : src_einsums)
  {
    const auto chains = FindEinsumDependencyChain(src, einsum);

    for (const auto& chain : chains)
    {
      auto accesses = workload_.Accesses(src, dspace);
      auto cur_tensor = dspace;
      for (const auto& e : chain)
      {
        if (e != src)
        {
          accesses = workload_.Accesses(e, cur_tensor).apply_range(accesses);
        }

        if (e == chain.back())
        {
          const auto dim_involved = isl_map_involves_dims(accesses.get(),
                                                          isl_dim_in,
                                                          dim_idx,
                                                          1);
          if (dim_involved == isl_bool_true)
          {
            return true;
          }
        }
        else
        {
          const auto cur_intermediate_tensor =
              *workload_.TensorsWrittenByEinsum(e).begin();
          const auto& write_of_cur_intermediate =
              workload_.Accesses(e, cur_intermediate_tensor);
          accesses = write_of_cur_intermediate.reverse().apply_range(accesses);
          cur_tensor = cur_intermediate_tensor;
        }
      }
    }
  }

  return false;
}

const std::set<DimensionId>&
FusedWorkloadDependencyAnalyzer::EinsumDimsDirectlyRelevantToTensor(
  EinsumId einsum,
  DataSpaceId dspace
) const
{
  const auto key = std::make_pair(einsum, dspace);
  auto it = directly_relevant_einsum_dim_memo_.find(key);
  if (it != directly_relevant_einsum_dim_memo_.end())
  {
    return it->second;
  }
  else
  {
    std::set<DimensionId> relevant_dims;
    for (const auto einsum_dim : workload_.EinsumOspaceDimensions(einsum))
    {
      if (EinsumDimIsDirectlyRelevantToTensor(einsum, einsum_dim, dspace))
      {
        relevant_dims.insert(einsum_dim);
      }
    }

    directly_relevant_einsum_dim_memo_.emplace_hint(
      it,
      std::make_pair(key, std::move(relevant_dims))
    );

    return directly_relevant_einsum_dim_memo_.at(key);
  }
}

const std::set<DimensionId>&
FusedWorkloadDependencyAnalyzer::EinsumDimsRelevantToTensor(
  EinsumId einsum,
  DataSpaceId dspace
) const
{
  const auto key = std::make_pair(einsum, dspace);
  auto it = relevant_einsum_dim_memo_.find(key);
  if (it != relevant_einsum_dim_memo_.end())
  {
    return it->second;
  }
  else
  {
    std::set<DimensionId> relevant_dims;
    for (const auto einsum_dim : workload_.EinsumOspaceDimensions(einsum))
    {
      if (EinsumDimIsRelevantToTensor(einsum, einsum_dim, dspace))
      {
        relevant_dims.insert(einsum_dim);
      }
    }

    relevant_einsum_dim_memo_.emplace_hint(
      it,
      std::make_pair(key, std::move(relevant_dims))
    );

    return relevant_einsum_dim_memo_.at(key);
  }
}
}