#pragma once

#include "isl-wrapper/isl-functions.hpp"
#include "loop-analysis/isl-ir.hpp"

struct SharedInputTensor
{
  problem::DataSpaceId shared_tensor;
};

struct NoSharedInputTensor {};

struct MultipleSharedInputTensor {};

using SharedInputTensorDetectionOutcome =
  std::variant<SharedInputTensor,
               NoSharedInputTensor,
               MultipleSharedInputTensor>;

template<typename EinsumsIterT>
SharedInputTensorDetectionOutcome DetectSharedInputTensor(
  EinsumsIterT einsums_begin, EinsumsIterT einsums_end,
  const problem::FusedWorkload& workload
)
{
  size_t n_einsums = 0;
  auto tensor_read_counts = std::map<problem::DataSpaceId, size_t>();
  for (auto it = einsums_begin; it != einsums_end; ++it)
  {
    const auto& einsum = *it;
    for (const auto& tensor : workload.TensorsReadByEinsum(einsum))
    {
      if (tensor_read_counts.find(tensor) != tensor_read_counts.end())
      {
        tensor_read_counts.at(tensor) += 1;
      }
      else
      {
        tensor_read_counts[tensor] = 1;
      }
    }
    ++n_einsums;
  }
  auto shared_input_tensor = std::optional<problem::DataSpaceId>();
  for (const auto& [tensor, counts] : tensor_read_counts)
  {
    if (counts == n_einsums)
    {
      if (!shared_input_tensor)
      {
        shared_input_tensor = tensor;
      }
      else
      {
        return MultipleSharedInputTensor();
      }
    }
  }

  if (!shared_input_tensor)
  {
    return NoSharedInputTensor();
  }
  else
  {
    return SharedInputTensor{*shared_input_tensor};
  }
}

template<typename EinsumsIterT>
void SharedInputBasedTileShapeInference(
  std::map<problem::EinsumId, isl::map>& tiling_info,
  EinsumsIterT einsums_begin, EinsumsIterT einsums_end,
  const problem::FusedWorkload& workload,
  problem::EinsumId tiled_einsum
)
{
  auto shared_tensor = std::get<SharedInputTensor>(
    DetectSharedInputTensor(einsums_begin, einsums_end, workload)
  ).shared_tensor;

  auto tiled_einsum_read_accesses = workload.ReadAccesses(tiled_einsum,
                                                          shared_tensor);
  auto read_data =
    tiling_info.at(tiled_einsum).apply_range(tiled_einsum_read_accesses);

  for (auto it = einsums_begin; it != einsums_end; ++it)
  {
    const auto einsum = *it;
    if (einsum == tiled_einsum)
    {
      continue;
    }

    auto read_accesses = workload.ReadAccesses(einsum, shared_tensor);
    auto executable_operations = read_data.apply_range(read_accesses.reverse());
    executable_operations = executable_operations.intersect_range(
      workload.EinsumOspaceBound(einsum)
    );

    // WARNING: mutation!
    auto& tiling = tiling_info.at(einsum);
    tiling = tiling.intersect(executable_operations);
  }
}
