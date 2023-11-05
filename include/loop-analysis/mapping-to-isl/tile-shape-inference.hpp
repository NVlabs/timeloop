#pragma once

#include "isl-wrapper/isl-functions.hpp"
#include "loop-analysis/isl-ir.hpp"

template<typename EinsumsIterT>
void SharedInputBasedTileShapeInference(
  std::map<problem::EinsumId, isl::map>& tiling_info,
  const std::map<problem::DataSpaceId, size_t>& dspace_to_reuse_level,
  EinsumsIterT einsums_begin, EinsumsIterT einsums_end,
  const problem::FusedWorkload& workload,
  problem::EinsumId tiled_einsum
)
{
  size_t n_einsums = 0;
  for (auto it = einsums_begin; it != einsums_end; ++it)
  {
    const auto& einsum = *it;
    auto tensor_read_counts = std::map<problem::DataSpaceId, size_t>();
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
        throw std::logic_error("Share only one tensor for valid inference");
      }
    }
  }
  if (!shared_input_tensor)
  {
    throw std::logic_error("There is tensor shared by all einsums");
  }

  auto queue = std::deque<problem::EinsumId>({tiled_einsum});
  while (queue.size() > 0)
  {
    auto einsum = queue.front();
    queue.pop_front();

    const auto& tiling = tiling_info.at(einsum);
    for (auto tensor : workload.TensorsReadByEinsum(einsum))
    {
      auto producer_einsum_opt = workload.WriterEinsum(tensor);
      if (!producer_einsum_opt) // Not an intermediate tensor
      {
        continue;
      }

      auto prod_einsum = *producer_einsum_opt;
      if (std::find(einsums_begin, einsums_end, prod_einsum) == einsums_end)
      {
        // Not in this fusion set
        continue;
      }

      auto read_accesses = workload.ReadAccesses(einsum, tensor);
      auto required_data = tiling.apply_range(read_accesses);

      isl::map computed_data = required_data;
      auto it = dspace_to_reuse_level.find(tensor);
      if (it != dspace_to_reuse_level.end())
      {
        auto reuse_level = it->second;
        auto shifter = isl::MapToPriorData(
          isl::dim(tiling, isl_dim_in),
          reuse_level
        );
        auto buffered_data = shifter.apply_range(required_data);
        computed_data =
          computed_data.subtract(buffered_data).coalesce();
      }

      auto producer_write_dep =
        workload.WriteAccesses(prod_einsum, tensor);
      auto required_ops =
        computed_data.apply_range(producer_write_dep.reverse());
      required_ops = required_ops.intersect_range(
        workload.EinsumOspaceBound(prod_einsum)
      );

      // WARNING: mutation!
      auto& prod_tiling = tiling_info.at(prod_einsum);
      prod_tiling = prod_tiling.intersect(required_ops);

      queue.push_back(prod_einsum);
    }
  }
}

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
  EinsumsIterT einsums_begin, EinsumsIterT einsums_end
)

template<typename EinsumsIterT>
void ConsumerBasedTileShapeInference(
  std::map<problem::EinsumId, isl::map>& tiling_info,
  const std::map<problem::DataSpaceId, size_t>& dspace_to_reuse_level,
  EinsumsIterT einsums_begin, EinsumsIterT einsums_end,
  const problem::FusedWorkload& workload,
  problem::EinsumId tiled_einsum
)
{
  auto queue = std::deque<problem::EinsumId>({tiled_einsum});
  while (queue.size() > 0)
  {
    auto einsum = queue.front();
    queue.pop_front();

    const auto& tiling = tiling_info.at(einsum);
    for (auto tensor : workload.TensorsReadByEinsum(einsum))
    {
      auto producer_einsum_opt = workload.WriterEinsum(tensor);
      if (!producer_einsum_opt) // Not an intermediate tensor
      {
        continue;
      }

      auto prod_einsum = *producer_einsum_opt;
      if (std::find(einsums_begin, einsums_end, prod_einsum) == einsums_end)
      {
        // Not in this fusion set
        continue;
      }

      auto read_accesses = workload.ReadAccesses(einsum, tensor);
      auto required_data = tiling.apply_range(read_accesses);

      isl::map computed_data = required_data;
      auto it = dspace_to_reuse_level.find(tensor);
      if (it != dspace_to_reuse_level.end())
      {
        auto reuse_level = it->second;
        auto shifter = isl::MapToPriorData(
          isl::dim(tiling, isl_dim_in),
          reuse_level
        );
        auto buffered_data = shifter.apply_range(required_data);
        computed_data =
          computed_data.subtract(buffered_data).coalesce();
      }

      auto producer_write_dep =
        workload.WriteAccesses(prod_einsum, tensor);
      auto required_ops =
        computed_data.apply_range(producer_write_dep.reverse());
      required_ops = required_ops.intersect_range(
        workload.EinsumOspaceBound(prod_einsum)
      );

      // WARNING: mutation!
      auto& prod_tiling = tiling_info.at(prod_einsum);
      prod_tiling = prod_tiling.intersect(required_ops);

      queue.push_back(prod_einsum);
    }
  }
}
