#include "loop-analysis/isl-to-legacy-adaptor.hpp"

#include <barvinok/isl.h>

#include "isl-wrapper/ctx-manager.hpp"
#include "isl-wrapper/isl-functions.hpp"

namespace analysis
{
/******************************************************************************
 * Local function declarations
 *****************************************************************************/

CompoundComputeNest GenerateCompoundComputeNest(
  const ReuseAnalysisOutput& isl_analysis_output,
  const std::vector<analysis::LoopState>& nest_state,
  const std::vector<uint64_t>& storage_tiling_boundaries
);

CompoundDataMovementNest GenerateCompoundDataMovementNest(
  ReuseAnalysisOutput isl_analysis_output,
  const std::vector<analysis::LoopState>& nest_state,
  const std::vector<uint64_t>& storage_tiling_boundaries,
  const std::vector<bool>& master_spatial_level,
  const std::vector<bool>& storage_boundary_level,
  const std::vector<uint64_t>& num_spatial_elems,
  const std::vector<uint64_t>& logical_fanouts,
  const problem::Workload& workload
);

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

std::pair<CompoundComputeNest, CompoundDataMovementNest>
GenerateLegacyNestAnalysisOutput(
  const ReuseAnalysisOutput& isl_analysis_output,
  const std::vector<analysis::LoopState>& nest_state,
  const std::vector<uint64_t>& storage_tiling_boundaries,
  const std::vector<bool>& master_spatial_level,
  const std::vector<bool>& storage_boundary_level,
  const std::vector<uint64_t>& num_spatial_elems,
  const std::vector<uint64_t>& logical_fanouts,
  const problem::Workload& workload
)
{
  return std::make_pair(
    GenerateCompoundComputeNest(isl_analysis_output,
                                nest_state,
                                storage_tiling_boundaries),
    GenerateCompoundDataMovementNest(isl_analysis_output,
                                     nest_state,
                                     storage_tiling_boundaries,
                                     master_spatial_level,
                                     storage_boundary_level,
                                     num_spatial_elems,
                                     logical_fanouts,
                                     workload)
  );
}


/******************************************************************************
 * Local function implementations
 *****************************************************************************/

CompoundComputeNest GenerateCompoundComputeNest(
  const ReuseAnalysisOutput& isl_analysis_output,
  const std::vector<analysis::LoopState>& nest_state,
  const std::vector<uint64_t>& storage_tiling_boundaries
)
{
  CompoundComputeNest compute_info_sets;

  size_t num_compute_units = 1;
  for (const auto& state : nest_state)
  {
    if (loop::IsSpatial(state.descriptor.spacetime_dimension))
    {
      num_compute_units *= state.descriptor.end;
    }
  }
  // Insert innermost level with number of iterations divided by spatial elements
  BufferID innermost_buf_id = storage_tiling_boundaries.size();

  uint64_t max_temporal_iterations = 1;
  for (auto& state : nest_state)
  {
    if (!loop::IsSpatial(state.descriptor.spacetime_dimension))
      max_temporal_iterations *= state.descriptor.end;
  }

  for (auto& [buf, stats] : isl_analysis_output.buf_to_stats)
  {
    const auto& occupancy = stats.occupancy;
    if (buf.buffer_id == innermost_buf_id)
    {
      auto compute_info = ComputeInfo();
      compute_info.replication_factor = num_compute_units;
      compute_info.accesses = isl::val_to_double(
        isl::get_val_from_singular_qpolynomial(
          isl::set_card(occupancy.map.domain())
        )
      ) / num_compute_units;
      compute_info.max_temporal_iterations = max_temporal_iterations;
      compute_info_sets.push_back(compute_info);
      break;
    }
  }
  for (long unsigned i = 0; i < nest_state.size() - 1; ++i)
  {
    auto compute_info = ComputeInfo();
    compute_info_sets.push_back(compute_info);
  }

  return compute_info_sets;
}

CompoundDataMovementNest GenerateCompoundDataMovementNest(
  ReuseAnalysisOutput isl_analysis_output,
  const std::vector<analysis::LoopState>& nest_state,
  const std::vector<uint64_t>& storage_tiling_boundaries,
  const std::vector<bool>& master_spatial_level,
  const std::vector<bool>& storage_boundary_level,
  const std::vector<uint64_t>& num_spatial_elems,
  const std::vector<uint64_t>& logical_fanouts,
  const problem::Workload& workload
)
{
  CompoundDataMovementNest working_sets;

  BufferID cur_buffer_id = storage_tiling_boundaries.size();
  bool first_loop = true;
  bool last_boundary_found = false;
  bool should_dump = false;
  for (const auto& cur : nest_state)
  {
    bool valid_level = !loop::IsSpatial(cur.descriptor.spacetime_dimension)
      || master_spatial_level[cur.level];
    if (!valid_level)
    {
      continue;
    }

    auto is_master_spatial = master_spatial_level[cur.level];
    auto is_boundary = storage_boundary_level[cur.level];

    if (first_loop)
    {
      last_boundary_found = false;
      should_dump = true;
    }
    else if (is_boundary && !last_boundary_found)
    {
      last_boundary_found = true;
      should_dump = false;
    }
    else if (is_boundary && last_boundary_found)
    {
      last_boundary_found = true;
      should_dump = true;
    }
    else if (is_master_spatial && last_boundary_found)
    {
      last_boundary_found = false;
      should_dump = true;
    }

    for (unsigned dspace_id = 0;
        dspace_id < workload.GetShape()->NumDataSpaces;
        ++dspace_id)
    {
      DataMovementInfo tile;
      tile.link_transfers = 0;
      tile.replication_factor = num_spatial_elems[cur.level];
      tile.fanout = logical_fanouts[cur.level];
      tile.is_on_storage_boundary = storage_boundary_level[cur.level];
      tile.is_master_spatial = master_spatial_level[cur.level];

      const auto& stats = isl_analysis_output.buf_to_stats.at(
        LogicalBuffer(cur_buffer_id, dspace_id, 0)
      );
      const auto& occ = stats.effective_occupancy;

      if (should_dump)
      {
        const auto& key_to_access_stats = stats.compat_access_stats;
        const auto& link_transfers = stats.link_transfer;

        for (const auto& [key, access_stats] : key_to_access_stats)
        {
          tile.access_stats.stats[key] = AccessStats{
            .accesses = access_stats.accesses / tile.replication_factor,
            .hops = access_stats.hops
          };
        }

        auto p_val = isl::get_val_from_singular_qpolynomial(
          isl::sum_map_range_card(link_transfers.map)
        );
        p_val = isl_val_div(
          p_val,
          isl_val_int_from_si(GetIslCtx().get(),
                              num_spatial_elems[cur.level])
        );
        tile.link_transfers = isl::val_to_double(p_val);
      }

      if (first_loop)
      {
        tile.size = 0;
      }
      else if (should_dump)
      {
        auto p_occ_map = occ.map.copy();
        auto p_occ_count = isl::get_val_from_singular_qpolynomial_fold(
          isl_pw_qpolynomial_bound(
            isl_map_card(
              isl_map_project_out(
                p_occ_map,
                isl_dim_in,
                isl_map_dim(p_occ_map, isl_dim_in)-2,
                2
              )
            ),
            isl_fold_max,
            nullptr
          )
        );
        tile.size = isl::val_to_double(p_occ_count);
      }
      else if (is_boundary)
      {
        auto p_occ_map = occ.map.copy();
        auto p_occ_count = isl::get_val_from_singular_qpolynomial_fold(
          isl_pw_qpolynomial_bound(isl_map_card(p_occ_map),
                                   isl_fold_max,
                                   nullptr)
        );
        tile.size = isl::val_to_double(p_occ_count);
      }
      else
      {
        tile.size = 0;
      }

      working_sets[dspace_id].push_back(tile);
    }
    
    if (should_dump)
    {
      should_dump = false;
      cur_buffer_id--;
    }

    first_loop = false;
  }

  return working_sets;
}

}; // namespace analysis