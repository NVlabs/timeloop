/**
 * @file mapping-to-isl.cpp
 * @author Michael Gilbert (gilbertm@mit.edu)
 * @brief Implements conversion between mapping and analysis IR
 * @version 0.1
 * @date 2023-02-07
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <stdexcept>
#include <isl/cpp.h>

#include "loop-analysis/isl-ir.hpp"
#include "isl-wrapper/ctx-manager.hpp"
#include "isl-wrapper/isl-functions.hpp"

namespace analysis
{

/******************************************************************************
 * Local declarations
 *****************************************************************************/

BranchTilings TilingFromMapping(const loop::Nest& nest);

std::vector<std::pair<LogicalBuffer, size_t>> 
BufferIterLevelsFromMapping(const loop::Nest& nest,
                            const problem::Workload& workload);

/**
 * @brief Utility to help TilingFromMapping track coefficients.
 */
struct TilingCoefTracker
{
  TilingCoefTracker();

  TilingCoefTracker&
  NewIterDim(size_t op_dim, const std::optional<size_t>& iter_dim_end);

 private:
  friend isl::map TilingCoefTrackerToMap(TilingCoefTracker&& tracker);

  std::vector<std::vector<std::optional<size_t>>> coefs_;
  size_t n_iter_dims_;

};

isl::map TilingCoefTrackerToMap(TilingCoefTracker&& tracker);

std::map<LogicalBuffer, Skew>
LogicalBufSkewsFromMapping(const loop::Nest& mapping,
                           const problem::Workload& workload);

std::map<DataSpaceID, isl::map>
OpsToDSpaceFromEinsum(const problem::Workload& workload);

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

std::map<LogicalBuffer, Occupancy>
OccupanciesFromMapping(const loop::Nest& mapping,
                       const problem::Workload& workload)
{
  auto ops_to_dspace = OpsToDSpaceFromEinsum(workload);
  auto branch_tiling = TilingFromMapping(mapping).at(0);
  auto buf_skew = LogicalBufSkewsFromMapping(mapping, workload);

  std::map<LogicalBuffer, Occupancy> result;
  for (auto& [buf, skew] : buf_skew)
  {
    auto occupancy = skew.map.apply_range(
      isl::project_dim_in_after(
        branch_tiling.apply_range(ops_to_dspace.at(buf.dspace_id)),
        isl::dim(skew.map, isl_dim_out)
      )
    );
    result.emplace(std::make_pair(
      buf,
      Occupancy(skew.dim_in_tags, std::move(occupancy)))
    );
  }

  return result;
}


/******************************************************************************
 * Local function implementations
 *****************************************************************************/

BranchTilings
TilingFromMapping(const loop::Nest& nest)
{
  const auto& loops = nest.loops;

  TilingCoefTracker coef_tracker;
  for (const auto& loop : loops)
  {
    const auto& ospace_dim = loop.dimension;
    coef_tracker.NewIterDim(ospace_dim, loop.end);
  }

  BranchTilings result;
  result.emplace(std::make_pair(
    0,
    TilingCoefTrackerToMap(std::move(coef_tracker))
  ));

  return result;
}

std::vector<std::pair<LogicalBuffer, size_t>> 
BufferIterLevelsFromMapping(const loop::Nest& nest,
                            const problem::Workload& workload)
{
  std::vector<std::pair<LogicalBuffer, size_t>> result;

  std::set<decltype(nest.storage_tiling_boundaries)::value_type>
    tiling_boundaries(nest.storage_tiling_boundaries.begin(),
                      nest.storage_tiling_boundaries.end());

  // TODO: for now, buffer id in loop nest is the arch level
  BufferID arch_level = 0;
  for (std::size_t loop_idx = 0; loop_idx < nest.loops.size(); ++loop_idx)
  {
    if (tiling_boundaries.find(loop_idx) != tiling_boundaries.end()
        || loop_idx == 0)
    {
      for (const auto& [dspace_id, _] : workload.GetShape()->DataSpaceIDToName)
      {
        result.emplace_back(std::make_pair(
          LogicalBuffer(arch_level, dspace_id, 0),
          loop_idx
        ));
      }
      ++arch_level;
    }
  }

  // Last one is compute and the level is defined differently
  // (under instead of above loop idx)
  for (auto& [buf, level] : result)
  {
    if (buf.buffer_id == arch_level - 1)
    {
      ++level;
    }
  }

  return result;
}

std::map<LogicalBuffer, Skew>
LogicalBufSkewsFromMapping(const loop::Nest& nest,  
                           const problem::Workload& workload)
{
  const auto& loops = nest.loops;
  const auto n_loops = loops.size();
  const std::set<decltype(nest.storage_tiling_boundaries)::value_type>
    tiling_boundaries(nest.storage_tiling_boundaries.begin(),
                      nest.storage_tiling_boundaries.end());

  std::map<LogicalBuffer, Skew> result;

  std::vector<SpaceTime> tags;
  auto p_aff_list = isl_aff_list_alloc(GetIslCtx().get(), n_loops);
  auto p_domain_space = isl_space_set_alloc(GetIslCtx().get(), 0, n_loops);
  int aff_list_size = 0;

  std::optional<int> last_x_idx_opt = std::nullopt;
  std::optional<int> last_y_idx_opt = std::nullopt;
  int arch_spatial_levels = 0;
  int arch_level = 0;

  int loop_idx = 0;
  int timeloop_loop_idx = n_loops - 1;
  isl_aff* p_aff = nullptr;
  for (auto loop_it = loops.rbegin(); loop_it != loops.rend(); ++loop_it)
  {
    auto spacetime_dim = loop_it->spacetime_dimension;

    auto is_boundary =
      tiling_boundaries.find(timeloop_loop_idx) != tiling_boundaries.end();
    if (is_boundary)
    {
      if (arch_spatial_levels == 0)  // i.e., no x loop between last and this
      {
        p_aff = isl_aff_zero_on_domain_space(isl_space_copy(p_domain_space));
        p_aff_list = isl_aff_list_add(p_aff_list, p_aff);
        aff_list_size++;
        tags.emplace_back(Spatial(0));
        arch_spatial_levels = 1;
      }
      if (arch_spatial_levels == 1)  // i.e., no x loop between last and this
      {
        p_aff = isl_aff_zero_on_domain_space(isl_space_copy(p_domain_space));
        p_aff_list = isl_aff_list_add(p_aff_list, p_aff);
        aff_list_size++;
        tags.emplace_back(Spatial(1));
        arch_spatial_levels = 0;
      }

      auto p_map = isl_map_from_basic_map(
        isl_basic_map_from_aff_list(isl_space_copy(p_domain_space),
                                    isl_aff_list_copy(p_aff_list))
      );
      auto map = isl::manage(p_map).reverse();

      for (const auto& [dspace_id, _] : workload.GetShape()->DataSpaceIDToName)
      {
        result.emplace(std::make_pair(LogicalBuffer(arch_level, dspace_id, 0),
                                      Skew(tags, map)));
      }

      ++arch_level;
    }

    if (spacetime_dim == spacetime::Dimension::Time)
    {
      p_aff = isl_aff_zero_on_domain_space(isl_space_copy(p_domain_space));
      p_aff = isl_aff_set_coefficient_si(p_aff, isl_dim_in, loop_idx, 1);
      p_aff_list = isl_aff_list_add(p_aff_list, p_aff);
      aff_list_size++;
      tags.emplace_back(Temporal());

      last_x_idx_opt = std::nullopt;
      last_y_idx_opt = std::nullopt;
      arch_spatial_levels = 0;
    }
    else if (spacetime_dim == spacetime::Dimension::SpaceX)
    {
      if (!last_x_idx_opt)
      {
        p_aff = isl_aff_zero_on_domain_space(isl_space_copy(p_domain_space));
        p_aff_list = isl_aff_list_add(p_aff_list, p_aff);
        aff_list_size++;
        tags.emplace_back(Spatial(0));
      }
      p_aff = isl_aff_list_get_aff(p_aff_list, aff_list_size-1);

      auto last_x_idx = last_x_idx_opt.value_or(loop_idx);
      unsigned long tile_size = loop_it->end - loop_it->start;
      for (auto i = last_x_idx; i < loop_idx; ++i)
      {
        p_aff = isl_aff_set_coefficient_val(
          p_aff,
          isl_dim_in,
          i,
          isl_val_mul_ui(isl_aff_get_coefficient_val(p_aff, isl_dim_in, i),
                         tile_size)
        );
      }
      p_aff = isl_aff_set_coefficient_si(p_aff, isl_dim_in, loop_idx, 1);
      p_aff_list = isl_aff_list_set_aff(p_aff_list, aff_list_size-1, p_aff);

      last_x_idx_opt = loop_idx;
      last_y_idx_opt = std::nullopt;
      arch_spatial_levels = 1;
    }
    else if (spacetime_dim == spacetime::Dimension::SpaceY)
    {
      if (arch_spatial_levels == 0)  // i.e., no x loop between last and this
      {
        p_aff = isl_aff_zero_on_domain_space(isl_space_copy(p_domain_space));
        p_aff_list = isl_aff_list_add(p_aff_list, p_aff);
        aff_list_size++;
        tags.emplace_back(Spatial(0));
      }

      if (!last_y_idx_opt)
      {
        p_aff = isl_aff_zero_on_domain_space(isl_space_copy(p_domain_space));
        p_aff_list = isl_aff_list_add(p_aff_list, p_aff);
        aff_list_size++;
        tags.emplace_back(Spatial(1));
      }
      p_aff = isl_aff_list_get_aff(p_aff_list, aff_list_size-1);

      auto last_y_idx = last_y_idx_opt.value_or(loop_idx);
      unsigned long tile_size = loop_it->end - loop_it->start;
      for (int i = last_y_idx; i < loop_idx; ++i)
      {
        p_aff = isl_aff_set_coefficient_val(
          p_aff,
          isl_dim_in,
          i,
          isl_val_mul_ui(isl_aff_get_coefficient_val(p_aff, isl_dim_in, i),
                         tile_size)
        );
      }
      p_aff = isl_aff_set_coefficient_si(p_aff, isl_dim_in, loop_idx, 1);
      p_aff_list = isl_aff_list_set_aff(p_aff_list, aff_list_size-1, p_aff);

      last_x_idx_opt = std::nullopt;
      last_y_idx_opt = loop_idx;
      arch_spatial_levels = 2;
    }

    loop_idx++;
    timeloop_loop_idx--;
  }

  // The compute level happens after the last loop.
  if (timeloop_loop_idx != -1)
  {
    throw std::logic_error(
      "compute should be at the innermost level, but "
      + std::to_string(timeloop_loop_idx)
    );
  }
  if (arch_spatial_levels == 0)  // i.e., no x loop between last and this
  {
    p_aff = isl_aff_zero_on_domain_space(isl_space_copy(p_domain_space));
    p_aff_list = isl_aff_list_add(p_aff_list, p_aff);
    aff_list_size++;
    tags.emplace_back(Spatial(0));
    arch_spatial_levels = 1;
  }
  if (arch_spatial_levels == 1)  // i.e., no y loop between last and this
  {
    p_aff = isl_aff_zero_on_domain_space(isl_space_copy(p_domain_space));
    p_aff_list = isl_aff_list_add(p_aff_list, p_aff);
    aff_list_size++;
    tags.emplace_back(Spatial(1));
    arch_spatial_levels = 0;
  }

  auto p_map = isl_map_from_basic_map(
    isl_basic_map_from_aff_list(isl_space_copy(p_domain_space),
                                isl_aff_list_copy(p_aff_list))
  );
  auto map = isl::manage(p_map).reverse();

  for (const auto& [dspace_id, _] : workload.GetShape()->DataSpaceIDToName)
  {
    result.emplace(std::make_pair(
      LogicalBuffer(arch_level, dspace_id, 0),
      Skew(tags, map)
    ));
  }

  isl_space_free(p_domain_space);
  isl_aff_list_free(p_aff_list);

  return result;
}

std::map<DataSpaceID, isl::map>
OpsToDSpaceFromEinsum(const problem::Workload& workload)
{
  const auto& workload_shape = *workload.GetShape();

  std::map<DataSpaceID, isl::map> dspace_id_to_ospace_to_dspace;

  for (const auto& [name, dspace_id] : workload_shape.DataSpaceNameToID)
  {
    const auto dspace_order = workload_shape.DataSpaceOrder.at(dspace_id);
    const auto& projection = workload_shape.Projections.at(dspace_id);

    auto space = isl::space_alloc(GetIslCtx(),
                                  0,
                                  workload_shape.NumFactorizedDimensions,
                                  dspace_order);

    auto multi_aff = space.zero_multi_aff();
    for (unsigned dspace_dim = 0; dspace_dim < dspace_order; ++dspace_dim)
    {
      auto aff = space.domain().zero_aff_on_domain();
      for (const auto& term : projection.at(dspace_dim))
      {
        const auto& coef_id = term.first;
        const auto& factorized_dim_id = term.second;
        if (coef_id != workload_shape.NumCoefficients)
        {
          aff = set_coefficient_si(aff,
                                   isl_dim_in,
                                   factorized_dim_id,
                                   workload.GetCoefficient(coef_id));
        }
        else // Last term is a constant
        {
          aff = set_coefficient_si(aff,
                                   isl_dim_in,
                                   factorized_dim_id,
                                   1);
        }
      }
      multi_aff = multi_aff.set_at(dspace_dim, aff);
    }
    dspace_id_to_ospace_to_dspace.emplace(std::make_pair(
      dspace_id,
      isl::map_from_multi_aff(multi_aff)
    ));
  }

  return dspace_id_to_ospace_to_dspace;
}

TilingCoefTracker::TilingCoefTracker() : coefs_(), n_iter_dims_(0) {}

TilingCoefTracker&
TilingCoefTracker::NewIterDim(size_t op_dim,
                              const std::optional<size_t>& iter_dim_end)
{
  ++n_iter_dims_;

  while (coefs_.size() < op_dim + 1)
  {
    coefs_.emplace_back(n_iter_dims_, 0);
  }

  for (auto& dim_coefs : coefs_)
  {
    for (size_t i = dim_coefs.size(); i < n_iter_dims_; ++i)
    {
      dim_coefs.emplace_back(0);
    }
  }

  coefs_.at(op_dim).back() = iter_dim_end;

  return *this;
}

// template<typename IterT>
// isl::map FromVectorInMap(IterT begin, IterT end,
//                          size_t in_dims, size_t out_dims)
// {
//   auto eq_maff = isl::multi_aff::zero(
//     isl::space_alloc(GetIslCtx(), 0, in_dims, out_dims)
//   );

//   for (auto it = begin; it != end; ++it)
//   {
//     const auto& out_dim = it->first;
//     const auto& expr = it->second;

//     auto eq_aff = eq_maff.get_at(out_dim);
//     for (auto term_it = expr.begin(); term_it != expr.end(); ++term_it)
//     {
//       const auto& in_dim = term_it->first;
//       const auto& coef = term_it->second;
//       eq_aff = isl::set_coefficient_si(eq_aff, isl_dim_in, in_dim, coef);
//     }
//   }

//   return isl::map_from_multi_aff(eq_maff);
// }

isl::map TilingCoefTrackerToMap(TilingCoefTracker&& tracker)
{
  auto eq_maff = isl::multi_aff::zero(
    isl::space_alloc(GetIslCtx(),
                     0,
                     tracker.n_iter_dims_,
                     tracker.coefs_.size())
  );

  auto iter_set = isl::set::universe(eq_maff.space().domain());
  auto identity = isl::multi_aff::identity_on_domain(eq_maff.space().domain()); 

  for (size_t op_dim = 0; op_dim < tracker.coefs_.size(); ++op_dim)
  {
    auto& dim_coefs = tracker.coefs_.at(op_dim);
    int last_coef = 1;

    auto eq_aff = eq_maff.get_at(op_dim);
    for (size_t iter_dim = 0; iter_dim < tracker.n_iter_dims_; ++iter_dim)
    {
      auto& coef_opt = dim_coefs.at(iter_dim);
      auto& coef = *coef_opt;
      if (coef != 0)
      {
        auto reversed_iter_dim = tracker.n_iter_dims_ - iter_dim - 1;
        eq_aff = isl::set_coefficient_si(eq_aff,
                                         isl_dim_in,
                                         reversed_iter_dim,
                                         last_coef);

        iter_set = iter_set.intersect(
          identity.get_at(reversed_iter_dim).ge_set(
            isl::si_on_domain(identity.space().domain(), 0))
        );
        iter_set = iter_set.intersect(
          identity.get_at(reversed_iter_dim).lt_set(
            isl::si_on_domain(identity.space().domain(), coef))
        );
        last_coef *= coef;
      }
    }
    eq_maff = eq_maff.set_at(op_dim, eq_aff);
  }

  auto map = isl::map_from_multi_aff(eq_maff).intersect_domain(iter_set);

  return map;
}

};
