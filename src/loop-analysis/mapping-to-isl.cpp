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
BufferIterLevelsFromMapping(mapping::FusedMapping& mapping);
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

isl::map TilingCoefTrackerToMap(const TilingCoefTracker& tracker);

LogicalBufTiling
LogicalBufTilingFromMapping(mapping::FusedMapping& mapping);
LogicalBufTiling
LogicalBufTilingFromMapping(const loop::Nest& nest,
                            const problem::Workload& workload);

LogicalBufSkews
LogicalBufSkewsFromMapping(mapping::FusedMapping& mapping);
LogicalBufSkews
LogicalBufSkewsFromMapping(const loop::Nest& mapping,
                           const problem::Workload& workload);

std::map<DataSpaceID, isl::map>
OpsToDSpaceFromEinsum(const problem::Workload& workload);

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

// LogicalBufOccupancies
// OccupanciesFromMapping(mapping::FusedMapping& mapping,
//                        const problem::FusedWorkload& workload_ir)
// {
//   (void) workload_ir;
//   // auto tiling = TilingFromMapping(mapping);

//   return LogicalBufOccupancies();
// }

LogicalBufOccupancies
OccupanciesFromMapping(const loop::Nest& mapping,
                       const problem::Workload& workload)
{
  auto ops_to_dspace = OpsToDSpaceFromEinsum(workload);
  auto branch_tiling = TilingFromMapping(mapping).at(0);
  auto buf_skew = LogicalBufSkewsFromMapping(mapping, workload);

  LogicalBufOccupancies result;
  for (auto& [buf, skew] : buf_skew)
  {
    result.emplace(std::make_pair(
      buf,
      skew.apply_range(
        isl::project_dim_in_after(
          branch_tiling.apply_range(ops_to_dspace.at(buf.dspace_id)),
          isl::dim(skew.map, isl_dim_out)
        )
      )
    ));
    std::cout << result.at(buf) << std::endl;
  }

  return result;
}

/******************************************************************************
 * Local function implementations
 *****************************************************************************/

BranchTilings TilingFromMapping(mapping::FusedMapping& mapping,
                                problem::FusedWorkload& workload)
{
  BranchTilings result;
  for (auto path : GetPaths(mapping))
  {
    std::map<problem::DimensionId, std::vector<std::pair<size_t, int>>>
    prob_id_to_expr;

    size_t cur_dim_idx = 0;
    problem::EinsumId einsum_id;
    mapping::NodeID leaf_id;
    for (const auto& node : path)
    {
      std::visit(
        [&prob_id_to_expr, &cur_dim_idx, &einsum_id, &leaf_id] (auto&& node) {
          using NodeT = std::decay_t<decltype(node)>;
          if constexpr (std::is_same_v<NodeT, mapping::For>
                        || std::is_same_v<NodeT, mapping::ParFor>)
          {
            if (node.tile_size)
            {
              prob_id_to_expr[node.op_dim].emplace_back(std::make_pair(
                cur_dim_idx,
                *node.tile_size
              ));
            }
            ++cur_dim_idx;
          } else if constexpr (std::is_same_v<NodeT, mapping::Compute>)
          {
            leaf_id = node.id;
            einsum_id = node.kernel;
          }
        },
        node
      );
    }

    auto eq_maff = isl::multi_aff::zero(
      isl::space_alloc(GetIslCtx(), 0, cur_dim_idx, prob_id_to_expr.size())
    );
    for (const auto& [prob_idx, expr] : prob_id_to_expr)
    {
      auto einsum_dim_idx = workload.EinsumDimToIdx(einsum_id, prob_idx);
      auto eq_aff = eq_maff.get_at(einsum_dim_idx);

      for (const auto& [iter_id, coef] : expr)
      {
        eq_aff = isl::set_coefficient_si(eq_aff, isl_dim_in, iter_id, coef);
      }

      eq_maff = eq_maff.set_at(einsum_dim_idx, eq_aff);
    }

    result.emplace(std::make_pair(leaf_id, isl::map_from_multi_aff(eq_maff)));
  }

  return result;
}

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

std::vector<std::pair<LogicalBuffer, size_t>>
BufferIterLevelsFromMapping(mapping::FusedMapping& mapping)
{
  std::vector<std::pair<LogicalBuffer, size_t>> result;
  for (auto path : GetPaths(mapping))
  {
    size_t iter_idx = 0;
    std::vector<std::pair<LogicalBuffer, size_t>> new_results;
    for (const auto& node : path)
    {
      std::visit(
        [&new_results, &iter_idx] (auto&& node) {
          using NodeT = std::decay_t<decltype(node)>;

          if constexpr (std::is_same_v<NodeT, mapping::Storage>)
          {
            auto buffer = LogicalBuffer(node.buffer, node.dspace, 0);
            new_results.emplace_back(
              std::make_pair(std::move(buffer), iter_idx)
            );
          } else if constexpr (std::is_same_v<NodeT, mapping::For>
                               || std::is_same_v<NodeT, mapping::ParFor>)
          {
            ++iter_idx;
          } else if constexpr (std::is_same_v<NodeT, mapping::Compute>)
          {
            for (auto& [buf, _] : new_results)
            {
              buf.branch_leaf_id = node.id;
            }
          }
        },
        node
      );
    }
    result.insert(result.end(), new_results.begin(), new_results.end());
  }

  return result;
}

// LogicalBufTiling
// LogicalBufTilingFromMapping(mapping::FusedMapping& mapping)
// {
//   // auto branch_tiling = TilingFromMapping(mapping);
//   auto buf_to_iter_level = BufferIterLevelsFromMapping(mapping);

//   LogicalBufTiling result;
//   for (auto& [buf, level] : buf_to_iter_level)
//   {
//     result.emplace(std::make_pair(
//       buf,
//       project_dim_in_after(isl::map(branch_tiling.at(buf.branch_leaf_id)),
//                            level)
//     ));
//   }

//   return result;
// }

LogicalBufSkews
LogicalBufSkewsFromMapping(const loop::Nest& nest,  
                           const problem::Workload& workload)
{
  const auto n_loops = nest.loops.size();

  std::set<decltype(nest.storage_tiling_boundaries)::value_type>
    tiling_boundaries(nest.storage_tiling_boundaries.begin(),
                      nest.storage_tiling_boundaries.end());

  LogicalBufSkews result;

  std::vector<spacetime::Dimension> tags;
  auto map = isl::map_from_multi_aff(isl::multi_aff::identity_on_domain(
    isl::space_alloc(GetIslCtx(), 0, 0, 0).domain()
  ));
  bool arch_has_spatial = false;
  // TODO: for now, buffer id in loop nest is the arch level.
  // Arch spec should define an ID
  BufferID arch_level = 0;
  auto loop_it = nest.loops.rbegin();
  for (std::size_t loop_idx = 0; loop_idx < n_loops; ++loop_idx)
  {
    auto spacetime_dim = loop_it->spacetime_dimension;

    auto tiling_boundary_it = tiling_boundaries.find(n_loops - loop_idx - 1);
    if (tiling_boundary_it != tiling_boundaries.end())
    {
      map = isl::insert_equal_dims(std::move(map),
                                   isl::dim(map, isl_dim_in),
                                   isl::dim(map, isl_dim_out),
                                   loop_idx - isl::dim(map, isl_dim_out));
      if (!arch_has_spatial)
      {
        // TODO: assumes 1D spatial array. Implement to infer from arch spec
        const size_t n_spatial_dims = 1;
        tags.emplace_back(spacetime::Dimension::SpaceX);
        map = isl::insert_dummy_dim_ins(std::move(map),
                                        isl::dim(map, isl_dim_in),
                                        n_spatial_dims);
      }

      for (const auto& [dspace_id, _] : workload.GetShape()->DataSpaceIDToName)
      {
        result.emplace(std::make_pair(
          LogicalBuffer(arch_level, dspace_id, 0),
          TaggedMap<isl::map, spacetime::Dimension>(map, tags)
        ));
      }
      ++arch_level;
      arch_has_spatial = false;
    }

    tags.emplace_back(spacetime_dim);
    if (spacetime_dim != spacetime::Dimension::Time)
    {
      arch_has_spatial = true;
    }

    ++loop_it;
  }

  // Last loop index is compute level
  map = isl::insert_equal_dims(std::move(map),
                                isl::dim(map, isl_dim_in),
                                isl::dim(map, isl_dim_out),
                                n_loops - isl::dim(map, isl_dim_out));
  if (!arch_has_spatial)
  {
    // TODO: assumes 1D spatial array. Implement to infer from arch spec
    const size_t n_spatial_dims = 1;
    tags.emplace_back(spacetime::Dimension::SpaceX);
    map = isl::insert_dummy_dim_ins(std::move(map),
                                    isl::dim(map, isl_dim_in),
                                    n_spatial_dims);
  }

  for (const auto& [dspace_id, _] : workload.GetShape()->DataSpaceIDToName)
  {
    result.emplace(std::make_pair(
      LogicalBuffer(arch_level, dspace_id, 0),
      TaggedMap<isl::map, spacetime::Dimension>(map, tags)
    ));
  }

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
