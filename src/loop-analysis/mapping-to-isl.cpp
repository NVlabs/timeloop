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

#include "loop-analysis/isl-ir.hpp"
#include "isl-wrapper/ctx-manager.hpp"

namespace analysis
{

/******************************************************************************
 * Local declarations
 *****************************************************************************/

BranchTilings TilingFromMapping(const mapping::FusedMapping& mapping);
BranchTilings TilingFromMapping(const loop::Nest& nest);


std::vector<std::pair<LogicalBuffer, size_t>> 
BufferIterLevelsFromMapping(const loop::Nest& nest,
                            const problem::Workload& workload);
std::vector<std::pair<LogicalBuffer, size_t>>
BufferIterLevelsFromMapping(const mapping::FusedMapping& mapping);

/**
 * @brief Utility to help TilingFromMapping track coefficients.
 */
struct TilingCoefTracker
{
  TilingCoefTracker();

  TilingCoefTracker&
  NewIterDim(size_t op_dim, const std::optional<size_t>& iter_dim_end);

 private:
  friend IslMap TilingCoefTrackerToMap(TilingCoefTracker&& tracker);

  std::vector<std::vector<std::optional<size_t>>> coefs_;
  size_t n_iter_dims_;

};

IslMap TilingCoefTrackerToMap(const TilingCoefTracker& tracker);

LogicalBufTiling
LogicalBufTilingFromMapping(const mapping::FusedMapping& mapping);
LogicalBufTiling
LogicalBufTilingFromMapping(const loop::Nest& nest,
                            const problem::Workload& workload);

LogicalBufSkews
LogicalBufSkewsFromMapping(const mapping::FusedMapping& mapping);
LogicalBufSkews
LogicalBufSkewsFromMapping(const loop::Nest& mapping,
                           const problem::Workload& workload);

std::map<DataSpaceID, IslMap>
OpsToDSpaceFromEinsum(const problem::Workload& workload);

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

LogicalBufOccupancies
OccupanciesFromMapping(const mapping::FusedMapping& mapping,
                       const problem::Workload& workload)
{
  auto ops_to_dspace = OpsToDSpaceFromEinsum(workload);
  auto buf_tiling = LogicalBufTilingFromMapping(mapping);
  auto buf_skew = LogicalBufSkewsFromMapping(mapping);

  LogicalBufOccupancies result;
  for (auto& [buf, tiling] : buf_tiling)
  {
    result.emplace(std::make_pair(
      buf,
      ApplyRange(
        std::move(buf_skew.at(buf)),
        ApplyRange(std::move(tiling), IslMap(ops_to_dspace.at(buf.dspace_id)))
      )
    ));
  }

  return result;
}

LogicalBufOccupancies
OccupanciesFromMapping(const loop::Nest& mapping,
                       const problem::Workload& workload)
{
  auto ops_to_dspace = OpsToDSpaceFromEinsum(workload);
  auto buf_tiling = LogicalBufTilingFromMapping(mapping, workload);
  auto buf_skew = LogicalBufSkewsFromMapping(mapping, workload);

  LogicalBufOccupancies result;
  for (auto& [buf, tiling] : buf_tiling)
  {
    std::cout << "proj: " << ops_to_dspace.at(buf.dspace_id) << std::endl;
    std::cout << "tiling: " << tiling << std::endl;
    std::cout << "skew: " << buf_skew.at(buf) << std::endl;

    result.emplace(std::make_pair(
      buf,
      ApplyRange(
        std::move(buf_skew.at(buf)),
        ApplyRange(std::move(tiling), IslMap(ops_to_dspace.at(buf.dspace_id)))
      )
    ));
  }

  return result;
}

/******************************************************************************
 * Local function implementations
 *****************************************************************************/

BranchTilings TilingFromMapping(const mapping::FusedMapping& mapping)
{
  BranchTilings result;
  for (const auto& path : GetPaths(mapping))
  {
    TilingCoefTracker coef_tracker;
    std::optional<IslPwMultiAff> explicit_tiling_spec;
    mapping::NodeID leaf_id;
    for (const auto& node : path)
    {
      std::visit(
        [&coef_tracker, &explicit_tiling_spec, &leaf_id] (auto&& node) {
          using NodeT = std::decay_t<decltype(node)>;
          if constexpr (std::is_same_v<NodeT, mapping::For>
                        || std::is_same_v<NodeT, mapping::ParFor>)
          {
            coef_tracker.NewIterDim(node.op_dim, node.end);
          } else if constexpr (std::is_same_v<NodeT, mapping::Compute>)
          {
            explicit_tiling_spec = node.tiling_spec;
            leaf_id = node.id;
          }
        },
        node
      );
    }

    if (explicit_tiling_spec)
    {
      result.emplace(std::make_pair(
        leaf_id,
        IslMap(std::move(*explicit_tiling_spec))
      ));
    } else
    {
      result.emplace(std::make_pair(
        leaf_id,
        TilingCoefTrackerToMap(coef_tracker)
      ));
    }
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
        std::cout << arch_level << ", " << dspace_id << std::endl;
        result.emplace_back(std::make_pair(
          LogicalBuffer{.buffer_id = arch_level,
                        .dspace_id = dspace_id,
                        .branch_leaf_id = 0 },
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
BufferIterLevelsFromMapping(const mapping::FusedMapping& mapping)
{
  std::vector<std::pair<LogicalBuffer, size_t>> result;
  for (const auto& path : GetPaths(mapping))
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
            auto buffer = LogicalBuffer();
            buffer.buffer_id = node.buffer;
            buffer.dspace_id = node.dspace;
            buffer.branch_leaf_id = 0;

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

LogicalBufTiling
LogicalBufTilingFromMapping(const mapping::FusedMapping& mapping)
{
  auto branch_tiling = TilingFromMapping(mapping);
  auto buf_to_iter_level = BufferIterLevelsFromMapping(mapping);

  LogicalBufTiling result;
  for (auto& [buf, level] : buf_to_iter_level)
  {
    result.emplace(std::make_pair(
      buf,
      ProjectDimInAfter(IslMap(branch_tiling.at(buf.branch_leaf_id)),
                        level)
    ));
  }

  return result;
}

LogicalBufTiling
LogicalBufTilingFromMapping(const loop::Nest& nest,
                            const problem::Workload& workload)
{
  auto branch_tiling = TilingFromMapping(nest);
  auto buf_to_iter_level = BufferIterLevelsFromMapping(nest, workload);

  std::cout << branch_tiling.begin()->second << std::endl;

  LogicalBufTiling result;
  for (auto& [buf, level] : buf_to_iter_level)
  {
    result.emplace(std::make_pair(
      buf,
      ProjectDimInAfter(IslMap(branch_tiling.at(buf.branch_leaf_id)),
                        level)
    ));
  }

  return result;
}

LogicalBufSkews
LogicalBufSkewsFromMapping(const loop::Nest& nest,  
                           const problem::Workload& workload)
{
  auto buf_to_iter_level = BufferIterLevelsFromMapping(nest, workload);

  LogicalBufSkews result;
  for (auto& [buf, level] : buf_to_iter_level)
  {
    std::vector<spacetime::Dimension> tags;
    size_t loop_idx = 0;
    // Hardcoded for now since spacetime::Dimension has SpaceX and SpaceY.
    // Should be inferred from architecture array spec.
    for (const auto& loop : nest.loops)
    {
      tags.emplace_back(loop.spacetime_dimension);
      if (loop_idx == level)
      {
        break;
      }
    }
    result.emplace(std::make_pair(
      buf,
      TaggedMap<IslMap, spacetime::Dimension>(
        IslMap(
          IslMultiAff::Identity(IslSpace::Alloc(GetIslCtx(), 0, level, level))
        ),
        std::move(tags)
      )
    ));
  }

  return result;
}

std::map<DataSpaceID, IslMap>
OpsToDSpaceFromEinsum(const problem::Workload& workload)
{
  const auto& workload_shape = *workload.GetShape();

  std::map<DataSpaceID, IslMap> dspace_id_to_ospace_to_dspace;

  for (const auto& [name, dspace_id] : workload_shape.DataSpaceNameToID)
  {
    const auto dspace_order = workload_shape.DataSpaceOrder.at(dspace_id);
    const auto& projection = workload_shape.Projections.at(dspace_id);

    auto space = IslSpace::Alloc(GetIslCtx(),
                                 0,
                                 workload_shape.NumFactorizedDimensions,
                                 dspace_order);
    for (const auto& [ospace_dim_name, ospace_dim_id] :
         workload_shape.FactorizedDimensionNameToID)
    {
      space.SetDimName(isl_dim_in, ospace_dim_id, ospace_dim_name);
    }
    for (unsigned dspace_dim = 0; dspace_dim < dspace_order; ++dspace_dim)
    {
      const auto isl_dspace_dim_name = name + "_" + std::to_string(dspace_dim);
      space.SetDimName(isl_dim_out, dspace_dim, isl_dspace_dim_name);
    }

    auto multi_aff = IslMultiAff::Zero(IslSpace(space));
    for (unsigned dspace_dim = 0; dspace_dim < dspace_order; ++dspace_dim)
    {
      auto aff = IslAff::ZeroOnDomainSpace(IslSpaceDomain(IslSpace(space)));
      for (const auto& term : projection.at(dspace_dim))
      {
        const auto& coef_id = term.first;
        const auto& factorized_dim_id = term.second;
        if (coef_id != workload_shape.NumCoefficients)
        {
          aff.SetCoefficientSi(isl_dim_in,
                               factorized_dim_id,
                               workload.GetCoefficient(coef_id));
        }
        else // Last term is a constant
        {
          aff.SetCoefficientSi(isl_dim_in, factorized_dim_id, 1);
        }
        multi_aff.SetAff(dspace_dim, std::move(aff));
      }
    }
    dspace_id_to_ospace_to_dspace.emplace(std::make_pair(
      dspace_id,
      IslMap(std::move(multi_aff))
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

IslMap TilingCoefTrackerToMap(TilingCoefTracker&& tracker)
{
  auto eq_maff = IslMultiAff::Zero(
    IslSpace::Alloc(GetIslCtx(),
                    0,
                    tracker.n_iter_dims_,
                    tracker.coefs_.size())
  );
  auto iter_set = IslSet::Universe(eq_maff.GetDomainSpace());
  auto identity = IslMultiAff::IdentityOnDomainSpace(eq_maff.GetDomainSpace());

  for (size_t op_dim = 0; op_dim < tracker.coefs_.size(); ++op_dim)
  {
    auto& dim_coefs = tracker.coefs_.at(op_dim);
    int last_coef = 1;

    auto eq_aff = eq_maff.GetAff(op_dim);
    for (size_t iter_dim = 0; iter_dim < tracker.n_iter_dims_; ++iter_dim)
    {
      auto& coef_opt = dim_coefs.at(iter_dim);
      auto& coef = *coef_opt;
      if (coef != 0)
      {
        auto reversed_iter_dim = tracker.n_iter_dims_ - iter_dim - 1;
        eq_aff.SetCoefficientSi(isl_dim_in,
                                reversed_iter_dim,
                                last_coef);
        iter_set = Intersect(
          IslSet(iter_set),
          GeSet(identity.GetAff(reversed_iter_dim),
                IslAff::ValOnDomainSpace(identity.GetDomainSpace(), IslVal(0)))
        );
        iter_set = Intersect(
          IslSet(iter_set),
          LtSet(identity.GetAff(reversed_iter_dim),
                IslAff::ValOnDomainSpace(identity.GetDomainSpace(),
                                         IslVal(coef)))
        );
        last_coef *= coef;
      }
    }
    eq_maff.SetAff(op_dim, std::move(eq_aff));
  }

  std::cout << "eq maff: " << eq_maff << std::endl;
  std::cout << "iter set: " << iter_set << std::endl;

  auto map = IntersectDomain(IslMap(std::move(eq_maff)), std::move(iter_set));
  std::cout << "map: " << map << std::endl;

  return map;
}

};