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

namespace analysis
{

/******************************************************************************
 * Local function declarations
 *****************************************************************************/

std::map<DataSpaceID, IslMap>
OpsToDSpaceFromEinsum(const problem::Workload& workload);


/**
 * @brief Utility to help TilingFromMapping track coefficients.
 */
struct TilingCoefTracker
{
  TilingCoefTracker();

  TilingCoefTracker&
  NewIterDim(const problem::Shape::FlattenedDimensionID& op_dim,
             const std::optional<size_t>& coef);

 private:
  friend IslMap TilingCoefTrackerToMap(TilingCoefTracker&& tracker);

  std::vector<std::vector<std::optional<size_t>>> coefs_;
};

IslMap TilingCoefTrackerToMap(TilingCoefTracker&& tracker);


/******************************************************************************
 * Global function implementations
 *****************************************************************************/

LogicalBufOccupancies OccupanciesFromMapping(const Mapping& nest,
                                             const problem::Workload& workload)
{
  auto ops_to_dspace = OpsToDSpaceFromEinsum(workload);
  auto branch_tiling = TilingFromMapping(nest);
}


/******************************************************************************
 * Local function implementations
 *****************************************************************************/

BranchTilings
TilingFromMapping(const mapping::FusedMapping& mapping,
                  const problem::Workload& workload)
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
        IslMap::FromMultiAff(std::move(*explicit_tiling_spec))
      ));
    } else
    {
      result.emplace(std::make_pair(
        leaf_id,
        TilingCoefTrackerToMap(std::move(coef_tracker))
      ));
    }
  }

  return result;
}

};
