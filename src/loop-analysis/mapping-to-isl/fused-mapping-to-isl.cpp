#include "loop-analysis/mapping-to-isl/fused-mapping-to-isl.hpp"
#include "loop-analysis/mapping-to-isl/tile-shape-inference.hpp"
#include "loop-analysis/mapping-to-isl/tiling.hpp"
#include "isl-wrapper/ctx-manager.hpp"
#include "isl-wrapper/isl-functions.hpp"

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <barvinok/isl.h>

bool gDumpIslIr =
  (getenv("TIMELOOP_DUMP_ISL_IR") != NULL) &&
  (strcmp(getenv("TIMELOOP_DUMP_ISL_IR"), "0") != 0);
bool gLogIslIr =
  (getenv("TIMELOOP_LOG_ISL_IR") != NULL) &&
  (strcmp(getenv("TIMELOOP_LOG_ISL_IR"), "0") != 0);

namespace analysis
{

/******************************************************************************
 * Local declarations
 *****************************************************************************/
std::map<mapping::NodeID, double>
GetParallelism(mapping::FusedMapping& mapping);

BranchTilings TilingFromMapping(mapping::FusedMapping& mapping,
                                const problem::FusedWorkload& workload);

std::map<LogicalBuffer, bool>
BufRightAboveSequential(mapping::FusedMapping& mapping);

struct SkewsInfo
{
  std::map<LogicalBuffer, Skew> lbuf_to_skew;
  std::map<LogicalComputeUnit, Skew> lcomp_to_skew;
};

SkewsInfo SkewsFromMapping(mapping::FusedMapping& mapping,
                           const problem::FusedWorkload& workload);

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

MappingAnalysisResult
OccupanciesFromMapping(mapping::FusedMapping& mapping,
                       const problem::FusedWorkload& workload)
{
  MappingAnalysisResult result;

  result.compute_latency_aggregator = CreateLatencyAggregatorFromMapping(
    mapping
  );
  result.compute_to_assumed_parallelism = GetParallelism(mapping);

  auto branch_tiling = analysis::TilingFromMapping(mapping, workload);
  if (gDumpIslIr)
  {
    for (const auto& [node_id, tiling] : branch_tiling)
    {
      std::cout << "[Tiling]Node(" << node_id << "): " << tiling << std::endl;
      std::cout << "[Ops]Node(" << node_id << "): "
        << isl_pw_qpolynomial_to_str(isl_pw_qpolynomial_sum(isl_map_card(tiling.copy())))
        << std::endl;
    }
  }

  result.buf_right_above_sequential = BufRightAboveSequential(mapping);

  std::map<LogicalBuffer, Occupancy> occupancies;
  auto skews = SkewsFromMapping(mapping, workload);
  for (auto& [buf, skew] : skews.lbuf_to_skew)
  {
    if (gDumpIslIr)
    {
      std::cout << buf << " has skew: " << skew << std::endl;
    }

    auto einsum =
      std::get<mapping::Compute>(mapping.NodeAt(buf.branch_leaf_id)).kernel;
    auto dspace = buf.dspace_id;

    const auto& tiling = branch_tiling.at(buf.branch_leaf_id);

    auto accesses_opt = std::optional<isl::map>();
    const auto& read_tensors = workload.TensorsReadByEinsum(einsum);
    const auto& write_tensors = workload.TensorsWrittenByEinsum(einsum);
    if (read_tensors.find(dspace) != read_tensors.end())
    {
      accesses_opt = workload.ReadAccesses(einsum, dspace);
    }
    else if (write_tensors.find(dspace) != write_tensors.end())
    {
      accesses_opt = workload.WriteAccesses(einsum, dspace);
    }
    else
    {
      continue;
    }

    auto accesses = *accesses_opt;
    auto occupancy = skew.map.apply_range(
      isl::project_dim_in_after(tiling.apply_range(accesses),
                                isl::dim(skew.map, isl_dim_out))
    );

    if (gDumpIslIr)
    {
      std::cout << buf << " has occupancy: " << occupancy << std::endl;
    }

    occupancies.emplace(std::make_pair(buf,
                                       Occupancy(skew.dim_in_tags,
                                                 std::move(occupancy))));
  }

  std::map<LogicalComputeUnit, OpOccupancy> op_occupancies;
  for (auto& [lcomp, skew] : skews.lcomp_to_skew)
  {
    const auto& tiling = branch_tiling.at(lcomp.branch_leaf_id);
    auto op_occupancy = skew.map.apply_range(
      isl::project_dim_in_after(tiling, isl::dim(skew.map, isl_dim_out))
    );
    op_occupancies.emplace(
      lcomp,
      OpOccupancy(skew.dim_in_tags, std::move(op_occupancy))
    );
  }

  result.lbuf_to_occupancy = std::move(occupancies);
  result.lcomp_to_occupancy = std::move(op_occupancies);
  result.branch_tiling = std::move(branch_tiling);

  return result;
}

/******************************************************************************
 * Local function implementations
 *****************************************************************************/

std::map<mapping::NodeID, double>
GetParallelism(mapping::FusedMapping& mapping)
{
  using namespace problem;

  auto result = std::map<mapping::NodeID, double>();

  auto root = mapping.GetRoot().id;
  auto dfs_stack = std::vector<mapping::NodeID>();
  dfs_stack.push_back(root);

  while (dfs_stack.size() > 0)
  {
    auto node_id = dfs_stack.back();
    dfs_stack.pop_back();

    const auto& node = mapping.NodeAt(node_id);
    std::visit(
      [&dfs_stack, &result, &node_id](auto&& node)
      {
        using T = std::decay_t<decltype(node)>;
        if constexpr (mapping::HasOneChildV<T>)
        {
          dfs_stack.push_back(*node.child);
        }
        else if constexpr (mapping::HasManyChildrenV<T>)
        {
          for (const auto& child : node.children)
          {
            dfs_stack.push_back(child);
          }
        }
        else if constexpr (std::is_same_v<T, mapping::Compute>)
        {
          if (node.parallelism)
          {
            result.emplace(std::make_pair(
              node_id,
              *node.parallelism
            ));
          }
        }
        else
        {
          throw std::logic_error("unknown mapping node type");
        }
      },
      node
    );
  }

  return result;
}
std::map<mapping::NodeID, std::set<problem::EinsumId>>
GetMappingGroupEinsums(mapping::FusedMapping& mapping)
{
  using namespace boost::adaptors;
  using namespace mapping;
  using namespace problem;

  // Each pair is a (current_node_id, last_non_branch_node_id)
  auto dfs_stack = std::vector<std::pair<NodeID, NodeID>>();
  // Each pair is a (last_non_branch_node_id, set_of_children_ids)
  auto child_stack = std::vector<std::pair<NodeID, std::set<NodeID>>>();
  auto result = std::map<NodeID, std::set<EinsumId>>();

  auto root = mapping.GetRoot().id;
  dfs_stack.push_back(std::make_pair(root, root));

  while (dfs_stack.size() > 0)
  {
    auto node_id = dfs_stack.back().first;
    auto last_non_branch = dfs_stack.back().second;
    dfs_stack.pop_back();

    const auto& node = mapping.NodeAt(node_id);
    std::visit(
      [&dfs_stack, &child_stack, &result, &last_non_branch](auto&& node)
      {
        using T = std::decay_t<decltype(node)>;
        if constexpr (mapping::HasOneChildV<T>)
        {
          dfs_stack.emplace_back(std::make_pair(*node.child, last_non_branch));
        }
        else if constexpr (mapping::HasManyChildrenV<T>)
        {
          child_stack.emplace_back(
            std::make_pair(last_non_branch,
                           std::set<NodeID>(node.children.begin(),
                                            node.children.end()))
          ); 

          for (auto child : node.children)
          {
            dfs_stack.emplace_back(std::make_pair(child, child));
          }
        }
        else if constexpr (mapping::HasNoChildV<T> &&
                           std::is_same_v<T, Compute>)
        {
          result.emplace(std::make_pair(last_non_branch,
                                        std::set<EinsumId>({node.kernel})));
        }
        else
        {
          throw std::logic_error("node with unknown number of children");
        }
      },
      node
    );
  }

  for (const auto& [node_id, children] : child_stack | reversed)
  {
    auto& einsum_set = result[node_id];
    for (const auto& child : children)
    {
      einsum_set.insert(result.at(child).begin(), result.at(child).end());
    }
  }

  return result;
}

std::set<problem::EinsumId>
GetHeadAmongEinsums(const std::set<problem::EinsumId>& einsum_set,
                    const problem::FusedWorkload& workload)
{
  using namespace problem;

  auto heads = std::set<EinsumId>();

  for (auto einsum_id : einsum_set)
  {
    auto is_head = true;

    for (auto out_dspace : workload.TensorsWrittenByEinsum(einsum_id))
    {
      for (auto consumer_einsum : workload.ReaderEinsums(out_dspace))
      {
        auto it = einsum_set.find(consumer_einsum);
        if (it != einsum_set.end())
        {
          is_head = false;
          break;
        }
      }
      if (!is_head)
      {
        break;
      }
    }

    if (is_head)
    {
      heads.insert(einsum_id);
    }
  }

  return heads;
}

BranchTilings TilingFromMapping(mapping::FusedMapping& mapping,
                                const problem::FusedWorkload& workload)
{
  using namespace mapping;
  using namespace problem;

  BranchTilings result;

  auto mapping_groups = GetMappingGroupEinsums(mapping);
  auto mapping_group_heads = std::map<NodeID, std::set<EinsumId>>();
  for (const auto& [node_id, group] : mapping_groups)
  {
    mapping_group_heads.emplace(std::make_pair(
      node_id,
      GetHeadAmongEinsums(group, workload)
    ));
  }
  auto dspace_to_reuse_level = std::map<DataSpaceId, size_t>();

  auto dfs_stack = std::vector<NodeID>();
  // maps last non branch to tiling of for each einsum in the group
  auto tiling_info = std::map<NodeID, std::map<EinsumId, isl::map>>();

  auto root = mapping.GetRoot().id;
  dfs_stack.emplace_back(root);
  for (const auto& [einsum_name, einsum_id] : workload.EinsumNameToId())
  {
    auto p_tiling = isl_map_from_range(
      workload.EinsumOspaceBound(einsum_id).copy()
    );
    if (gDumpIslIr)
    {
      std::cout << "Tiling: " << isl_map_to_str(p_tiling) << std::endl;
    }
    tiling_info[root][einsum_id] = isl::manage(p_tiling);
  }

  while (dfs_stack.size() > 0)
  {
    auto node_id = dfs_stack.back();
    dfs_stack.pop_back();

    auto heads = mapping_group_heads.at(node_id);

    auto cur_node_id = node_id;
    bool is_tiling = true;
    while (is_tiling)
    {
      const auto& node = mapping.NodeAt(cur_node_id);
      std::visit(
        [&is_tiling, &cur_node_id, &dfs_stack, &tiling_info, &result,
         &dspace_to_reuse_level, &heads, &mapping_groups, &workload, &node_id]
        (auto&& node)
        {
          using T = std::decay_t<decltype(node)>;
          if constexpr (IsLoopV<T>)
          {
            if (heads.size() != 1)
            {
              throw std::logic_error("cannot tile fused set with " +
                                     std::to_string(heads.size()) + " heads");
            }

            auto dim = node.op_dim;
            auto head = *heads.begin();

            auto p_old_tiling = tiling_info.at(node_id).at(head).release();
            auto dim_idx = workload.EinsumDimToIdx(head).at(dim);

            decltype(p_old_tiling) p_new_tiling = nullptr;
            if (node.tile_size)
            {
              p_new_tiling = AddNewTileDim(p_old_tiling,
                                           dim_idx,
                                           *node.tile_size);
            }
            else
            {
              throw std::logic_error("tile size analysis not implemented");
            }

            auto iter_set = isl::manage(
              isl_map_domain(isl_map_copy(p_new_tiling))
            );

            tiling_info.at(node_id).at(head) = isl::manage(p_new_tiling);

            for (auto einsum : mapping_groups.at(node_id))
            {
              if (einsum == head)
              {
                continue;
              }

              auto& tiling = tiling_info.at(node_id).at(einsum);
              auto p_tiling = tiling.release();
              tiling = isl::manage(isl_map_insert_dims(
                p_tiling,
                isl_dim_in,
                isl_map_dim(p_tiling, isl_dim_in),
                1
              ));
              tiling = tiling.intersect_domain(iter_set);
            }

            cur_node_id = *node.child;
          }
          else if constexpr (std::is_same_v<T, Storage>)
          {
            // Check if highest level storage (determine reuse level)
            auto it = dspace_to_reuse_level.find(node.dspace);
            if (it == dspace_to_reuse_level.end() && node.exploits_reuse)
            {
              const auto& random_einsum = *mapping_groups.at(node_id).begin();
              const auto& tiling = tiling_info.at(node_id).at(random_einsum);
              dspace_to_reuse_level[node.dspace] =
                isl::dim(tiling, isl_dim_in);
            }
            cur_node_id = *node.child;
          }
          else if constexpr (std::is_same_v<T, Root>)
          {
            cur_node_id = *node.child;
          }
          else if constexpr (std::is_same_v<T, Compute>)
          {
            result.emplace(std::make_pair(
              cur_node_id,
              tiling_info.at(node_id).at(node.kernel)
            ));
            is_tiling = false;
          }
          else if constexpr (IsBranchV<T>)
          {
            const auto& fused_set = mapping_groups.at(node_id);
            if (heads.size() != 1)
            {
              // There cannot be tiling, so no inference to be done
              return;
            }

            auto shared_input_tensor = DetectSharedInputTensor(
              fused_set.begin(), fused_set.end(),
              workload
            );

            if (std::holds_alternative<SharedInputTensor>(shared_input_tensor))
            {
              SharedInputBasedTileShapeInference(
                tiling_info.at(node_id),
                fused_set.begin(), fused_set.end(),
                workload,
                *heads.begin()
              );
            }
            else
            {
              ConsumerBasedTileShapeInference(tiling_info.at(node_id),
                                              dspace_to_reuse_level,
                                              fused_set.begin(), fused_set.end(),
                                              workload,
                                              *heads.begin());
            }

            for (auto it : node.children | boost::adaptors::indexed(0))
            {
              auto child = it.value();
              auto idx = it.index();

              // Each child needs tilings for all Einsums in its group
              const auto& group = mapping_groups.at(child);
              auto tilings = std::map<EinsumId, isl::map>();
              for (auto einsum_id : group)
              {
                auto& tiling = tiling_info.at(node_id).at(einsum_id);
                auto new_tiling = isl::add_dims(tiling, isl_dim_in, 1);
                tilings[einsum_id] = isl::fix_si(
                  std::move(new_tiling),
                  isl_dim_in,
                  isl::dim(new_tiling, isl_dim_in)-1,
                  idx
                );
              }
              tiling_info.emplace(std::make_pair(
                child,
                std::move(tilings)
              ));

              dfs_stack.emplace_back(child);
            }
            is_tiling = false;
          }
        },
        node
      );
    }
  }

  return result;
}

std::map<LogicalBuffer, bool>
BufRightAboveSequential(mapping::FusedMapping& mapping)
{
  std::map<LogicalBuffer, bool> buf_right_above_sequential;
  for (auto path : GetPaths(mapping))
  {
    const auto& leaf = path.back();
    auto last_bufs = std::vector<LogicalBuffer>();
    for (const auto& node : path)
    {
      std::visit(
        [&last_bufs, &leaf, &buf_right_above_sequential] (auto&& node) {
          using NodeT = std::decay_t<decltype(node)>;

          if constexpr (std::is_same_v<NodeT, mapping::Storage>)
          {
            last_bufs.emplace_back(
              LogicalBuffer(node.buffer, node.dspace, GetNodeId(leaf))
            );
          }
          else if constexpr (std::is_same_v<NodeT, mapping::Sequential>)
          {
            for (const auto& buf : last_bufs)
            {
              buf_right_above_sequential.emplace(std::make_pair(
                buf,
                true
              ));
            }
            last_bufs.clear();
          }
          else 
          {
            for (const auto& buf : last_bufs)
            {
              buf_right_above_sequential.emplace(std::make_pair(
                buf,
                false
              ));
            }
            last_bufs.clear();
          }
        },
        node
      );
    }
  }

  return buf_right_above_sequential;
}

SkewsInfo SkewsFromMapping(mapping::FusedMapping& mapping,
                           const problem::FusedWorkload& workload)
{
  std::map<LogicalComputeUnit, Skew> lcomp_to_skew;
  std::map<LogicalBuffer, Skew> lbuf_to_skew;

  for (auto path : GetPaths(mapping))
  {
    const auto& leaf = path.back();

    // Get the last storage node in path for a particular buffer_id
    std::map<BufferId, mapping::NodeID> buf_to_last_storage_node_id;
    std::vector<std::tuple<BufferId, mapping::NodeID>> buf_node_id;
    std::vector<std::tuple<BufferId, DataSpaceId>> all_buffer_dspaces;
    for (const auto& node : path)
    {
      std::visit(
        [&buf_to_last_storage_node_id, &all_buffer_dspaces, &buf_node_id]
        (auto&& node) {
          using NodeT = std::decay_t<decltype(node)>;
          if constexpr (std::is_same_v<NodeT, mapping::Storage>)
          {
            const auto buf_id = node.buffer;
            buf_to_last_storage_node_id[buf_id] = node.id;
            all_buffer_dspaces.emplace_back(std::tie(buf_id, node.dspace));
            buf_node_id.emplace_back(std::tie(buf_id, node.id));
          }
          else if constexpr (std::is_same_v<NodeT, mapping::Compute>)
          {
            const auto compute_id = node.compute;
            buf_to_last_storage_node_id[compute_id] = node.id;
            buf_node_id.emplace_back(std::tie(compute_id, node.id));
          }
        },
        node
      );
    }

    std::map<mapping::NodeID, BufferId> node_to_cur_buf;
    std::size_t cur_buf_idx = 0;
    for (const auto& node : path)
    {
      std::visit(
        [&node_to_cur_buf, &buf_node_id, &cur_buf_idx] (auto&& node) {
          const auto& [cur_buf, cur_buf_last_node] = buf_node_id.at(cur_buf_idx);
          const auto node_id = mapping::GetNodeId(node);

          node_to_cur_buf[node_id] = cur_buf;

          if (node_id == cur_buf_last_node)
          {
            cur_buf_idx += 1;
          }
        },
        node
      );
    }

    // Generate tags, map, and which dims (and tags) should be removed for each
    // buffer
    std::vector<SpaceTime> tags;
    auto map = isl::map_from_multi_aff(isl::multi_aff::identity_on_domain(
      isl::space_alloc(GetIslCtx(), 0, 0, 0).domain()
    ));
    std::set<std::tuple<BufferId, DataSpaceId>> buf_storage_past;
    std::set<BufferId> buf_fully_complete;
    std::map<std::tuple<BufferId, DataSpaceId>, std::vector<bool>> buf_to_dim_removal_mask;
    for (const auto& node : path)
    {
      std::visit(
        [&buf_to_last_storage_node_id, &tags, &map, &buf_fully_complete,
         &buf_storage_past, &buf_to_dim_removal_mask, &all_buffer_dspaces,
         &node_to_cur_buf]
        (auto&& node)
        {
          using NodeT = std::decay_t<decltype(node)>;
          if constexpr (std::is_same_v<NodeT, mapping::Storage>)
          {
            buf_storage_past.emplace(std::tie(node.buffer, node.dspace));
            if (node.id == buf_to_last_storage_node_id.at(node.buffer))
            {
              buf_fully_complete.emplace(node.buffer);
            }
          }
          else if constexpr (mapping::IsLoopV<NodeT>)
          {
            if constexpr(std::is_same_v<NodeT, mapping::For>)
            {
              tags.push_back(Temporal());
            }
            else if constexpr (std::is_same_v<NodeT, mapping::ParFor>)
            {
              tags.push_back(Spatial(0, node_to_cur_buf.at(node.id)));
            }
            else
            {
              throw std::logic_error("unreachable");
            }
            map = isl::insert_equal_dims(std::move(map),
                                         isl::dim(map, isl_dim_in),
                                         isl::dim(map, isl_dim_out),
                                         1);

            for (const auto& [buf, dspace] : all_buffer_dspaces)
            {
              const auto buf_dspace = std::tie(buf, dspace);
              auto& removal_mask = buf_to_dim_removal_mask[buf_dspace];
              if (buf_fully_complete.find(buf) != buf_fully_complete.end())
              {
                removal_mask.emplace_back(true);
              }
              else if (buf_storage_past.find(buf_dspace) != buf_storage_past.end())
              {
                removal_mask.emplace_back(std::is_same_v<NodeT, mapping::For>);
              }
              else
              {
                removal_mask.emplace_back(false);
              }
            }
          }
          else if constexpr (std::is_same_v<NodeT, mapping::Pipeline>)
          {
            tags.push_back(PipelineSpatial());
            map = isl::insert_equal_dims(std::move(map),
                                         isl::dim(map, isl_dim_in),
                                         isl::dim(map, isl_dim_out),
                                         1);

            for (const auto& [buf, dspace] : all_buffer_dspaces)
            {
              const auto buf_dspace = std::tie(buf, dspace);
              auto& removal_mask = buf_to_dim_removal_mask[buf_dspace];
              if (buf_fully_complete.find(buf) != buf_fully_complete.end())
              {
                removal_mask.emplace_back(true);
              }
              else
              {
                removal_mask.emplace_back(false);
              }
            }
          }
          else if constexpr (std::is_same_v<NodeT, mapping::Sequential>)
          {
            tags.push_back(Sequential());
            map = isl::insert_equal_dims(std::move(map),
                                         isl::dim(map, isl_dim_in),
                                         isl::dim(map, isl_dim_out),
                                         1);

            for (const auto& [buf, dspace] : all_buffer_dspaces)
            {
              const auto buf_dspace = std::tie(buf, dspace);
              auto& removal_mask = buf_to_dim_removal_mask[buf_dspace];
              if (buf_fully_complete.find(buf) != buf_fully_complete.end())
              {
                removal_mask.emplace_back(true);
              }
              else
              {
                removal_mask.emplace_back(false);
              }
            }
          }
        },
        node
      );
    }

    for (const auto& [buf, dspace] : all_buffer_dspaces)
    {
      const auto& mask = buf_to_dim_removal_mask.at(std::tie(buf, dspace));
      const auto domain = map.domain();
      auto p_projector = isl::dim_projector(domain.get_space().release(), mask);
      auto p_map = isl_map_apply_range(p_projector, map.copy());

      auto buf_tags = std::vector<SpaceTime>();
      for (std::size_t i = 0; i < tags.size(); ++i)
      {
        const auto should_remove = mask.at(i);
        if (!should_remove)
        {
          buf_tags.emplace_back(tags.at(i));
        }
      }

      lbuf_to_skew.emplace(std::make_pair(
        LogicalBuffer(buf, dspace, GetNodeId(leaf)),
        Skew(buf_tags, isl::manage(p_map))
      ));
    }

    auto compute_id = std::get<mapping::Compute>(leaf).compute;
    lcomp_to_skew.emplace(std::make_pair(
      LogicalComputeUnit(compute_id, GetNodeId(leaf)),
      Skew(tags, map)
    ));
    auto einsum_id = std::get<mapping::Compute>(leaf).kernel;
    for (const auto tensor : workload.TensorsReadByEinsum(einsum_id))
    {
      lbuf_to_skew.emplace(std::make_pair(
        LogicalBuffer(compute_id, tensor, GetNodeId(leaf)),
        Skew(tags, map)
      ));
    }
    for (const auto tensor : workload.TensorsWrittenByEinsum(einsum_id))
    {
      lbuf_to_skew.emplace(std::make_pair(
        LogicalBuffer(compute_id, tensor, GetNodeId(leaf)),
        Skew(tags, map)
      ));
    }
  }

  return SkewsInfo{lbuf_to_skew, lcomp_to_skew};
}

} // namespace analysis