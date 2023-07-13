#include "loop-analysis/isl-analysis/capacity-analysis.hpp"

#include <barvinok/isl.h>
#include <boost/range/adaptor/reversed.hpp>

#include "isl-wrapper/isl-functions.hpp"

namespace analysis
{

struct BufferInfo
{
  BufferId buffer_id;
  DataSpaceID dspace_id;
  bool exploits_reuse;
  bool right_above_branch;
};

std::map<BufferId, isl_pw_qpolynomial*> ComputeCapacityFromMapping(
  mapping::NodeID cur_node_id,
  size_t n_loops,
  std::vector<BufferInfo>& buffers,
  mapping::FusedMapping& mapping,
  const std::map<LogicalBuffer, Occupancy>& occupancies,
  const problem::FusedWorkload& workload,
  const std::map<mapping::NodeID, std::set<EinsumID>> node_to_einsums
);

template<typename RangeT>
isl_pw_qpolynomial*
ReduceCapacityToLastBranch(__isl_take isl_pw_qpolynomial* p_cap,
                           RangeT dim_in_tags_reversed)
{
  auto mask = std::vector<bool>(isl_pw_qpolynomial_dim(p_cap, isl_dim_in),
                                false);
  auto i = mask.size()-1;
  for (const auto& tag : dim_in_tags_reversed)
  {
    if (std::holds_alternative<Sequential>(tag)
        || std::holds_alternative<PipelineSpatial>(tag))
    {
      break;
    }
    if (IsTemporal(tag))
    {
      mask[i] = true;
    }
    --i;
  }

  auto p_proj = isl::dim_projector(isl_pw_qpolynomial_get_domain_space(p_cap),
                                   mask);
  auto p_cap_fold = isl_pw_qpolynomial_fold_from_pw_qpolynomial(isl_fold_max,
                                                                p_cap);
  isl_bool tight;
  p_cap_fold = isl_map_apply_pw_qpolynomial_fold(p_proj, p_cap_fold, &tight);
  return isl::gather_pw_qpolynomial_from_fold(p_cap_fold);
}

template<typename RangeT>
isl_pw_qpolynomial*
ReduceCapacity(__isl_take isl_pw_qpolynomial* p_cap,
               RangeT dim_in_tags_reversed)
{
  auto mask = std::vector<bool>(isl_pw_qpolynomial_dim(p_cap, isl_dim_in),
                                false);
  auto i = mask.size()-1;
  for (const auto& tag : dim_in_tags_reversed)
  {
    if (IsTemporal(tag) || std::holds_alternative<Sequential>(tag)
        || std::holds_alternative<PipelineTemporal>(tag))
    {
      mask[i] = true;
    }
    --i;
  }

  auto p_proj = isl::dim_projector(isl_pw_qpolynomial_get_domain_space(p_cap),
                                   mask);
  auto p_cap_fold = isl_pw_qpolynomial_fold_from_pw_qpolynomial(isl_fold_max,
                                                                p_cap);
  isl_bool tight;
  p_cap_fold = isl_map_apply_pw_qpolynomial_fold(p_proj, p_cap_fold, &tight);
  return isl::gather_pw_qpolynomial_from_fold(p_cap_fold);
}

std::map<mapping::NodeID, std::set<EinsumID>>
GatherEinsumFromLeaves(mapping::FusedMapping& mapping)
{
  auto result = std::map<mapping::NodeID, std::set<EinsumID>>();
  for (auto path : mapping::GetPaths(mapping))
  {
    auto einsum_id = std::get<mapping::Compute>(path.back()).kernel;
    for (const auto& node_id : path)
    {
      result[mapping::GetNodeId(node_id)].insert(einsum_id);
    }
  }
  return result;
}


/**
 * @brief Compute capacity usage for each buffer id.
 * 
 */
std::map<mapping::BufferId, isl_pw_qpolynomial*> ComputeCapacityFromMapping(
  mapping::FusedMapping& mapping,
  const std::map<LogicalBuffer, Occupancy>& occupancies,
  const problem::FusedWorkload& workload
)
{
  auto final_result = std::map<mapping::BufferId, isl_pw_qpolynomial*>();
  auto node_to_einsums = GatherEinsumFromLeaves(mapping);
  auto buffers = std::vector<BufferInfo>();
  return ComputeCapacityFromMapping(
    mapping.GetRoot().id,
    0,
    buffers,
    mapping,
    occupancies,
    workload,
    node_to_einsums
  );
}

std::map<BufferId, isl_pw_qpolynomial*> ComputeCapacityFromMapping(
  mapping::NodeID cur_node_id,
  size_t n_loops,
  std::vector<BufferInfo>& buffers,
  mapping::FusedMapping& mapping,
  const std::map<LogicalBuffer, Occupancy>& occupancies,
  const problem::FusedWorkload& workload,
  const std::map<mapping::NodeID, std::set<EinsumID>> node_to_einsums
)
{
  auto to_aggregate = std::map<BufferId, isl_pw_qpolynomial*>();

  const auto& node = mapping.NodeAt(cur_node_id);
  std::visit(
    [&cur_node_id, &buffers, &mapping, &occupancies, &workload,
     &node_to_einsums, &n_loops, &to_aggregate]
    (auto&& node)
    {
      using T = std::decay_t<decltype(node)>;

      if constexpr (std::is_same_v<T, mapping::Storage>)
      {
        buffers.emplace_back(
          BufferInfo{node.buffer, node.dspace, node.exploits_reuse, true}
        );
      }
      else if constexpr (mapping::IsLoopV<T>)
      {
        for (auto& b : buffers)
        {
          b.right_above_branch = false;
        }
      }

      auto to_agg_from_children = std::map<BufferId, isl_pw_qpolynomial*>();
      if constexpr (mapping::HasOneChildV<T>)
      {
        auto child_n_loops = n_loops;
        if constexpr (mapping::IsLoopV<T>)
        {
          ++child_n_loops;
        }
        to_agg_from_children = 
          ComputeCapacityFromMapping(*node.child,
                                     child_n_loops,
                                     buffers,
                                     mapping,
                                     occupancies,
                                     workload,
                                     node_to_einsums);
      }
      else if constexpr (mapping::HasManyChildrenV<T>)
      {
        for (auto child : node.children)
        {
          auto child_buffers = std::vector<BufferInfo>();
          auto to_agg_from_child = 
            ComputeCapacityFromMapping(child,
                                       n_loops,
                                       child_buffers,
                                       mapping,
                                       occupancies,
                                       workload,
                                       node_to_einsums);
          for (auto [buf_id, cap] : to_agg_from_child)
          {
            to_agg_from_children[buf_id] = cap;
          }
        }
      }

      if constexpr (std::is_same_v<T, mapping::Compute>)
      {
        for (const auto& b : buffers)
        {
          isl_map* p_total_occ = nullptr;
          std::vector<SpaceTime> dim_in_tags;
          for (const auto& [lb, occ] : occupancies)
          {
            if (lb.buffer_id == b.buffer_id && lb.dspace_id == b.dspace_id)
            {
              dim_in_tags = occ.dim_in_tags;
              auto p_occ = occ.map.copy();
              if (!p_total_occ)
              {
                p_total_occ = p_occ;
              }
              else
              {
                p_total_occ = isl_map_union(p_total_occ, p_occ);
              }
            }
          }
          std::reverse(dim_in_tags.begin(), dim_in_tags.end());
          auto p_cap = ReduceCapacityToLastBranch(isl_map_card(p_total_occ),
                                                  dim_in_tags);

          auto it = to_aggregate.find(b.buffer_id);
          if (it == to_aggregate.end())
          {
            to_aggregate[b.buffer_id] = p_cap;
          }
          else
          {
            auto p_cur_cap = to_aggregate[b.buffer_id];
            p_cur_cap = isl_pw_qpolynomial_add(p_cur_cap, p_cap);
            to_aggregate[b.buffer_id] = p_cur_cap;
          }
        }
      }
      else if constexpr (mapping::IsBranchV<T>)
      {
        auto dspace_start = std::map<DataSpaceID, size_t>();
        auto dspace_end = std::map<DataSpaceID, size_t>();
        size_t i = 0;
        for (auto child_id : node.children)
        {
          for (const auto& e : node_to_einsums.at(child_id))
          {
            for (const auto& d : workload.TensorsReadByEinsum(e))
            {
              auto it = dspace_start.find(d);
              if (it == dspace_start.end())
              {
                dspace_start.emplace_hint(it, d, i);
                dspace_end.emplace(d, i);
              }
              else
              {
                dspace_end[d] = i;
              }
            }
            for (const auto& d : workload.TensorsWrittenByEinsum(e))
            {
              auto it = dspace_start.find(d);
              if (it == dspace_start.end())
              {
                dspace_start.emplace_hint(it, d, i);
                dspace_end.emplace(d, i);
              }
              else
              {
                dspace_end[d] = i;
              }
            }
          }
          ++i;
        }

        for (const auto& b : buffers)
        {
          isl_map* p_total_occ = nullptr;
          std::vector<SpaceTime> dim_in_tags;
          for (const auto& [lb, occ] : occupancies)
          {
            if (lb.buffer_id == b.buffer_id && lb.dspace_id == b.dspace_id)
            {
              dim_in_tags = occ.dim_in_tags;
              auto p_occ = occ.map.copy();
              if (!p_total_occ)
              {
                p_total_occ = p_occ;
              }
              else
              {
                p_total_occ = isl_map_union(p_total_occ, p_occ);
              }
            }
          }

          auto n_temp = std::count_if(
            dim_in_tags.begin(),
            dim_in_tags.end(),
            [](auto&& tag) { return std::holds_alternative<Temporal>(tag); }
          );

          p_total_occ = isl_map_insert_dims(p_total_occ,
                                            isl_dim_in,
                                            dim_in_tags.size(),
                                            n_loops-n_temp+1);
          if constexpr (std::is_same_v<T, mapping::Sequential>)
          {
            for (size_t i = 0; i < n_loops-n_temp; ++i)
            {
              dim_in_tags.push_back(Temporal());
            }
            dim_in_tags.push_back(Sequential());
          }
          else
          {
            for (size_t i = 0; i < n_loops-n_temp; ++i)
            {
              dim_in_tags.push_back(PipelineTemporal());
            }
            dim_in_tags.push_back(PipelineSpatial());
          }
          auto start = 0;
          auto end = node.children.size();
          if constexpr (std::is_same_v<T, mapping::Sequential>)
          {
            if (!b.exploits_reuse && b.right_above_branch)
            {
              start = dspace_start.at(b.dspace_id);
              end = dspace_end.at(b.dspace_id);
            }
          }
          p_total_occ = isl::bound_dim_si(p_total_occ,
                                          isl_dim_in,
                                          dim_in_tags.size()-1,
                                          start,
                                          end);
          auto p_cap = isl_map_card(p_total_occ);

          auto it = to_aggregate.find(b.buffer_id);
          if (it == to_aggregate.end())
          {
            to_aggregate[b.buffer_id] = p_cap;
          }
          else
          {
            auto p_cur_cap = to_aggregate[b.buffer_id];
            p_cur_cap = isl_pw_qpolynomial_add(p_cur_cap, p_cap);
            to_aggregate[b.buffer_id] = p_cur_cap;
          }
        }

        for (const auto& [buf_id, child_cap] : to_agg_from_children)
        {
          auto it = to_aggregate.find(buf_id);
          if (it == to_aggregate.end())
          {
            to_aggregate[buf_id] = child_cap;
          }
          else
          {
            auto p_cur_cap = to_aggregate[buf_id];
            p_cur_cap = isl_pw_qpolynomial_add(p_cur_cap, child_cap);
            to_aggregate[buf_id] = p_cur_cap;
          }
        }
      }
      else
      {
        to_aggregate = to_agg_from_children;
      }
    },
    node
  );

  return to_aggregate;
}

}; // namespace analysis