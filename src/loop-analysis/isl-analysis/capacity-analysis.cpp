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

struct IntermediateResult
{
  std::set<EinsumID> einsums;
};

IntermediateResult ComputeCapacityFromMapping(
  mapping::NodeID cur_node_id,
  mapping::FusedMapping& mapping,
  const std::map<LogicalBuffer, Occupancy>& occupancies,
  const problem::FusedWorkload& workload
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
    if (IsTemporal(tag) || std::holds_alternative<Sequential>
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

std::map<mapping::BufferId, isl_pw_qpolynomial*>
ComputeCapacityFromMapping(
  mapping::FusedMapping& mapping,
  const std::map<LogicalBuffer, Occupancy>& occupancies,
  const problem::FusedWorkload& workload
)
{
  ComputeCapacityFromMapping(mapping.GetRoot().id, mapping, occupancies, workload);
  return std::map<mapping::BufferId, isl_pw_qpolynomial*>();
}

IntermediateResult ComputeCapacityFromMapping(
  mapping::NodeID cur_node_id,
  mapping::FusedMapping& mapping,
  const std::map<LogicalBuffer, Occupancy>& occupancies,
  const problem::FusedWorkload& workload
)
{
  IntermediateResult result;
  std::vector<BufferInfo> buffers;
  auto keep_going = true;
  while (keep_going)
  {
    const auto& node = mapping.NodeAt(cur_node_id);
    std::visit(
      [&cur_node_id, &buffers, &keep_going, &result, &mapping, &occupancies,
       &workload]
      (auto&& node)
      {
        using T = std::decay_t<decltype(node)>;

        if constexpr (std::is_same_v<T, mapping::Storage>)
        {
          buffers.emplace_back(
            BufferInfo{node.buffer, node.dspace, node.exploits_reuse, true}
          );
        }
        else if constexpr (std::is_same_v<T, mapping::Compute>)
        {
          result.einsums.insert(node.kernel);

          for (const auto& b : buffers)
          {
            isl_map* p_total_occ = nullptr;
            std::vector<SpaceTime> dim_in_tags;
            for (const auto& [lb, occ] : occupancies)
            {
              if (lb.buffer_id == b.buffer_id && lb.dspace_id == b.dspace_id)
              {
                dim_in_tags = occ.dim_in_tags;
                auto p_lb_cap = ReduceCapacityToLastBranch(
                  isl_map_card(occ.map.copy()),
                  occ.dim_in_tags
                );
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
            auto p_cap = ReduceCapacity(isl_map_card(p_total_occ), dim_in_tags);
            std::cout << b.buffer_id << ", " << b.dspace_id << isl_pw_qpolynomial_to_str(p_cap) << std::endl;
          }
        }
        else if constexpr (mapping::IsBranchV<T>)
        {
          auto dspace_start = std::map<DataSpaceID, size_t>();
          auto dspace_end = std::map<DataSpaceID, size_t>();
          size_t i = 0;
          for (auto child_id : node.children)
          {
            auto child_res =
              ComputeCapacityFromMapping(child_id, mapping, occupancies, workload);
            for (const auto& e : child_res.einsums)
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
            result.einsums.merge(std::move(child_res.einsums));
            ++i;
          }

          if constexpr (std::is_same_v<T, mapping::Sequential>)
          {
            for (const auto& b : buffers)
            {
              if (!b.exploits_reuse && b.right_above_branch)
              {
                auto start = dspace_start.at(b.dspace_id);
                auto end = dspace_end.at(b.dspace_id);

                isl_map* p_total_occ = nullptr;
                std::vector<SpaceTime> dim_in_tags;
                for (const auto& [lb, occ] : occupancies)
                {
                  if (lb.buffer_id == b.buffer_id && lb.dspace_id == b.dspace_id)
                  {
                    dim_in_tags = occ.dim_in_tags;
                    auto p_lb_cap = ReduceCapacityToLastBranch(
                      isl_map_card(occ.map.copy()),
                      occ.dim_in_tags
                    );
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
                auto p_cap = ReduceCapacity(isl_map_card(p_total_occ), dim_in_tags);
                std::cout << b.buffer_id << ", " << b.dspace_id << isl_pw_qpolynomial_to_str(p_cap) << std::endl;
              }
              else
              {
                isl_map* p_total_occ = nullptr;
                std::vector<SpaceTime> dim_in_tags;
                for (const auto& [lb, occ] : occupancies)
                {
                  if (lb.buffer_id == b.buffer_id && lb.dspace_id == b.dspace_id)
                  {
                    dim_in_tags = occ.dim_in_tags;
                    auto p_lb_cap = ReduceCapacityToLastBranch(
                      isl_map_card(occ.map.copy()),
                      occ.dim_in_tags
                    );
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
                auto p_cap = ReduceCapacity(isl_map_card(p_total_occ), dim_in_tags);
                std::cout << b.buffer_id << ", " << b.dspace_id << isl_pw_qpolynomial_to_str(p_cap) << std::endl;
              }
            }
          }
          else if constexpr (std::is_same_v<T, mapping::Pipeline>)
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
                  auto p_lb_cap = ReduceCapacityToLastBranch(
                    isl_map_card(occ.map.copy()),
                    occ.dim_in_tags
                  );
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
              auto p_cap = ReduceCapacity(isl_map_card(p_total_occ), dim_in_tags);
              std::cout << b.buffer_id << ", " << b.dspace_id << isl_pw_qpolynomial_to_str(p_cap) << std::endl;
            }
          }
          else
          {
            throw std::logic_error("unknown mapping branch node");
          }
        }
        else if constexpr (mapping::IsLoopV<T>)
        {
          for (auto& b : buffers)
          {
            b.right_above_branch = false;
          }
        }

        if constexpr(mapping::HasOneChildV<T>)
        {
          cur_node_id = *node.child;
        }
        else
        {
          keep_going = false;
        }
      },
      node
    );
  }

  return result;
}
}; // namespace analysis