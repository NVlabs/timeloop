#include "mapping/fused-mapping.hpp"
#include "loop-analysis/isl-ir.hpp"

namespace mapping
{

NodeID CompoundConfigNodeToMapping(FusedMapping& mapping,
                                   const problem::FusedWorkload& workload,
                                   NodeID parent_id,
                                   const config::CompoundConfigNode& cfg)
{
  std::string type;
  BufferID target;
  std::string dspace_name;
  std::string dim_name;
  std::string einsum_name;
  int factor;

  cfg.lookupValue("type", type);
  if (type == "temporal")
  {
    cfg.lookupValue("dimension", dim_name);
    std::cout << "dim name: " << dim_name << std::endl;
    auto dim_id = workload.DimensionNameToId().at(dim_name);
    cfg.lookupValue("factor", factor);
    if (factor == 0)
    {
      return mapping.AddChild<For>(parent_id, "", dim_id);
    }
    else
    {
      return mapping.AddChild<For>(parent_id, "", dim_id, 0, factor);
    }
  }
  else if (type == "spatial")
  {
    cfg.lookupValue("dimension", dim_name);
    auto dim_id = workload.DimensionNameToId().at(dim_name);
    cfg.lookupValue("factor", factor);
    if (factor == 0)
    {
      return mapping.AddChild<ParFor>(parent_id, "", dim_id);
    }
    else
    {
      return mapping.AddChild<ParFor>(parent_id, "", dim_id, 0, factor);
    }
  }
  else if (type == "storage")
  {
    cfg.lookupValue("target", target);
    std::cout << "storage target: " << target << std::endl;
    std::vector<std::string> dspace_names;
    cfg.lookupArrayValue("dspace", dspace_names);
    NodeID node;
    for (const auto& dspace_name : dspace_names)
    {
      auto dspace = workload.DataSpaceNameToId().at(dspace_name);
      node = mapping.AddChild<Storage>(parent_id, target, dspace);
    }
    return node;
  }
  else if (type == "compute")
  {
    cfg.lookupValue("einsum", einsum_name);
    auto einsum = workload.EinsumNameToId().at(einsum_name);
    return mapping.AddChild<Compute>(parent_id, einsum);
  }
  else if (type == "pipeline")
  {
    return mapping.AddChild<Pipeline>(parent_id);
  }
  else if (type == "sequential")
  {
    return mapping.AddChild<Sequential>(parent_id);
  }
  else
  {
    throw std::logic_error("unknown node type: " + type);
  }
}

FusedMapping ParseMapping(const config::CompoundConfigNode& cfg,
                          const problem::FusedWorkload& workload)
{
  std::string mapping_type;
  if (!cfg.lookupValue("type", mapping_type) || mapping_type!="fused")
  {
    throw std::logic_error("wrong mapping type");
  }

  auto nodes = cfg.lookup("nodes");

  FusedMapping mapping;
  auto parent_id = mapping.GetRoot().id;
  for (int i = 0; i < nodes.getLength(); ++i)
  {
    auto cur_node = nodes[i];
    parent_id =
      CompoundConfigNodeToMapping(mapping, workload, parent_id, cur_node);
  }

  return mapping;
}

} // namespace mapping