/**
 * @file mapping/fused-mapping/parser.cpp
 * @author Michael Gilbert (gilbertm@mit.edu)
 * @brief Implementation of fused mapping parser for LoopTree.
 * 
 * To add a new node type, implement a function with signature that matches
 * the `NodeParser` type, then add it to `NODE_TYPE_TO_PARSER`.
 */
#include "mapping/fused-mapping.hpp"
#include "loop-analysis/isl-ir.hpp"
#include "model/topology.hpp"

namespace mapping
{

/******************************************************************************
 * Local declarations
 *****************************************************************************/

void InsertToMapping(NodeID parent_id,
                     FusedMapping& mapping,
                     const config::CompoundConfigNode& nodes_cfg,
                     const problem::FusedWorkload& workload);

NodeID CompoundConfigNodeToMapping(NodeID parent_id,
                                   FusedMapping& mapping,
                                   const config::CompoundConfigNode& cfg,
                                   const problem::FusedWorkload& workload);

typedef NodeID (*NodeParser)(NodeID,
                             FusedMapping&,
                             const config::CompoundConfigNode&,
                             const problem::FusedWorkload&);

#define DEFINE_NODE_PARSER(name)       \
  NodeID name(                         \
    NodeID,                            \
    FusedMapping&,                     \
    const config::CompoundConfigNode&, \
    const problem::FusedWorkload&)

DEFINE_NODE_PARSER(ParseTemporalNode);
DEFINE_NODE_PARSER(ParseSpatialNode);
DEFINE_NODE_PARSER(ParseStorageNode);
DEFINE_NODE_PARSER(ParseComputeNode);
DEFINE_NODE_PARSER(ParsePipelineNode);
DEFINE_NODE_PARSER(ParseSequentialNode);

const std::map<std::string, NodeParser> NODE_TYPE_TO_PARSER = {
  std::make_pair("temporal",   &ParseTemporalNode),
  std::make_pair("spatial",    &ParseSpatialNode),
  std::make_pair("storage",    &ParseStorageNode),
  std::make_pair("compute",    &ParseComputeNode),
  std::make_pair("pipeline",   &ParsePipelineNode),
  std::make_pair("sequential", &ParseSequentialNode),
};

/******************************************************************************
 * Global implementations
 *****************************************************************************/

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
  InsertToMapping(parent_id, mapping, nodes, workload);

  return mapping;
}

/******************************************************************************
 * Local implementations
 *****************************************************************************/

NodeID CompoundConfigNodeToMapping(NodeID parent_id,
                                   FusedMapping& mapping,
                                   const config::CompoundConfigNode& cfg,
                                   const problem::FusedWorkload& workload)
{
  std::string type;
  cfg.lookupValue("type", type);

  auto node_parser_it = NODE_TYPE_TO_PARSER.find(type);
  if (node_parser_it == NODE_TYPE_TO_PARSER.end())
  {
    throw std::logic_error("node type unknown");
  }

  auto node_parser = node_parser_it->second;
  return node_parser(parent_id, mapping, cfg, workload);
}

NodeID ParseTemporalNode(NodeID parent_id,
                         FusedMapping& mapping,
                         const config::CompoundConfigNode& cfg,
                         const problem::FusedWorkload& workload)
{
  const auto& dim_name_to_id = workload.DimensionNameToId();

  std::string iterator_name = "";
  if (cfg.exists("iterator_name"))
  {
    cfg.lookupValue("iterator_name", iterator_name);
  }

  std::string dim_name;
  if (!cfg.exists("rank"))
  {
    throw std::runtime_error("temporal node missing rank");
  }
  cfg.lookupValue("rank", dim_name);
  auto dim_id_it = dim_name_to_id.find(dim_name);
  if (dim_id_it == dim_name_to_id.end())
  {
    throw std::out_of_range("Could not find dim " + dim_name + " in workload");
  }
  auto dim_id = dim_id_it->second;

  if (cfg.exists("factor"))
  {
    int factor = 0;
    cfg.lookupValue("factor", factor);
    if (factor == 0)
    {
      return mapping.AddChild<For>(parent_id, iterator_name, dim_id);
    }
    else
    {
      return mapping.AddChild<For>(parent_id,
                                   iterator_name,
                                   dim_id,
                                   0,
                                   factor);
    }
  }
  else
  {
    std::string tile_size_str;
    if (!cfg.exists("tile_shape"))
    {
      throw std::runtime_error("temporal node missing tile_shape");
    }
    cfg.lookupValue("tile_shape", tile_size_str);
    int tile_size = std::stoi(tile_size_str);
    return mapping.AddChild(For::WithTileSize,
                            parent_id,
                            iterator_name,
                            dim_id,
                            tile_size);
  }
}

NodeID ParseSpatialNode(NodeID parent_id,
                        FusedMapping& mapping,
                        const config::CompoundConfigNode& cfg,
                        const problem::FusedWorkload& workload)
{
  const auto& dim_name_to_id = workload.DimensionNameToId();

  std::string iterator_name = "";
  if (cfg.exists("iterator_name"))
  {
    cfg.lookupValue("iterator_name", iterator_name);
  }

  std::string dim_name;
  cfg.lookupValue("rank", dim_name);
  auto dim_id = dim_name_to_id.at(dim_name);

  int spatial = 0;
  if (cfg.exists("spatial"))
  {
    cfg.lookupValue("spatial", spatial);
  }

  if (cfg.exists("factor"))
  {
    int factor = 0;
    cfg.lookupValue("factor", factor);
    if (factor == 0)
    {
      return mapping.AddChild<ParFor>(parent_id,
                                      iterator_name,
                                      dim_id,
                                      spatial);
    }
    else
    {
      return mapping.AddChild<ParFor>(parent_id,
                                      iterator_name,
                                      dim_id,
                                      spatial,
                                      0,
                                      factor);
    }
  }
  else
  {
    std::string tile_size_str;
    cfg.lookupValue("tile_size", tile_size_str);
    int tile_size = std::stoi(tile_size_str);
    return mapping.AddChild(ParFor::WithTileSize,
                            parent_id,
                            iterator_name,
                            dim_id,
                            spatial,
                            tile_size);
  }
}

NodeID ParseStorageNode(NodeID parent_id,
                        FusedMapping& mapping,
                        const config::CompoundConfigNode& cfg,
                        const problem::FusedWorkload& workload)
{
  const auto& dspace_name_to_id = workload.DataSpaceNameToId();

  std::string target = "";
  if (!cfg.exists("target"))
  {
    throw std::runtime_error("storage node missing target");
  }
  cfg.lookupValue("target", target);

  // If target is an integer, we mean buffer_id; otherwise, we mean buffer name
  BufferId buffer_id;
  buffer_id = std::stoi(target);

  std::vector<std::string> dspace_names;
  cfg.lookupArrayValue("dspace", dspace_names);

  bool exploits_reuse = true;
  if (cfg.exists("exploits_reuse"))
  {
    cfg.lookupValue("exploits_reuse", exploits_reuse);
  }

  auto node = parent_id;
  for (const auto& dspace_name : dspace_names)
  {
    auto dspace = dspace_name_to_id.at(dspace_name);
    node = mapping.AddChild<Storage>(node, buffer_id, dspace, exploits_reuse);
  }

  return node;
}

NodeID ParseComputeNode(NodeID parent_id,
                        FusedMapping& mapping,
                        const config::CompoundConfigNode& cfg,
                        const problem::FusedWorkload& workload)
{
  std::string einsum_name = "";
  cfg.lookupValue("einsum", einsum_name);
  auto einsum = workload.EinsumNameToId().at(einsum_name);

  auto parallelism = std::optional<double>();
  if (cfg.exists("parallelism"))
  {
    double parallelism_val;
    cfg.lookupValue("parallelism", parallelism_val);
    parallelism = parallelism_val;
  }

  std::string target_str = "";
  cfg.lookupValue("target", target_str);
  BufferId compute = std::stoi(target_str);

  return mapping.AddChild<Compute>(
    parent_id,
    einsum,
    compute,
    parallelism,
    std::nullopt
  );
}

NodeID ParsePipelineNode(NodeID parent_id,
                         FusedMapping& mapping,
                         const config::CompoundConfigNode& cfg,
                         const problem::FusedWorkload& workload)
{
  (void) cfg;
  (void) workload;
  return mapping.AddChild<Pipeline>(parent_id);
}

NodeID ParseSequentialNode(NodeID parent_id,
                           FusedMapping& mapping,
                           const config::CompoundConfigNode& cfg,
                           const problem::FusedWorkload& workload)
{
  (void) cfg;
  (void) workload;
  return mapping.AddChild<Sequential>(parent_id);
}

void InsertToMapping(NodeID parent_id,
                     FusedMapping& mapping,
                     const config::CompoundConfigNode& nodes_cfg,
                     const problem::FusedWorkload& workload)
{
  const auto last_node_idx = nodes_cfg.getLength()-1;
  for (int i = 0; i <= last_node_idx; ++i)
  {
    auto cur_node = nodes_cfg[i];
    parent_id = CompoundConfigNodeToMapping(parent_id,
                                            mapping,
                                            cur_node,
                                            workload);

    if (cur_node.exists("branches"))
    {
      auto branches_cfg = cur_node.lookup("branches");
      for (int j = 0; j < branches_cfg.getLength(); ++j)
      {
        InsertToMapping(parent_id,
                        mapping,
                        branches_cfg[j],
                        workload);
      }
    }

    bool cur_node_is_branch_or_leaf = std::visit(
      [](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        return mapping::HasNoChildV<T> || mapping::IsBranchV<T>;
      },
      mapping.NodeAt(parent_id)
    );

    if (i == last_node_idx && !cur_node_is_branch_or_leaf)
    {
      throw std::logic_error("Last node in a list in mapping must be a branch or a leaf.");
    }
  }
}

} // namespace mapping