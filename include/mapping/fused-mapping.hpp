#pragma once

#include <memory>
#include <variant>
#include <optional>
#include <isl/cpp.h>

#include "compound-config/compound-config.hpp"
#include "util/metaprogramming.hpp"
#include "workload/workload.hpp"
#include "workload/fused-workload.hpp"

namespace mapping
{
using NodeID = size_t;
using BufferId = int;

class FusedMapping;
class MappingPath;
class MappingPaths;

struct Root
{
  NodeID id;
  std::optional<NodeID> child;

  Root(const NodeID& id);
};

struct For
{
  std::string iterator_name;
  problem::DimensionId op_dim;
  std::optional<size_t> begin;
  std::optional<size_t> end;
  std::optional<size_t> tile_size;

  NodeID id;
  std::optional<NodeID> child;

  For(const NodeID& id,
      const std::string& iterator_name,
      const problem::DimensionId& op_dim,
      std::optional<size_t>&& begin = std::nullopt,
      std::optional<size_t>&& end = std::nullopt);

  static For WithTileSize(const NodeID& id,
                          const std::string& iterator_name,
                          const problem::DimensionId& op_dim,
                          size_t tile_size);
};

struct ParFor
{
  std::string iterator_name;
  problem::DimensionId op_dim;
  int spatial;
  std::optional<size_t> begin;
  std::optional<size_t> end;
  std::optional<size_t> tile_size;

  NodeID id;
  std::optional<NodeID> child;

  ParFor(const NodeID& id,
         const std::string& iterator_name,
         const problem::DimensionId& op_dim,
         int spatial,
         std::optional<size_t>&& begin = std::nullopt,
         std::optional<size_t>&& end = std::nullopt);

  static ParFor WithTileSize(const NodeID& id,
                             const std::string& iterator_name,
                             const problem::DimensionId& op_dim,
                             int spatial,
                             size_t tile_size);
};

struct Storage
{
  BufferId buffer;
  problem::DataSpaceId dspace;
  std::vector<std::pair<NodeID, isl::map>> logical_buf_occupancy;
  bool exploits_reuse;

  NodeID id;
  std::optional<NodeID> child;

  Storage(const NodeID& id,
          const BufferId& buffer,
          const problem::DataSpaceId& dspace,
          bool exploits_reuse=true);
};

struct Compute
{
  problem::EinsumId kernel;
  /**
   * @brief An explicit tiling specifiction. E.g., [p_1, p_0] -> [4*p_1+p_0]
   * 
   * If given, bounds are not used to infer tiling map.
   */
  std::optional<isl::pw_multi_aff> tiling_spec;
  std::optional<double> parallelism;

  NodeID id;

  Compute(const NodeID& id,
          const problem::EinsumId& einsum,
          const std::optional<double> paralellism = std::nullopt,
          const std::optional<isl::pw_multi_aff>&& tiling_spec = std::nullopt);
};

struct Pipeline
{
  NodeID id;
  std::vector<NodeID> children;

  Pipeline(const NodeID& id);
};

struct Sequential
{
  NodeID id;
  std::vector<NodeID> children;

  Sequential(const NodeID& id);
};

using MappingNodeTypes
    = std::variant<Root, For, ParFor, Storage, Compute, Pipeline, Sequential>;

template<typename T>
inline constexpr bool IsLoopV = std::is_same_v<T, For> ||
                                std::is_same_v<T, ParFor>;

template<typename T>
inline constexpr bool IsBranchV = std::is_same_v<T, Sequential> ||
                                  std::is_same_v<T, Pipeline>;

template<typename T>
inline constexpr bool HasOneChildV = std::is_same_v<T, For> ||
                                     std::is_same_v<T, ParFor> ||
                                     std::is_same_v<T, Storage> ||
                                     std::is_same_v<T, Root>;

template<typename T>
inline constexpr bool HasManyChildrenV = IsBranchV<T>;

template<typename T>
inline constexpr bool HasNoChildV = std::is_same_v<T, Compute>;

NodeID GetNodeId(const MappingNodeTypes& node);

class FusedMappingNodeIterator
{
 public:
  FusedMappingNodeIterator& operator++();
  bool operator!=(const FusedMappingNodeIterator& other) const;
  MappingNodeTypes& operator*();

 private:
  friend FusedMapping;

  FusedMappingNodeIterator(std::map<NodeID, MappingNodeTypes>::iterator iter);

 private:
  std::map<NodeID, MappingNodeTypes>::iterator cur_;
};

struct AddChildToNode
{
  NodeID child_id_;
  AddChildToNode(NodeID child_id) : child_id_(child_id) {}

  void operator()(Root& node) { node.child = child_id_; }
  void operator()(For& node) { node.child = child_id_; }
  void operator()(ParFor& node) { node.child = child_id_; }
  void operator()(Storage& node) { node.child = child_id_; }
  void operator()(Compute& node)
  {
    (void) node;
    throw std::logic_error("compute has to be leaf");
  }
  void operator()(Pipeline& node) { node.children.emplace_back(child_id_); }
  void operator()(Sequential& node) { node.children.emplace_back(child_id_); }
};

class FusedMapping
{
 public:
  using Iterator = FusedMappingNodeIterator;

 public:
  FusedMapping();

  template<typename NodeT, typename... ArgsT>
  NodeID AddChild(NodeID parent_id, ArgsT... args)
  {
    auto [it, _] = nodes_.emplace(std::make_pair(
      nodes_.size(),
      MappingNodeTypes(NodeT(nodes_.size(), args...))
    ));

    auto child_id = it->first;
    std::visit(AddChildToNode(child_id), NodeAt(parent_id));

    return child_id;
  }

  template<typename FactoryT, typename... ArgsT>
  NodeID AddChild(FactoryT factory, NodeID parent_id, ArgsT... args)
  {
    auto [it, _] = nodes_.emplace(std::make_pair(
      nodes_.size(),
      MappingNodeTypes(factory(nodes_.size(), args...))
    ));

    auto child_id = it->first;
    std::visit(AddChildToNode(child_id), NodeAt(parent_id));

    return child_id;
  }

  const MappingNodeTypes& NodeAt(const NodeID& node_id) const;
  MappingNodeTypes& NodeAt(const NodeID& node_id);

  const Root& GetRoot() const;
  Root& GetRoot();

  Iterator begin();
  Iterator end();

 private:
  std::map<NodeID, MappingNodeTypes> nodes_;
};

class MappingPathsIterator
{
 public:
  MappingPath operator*();
  bool operator==(const MappingPathsIterator& other) const;
  bool operator!=(const MappingPathsIterator& other) const;
  MappingPathsIterator& operator++();

 private:
  struct DfsRecord
  {
    size_t path_backtrack_idx;
    std::reference_wrapper<MappingNodeTypes> ref_node;

    DfsRecord(size_t backtrack_idx, MappingNodeTypes& node);
  };

 private:
  FusedMapping& mapping_;
  std::vector<DfsRecord> dfs_stack_;
  std::vector<std::reference_wrapper<MappingNodeTypes>> path_;
  size_t idx_;
  bool done_;

  MappingPathsIterator(FusedMapping& paths, bool done=false);

  void GetNextPath();

 private:
  friend MappingPaths;
};

class MappingPaths
{
 public:
  using Iterator = MappingPathsIterator;

 public:
  Iterator begin();
  Iterator end();

 private:
  FusedMapping& fused_mapping_;

  MappingPaths(FusedMapping& mapping);

 private:
  friend MappingPaths GetPaths(FusedMapping& mapping);
};

class MappingPathNodeIterator
{
 public:
  MappingNodeTypes& operator*();
  bool operator==(const MappingPathNodeIterator& other) const;
  bool operator!=(const MappingPathNodeIterator& other) const;
  MappingPathNodeIterator& operator++();
 
 private:
  MappingPath& path_;
  size_t idx_;

  MappingPathNodeIterator(MappingPath& path, size_t idx=0);

 private:
  friend MappingPath;
};

class MappingPath
{
 public:
  using Iterator = MappingPathNodeIterator;

 public:
  Iterator begin();
  Iterator end();

  MappingNodeTypes& back();

 private:
  std::vector<std::reference_wrapper<MappingNodeTypes>> ref_nodes_;

  MappingPath(std::vector<std::reference_wrapper<MappingNodeTypes>> ref_nodes);

 private:
  friend MappingPathsIterator;
  friend MappingPathNodeIterator;
};

MappingPaths GetPaths(FusedMapping& mapping);

struct DfsRange;

DfsRange IterateInDfsOrder(mapping::FusedMapping&);

struct DfsIterator
{
  bool operator==(const DfsIterator& other);
  bool operator!=(const DfsIterator& other);

  DfsIterator& operator++();

  /**
   * @brief Id of current node, the id of its parent, and the number of loops
   *    above this node.
   */
  std::tuple<NodeID, NodeID, size_t> operator*();

 private:
  FusedMapping& mapping_;
  std::function<bool(const MappingNodeTypes&)> filter_;
  std::vector<NodeID> stack_;
  std::map<NodeID, NodeID> child_to_parent_;
  std::map<NodeID, size_t> node_to_n_loops_;

  DfsIterator(FusedMapping& mapping,
              const std::vector<NodeID>& stack,
              std::function<bool(const MappingNodeTypes&)> filter);

  friend DfsRange;
};

struct DfsRange
{
  DfsIterator begin();
  DfsIterator end();

 private:
  FusedMapping& mapping_;
  std::function<bool(const MappingNodeTypes&)> filter_;

  DfsRange(FusedMapping& mapping,
           std::function<bool(const MappingNodeTypes&)> filter);

  friend DfsRange
  IterateInDfsOrder(FusedMapping&,
                    std::function<bool(const MappingNodeTypes&)>);
};

DfsRange
IterateInDfsOrder(FusedMapping& mapping,
                  std::function<bool(const MappingNodeTypes&)> filter);

template<typename... IncludedNodesT>
DfsRange IterateInDfsOrder(FusedMapping& mapping)
{
  auto filter =
    [](const MappingNodeTypes& node) -> bool
    {
      return std::visit(
        [](auto&& node)
        {
          using T = std::decay_t<decltype(node)>;
          return IsAnyOfV<T, IncludedNodesT...>;
        },
        node
      );
    };
  
  return IterateInDfsOrder(mapping, filter);
}

template<> DfsRange IterateInDfsOrder<>(FusedMapping& mapping);

FusedMapping ParseMapping(const config::CompoundConfigNode& cfg,
                          const problem::FusedWorkload& workload);

}; // namespace mapping