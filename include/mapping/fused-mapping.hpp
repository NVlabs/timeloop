#pragma once

#include <memory>
#include <variant>
#include <optional>
#include <isl/cpp.h>

#include "compound-config/compound-config.hpp"
#include "workload/workload.hpp"
#include "workload/fused-workload.hpp"

namespace mapping
{
using NodeID = size_t;
using BufferID = int;

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
  problem::Shape::FlattenedDimensionID op_dim;
  std::optional<size_t> begin;
  std::optional<size_t> end;

  NodeID id;
  std::optional<NodeID> child;

  For(const NodeID& id,
      const std::string& iterator_name,
      const problem::Shape::FlattenedDimensionID& op_dim,
      std::optional<size_t>&& begin = std::nullopt,
      std::optional<size_t>&& end = std::nullopt);
};

struct ParFor
{
  std::string iterator_name;
  problem::Shape::FlattenedDimensionID op_dim;
  // TODO: missing spacetime_dim
  std::optional<size_t> begin;
  std::optional<size_t> end;

  NodeID id;
  std::optional<NodeID> child;

  ParFor(const NodeID& id,
         const std::string& iterator_name,
         const problem::Shape::FlattenedDimensionID& op_dim,
         std::optional<size_t>&& begin = std::nullopt,
         std::optional<size_t>&& end = std::nullopt);
};

struct Storage
{
  BufferID buffer;
  problem::Shape::DataSpaceID dspace;
  std::vector<std::pair<NodeID, isl::map>> logical_buf_occupancy;

  NodeID id;
  std::optional<NodeID> child;

  Storage(const NodeID& id,
          const BufferID& buffer,
          const problem::Shape::DataSpaceID& dspace);
};

struct Compute
{
  // TODO: This should only be defined once somewhere, but can be found here
  // and in isl-ir.hpp
  problem::EinsumID kernel;
  /**
   * @brief An explicit tiling specifiction. E.g., [p_1, p_0] -> [4*p_1+p_0]
   * 
   * If given, bounds are not used to infer tiling map.
   */
  std::optional<isl::pw_multi_aff> tiling_spec;

  NodeID id;

  Compute(const NodeID& id,
          const problem::EinsumID& einsum,
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

  template<typename LoopT, typename... ArgsT>
  NodeID AddChild(NodeID parent_id, ArgsT... args)
  {
    auto [it, _] = nodes_.emplace(std::make_pair(
      nodes_.size(),
      MappingNodeTypes(std::in_place_type<LoopT>, nodes_.size(), args...)
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

 private:
  std::vector<std::reference_wrapper<MappingNodeTypes>> ref_nodes_;

  MappingPath(std::vector<std::reference_wrapper<MappingNodeTypes>> ref_nodes);

 private:
  friend MappingPathsIterator;
  friend MappingPathNodeIterator;
};

MappingPaths GetPaths(FusedMapping& mapping);

FusedMapping ParseMapping(const config::CompoundConfig& cfg);

}; // namespace mapping