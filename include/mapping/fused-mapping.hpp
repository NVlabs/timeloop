#pragma once

#include <memory>
#include <variant>

#include "isl-wrapper/isl-wrapper.hpp"
#include "workload/workload.hpp"

namespace mapping
{
using NodeID = size_t;
using BufferID = size_t;

class FusedMapping;

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
  std::optional<IslAff> begin;
  std::optional<IslAff> end;

  NodeID id;
  std::optional<NodeID> child;

  For(const NodeID& id,
      const std::string& iterator_name,
      const problem::Shape::FlattenedDimensionID& op_dim,
      std::optional<IslAff>&& begin = std::nullopt,
      std::optional<IslAff>&& end = std::nullopt);
};

struct ParFor
{
  std::string iterator_name;
  problem::Shape::FlattenedDimensionID op_dim;
  // TODO: missing spacetime_dim
  std::optional<IslAff> begin;
  std::optional<IslAff> end;

  NodeID id;
  NodeID child;

  ParFor(const NodeID& id,
         const std::string& iterator_name,
         const problem::Shape::FlattenedDimensionID& op_dim,
         std::optional<IslAff>&& begin = std::nullopt,
         std::optional<IslAff>&& end = std::nullopt);
};

struct Storage
{
  BufferID buffer;
  problem::Shape::DataSpaceID dspace;
  std::vector<std::pair<NodeID, IslMap>> logical_buf_occupancy;

  NodeID id;
  NodeID child;

  Storage(const NodeID& id,
        const BufferID& buffer,
        const problem::Shape::DataSpaceID& dspace);
};

struct Compute
{
  problem::KernelID kernel;
  /**
   * @brief An explicit tiling specifiction. E.g., [p_1, p_0] -> [4*p_1+p_0]
   * 
   * If given, bounds are not used to infer tiling map.
   */
  IslPwMultiAff tiling_spec;

  NodeID id;

  Compute(const NodeID& id,
          const problem::KernelID& kernel,
          const std::optional<IslPwMultiAff>&& tiling_spec);
};

struct Pipeline
{
  std::vector<NodeID> children;

  Pipeline(const NodeID& id);
};

using MappingNodeTypes
    = std::variant<Root, For, ParFor, Storage, Compute, Pipeline>;

class MappingNodeIterator
{
 public:
  MappingNodeIterator& operator++()
  {
    ++cur_;
    return *this;
  }

  bool operator!=(const MappingNodeIterator& other) const
  {
    return cur_ != other.cur_;
  }

  MappingNodeTypes& operator*()
  {
    return cur_->second;
  }

 private:
  friend FusedMapping;

  MappingNodeIterator(std::map<NodeID, MappingNodeTypes>::iterator iter);

 private:
  std::map<NodeID, MappingNodeTypes>::iterator cur_;
};

class FusedMapping
{
 public:
  FusedMapping()
  {
    nodes_.emplace(
      std::make_pair(0, MappingNodeTypes(std::in_place_type<Root>, 0)));
  }

  template<typename LoopT, typename... ArgsT>
  NodeID AddChild(NodeID parent_id, ArgsT... args)
  {
    auto [it, _] = nodes_.emplace(
      MappingNodeTypes(std::in_place_type<LoopT>, nodes_.size(), args...)
    );

    return it->first;
  }

  const MappingNodeTypes& NodeAt(const NodeID& node_id) const
  {
    return nodes_.at(node_id);
  }
  MappingNodeTypes& NodeAt(const NodeID& node_id)
  {
    return nodes_.at(node_id);
  }

  const Root& GetRoot() const
  {
    return std::get<Root>(nodes_.at(0));
  }

  MappingNodeIterator begin()
  {
    return MappingNodeIterator(nodes_.begin());
  }

  MappingNodeIterator end()
  {
    return MappingNodeIterator(nodes_.end());
  }

 private:
  std::map<NodeID, MappingNodeTypes> nodes_;
};

struct MappingPathNodeIterator
{
  MappingNodeTypes& operator*();
  bool operator==(const MappingPathNodeIterator& other) const;
  bool operator!=(const MappingPathNodeIterator& other) const;
  MappingNodeTypes& operator++();
};

struct MappingPath
{
  MappingPathNodeIterator begin() const;
  MappingPathNodeIterator end() const;
};

struct MappingPathsIterator
{
  MappingPath operator*();
  bool operator==(const MappingPathsIterator& other) const;
  bool operator!=(const MappingPathsIterator& other) const;
  MappingNodeTypes& operator++();
};

struct MappingPaths
{
  MappingPathsIterator begin() const;
  MappingPathsIterator end() const;
};

MappingPaths GetPaths(const FusedMapping& mapping);

}; // namespace mapping