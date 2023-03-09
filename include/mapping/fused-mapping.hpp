#pragma once

#include <memory>
#include <variant>
#include <optional>
#include <isl/cpp.h>

#include "workload/workload.hpp"
#include "workload/fused-workload.hpp"

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
  std::optional<size_t> begin;
  std::optional<size_t> end;

  NodeID id;
  std::optional<NodeID> child;

  For(const NodeID& id,
      const std::string& iterator_name,
      const problem::Shape::FlattenedDimensionID& op_dim,
      std::optional<isl::aff>&& begin = std::nullopt,
      std::optional<isl::aff>&& end = std::nullopt);
};

struct ParFor
{
  std::string iterator_name;
  problem::Shape::FlattenedDimensionID op_dim;
  // TODO: missing spacetime_dim
  std::optional<size_t> begin;
  std::optional<size_t> end;

  NodeID id;
  NodeID child;

  ParFor(const NodeID& id,
         const std::string& iterator_name,
         const problem::Shape::FlattenedDimensionID& op_dim,
         std::optional<isl::aff>&& begin = std::nullopt,
         std::optional<isl::aff>&& end = std::nullopt);
};

struct Storage
{
  BufferID buffer;
  problem::Shape::DataSpaceID dspace;
  std::vector<std::pair<NodeID, isl::map>> logical_buf_occupancy;

  NodeID id;
  NodeID child;

  Storage(const NodeID& id,
          const BufferID& buffer,
          const problem::Shape::DataSpaceID& dspace);
};

struct Compute
{
  problem::EinsumID kernel;
  /**
   * @brief An explicit tiling specifiction. E.g., [p_1, p_0] -> [4*p_1+p_0]
   * 
   * If given, bounds are not used to infer tiling map.
   */
  isl::pw_multi_aff tiling_spec;

  NodeID id;

  Compute(const NodeID& id,
          const problem::EinsumID& einsum,
          const std::optional<isl::pw_multi_aff>&& tiling_spec);
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
  MappingNodeIterator& operator++();
  bool operator!=(const MappingNodeIterator& other) const;
  MappingNodeTypes& operator*();

 private:
  friend FusedMapping;

  MappingNodeIterator(std::map<NodeID, MappingNodeTypes>::iterator iter);

 private:
  std::map<NodeID, MappingNodeTypes>::iterator cur_;
};

class FusedMapping
{
 public:
  FusedMapping();

  template<typename LoopT, typename... ArgsT>
  NodeID AddChild(NodeID parent_id, ArgsT... args)
  {
    auto [it, _] = nodes_.emplace(
      MappingNodeTypes(std::in_place_type<LoopT>, nodes_.size(), args...)
    );

    return it->first;
  }

  const MappingNodeTypes& NodeAt(const NodeID& node_id) const;
  MappingNodeTypes& NodeAt(const NodeID& node_id);

  const Root& GetRoot() const;

  MappingNodeIterator begin();
  MappingNodeIterator end();

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