#include "mapping/fused-mapping.hpp"

namespace mapping
{

Root::Root(const NodeID& id) : id(id) {}

For::For(const NodeID& id,
         const std::string& iterator_name,
         const problem::Shape::FlattenedDimensionID& op_dim,
         std::optional<isl::aff>&& begin,
         std::optional<isl::aff>&& end) :
  iterator_name(iterator_name),
  op_dim(op_dim),
  begin(begin),
  end(end),
  id(id)
{
}

ParFor::ParFor(const NodeID& id,
               const std::string& iterator_name,
               const problem::Shape::FlattenedDimensionID& op_dim,
               std::optional<isl::aff>&& begin,
               std::optional<isl::aff>&& end) :
  iterator_name(iterator_name),
  op_dim(op_dim),
  begin(begin),
  end(end),
  id(id)
{
}

Storage::Storage(const NodeID& id,
        const BufferID& buffer,
        const problem::Shape::DataSpaceID& dspace) :
  buffer(buffer), dspace(dspace), id(id) {}

Compute::Compute(const NodeID& id,
        const problem::EinsumID& einsum,
        const std::optional<isl::pw_multi_aff>&& tiling_spec) :
  kernel(einsum), tiling_spec(tiling_spec), id(id)
{
}

Pipeline::Pipeline(const NodeID& id) 
{
  children.push_back(id);
}

FusedMappingNodeIterator& FusedMappingNodeIterator::operator++()
{
  ++cur_;
  return *this;
}

bool
FusedMappingNodeIterator::operator!=(
  const FusedMappingNodeIterator& other
) const
{
  return cur_ != other.cur_;
}

MappingNodeTypes& FusedMappingNodeIterator::operator*()
{
  return cur_->second;
}

FusedMappingNodeIterator::FusedMappingNodeIterator(
  std::map<NodeID, MappingNodeTypes>::iterator iter
) : cur_(iter)
{
}

FusedMapping::FusedMapping()
{
  nodes_.emplace(
    std::make_pair(0, MappingNodeTypes(std::in_place_type<Root>, 0)));
}

const MappingNodeTypes& FusedMapping::NodeAt(const NodeID& node_id) const
{
  return nodes_.at(node_id);
}

MappingNodeTypes& FusedMapping::NodeAt(const NodeID& node_id)
{
  return nodes_.at(node_id);
}

const Root& FusedMapping::GetRoot() const
{
  return std::get<Root>(nodes_.at(0));
}

Root& FusedMapping::GetRoot()
{
  return std::get<Root>(nodes_.at(0));
}

FusedMappingNodeIterator FusedMapping::begin()
{
  return FusedMappingNodeIterator(nodes_.begin());
}

FusedMappingNodeIterator FusedMapping::end()
{
  return FusedMappingNodeIterator(nodes_.end());
}

};