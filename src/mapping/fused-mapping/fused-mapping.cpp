#include "mapping/fused-mapping.hpp"

namespace mapping
{

Root::Root(const NodeID& id) : id(id) {}

For::For(const NodeID& id,
         const std::string& iterator_name,
         const problem::DimensionId& op_dim,
         std::optional<size_t>&& begin,
         std::optional<size_t>&& end) :
  iterator_name(iterator_name),
  op_dim(op_dim),
  begin(begin),
  end(end),
  id(id)
{
}

For For::WithTileSize(const NodeID& id,
                      const std::string& iterator_name,
                      const problem::DimensionId& op_dim,
                      size_t tile_size)
{
  For node(id, iterator_name, op_dim);
  node.tile_size = tile_size;
  return node;
}

ParFor::ParFor(const NodeID& id,
               const std::string& iterator_name,
               const problem::DimensionId& op_dim,
               std::optional<size_t>&& begin,
               std::optional<size_t>&& end) :
  iterator_name(iterator_name),
  op_dim(op_dim),
  begin(begin),
  end(end),
  id(id)
{
}

ParFor ParFor::WithTileSize(const NodeID& id,
                            const std::string& iterator_name,
                            const problem::DimensionId& op_dim,
                            size_t tile_size)
{
  ParFor node(id, iterator_name, op_dim);
  node.tile_size = tile_size;
  return node;
}

Storage::Storage(const NodeID& id,
        const BufferID& buffer,
        const problem::DataSpaceId& dspace) :
  buffer(buffer), dspace(dspace), id(id) {}

Compute::Compute(const NodeID& id,
        const problem::EinsumId& einsum,
        const std::optional<isl::pw_multi_aff>&& tiling_spec) :
  kernel(einsum), tiling_spec(tiling_spec), id(id)
{
}

Pipeline::Pipeline(const NodeID& id) : id(id) {}
Sequential::Sequential(const NodeID& id) : id(id) {}

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